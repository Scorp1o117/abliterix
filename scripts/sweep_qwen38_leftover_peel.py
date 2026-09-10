#!/usr/bin/env python3
"""Second-direction peel: only the prompts that still refuse after ORBA o8.7.

Earlier residual extracts averaged ALL harmful states, so leftover refusals
were drowned. Here we generate on train[:400], keep keyword-refusals, and
mean-diff those vs benign — then peel r2 on top of o8.7. In-memory, no merge.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline  # noqa: E402
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402
from abliterix.vectors import compute_configured_steering_vectors  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_leftover_peel.json")
T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982
VINDEX = 50.79667354766495
O87 = 8.7


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["leftover_peel", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _orba87() -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=O87,
            max_weight_position=T24_POS,
            min_weight=O87 * 0.5,
            min_weight_distance=T24_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.37,
            max_weight_position=37.89,
            min_weight=0.25,
            min_weight_distance=30.46,
        ),
    }


def _band(o: float, dist: float = T24_DIST) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o),
            max_weight_position=T24_POS,
            min_weight=max(0.0, float(o) * 0.4),
            min_weight_distance=float(dist),
        )
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_leftover_peel.json").write_text(text, encoding="utf-8")


def _snapshot(engine) -> dict[str, torch.Tensor]:
    snap = {}
    for n, p in engine.model.named_parameters():
        if p.ndim == 2 and any(s in n for s in (".o_proj.weight", ".out_proj.weight", ".down_proj.weight")):
            if "lora_" in n or "base_layer" in n:
                continue
            snap[n] = p.detach().clone()
    return snap


def _restore_snap(engine, snap: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for n, p in engine.model.named_parameters():
            if n in snap:
                p.data.copy_(snap[n])


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "save_pretrained": False,
        "base": "ORBA o8.7",
        "points": [],
    }
    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    apply_steering(
        engine,
        cache["vectors"],
        VINDEX,
        _orba87(),
        cfg,
        benign_states=cache.get("benign_states"),
    )
    base_snap = _snapshot(engine)
    print(f"o8.7 snap tensors {len(base_snap)}", flush=True)

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    kl0 = float(scorer.measure_kl_divergence(engine))
    ref0, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    print(f"SCORE o8.7_base: {ref0}/{n} @ {kl0:.4f}", flush=True)
    payload["points"].append(
        {
            "tag": "o8.7_base",
            "keyword_refusals": int(ref0),
            "n": n,
            "full_distribution_kl_3token_vs_original": kl0,
            "in_budget": bool(8 <= int(ref0) <= 12 and kl0 <= 0.1),
        }
    )
    _dump(payload)

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[:400]
    train_b = load_prompt_dataset(cfg, cfg.benign_prompts)[:200]
    print(f"generating train harmful n={len(train_h)} under o8.7...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover = [m for m, t in zip(train_h, texts) if detector.detect_refusal(t or "")]
    print(f"leftover refusals {len(leftover)}/{len(train_h)}", flush=True)
    payload["leftover_n"] = len(leftover)
    if len(leftover) < 8:
        print("too few leftover refusals to estimate r2", flush=True)
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        return

    print("extracting leftover vs benign residuals...", flush=True)
    tgt = engine.extract_hidden_states_batched(leftover)
    beni = engine.extract_hidden_states_batched(train_b)
    r2 = compute_configured_steering_vectors(beni, tgt, cfg)
    print(f"r2 vectors {tuple(r2.shape)}", flush=True)
    # cosine vs original mean-diff at layer 50
    v1 = cache["vectors"]
    if v1.ndim == 2 and r2.ndim == 2:
        i = min(51, v1.shape[0] - 1, r2.shape[0] - 1)
        cos = float(torch.nn.functional.cosine_similarity(v1[i].float(), r2[i].float(), dim=0))
        print(f"cos(r_mean, r2) layer~50 = {cos:.4f}", flush=True)
        payload["cos_r1_r2"] = cos

    cfg.steering.direct_transform = DirectTransform.ORBA
    for o, dist in ((0.8, T24_DIST), (1.6, T24_DIST), (2.4, T24_DIST), (3.5, T24_DIST), (2.4, 10.0), (4.5, T24_DIST)):
        _restore_snap(engine, base_snap)
        apply_steering(
            engine,
            r2,
            None,
            _band(o, dist),
            cfg,
            benign_states=beni,
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        tag = f"o8.7+r2_o{o:.1f}_d{dist:.0f}"
        row = {
            "tag": tag,
            "keyword_refusals": int(refusals),
            "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
            "r2_o": o,
            "r2_dist": dist,
        }
        print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {tag} (not saving)", flush=True)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
