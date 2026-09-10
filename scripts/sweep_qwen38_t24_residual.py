#!/usr/bin/env python3
"""Re-extract a refusal direction on merged t24, then ORBA/direct peel.

Peeling original mean-diff on t24 undid the bake (90–100 refusals). The
residual direction must be measured on t24 itself.
"""

from __future__ import annotations

import json
import os
import shutil
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
T24 = "/run/media/s117/OS/Models/Qwen3.8-27B-t24"
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_t24_residual_sweep.json")
N = 96
T24_POS = 49.611242819678694
T24_DIST = 12.0  # wider than LoRA envelope for residual


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["t24_residual", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.model.model_id = T24
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.STANDARD
    cfg.steering.vector_method = cfg.steering.vector_method
    return cfg


def _band(o_max: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=T24_POS,
            min_weight=max(0.0, float(o_max) * 0.35),
            min_weight_distance=T24_DIST,
        )
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_t24_residual_sweep.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base": T24,
        "points": [],
    }
    print("loading t24...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    kl0 = float(scorer.measure_kl_divergence(engine))
    ref0, _ = scorer.measure_compliance_objective(engine)
    print(f"SCORE t24_base: {ref0}/{len(scorer.target_msgs)} @ {kl0:.4f}", flush=True)
    payload["points"].append(
        {
            "tag": "t24_base",
            "keyword_refusals": int(ref0),
            "n": len(scorer.target_msgs),
            "full_distribution_kl_3token_vs_original": kl0,
            "in_budget": bool(8 <= int(ref0) <= 12 and kl0 <= 0.1),
        }
    )
    _dump(payload)

    harm = load_prompt_dataset(cfg, cfg.target_prompts)[:N]
    beni = load_prompt_dataset(cfg, cfg.benign_prompts)[:N]
    print(f"extracting residual states n={N}+{N}...", flush=True)
    target_states = engine.extract_hidden_states_batched(harm)
    benign_states = engine.extract_hidden_states_batched(beni)
    vectors = compute_configured_steering_vectors(benign_states, target_states, cfg)
    print(f"residual vectors {tuple(vectors.shape)}", flush=True)

    jobs = [
        ("std", DirectTransform.STANDARD, 0.8),
        ("std", DirectTransform.STANDARD, 1.5),
        ("std", DirectTransform.STANDARD, 2.4),
        ("orba", DirectTransform.ORBA, 1.2),
        ("orba", DirectTransform.ORBA, 2.0),
        ("orba", DirectTransform.ORBA, 3.0),
    ]
    hit = None
    for kind, transform, o in jobs:
        engine.restore_baseline()
        cfg.steering.direct_transform = transform
        apply_steering(
            engine,
            vectors,
            None,
            _band(o),
            cfg,
            benign_states=benign_states if transform == DirectTransform.ORBA else None,
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        tag = f"t24_resid_{kind}_{o:.2f}"
        row = {
            "tag": tag,
            "keyword_refusals": int(refusals),
            "n": int(n),
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            hit = tag
            payload["named_candidate"] = tag
            break

    if hit is None:
        payload["named_candidate"] = None
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("sweep finished, no in-budget point", flush=True)
        return

    print(f"HIT {hit} — saving merged...", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in ("chat_template.jinja", "generation_config.json"):
        src = Path(ORIGINAL) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    payload["merged"] = str(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
