#!/usr/bin/env python3
"""OrcaRouter-style Arditi edit: mergeable W' = W − r(rᵀW).

Single refusal direction from layer 38 (round(0.6×64)), massive-activation
masked mean-diff, then float32 projection on every residual-writing matrix
plus embed_tokens. Vision tower left untouched.
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
from torch import Tensor  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringMode  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
CACHE = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_arditi_orcarouter.json")
LAYER = 38  # round(0.6 * 64)
# s=1.0 already scored 72/100 @ 0.1049 on our meter; search the
# overshoot side to drop refusals toward 8–12.
STRENGTHS = (1.10, 1.18, 1.22)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["arditi_orca", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    return cfg


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_arditi_orcarouter.json").write_text(text, encoding="utf-8")


def massive_masked_direction(harm: Tensor, benign: Tensor, q: float = 0.99) -> Tensor:
    """Mean-diff at one layer with massive-activation dims zeroed."""
    both = torch.cat([harm, benign], dim=0).float()
    peak = both.abs().amax(dim=0)
    keep = peak <= torch.quantile(peak, q)
    h = harm.float().clone()
    b = benign.float().clone()
    h[:, ~keep] = 0
    b[:, ~keep] = 0
    r = h.mean(0) - b.mean(0)
    r = r / r.norm().clamp(min=1e-8)
    print(
        f"layer-38 r: kept {int(keep.sum())}/{keep.numel()} dims, "
        f"|r|_max={float(r.abs().max()):.4f}",
        flush=True,
    )
    return r


def _is_residual_writer(name: str) -> bool:
    if "visual." in name or "vision" in name:
        return False
    if not name.endswith("weight"):
        return False
    if "lora_" in name or "base_layer" in name:
        return False
    return any(
        s in name
        for s in (
            ".o_proj.weight",
            ".out_proj.weight",
            ".down_proj.weight",
            "embed_tokens.weight",
        )
    )


def _project(weight: Tensor, r: Tensor, strength: float) -> None:
    W = weight.data.to(torch.float32)
    v = r.to(device=W.device, dtype=torch.float32)
    v = v / v.norm().clamp(min=1e-8)
    out_f, in_f = W.shape
    if v.numel() == out_f:
        # W' = (I − s rrᵀ) W
        W_new = W - strength * v.unsqueeze(1) * (v @ W).unsqueeze(0)
    elif v.numel() == in_f:
        # W' = W (I − s rrᵀ)  (embed row space)
        W_new = W - strength * (W @ v).unsqueeze(1) * v.unsqueeze(0)
    else:
        return
    weight.data.copy_(W_new.to(weight.dtype))
    del W, W_new


def _collect(model) -> list[tuple[str, Tensor]]:
    out = []
    for n, p in model.named_parameters():
        if p.ndim == 2 and _is_residual_writer(n):
            out.append((n, p))
    return out


def _score(engine, cfg, tag: str) -> dict:
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    with torch.no_grad():
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
        flush=True,
    )
    return row


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": "Arditi k=1 layer38 massive-masked, all residual writers + embed",
        "layer": LAYER,
        "points": [],
    }
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    # states[:, 0] = embed; states[:, 1] = layer 0 → layer 38 is index 39
    idx = LAYER + 1
    r = massive_masked_direction(
        cache["target_states"][:, idx, :],
        cache["benign_states"][:, idx, :],
    )
    del cache

    print("loading original (no LoRA)...", flush=True)
    engine = SteeringEngine(cfg)
    writers = _collect(engine.model)
    print(f"residual writers {len(writers)}", flush=True)
    for n, _p in writers:
        print(f"  {n}", flush=True)
    originals = {n: p.detach().clone() for n, p in writers}

    hit = None
    for s in STRENGTHS:
        for n, p in writers:
            p.data.copy_(originals[n])
            _project(p, r, s)
        row = _score(engine, cfg, f"arditi_l38_s{s:.2f}")
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            hit = row
            break

    if hit is None:
        payload["named_candidate"] = None
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("sweep finished, no in-budget point", flush=True)
        return

    print(f"HIT {hit['tag']} — saving merged {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in (
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = Path(ORIGINAL) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    payload["named_candidate"] = hit["tag"]
    payload["merged"] = str(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
