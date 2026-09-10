#!/usr/bin/env python3
"""ORBA on 8 layer bands. Find Δrefusal / ΔKL. In-memory, no merge."""

from __future__ import annotations

import json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline
from abliterix.scriptlib import setup_io

setup_io()
import torch
from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.eval.detector import RefusalDetector
from abliterix.eval.scorer import TrialScorer
from abliterix.settings import AbliterixConfig
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_layer_bands.json")
VINDEX = 50.79667354766495
O_MAX = 10.0


def _cfg():
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["layer_bands", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _band(center: float) -> dict[str, SteeringProfile]:
    # min=max → only layers within distance of center get weight.
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=O_MAX,
            max_weight_position=float(center),
            min_weight=O_MAX,
            min_weight_distance=4.2,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.5,
            max_weight_position=float(center),
            min_weight=0.5,
            min_weight_distance=4.2,
        ),
    }


def main():
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {"started_utc": datetime.now(timezone.utc).isoformat(), "save_pretrained": False, "points": []}
    engine = SteeringEngine(cfg)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    centers = [3.5, 11.5, 19.5, 27.5, 35.5, 43.5, 51.5, 59.5]
    for c in centers:
        engine.restore_baseline()
        apply_steering(
            engine, cache["vectors"], VINDEX, _band(c), cfg,
            benign_states=cache.get("benign_states"),
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        tag = f"band_{int(c-3.5):02d}_{int(c+4.5):02d}"
        row = {
            "tag": tag,
            "center": c,
            "keyword_refusals": int(refusals),
            "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "delta_refusals": 100 - int(refusals),
            "kl_per_refusal_drop": (kl / max(1, 100 - int(refusals))),
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(
            f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  drop={row['delta_refusals']} "
            f"kl/drop={row['kl_per_refusal_drop']:.4f}",
            flush=True,
        )
        payload["points"].append(row)
        text = json.dumps(payload, indent=2)
        OUT.write_text(text)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        (SCRATCH / "qwen38_layer_bands.json").write_text(text)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            OUT.write_text(json.dumps(payload, indent=2))
            print(f"HIT {tag}", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload, indent=2))
    (SCRATCH / "qwen38_layer_bands.json").write_text(json.dumps(payload, indent=2))
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
