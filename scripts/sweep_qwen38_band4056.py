#!/usr/bin/env python3
"""ORBA on the two efficient bands 40–56 as one envelope. No merge.

48–56 o=11: 17@0.160 (overshoots if stronger). 40–48 alone: 22-drop @ 0.064.
Cover 40–56 together.
"""

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
OUT = Path("logs/qwen38_band4056.json")
VINDEX = 50.79667354766495


def _cfg():
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["band4056", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _prof(o, down, pos, dist, flat):
    mn = float(o) if flat else max(0.0, float(o) * 0.45)
    dn = float(down) if flat else max(0.0, float(down) * 0.45)
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o), max_weight_position=float(pos),
            min_weight=mn, min_weight_distance=float(dist),
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=float(down), max_weight_position=float(pos),
            min_weight=dn, min_weight_distance=float(dist),
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
    jobs = [
        ("p48_d8_o9", 9.0, 0.45, 48.0, 8.0, False),
        ("p48_d8_o10", 10.0, 0.45, 48.0, 8.0, False),
        ("p48_d8_o11", 11.0, 0.5, 48.0, 8.0, False),
        ("p48_d8_o10flat", 10.0, 0.5, 48.0, 8.0, True),
        ("p46_d10_o10", 10.0, 0.45, 46.0, 10.0, False),
    ]
    for tag, o, down, pos, dist, flat in jobs:
        engine.restore_baseline()
        apply_steering(
            engine, cache["vectors"], VINDEX, _prof(o, down, pos, dist, flat),
            cfg, benign_states=cache.get("benign_states"),
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": tag, "o_proj": o, "down": down, "pos": pos, "dist": dist, "flat": flat,
            "keyword_refusals": int(refusals), "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        text = json.dumps(payload, indent=2)
        OUT.write_text(text)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        (SCRATCH / "qwen38_band4056.json").write_text(text)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            OUT.write_text(json.dumps(payload, indent=2))
            print(f"HIT {tag} (not saving)", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload, indent=2))
    (SCRATCH / "qwen38_band4056.json").write_text(json.dumps(payload, indent=2))
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
