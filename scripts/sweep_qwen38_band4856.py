#!/usr/bin/env python3
"""ORBA only on layers ~48–56, raise strength toward 8–12. No merge.

Band 48–56 at o=10: 30/100 @ 0.135 (best kl-per-drop). t24-band o12.2
needed 0.210 to reach 12 refusals. If efficiency holds, this band at
higher o should hit ~10 refusals nearer KL 0.17.
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
OUT = Path("logs/qwen38_band4856.json")
VINDEX = 50.79667354766495
CENTER = 51.5
DIST = 4.2


def _cfg():
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["band4856", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _prof(o, down):
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o), max_weight_position=CENTER,
            min_weight=float(o), min_weight_distance=DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=float(down), max_weight_position=CENTER,
            min_weight=float(down), min_weight_distance=DIST,
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
    jobs = [(11.0, 0.5), (12.0, 0.5), (13.0, 0.5), (14.5, 0.6), (12.0, 0.9)]
    for o, down in jobs:
        engine.restore_baseline()
        apply_steering(engine, cache["vectors"], VINDEX, _prof(o, down), cfg, benign_states=cache.get("benign_states"))
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        tag = f"b4856_o{o:.1f}_d{down:.1f}"
        row = {
            "tag": tag, "o_proj": o, "down": down,
            "keyword_refusals": int(refusals), "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        text = json.dumps(payload, indent=2)
        OUT.write_text(text)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        (SCRATCH / "qwen38_band4856.json").write_text(text)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            OUT.write_text(json.dumps(payload, indent=2))
            print(f"HIT {tag} (not saving)", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload, indent=2))
    (SCRATCH / "qwen38_band4856.json").write_text(json.dumps(payload, indent=2))
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
