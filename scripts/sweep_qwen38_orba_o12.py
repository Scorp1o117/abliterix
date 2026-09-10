#!/usr/bin/env python3
"""In-memory ORBA o12.5–13.2. o12.2 already 12/100 @ 0.210. No merge."""

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
OUT = Path("logs/qwen38_orba_o12.json")
T24_POS = 49.611242819678694
VINDEX = 50.79667354766495


def _cfg():
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["orba_o12", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _prof(o):
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o),
            max_weight_position=T24_POS,
            min_weight=max(0.0, float(o) * 0.5),
            min_weight_distance=5.105,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.37,
            max_weight_position=37.89,
            min_weight=0.25,
            min_weight_distance=30.46,
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
    for o in (12.5, 12.8, 13.2):
        engine.restore_baseline()
        apply_steering(
            engine, cache["vectors"], VINDEX, _prof(o), cfg,
            benign_states=cache.get("benign_states"),
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": f"orba_o{o:.1f}",
            "o_proj": o,
            "keyword_refusals": int(refusals),
            "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(f"SCORE {row['tag']}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        text = json.dumps(payload, indent=2)
        OUT.write_text(text)
        SCRATCH.mkdir(parents=True, exist_ok=True)
        (SCRATCH / "qwen38_orba_o12.json").write_text(text)
        if row["in_budget"]:
            payload["named_candidate"] = row["tag"]
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            OUT.write_text(json.dumps(payload, indent=2))
            print(f"HIT {row['tag']} (not saving)", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    OUT.write_text(json.dumps(payload, indent=2))
    (SCRATCH / "qwen38_orba_o12.json").write_text(json.dumps(payload, indent=2))
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
