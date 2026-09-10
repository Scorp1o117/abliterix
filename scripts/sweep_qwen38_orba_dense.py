#!/usr/bin/env python3
"""In-memory ORBA densify on the 8–10 o_proj band. Does not write 52G.

Known: o8.0 31@0.110, o9.0 21@0.131. Fill the gap and a bit past 9
toward ~10–16 refusals without Arditi's KL cliff.
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
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_orba_dense.json")
T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982
VINDEX = 50.79667354766495
STRENGTHS = (8.4, 8.7, 9.2, 9.5, 10.0, 10.5)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["orba_dense", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _profiles(o_max: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=T24_POS,
            min_weight=max(0.0, float(o_max) * 0.5),
            min_weight_distance=T24_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.37,
            max_weight_position=37.89,
            min_weight=0.25,
            min_weight_distance=30.46,
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_orba_dense.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "save_pretrained": False,
        "points": [],
    }
    print("loading original (in-memory only, no merge)...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    for o in STRENGTHS:
        engine.restore_baseline()
        apply_steering(
            engine,
            cache["vectors"],
            VINDEX,
            _profiles(o),
            cfg,
            benign_states=cache.get("benign_states"),
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": f"orba_o{o:.1f}",
            "o_proj": o,
            "keyword_refusals": int(refusals),
            "n": int(n),
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(
            f"SCORE {row['tag']}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
            flush=True,
        )
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = row["tag"]
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {row['tag']} (not saving weights)", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
