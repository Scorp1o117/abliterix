#!/usr/bin/env python3
"""Apply the learned RDO direction as a flat direct projection.

RDO+LoRA at pocket strengths (o≈4–6) all 30/30-pruned: the RDO vector is
nearly orthogonal to mean-diff (global cos ~0.11). The paper applies r as
h ← h − (h·r̂)r̂ at every layer, i.e. steering_mode=direct with a flat
profile. Sweep that magnitude vs original 3-token KL + keyword refusals.
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

from eval_qwen38_vs_original import (  # noqa: E402
    POCKET_BASELINE,
    _inject_original_baseline,
)
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_rdo.toml"
VECTORS = Path(
    "checkpoints_qwen38_27b_rdo/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT = Path("logs/qwen38_rdo_direct_sweep.json")
# Paper applies unit r at every layer. Strength is the projection scale.
STRENGTHS = (0.30, 0.60, 0.90, 1.20, 1.60, 2.20, 3.00)


def _load_cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_rdo_direct", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    return cfg


def _flat(strength: float) -> dict[str, SteeringProfile]:
    # Cover all 64 layers: position mid, distance > 32.
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=32.0,
            min_weight=float(strength),
            min_weight_distance=40.0,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.0,
            max_weight_position=32.0,
            min_weight=0.0,
            min_weight_distance=40.0,
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_rdo_direct_sweep.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _load_cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "vectors": str(VECTORS),
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "apply": "direct flat o_proj, down=0, all layers",
        "points": [],
    }
    print("loading engine...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    vectors = cache["vectors"]
    print(f"RDO vectors {tuple(vectors.shape)}", flush=True)

    for strength in STRENGTHS:
        engine.restore_baseline()
        apply_steering(engine, vectors, None, _flat(strength), cfg)
        tag = f"rdo_direct_o{strength:.2f}"
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": tag,
            "o_proj": strength,
            "keyword_refusals": int(refusals),
            "n": int(n),
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(
            f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
            flush=True,
        )
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {tag}", flush=True)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
