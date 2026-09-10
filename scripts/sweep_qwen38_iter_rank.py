#!/usr/bin/env python3
"""Score rank-1/2/3 slices of the iterative subspace at small direct strengths.

The 20-trial search applied all 8 directions at once and jumped to KL 0.4–1.1
as soon as refusals dropped. This sweep uses the cached
``checkpoints_qwen38_27b_iter`` vectors and peels only the first k directions.
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

CONFIG = "configs/qwen38_27b_rocm_iter.toml"
VECTORS = Path(
    "checkpoints_qwen38_27b_iter/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT = Path("logs/qwen38_iter_rank_sweep.json")
RANKS = (1, 2, 3)
STRENGTHS = (0.50, 0.80, 1.00, 1.20, 1.50)


def _load_cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_iter_rank", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    return cfg


def _profiles(strength: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=50.0,
            min_weight=float(max(0.05, strength * 0.25)),
            min_weight_distance=20.0,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.0,
            max_weight_position=50.0,
            min_weight=0.0,
            min_weight_distance=20.0,
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_iter_rank_sweep.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _load_cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "vectors": str(VECTORS),
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "points": [],
    }
    print("loading engine...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    all_vectors = cache["vectors"]
    print(f"cached vectors {tuple(all_vectors.shape)}", flush=True)

    for rank in RANKS:
        sliced = all_vectors[:rank]
        for strength in STRENGTHS:
            engine.restore_baseline()
            apply_steering(
                engine,
                sliced,
                None,
                _profiles(strength),
                cfg,
            )
            tag = f"iter_rank{rank}_o{strength:.2f}"
            kl = float(scorer.measure_kl_divergence(engine))
            refusals, _ = scorer.measure_compliance_objective(engine)
            n = len(scorer.target_msgs)
            row = {
                "tag": tag,
                "rank": rank,
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
