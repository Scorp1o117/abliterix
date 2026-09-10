#!/usr/bin/env python3
"""All-layer concept-gated angular with lower thresholds.

t24-band prompt-latch at 0.5 never opened (98/100 @ 0.000). Scorers exist
for 64/64 layers. This sweep installs gated hooks on every decoder block
and tries token + prompt + global scopes at 0.20–0.50.
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

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
POCKET_VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCORER_CACHE = Path(
    "checkpoints_qwen38_27b_cga/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_concept_scorers.pt"
)
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT = Path("logs/qwen38_cga_all_sweep.json")


def _load_cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_cga_all", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.concept_gate_angular_overrotation = False
    cfg.steering.concept_gate_positive_alignment_only = True
    # -1 is a sentinel; apply_steering requires a real trained layer index.
    cfg.steering.concept_gate_global_decision_layer = 32
    return cfg


def _all_layers(strength: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=32.0,
            min_weight=float(strength),
            min_weight_distance=40.0,
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_all_sweep.json").write_text(text, encoding="utf-8")


def _score(scorer: TrialScorer, engine: SteeringEngine, tag: str, extra: dict) -> dict:
    kl = float(scorer.measure_kl_divergence(engine))
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        **extra,
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
        flush=True,
    )
    return row


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _load_cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "vectors": str(POCKET_VECTORS),
        "scorers": str(SCORER_CACHE),
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "apply": "all-layer concept-gated angular",
        "points": [],
    }
    print("loading engine (concept_gated_angular)...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    vectors = torch.load(POCKET_VECTORS, map_location="cpu", weights_only=False)["vectors"]
    blob = torch.load(SCORER_CACHE, map_location="cpu", weights_only=False)
    engine._concept_scorers = blob["scorers"] if isinstance(blob, dict) and "scorers" in blob else blob
    print(f"scorers={len(engine._concept_scorers)} vectors={tuple(vectors.shape)}", flush=True)

    jobs = [
        ("token", 0.20, 1.00),
        ("token", 0.35, 1.00),
        ("token", 0.50, 1.00),
        ("prompt", 0.20, 1.00),
        ("prompt", 0.35, 1.00),
        ("prompt", 0.50, 1.00),
        ("global_prompt", 0.35, 1.00),
    ]

    for scope, thresh, strength in jobs:
        engine.restore_baseline()
        engine._concept_scorers = (
            blob["scorers"] if isinstance(blob, dict) and "scorers" in blob else blob
        )
        cfg.steering.concept_gate_scope = scope
        cfg.steering.concept_gate_threshold = float(thresh)
        apply_steering(engine, vectors, None, _all_layers(strength), cfg)
        n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
        tag = f"cga_all_{scope}_t{thresh:.2f}_{strength:.2f}"
        print(f"hooks={n_hooks} {tag}", flush=True)
        row = _score(
            scorer,
            engine,
            tag,
            {
                "scope": scope,
                "threshold": thresh,
                "strength": strength,
                "n_hooks": n_hooks,
            },
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
