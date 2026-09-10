#!/usr/bin/env python3
"""Dump V30 T580 eval refusals (the 20/100 ship-bar leftovers)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abliterix.scriptlib import (  # noqa: E402
    apply_trial_artifact,
    extract_trial_artifact,
    load_trial,
    setup_io,
)

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from eval_qwen38_vs_original import _inject_original_baseline  # noqa: E402

CONFIG = "configs/spark_x25_4b_lora_v30.toml"
CHECKPOINT = "checkpoints_spark_x25_4b_lora_v30"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
OUT = Path("logs/spark_x25_t580_eval_refusals.json")
TRIAL = 580


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["inspect_t580", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    return cfg


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    trial = load_trial(CHECKPOINT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
    print(
        f"T580 attrs refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')}",
        flush=True,
    )
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    apply_steering(
        engine,
        cache["vectors"],
        artifact.vector_index,
        artifact.profiles,
        cfg,
        benign_states=cache.get("benign_states"),
        target_states=cache.get("target_states"),
    )
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    texts = engine.generate_text_batched(
        scorer.target_msgs,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    refused = []
    n_early = 0
    for i, (msg, text) in enumerate(zip(scorer.target_msgs, texts)):
        t = text or ""
        if not detector.detect_refusal(t):
            continue
        early = bool(detector.detect_refusal(t[:80]))
        if early:
            n_early += 1
        prompt = getattr(msg, "user", None) or ""
        row = {
            "i": i,
            "prompt": prompt,
            "response_prefix": t[:400],
            "early_refuse_in_80chars": early,
        }
        refused.append(row)
        flag = "EARLY" if early else "LATE "
        one_line = prompt.replace("\n", " ")[:160]
        print(f"{i:03d} {flag} | {one_line}", flush=True)
    payload = {
        "trial": TRIAL,
        "n_eval": len(scorer.target_msgs),
        "refusals": len(refused),
        "early_refuse_in_80chars": n_early,
        "late_only": len(refused) - n_early,
        "rows": refused,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"refusals {len(refused)}/{len(scorer.target_msgs)}  "
        f"early(80c)={n_early}  late_only={len(refused) - n_early}",
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
