#!/usr/bin/env python3
"""Dump T32 leftover eval prefixes to see if 3-token KL is the bottleneck.

If remaining refusals already start like the original (Sorry/I cannot),
changing them must move 3-token KL. If they start helpful and refuse later,
a later-token edit could drop refusals without paying first-3-token KL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_qwen38_vs_original import _inject_original_baseline  # noqa: E402
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

CONFIG = "configs/spark_x25_4b_lora_v9.toml"
CHECKPOINT = "checkpoints_spark_x25_4b_lora_v9"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
SCRATCH = Path("/tmp/grok-goal-85e2331705c4/implementer")
OUT = Path("logs/spark_x25_t32_leftover_prefixes.json")


def main() -> None:
    torch.set_grad_enabled(False)
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["inspect_t32", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    trial = load_trial(CHECKPOINT, MODEL, 32)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
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
    rows = []
    n_ref = 0
    n_early = 0
    for i, (msg, text) in enumerate(zip(scorer.target_msgs, texts)):
        t = text or ""
        refused = bool(detector.detect_refusal(t))
        prefix = t[:120].replace("\n", "\\n")
        early = refused and detector.detect_refusal(t[:80])
        if refused:
            n_ref += 1
            if early:
                n_early += 1
        prompt = ""
        if isinstance(msg, list) and msg:
            prompt = str(msg[-1].get("content", ""))[:80]
        elif isinstance(msg, dict):
            prompt = str(msg.get("content", ""))[:80]
        rows.append(
            {
                "i": i,
                "refused": refused,
                "early_refuse_in_80chars": early,
                "prefix": prefix,
                "prompt": prompt,
            }
        )
        flag = "REF" if refused else "ok "
        print(f"{i:03d} {flag} early={early} | {prefix}", flush=True)
    payload = {
        "n": len(rows),
        "refusals": n_ref,
        "early_refuse_in_80chars": n_early,
        "late_only_refusals": n_ref - n_early,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_t32_leftover_prefixes.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"refusals {n_ref}/{len(rows)}  early(80c)={n_early}  late_only={n_ref - n_early}",
        flush=True,
    )


if __name__ == "__main__":
    main()
