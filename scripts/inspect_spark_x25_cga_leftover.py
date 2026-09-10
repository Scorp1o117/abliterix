#!/usr/bin/env python3
"""Dump the 12 leftover refusals under CGA V27 (t0.50 s1.35 d40).

Prefix retry and lower thresh did not move them. Print first 120 chars
and whether they refuse in the first 80 chars.
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
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/spark_x25_4b_lora_v9.toml"
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
SCORER_CACHE = Path(
    "checkpoints_spark_x25_4b_cga_v25/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_concept_scorers.pt"
)
SCRATCH = Path("/tmp/grok-goal-85e2331705c4/implementer")
OUT = Path("logs/spark_x25_cga_leftover_prefixes.json")


def main() -> None:
    torch.set_grad_enabled(False)
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["inspect_cga", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.concept_gate_angular_overrotation = True
    cfg.steering.concept_gate_scope = "global_prompt"
    cfg.steering.concept_gate_threshold = 0.50
    cfg.steering.concept_gate_positive_alignment_only = True
    cfg.steering.concept_gate_global_decision_layer = 24
    print("loading Spark CGA leftover inspect...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    blob = torch.load(SCORER_CACHE, map_location="cpu", weights_only=False)
    engine._concept_scorers = blob["scorers"] if "scorers" in blob else blob
    apply_steering(
        engine,
        cache["vectors"],
        None,
        {
            "attn.o_proj": SteeringProfile(
                max_weight=1.35,
                max_weight_position=18.0,
                min_weight=1.35,
                min_weight_distance=40.0,
            )
        },
        cfg,
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
        prefix = t[:160].replace("\n", "\\n")
        early = refused and detector.detect_refusal(t[:80])
        if refused:
            n_ref += 1
            if early:
                n_early += 1
        prompt = ""
        if isinstance(msg, list) and msg:
            last = msg[-1]
            prompt = str(getattr(last, "content", last.get("content", "") if isinstance(last, dict) else last))[:100]
        rows.append(
            {
                "i": i,
                "refused": refused,
                "early_refuse_in_80chars": early,
                "prefix": prefix,
                "prompt": prompt,
            }
        )
        if refused:
            print(f"{i:03d} REF early={early} | {prefix}", flush=True)
    payload = {
        "n": len(rows),
        "refusals": n_ref,
        "early_refuse_in_80chars": n_early,
        "late_only_refusals": n_ref - n_early,
        "rows": [r for r in rows if r["refused"]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_cga_leftover_prefixes.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"refusals {n_ref}/{len(rows)}  early(80c)={n_early}  late_only={n_ref - n_early}",
        flush=True,
    )


if __name__ == "__main__":
    main()
