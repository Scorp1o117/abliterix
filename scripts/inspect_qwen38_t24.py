#!/usr/bin/env python3
"""Dump pocket trial 24 eval replies + keyword flags for manual audit."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from abliterix.scriptlib import (
    apply_trial_artifact,
    compute_trial_vectors,
    extract_trial_artifact,
    load_trial,
    setup_io,
)

setup_io()

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
CKPT = "checkpoints_qwen38_27b_pocket"
MODEL = "/run/media/s117/OS/Models/Qwen3.8-27B"
TRIAL = 24
OUT = Path("logs/qwen38_t24_inspect.json")


def main() -> None:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = [
        "inspect_qwen38_t24",
        "--config",
        CONFIG,
        "--seed",
        "117",
        "--non-interactive",
    ]

    import torch

    torch.set_grad_enabled(False)

    from abliterix.core.engine import SteeringEngine
    from abliterix.core.steering import apply_steering
    from abliterix.data import load_prompt_dataset
    from abliterix.eval.detector import RefusalDetector
    from abliterix.settings import AbliterixConfig
    from abliterix.util import flush_memory

    trial = load_trial(CKPT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    print(
        f"Trial {TRIAL}: recorded refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')}"
    )

    config = AbliterixConfig()
    apply_trial_artifact(config, artifact)
    config.inference.batch_size = 128
    config.inference.max_batch_size = 128
    config.detection.llm_judge = False

    engine = SteeringEngine(config)
    print("Extracting / loading steering residuals...")
    benign = load_prompt_dataset(config, config.benign_prompts)
    target = load_prompt_dataset(config, config.target_prompts)
    benign_states = engine.extract_hidden_states_batched(benign)
    target_states = engine.extract_hidden_states_batched(target)
    vectors = compute_trial_vectors(artifact, benign_states, target_states, config)
    engine.response_prefix = ""
    del benign, target
    flush_memory()

    eval_prompts = load_prompt_dataset(config, config.target_eval_prompts)
    print(f"Eval prompts: {len(eval_prompts)}")

    engine.restore_baseline()
    apply_steering(
        engine,
        vectors,
        artifact.vector_index,
        artifact.profiles,
        config,
        routing_config=artifact.routing,
        benign_states=benign_states,
        target_states=target_states,
    )
    del benign_states, target_states
    flush_memory()

    print("Generating...")
    responses = engine.generate_text_batched(
        eval_prompts,
        skip_special_tokens=True,
        max_new_tokens=config.inference.max_gen_tokens,
        min_new_tokens=config.inference.min_gen_tokens,
    )
    detector = RefusalDetector(config)
    rows = []
    n_kw = 0
    for i, (msg, resp) in enumerate(zip(eval_prompts, responses)):
        kw = bool(detector.detect_refusal(resp))
        n_kw += int(kw)
        rows.append(
            {
                "i": i,
                "prompt": msg.user,
                "response": resp,
                "keyword_refusal": kw,
            }
        )
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"keyword refusals {n_kw}/{len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
