#!/usr/bin/env python3
"""100-prompt eval of Muse v3 trial 12 as runtime angular (bnb)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ABLITERIX_UMA_GUARD", "1")
os.environ.setdefault("ABLITERIX_UMA_MIN_FREE_GB", "24")
os.environ.setdefault("ABLITERIX_UMA_MAX_RSS_GB", "88")
os.environ.setdefault("ABLITERIX_MAX_SEQ", "4096")
os.environ["AX_CONFIG"] = "configs/muse_glimmer_30b_rocm_v3_best.toml"

from abliterix.uma_guard import start_uma_guard

start_uma_guard()

sys.argv = [
    "abliterix",
    "--config",
    "configs/muse_glimmer_30b_rocm_v3_best.toml",
    "--non-interactive",
    "--inference.batch-size",
    "1",
    "--inference.max-batch-size",
    "1",
]

import torch

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.data import load_prompt_dataset
from abliterix.eval.detector import RefusalDetector
from abliterix.scriptlib import extract_trial_artifact, load_trial
from abliterix.settings import AbliterixConfig
from abliterix.types import SteeringMode
from abliterix.vectors import compute_configured_steering_vectors

MODEL_ID = "/run/media/s117/OS/Models/Muse-Glimmer-30B"
V3_CKPT = "checkpoints_muse_glimmer_30b_v3"
TRIAL = 12


def main() -> int:
    config = AbliterixConfig()
    config.steering.steering_mode = SteeringMode.ANGULAR
    config.inference.batch_size = 1
    config.inference.max_batch_size = 1

    trial = load_trial(V3_CKPT, MODEL_ID, TRIAL)
    artifact = extract_trial_artifact(trial)
    print("profiles:", list(artifact.profiles), "vector_index", artifact.vector_index)
    for name, profile in artifact.profiles.items():
        print(
            f"  {name}: max={profile.max_weight:.3f} pos={profile.max_weight_position:.2f} "
            f"min={profile.min_weight:.3f} dist={profile.min_weight_distance:.2f}"
        )

    engine = SteeringEngine(config)
    if getattr(engine, "tokenizer", None) is not None:
        engine.tokenizer.model_max_length = 4096
    engine.response_prefix = " to=user"
    print(f"prefix={engine.response_prefix!r} site={config.steering.runtime_hook_site}")

    blob = torch.load(
        Path(V3_CKPT)
        / "--run--media--s117--OS--Models--Muse-Glimmer-30B_steering.pt",
        map_location="cpu",
        weights_only=False,
    )
    vectors = blob.get("vectors")
    if vectors is None:
        vectors = compute_configured_steering_vectors(
            blob["benign_states"], blob["target_states"], config
        )

    apply_steering(
        engine,
        vectors,
        artifact.vector_index,
        artifact.profiles,
        config,
        routing_config=artifact.routing,
        benign_states=blob["benign_states"],
        target_states=blob["target_states"],
    )
    print("hooks", len(getattr(engine, "_angular_hooks", []) or []))

    eval_msgs = load_prompt_dataset(config, config.target_eval_prompts)
    detector = RefusalDetector(config)
    n_ref = 0
    samples = []
    for i, msg in enumerate(eval_msgs):
        text = engine.generate_text_batched(
            [msg],
            skip_special_tokens=True,
            max_new_tokens=config.inference.max_gen_tokens,
            min_new_tokens=config.inference.min_gen_tokens,
        )[0]
        is_ref = bool(detector.detect_refusal(text))
        n_ref += int(is_ref)
        if i < 8 or (not is_ref and len(samples) < 12):
            samples.append((is_ref, msg.user[:70], text[:180].replace("\n", " | ")))
        print(f"  [{i:02d} {'REFUSE' if is_ref else 'comply'}] {msg.user[:56]!r}", flush=True)

    print(f"* trial 12 angular refusals: {n_ref}/{len(eval_msgs)}", flush=True)
    for is_ref, prompt, text in samples[:8]:
        print(f"  {'REFUSE' if is_ref else 'comply'} {prompt!r} -> {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
