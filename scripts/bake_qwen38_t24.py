#!/usr/bin/env python3
"""Merge pocket trial 24 LoRA into a standalone BF16 checkpoint.

CGA distill collapsed; ORBA missed 8-12 @ KL≤0.1. t24 is the best
mergeable point on record (50/100 @ 0.0866 vs original).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[1])
os.environ.setdefault("AX_CONFIG", "configs/qwen38_27b_rocm_pocket.toml")
sys.argv = ["bake_t24", "--config", "configs/qwen38_27b_rocm_pocket.toml", "--seed", "117"]

import torch

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.scriptlib import extract_trial_artifact, load_trial, setup_io
from abliterix.settings import AbliterixConfig
from abliterix.util import slugify_model_name

setup_io()

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "/run/media/s117/OS/Models/Qwen3.8-27B"
CKPT = ROOT / "checkpoints_qwen38_27b_pocket"
TRIAL = 24
LORA_DIR = Path("/run/media/s117/OS/Models/Qwen3.8-27B-t24-lora")
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-t24")


def main() -> None:
    cfg = AbliterixConfig()
    trial = load_trial(str(CKPT), MODEL_ID, TRIAL)
    artifact = extract_trial_artifact(trial)
    print(
        f"trial {TRIAL} vector_index={artifact.vector_index} "
        f"refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')}",
        flush=True,
    )
    engine = SteeringEngine(cfg)
    slug = slugify_model_name(MODEL_ID)
    cache = torch.load(CKPT / f"{slug}_steering.pt", map_location="cpu", weights_only=False)
    apply_steering(engine, cache["vectors"], artifact.vector_index, artifact.profiles, cfg)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    engine.export_adapter(LORA_DIR)
    print(f"wrote adapter {LORA_DIR}", flush=True)
    print("merging LoRA into BF16...", flush=True)
    merged = engine.export_merged()
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ):
        src = Path(MODEL_ID) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
