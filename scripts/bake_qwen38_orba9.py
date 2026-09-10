#!/usr/bin/env python3
"""Bake the best mergeable Pareto point: ORBA o_proj=9.0 t24-band.

In-memory score was 21/100 @ 0.131 vs original. Lower refusals than t24
(50@0.087) without the Arditi s=1.25 KL cliff (16@0.279).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

os.environ["AX_CONFIG"] = "configs/qwen38_27b_rocm_pocket.toml"
sys.argv = ["bake_orba9", "--config", "configs/qwen38_27b_rocm_pocket.toml", "--seed", "117"]

from abliterix.scriptlib import setup_io

setup_io()

import torch

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.settings import AbliterixConfig
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile

VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982
O_MAX = 9.0
VINDEX = 50.79667354766495


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=O_MAX,
            max_weight_position=T24_POS,
            min_weight=O_MAX * 0.5,
            min_weight_distance=T24_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.37,
            max_weight_position=37.89,
            min_weight=0.25,
            min_weight_distance=30.46,
        ),
    }
    apply_steering(
        engine,
        cache["vectors"],
        VINDEX,
        profiles,
        cfg,
        benign_states=cache.get("benign_states"),
    )
    print("ORBA o9.0 applied — saving merged BF16...", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in (
        "chat_template.jinja",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = Path(ORIGINAL) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
