#!/usr/bin/env python3
"""Load Muse Glimmer-30B with the locked trial 12 angular hooks.

Uses the sidecar from ``export_muse_glimmer_t12_artifact.py`` and the
v3_best TOML (bnb by default). Returns (engine, artifact_meta).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("AX_CONFIG", str(ROOT / "configs/muse_glimmer_30b_rocm_v3_best.toml"))
os.environ.setdefault("ABLITERIX_MAX_SEQ", "4096")

sys.argv = [
    "abliterix",
    "--config",
    str(ROOT / "configs/muse_glimmer_30b_rocm_v3_best.toml"),
    "--non-interactive",
    "--inference.batch-size",
    "1",
]

import torch

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.settings import AbliterixConfig
from abliterix.types import SteeringMode, SteeringProfile

ARTIFACT = ROOT / "artifacts" / "muse_glimmer_t12_angular.pt"


def load(artifact_path: Path = ARTIFACT):
    payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    config = AbliterixConfig()
    config.steering.steering_mode = SteeringMode.ANGULAR
    config.steering.runtime_hook_site = payload["runtime_hook_site"]
    config.inference.batch_size = 1
    config.inference.max_batch_size = 1
    engine = SteeringEngine(config)
    if getattr(engine, "tokenizer", None) is not None:
        engine.tokenizer.model_max_length = 4096
    engine.response_prefix = payload["response_prefix"]
    profiles = {
        name: SteeringProfile(**spec) for name, spec in payload["profiles"].items()
    }
    apply_steering(
        engine,
        payload["vectors"],
        payload["vector_index"],
        profiles,
        config,
    )
    return engine, {k: v for k, v in payload.items() if k != "vectors"}


if __name__ == "__main__":
    _engine, meta = load()
    print(meta["eval"])
    print("hooks", len(getattr(_engine, "_angular_hooks", []) or []))
