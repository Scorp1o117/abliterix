#!/usr/bin/env python3
"""Write the locked Muse v3 trial 12 angular sidecar.

Does not copy the 56G base weights. Load with
``scripts/load_muse_glimmer_t12.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from abliterix.scriptlib import extract_trial_artifact, load_trial

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "/run/media/s117/OS/Models/Muse-Glimmer-30B"
V3 = ROOT / "checkpoints_muse_glimmer_30b_v3"
OUT = ROOT / "artifacts" / "muse_glimmer_t12_angular.pt"


def main() -> None:
    trial = load_trial(str(V3), MODEL_ID, 12)
    artifact = extract_trial_artifact(trial)
    cache = torch.load(
        V3 / "--run--media--s117--OS--Models--Muse-Glimmer-30B_steering.pt",
        map_location="cpu",
        weights_only=False,
    )
    profiles = {
        name: {
            "max_weight": float(p.max_weight),
            "max_weight_position": float(p.max_weight_position),
            "min_weight": float(p.min_weight),
            "min_weight_distance": float(p.min_weight_distance),
        }
        for name, p in artifact.profiles.items()
    }
    payload = {
        "schema": 1,
        "model_id": MODEL_ID,
        "trial": 12,
        "steering_mode": "angular",
        "runtime_hook_site": "decoder_block",
        "response_prefix": " to=user",
        "reasoning_strength": "low",
        "vector_index": artifact.vector_index,
        "profiles": profiles,
        "vectors": cache["vectors"].cpu(),
        "eval": {
            "split": "harmful_1000 train[900:]",
            "n": 100,
            "refusals": 65,
            "baseline_refusals": 95,
            "search_kl_nats_per_token": 0.0274,
            "prescreen_16": 10,
        },
        "note": (
            "Angular rotates the full decoder-block output (including skip). "
            "It is not a rank-1 edit of o_proj/down_proj. down_proj envelope "
            "is unused at apply time; hook strength comes from attn.o_proj."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT)
    meta = {k: v for k, v in payload.items() if k != "vectors"}
    (OUT.with_suffix(".json")).write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT}  vectors={tuple(cache['vectors'].shape)}")


if __name__ == "__main__":
    main()
