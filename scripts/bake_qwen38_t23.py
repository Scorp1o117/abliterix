# Apply Qwen3.8-27B v1 trial 23 (17/100, KL 0.18) and write:
#   artifacts/qwen38_t23_lora/     — LoRA adapter
#   /run/media/s117/OS/Models/Qwen3.8-27B-t23  — merged BF16 for v2 peel

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AX_CONFIG", "configs/qwen38_27b_rocm.toml")
sys.argv = ["abliterix", "--config", "configs/qwen38_27b_rocm.toml", "--seed", "117"]

import torch

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.scriptlib import extract_trial_artifact, load_trial
from abliterix.settings import AbliterixConfig
from abliterix.util import slugify_model_name

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "/run/media/s117/OS/Models/Qwen3.8-27B"
CKPT = ROOT / "checkpoints_qwen38_27b"
TRIAL = 23
LORA_DIR = ROOT / "artifacts" / "qwen38_t23_lora"
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-t23")


def main() -> None:
    cfg = AbliterixConfig()
    trial = load_trial(str(CKPT), MODEL_ID, TRIAL)
    artifact = extract_trial_artifact(trial)
    print(
        f"trial {TRIAL}  vector_index={artifact.vector_index}  "
        f"refusals={trial.user_attrs.get('refusals')}  "
        f"kl={trial.user_attrs.get('kl_divergence')}"
    )
    for name, p in artifact.profiles.items():
        print(
            f"  {name}: max={p.max_weight:.3f}@{p.max_weight_position:.2f}  "
            f"min={p.min_weight:.3f} d={p.min_weight_distance:.2f}"
        )

    engine = SteeringEngine(cfg)
    slug = slugify_model_name(MODEL_ID)
    cache = torch.load(CKPT / f"{slug}_steering.pt", map_location="cpu", weights_only=False)
    apply_steering(
        engine,
        cache["vectors"],
        artifact.vector_index,
        artifact.profiles,
        cfg,
    )

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    engine.export_adapter(LORA_DIR)
    print(f"wrote adapter {LORA_DIR}")

    print("merging LoRA into BF16 on CPU (slow, ~52G)...")
    merged = engine.export_merged()
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True)
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
    print(f"wrote merged {MERGED}")


if __name__ == "__main__":
    main()
