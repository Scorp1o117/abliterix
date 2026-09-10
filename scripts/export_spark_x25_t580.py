#!/usr/bin/env python3
"""Merge V30 T580 and save Spark-X2.5-4B-abliterix.

User-requested export. Not a dual HIT (20/100 @ 3-token KL 0.0924).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
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
from abliterix.settings import AbliterixConfig  # noqa: E402


def _sanitize_generation_config(model) -> None:
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None or getattr(gen_cfg, "do_sample", False):
        return
    for field in ("temperature", "top_p", "top_k", "min_p", "typical_p"):
        if getattr(gen_cfg, field, None) is not None:
            try:
                setattr(gen_cfg, field, None)
            except Exception:
                pass

CONFIG = "configs/spark_x25_4b_lora_v30.toml"
CHECKPOINT = "checkpoints_spark_x25_4b_lora_v30"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
MERGED = Path("/run/media/s117/OS/Models/Spark-X2.5-4B-abliterix")
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
OUT = Path("logs/spark_x25_t580_export.json")
TRIAL = 580


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["export_t580", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    return cfg


def _copy_spark_sidecars(dst: Path) -> None:
    src = Path(MODEL)
    for name in (
        "configuration_spark.py",
        "modeling_spark.py",
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
        "README.md",
    ):
        s = src / name
        if s.is_file() and not (dst / name).exists():
            shutil.copy2(s, dst / name)


def main() -> None:
    torch.set_grad_enabled(False)
    os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
    os.environ.setdefault("ABLITERIX_MAX_SEQ", "4096")
    cfg = _cfg()
    trial = load_trial(CHECKPOINT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "trial": TRIAL,
        "checkpoint": CHECKPOINT,
        "model": MODEL,
        "merged": str(MERGED),
        "keyword_refusals": trial.user_attrs.get("refusals"),
        "full_distribution_kl_3token_vs_original": trial.user_attrs.get(
            "kl_divergence"
        ),
        "vector_index": artifact.vector_index,
        "dual_hit_005": False,
        "note": "User-requested export of V30 T580. Not dual HIT (need ≤10/100 and KL≤0.05).",
    }
    print(
        f"exporting T{TRIAL}: refusals={payload['keyword_refusals']} "
        f"kl={payload['full_distribution_kl_3token_vs_original']} "
        f"vector_index={artifact.vector_index}",
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
    print(f"merging LoRA and saving {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    if hasattr(engine.model, "merge_and_unload"):
        model = engine.model.merge_and_unload()
        engine.needs_reload = True
    else:
        model = engine.export_merged()
    _sanitize_generation_config(model)
    model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    _copy_spark_sidecars(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    payload["baked"] = True
    (MERGED / "trial_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {MERGED}", flush=True)
    print("EXPORT_OK", flush=True)


if __name__ == "__main__":
    main()
