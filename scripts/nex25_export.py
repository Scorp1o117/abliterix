#!/usr/bin/env python3
"""Export an abliterix direct-mode trial without merging LoRA adapters.

Why this exists: `engine.export_merged()` calls `merge_and_unload()`, which
materialises a second copy of the weights. On a 122 GB unified-memory box the
70 GB Nex-N2.5-mini checkpoint already occupies ~76 GB of GTT there, so the
merge (plus shard buffers) trips the UMA guard and the export dies mid-write.

In direct mode the LoRA adapters are identity (zero-init, never trained): the
steering lives in the *base* weights, edited in place by `apply_steering`. So
the faithful export is simply the base HF model — no merge, no doubling. This
script applies the chosen trial's steering and saves the base model, then
records a small manifest (weight SHA256s + trial attributes + recipe) next to
the weights.

Usage:
    python scripts/nex25_export.py --trial 0 \
        --checkpoint checkpoints_nex25_mini_v5_256cap \
        --config configs/nex25_mini_rocm_v5_256cap.toml \
        --out /run/media/s117/OS/Models/Nex-N2.5-mini-abliterix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

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

SIDECARS = (
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
    "README.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-hashes", action="store_true")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    os.environ["AX_CONFIG"] = args.config
    sys.argv = ["nex25_export", "--config", args.config, "--seed", "117"]
    cfg = AbliterixConfig()

    print(f"loading trial {args.trial} from {args.checkpoint}...", flush=True)
    trial = load_trial(args.checkpoint, cfg.model.model_id, args.trial)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
    attrs = {
        "refusals": trial.user_attrs.get("refusals"),
        "kl_divergence": trial.user_attrs.get("kl_divergence"),
        "validation_kl": trial.user_attrs.get("validation_kl"),
        "vector_index": artifact.vector_index,
        "params": {k: v for k, v in trial.params.items()},
    }
    print(f"trial {args.trial}: refusals={attrs['refusals']} kl={attrs['kl_divergence']}")

    print("loading model and applying steering...", flush=True)
    engine = SteeringEngine(cfg)
    cache_path = next(Path(args.checkpoint).glob("*_steering.pt"))
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    apply_steering(
        engine,
        cache["vectors"],
        artifact.vector_index,
        artifact.profiles,
        cfg,
        benign_states=cache.get("benign_states"),
        target_states=cache.get("target_states"),
    )

    # Direct mode: the base weights already carry the edit and the LoRA adapters
    # are identity, so export the base model rather than merging (which would
    # double the resident weight memory and trip the UMA guard).
    base = engine.model.get_base_model() if hasattr(engine.model, "get_base_model") else engine.model
    print(f"saving base model ({type(base).__name__}) to {out}...", flush=True)
    base.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(out))
    for name in SIDECARS:
        src = Path(cfg.model.model_id) / name
        if src.is_file() and not (out / name).exists():
            shutil.copy2(src, out / name)

    # PEFT keeps its module replacement in the saved state dict: wrapped Linears
    # come out as `…base_layer.weight` plus `…lora_A/B`, so a plain reload would
    # silently re-initialise all 110 of them (observed: KL 12.44 on reload).
    # Direct mode's adapters are identity, so stripping the marker and dropping
    # the adapters is lossless.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fix_peft_export import fix as fix_peft_keys

    print("normalising PEFT-wrapped keys...", flush=True)
    fix_peft_keys(out)

    weights = sorted(out.glob("*.safetensors"))
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": cfg.model.model_id,
        "checkpoint": args.checkpoint,
        "trial": args.trial,
        "steering_mode": str(cfg.steering.steering_mode),
        "weight_normalization": str(cfg.steering.weight_normalization),
        "max_gen_tokens": cfg.inference.max_gen_tokens,
        "trial_attrs": attrs,
        "weight_files": [p.name for p in weights],
    }
    if not args.skip_hashes:
        print(f"hashing {len(weights)} weight file(s)...", flush=True)
        manifest["weight_sha256"] = {p.name: _sha256(p) for p in weights}
    (out / "nex25_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    total = sum(p.stat().st_size for p in weights)
    print(f"wrote {len(weights)} shard(s), {total / 1e9:.2f} GB to {out}", flush=True)
    print(f"manifest: {out / 'nex25_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
