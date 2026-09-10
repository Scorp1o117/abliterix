#!/usr/bin/env python3
"""Fix a PEFT-wrapped safetensors export in place.

Saving ``PeftModel.get_base_model()`` keeps PEFT's module replacement: the
target Linears appear as ``…base_layer.weight`` plus ``…lora_A/B.default.weight``
instead of the plain ``…weight`` the architecture expects. Reloading such a
checkpoint silently re-initialises those modules (110 of them on Nex-N2.5-mini),
which destroys the model.

For direct-mode abliteration the LoRA adapters are identity (zero-init B), so
the correct fix is lossless:
  * strip the ``.base_layer`` marker so the real (steered) weights load, and
  * drop the ``lora_*`` tensors.

Rewrites each shard in place and rebuilds ``model.safetensors.index.json``.

Usage:
    python scripts/fix_peft_export.py /run/media/s117/OS/Models/Nex-N2.5-mini-abliterix
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def fix(out_dir: Path) -> None:
    shards = sorted(glob.glob(str(out_dir / "*.safetensors")))
    if not shards:
        raise SystemExit(f"no safetensors under {out_dir}")
    index_path = out_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    weight_map: dict[str, str] = {}
    total = 0
    renamed = dropped = 0

    for shard in shards:
        with safe_open(shard, framework="pt") as fh:
            keys = list(fh.keys())
        tensors = load_file(shard)
        clean: dict = {}
        for key in keys:
            if ".lora_A." in key or ".lora_B." in key or "lora_magnitude" in key:
                dropped += 1
                continue
            new_key = key.replace(".base_layer.", ".")
            if new_key != key:
                renamed += 1
            clean[new_key] = tensors[key]
            weight_map[new_key] = Path(shard).name
            total += tensors[key].numel() * tensors[key].element_size()
        tmp = Path(shard).with_suffix(".safetensors.tmp")
        save_file(clean, str(tmp), metadata={"format": "pt"})
        os.replace(tmp, shard)
        del tensors, clean
        print(f"  {Path(shard).name}: {len(keys)} -> {len(weight_map)} keys (running)", flush=True)

    index["metadata"] = {"total_size": total}
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"renamed {renamed} base_layer key(s), dropped {dropped} lora key(s)")
    print(f"index rewritten: {len(weight_map)} tensors, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    fix(ap.parse_args().out_dir)
