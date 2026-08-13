#!/usr/bin/env python3
"""Stream-merge the v6 t21 PEFT adapter into Ling-3.0-flash shards."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ADAPTER_PREFIX = "base_model.model."
SCALE = 1.0  # lora_alpha / r == 3 / 3


def _module_from_adapter_key(key: str) -> tuple[str, str]:
    if not key.startswith(ADAPTER_PREFIX):
        raise ValueError(f"unexpected adapter key: {key}")
    rest = key[len(ADAPTER_PREFIX) :]
    if rest.endswith(".lora_A.weight"):
        return rest[: -len(".lora_A.weight")] + ".weight", "A"
    if rest.endswith(".lora_B.weight"):
        return rest[: -len(".lora_B.weight")] + ".weight", "B"
    raise ValueError(f"adapter key is not lora A/B: {key}")


def load_adapter_pairs(adapter_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(adapter_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            weight_name, slot = _module_from_adapter_key(key)
            pairs[weight_name][slot] = handle.get_tensor(key)
    incomplete = [name for name, slots in pairs.items() if set(slots) != {"A", "B"}]
    if incomplete:
        raise ValueError(f"adapter missing A/B pair for {incomplete[:5]}")
    return pairs


def merge_delta(weight: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
    delta = lora_b.float() @ lora_a.float()
    if delta.shape != weight.shape:
        raise ValueError(
            f"LoRA delta {tuple(delta.shape)} does not match base {tuple(weight.shape)}"
        )
    return (weight.float() + SCALE * delta).to(dtype=weight.dtype)


def main() -> None:
    base = Path("/run/media/s117/OS/Models/Ling-3.0-flash")
    adapter_dir = Path(
        "/run/media/s117/OS/Models/LoRA/Ling-3.0-flash-LoRA-Trial21-Refusals17-KL0.093"
    )
    out = Path("/run/media/s117/OS/Models/Ling-3.0-flash-abliterix")
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_adapter_pairs(adapter_dir / "adapter_model.safetensors")
    print(f"adapter pairs: {len(pairs)}")

    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    missing = sorted(name for name in pairs if name not in weight_map)
    if missing:
        raise ValueError(f"{len(missing)} adapter targets missing from base, e.g. {missing[:5]}")

    shards: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        shards[shard].append(name)

    merged_names: set[str] = set()
    for shard_i, (shard_name, tensor_names) in enumerate(sorted(shards.items()), start=1):
        print(f"[{shard_i}/{len(shards)}] {shard_name} ({len(tensor_names)} tensors)")
        merged: dict[str, torch.Tensor] = {}
        with safe_open(base / shard_name, framework="pt", device="cpu") as handle:
            for name in tensor_names:
                tensor = handle.get_tensor(name)
                pair = pairs.get(name)
                if pair is not None:
                    tensor = merge_delta(tensor, pair["A"], pair["B"])
                    merged_names.add(name)
                merged[name] = tensor
        save_file(merged, out / shard_name)
        del merged

    leftover = sorted(set(pairs) - merged_names)
    if leftover:
        raise RuntimeError(f"{len(leftover)} adapter weights were not applied, e.g. {leftover[:5]}")

    shutil.copy2(base / "model.safetensors.index.json", out / "model.safetensors.index.json")
    for name in [
        "config.json",
        "configuration_bailing_moe_v3.py",
        "modeling_bailing_moe_v3.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]:
        src = base / name
        if src.exists():
            shutil.copy2(src, out / name)

    (out / "README.md").write_text(
        "# Ling-3.0-flash-abliterix\n\n"
        "Merged Abliterix v6 trial 21 LoRA into inclusionAI Ling-3.0-flash.\n\n"
        "- Source adapter: `Models/LoRA/Ling-3.0-flash-LoRA-Trial21-Refusals17-KL0.093`\n"
        "- Trial metrics: refusals 17/98 (17.3%), KL 0.0933 nats/token\n"
        "- Write path: MPOA `weight_normalization=full`, LoRA rank 3, `o_proj` + `down_proj`\n"
        "- This is the mergeable static recipe, not the v36 runtime gate.\n",
        encoding="utf-8",
    )
    print(f"merged {len(merged_names)} tensors -> {out}")


if __name__ == "__main__":
    main()
