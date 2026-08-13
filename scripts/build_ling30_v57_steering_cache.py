#!/usr/bin/env python3
"""Compose exact v43 primary + Zhao harmfulness for the v57 probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abliterix.harmfulness import compose_exact_primary_with_harmfulness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--rank2-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-band-lo", type=float, default=0.3)
    parser.add_argument("--layer-band-hi", type=float, default=0.7)
    args = parser.parse_args()

    legacy_payload = torch.load(args.legacy, map_location="cpu", weights_only=False)
    template_payload = torch.load(
        args.rank2_template, map_location="cpu", weights_only=False
    )
    primary = legacy_payload["vectors"]
    if primary.ndim != 2:
        raise ValueError(f"expected legacy rank-1 vectors, got {tuple(primary.shape)}")

    composed = compose_exact_primary_with_harmfulness(
        primary,
        legacy_payload["benign_states"],
        legacy_payload["target_states"],
        layer_band=(args.layer_band_lo, args.layer_band_hi),
        projected_abliteration=True,
    )

    output_payload = dict(template_payload)
    output_payload["vectors"] = composed
    output_payload["benign_states"] = legacy_payload["benign_states"]
    output_payload["target_states"] = legacy_payload["target_states"]
    output_payload["composition"] = {
        "primary": "exact v43 legacy rank-1 vector cache",
        "secondary": (
            "Zhao harmfulness PCA-1 of centered train[:800] target states, "
            "orthogonalized to the exact primary"
        ),
        "layer_band": [args.layer_band_lo, args.layer_band_hi],
        "projected_abliteration": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    template_secondary = template_payload["vectors"]
    secondary_cosine = None
    if (
        isinstance(template_secondary, torch.Tensor)
        and template_secondary.ndim == 3
        and template_secondary.shape[0] >= 2
    ):
        secondary_cosine = F.cosine_similarity(
            composed[1].float(), template_secondary[1].float(), dim=-1
        )
    primary_cosine = F.cosine_similarity(composed[0].float(), primary.float(), dim=-1)
    orthogonality = (composed[0].float() * composed[1].float()).sum(dim=-1).abs()
    print(
        {
            "output": str(args.output),
            "primary_exact": bool(torch.equal(composed[0], primary)),
            "primary_cosine_min": float(primary_cosine.min()),
            "secondary_dot_max": float(orthogonality.max()),
            "vs_svd_secondary_cosine_mean": (
                None if secondary_cosine is None else float(secondary_cosine.mean())
            ),
            "cache_key": output_payload.get("cache_key"),
            "shape": tuple(composed.shape),
        }
    )


if __name__ == "__main__":
    main()
