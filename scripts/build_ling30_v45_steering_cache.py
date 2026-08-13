#!/usr/bin/env python3
"""Compose exact legacy mean + train-derived orthogonal residual direction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--rank2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy_payload = torch.load(args.legacy, map_location="cpu", weights_only=False)
    rank2_payload = torch.load(args.rank2, map_location="cpu", weights_only=False)
    legacy = legacy_payload["vectors"].float()
    rank2 = rank2_payload["vectors"].float()
    if legacy.ndim != 2 or rank2.ndim != 3 or rank2.shape[0] != 2:
        raise ValueError(
            f"expected legacy (layers, hidden) and rank2 (2, layers, hidden), "
            f"got {tuple(legacy.shape)} and {tuple(rank2.shape)}"
        )
    if legacy.shape != rank2.shape[1:]:
        raise ValueError("legacy and rank2 layer/hidden shapes differ")

    primary = legacy
    secondary = rank2[1]
    primary_norm_sq = primary.square().sum(dim=-1, keepdim=True)
    secondary = secondary - (
        (secondary * primary).sum(dim=-1, keepdim=True)
        / primary_norm_sq.clamp(min=1e-12)
    ) * primary
    secondary_norm = secondary.norm(dim=-1, keepdim=True)
    secondary = torch.where(
        secondary_norm > 1e-8,
        secondary / secondary_norm.clamp(min=1e-8),
        torch.zeros_like(secondary),
    )
    composed = torch.stack([primary, secondary], dim=0).to(
        rank2_payload["vectors"].dtype
    )

    output_payload = dict(rank2_payload)
    output_payload["vectors"] = composed
    output_payload["composition"] = {
        "primary": "exact legacy rank-1 vector cache",
        "secondary": "rank-2 residual direction reorthogonalized to primary",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    cosine = F.cosine_similarity(composed[0].float(), legacy, dim=-1)
    orthogonality = (composed[0].float() * composed[1].float()).sum(dim=-1).abs()
    print(
        {
            "output": str(args.output),
            "primary_exact": bool(torch.equal(composed[0].float(), legacy)),
            "primary_cosine_min": float(cosine.min()),
            "secondary_dot_max": float(orthogonality.max()),
            "shape": tuple(composed.shape),
        }
    )


if __name__ == "__main__":
    main()
