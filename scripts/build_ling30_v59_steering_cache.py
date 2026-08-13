#!/usr/bin/env python3
"""Compose exact primary + steered residual peel for the v59 formal probe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abliterix.vectors import compose_exact_primary_with_steered_peel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--rank2-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-source-indices",
        type=int,
        nargs="*",
        default=[],
        help="Gate-off or otherwise non-steering failures to drop from peel labels.",
    )
    args = parser.parse_args()

    legacy_payload = torch.load(args.legacy, map_location="cpu", weights_only=False)
    dump_payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    template_payload = torch.load(
        args.rank2_template, map_location="cpu", weights_only=False
    )
    primary = legacy_payload["vectors"]
    excluded = set(args.exclude_source_indices)
    refusal_indices = [
        int(idx) for idx in dump_payload["refusal_indices"] if int(idx) not in excluded
    ]
    compliance_indices = [int(idx) for idx in dump_payload["compliance_indices"]]
    composed = compose_exact_primary_with_steered_peel(
        primary,
        dump_payload["residuals"],
        [int(idx) for idx in dump_payload["source_indices"]],
        refusal_indices,
        compliance_indices,
    )

    output_payload = dict(template_payload)
    output_payload["vectors"] = composed
    output_payload["composition"] = {
        "primary": "exact v43 legacy rank-1 vector cache",
        "secondary": (
            "steered failure-minus-success residual from v58 post-hook "
            "final-prefill dump, orthogonalized to the exact primary"
        ),
        "excluded_source_indices": sorted(excluded),
        "refusal_indices": refusal_indices,
        "compliance_count": len(compliance_indices),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    cosine = F.cosine_similarity(composed[0].float(), primary.float(), dim=-1)
    orthogonality = (composed[0].float() * composed[1].float()).sum(dim=-1).abs()
    peel_norm = composed[1].float().norm(dim=-1)
    print(
        {
            "output": str(args.output),
            "primary_exact": bool(torch.equal(composed[0], primary)),
            "primary_cosine_min": float(cosine.min()),
            "secondary_dot_max": float(orthogonality.max()),
            "peel_active_layers": int((peel_norm > 1e-6).sum()),
            "excluded_source_indices": sorted(excluded),
            "shape": tuple(composed.shape),
        }
    )


if __name__ == "__main__":
    main()
