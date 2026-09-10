#!/usr/bin/env python3
"""Fernflower / AEON SSM conv1d outlier repair for Qwen3.8-27B.

Per AEON-7: σ of linear_attn.conv1d.weight across GDN layers; flag
σ > 1.5 × median; rescale those weights by α = median_σ / σ.
Writes a sibling model dir (symlinks + rewritten shards) so Abliterix
can load it as a normal HF path. Does not touch the original.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("/run/media/s117/OS/Models/Qwen3.8-27B")
DST = Path("/run/media/s117/OS/Models/Qwen3.8-27B-ssm-repaired")
ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "logs" / "qwen38_conv1d_repair.json"
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer") / "qwen38_conv1d_repair.json"
THRESH = 1.5


def _scan() -> tuple[list[dict], float]:
    idx = json.loads((SRC / "model.safetensors.index.json").read_text())
    conv = {
        k: v
        for k, v in idx["weight_map"].items()
        if k.endswith("linear_attn.conv1d.weight")
    }
    by_shard: dict[str, list[str]] = defaultdict(list)
    for k, shard in conv.items():
        by_shard[shard].append(k)
    rows = []
    for shard, keys in by_shard.items():
        with safe_open(str(SRC / shard), framework="pt") as f:
            for k in keys:
                layer = int(re.search(r"layers\.(\d+)\.", k).group(1))
                t = f.get_tensor(k)
                sigma = float(t.float().std())
                rows.append(
                    {
                        "layer": layer,
                        "sigma": sigma,
                        "shape": list(t.shape),
                        "key": k,
                        "shard": shard,
                    }
                )
    rows.sort(key=lambda r: r["layer"])
    sigs = torch.tensor([r["sigma"] for r in rows])
    med = float(sigs.median())
    for r in rows:
        r["ratio"] = r["sigma"] / med
        r["outlier"] = r["sigma"] > THRESH * med
        r["alpha"] = (med / r["sigma"]) if r["outlier"] else 1.0
    return rows, med


def _rewrite_shards(outliers: list[dict]) -> list[str]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for r in outliers:
        by_shard[r["shard"]].append(r)
    written = []
    for shard, items in by_shard.items():
        scale = {r["key"]: r["alpha"] for r in items}
        tensors = {}
        metadata = None
        with safe_open(str(SRC / shard), framework="pt") as f:
            metadata = dict(f.metadata() or {})
            for k in f.keys():
                t = f.get_tensor(k)
                if k in scale:
                    alpha = scale[k]
                    t = (t.float() * alpha).to(t.dtype).contiguous()
                    print(f"  scale {k} α={alpha:.4f}", flush=True)
                tensors[k] = t
        dest = DST / shard
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        save_file(tensors, str(dest), metadata=metadata or None)
        written.append(shard)
        print(f"  wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        del tensors
    return written


def _symlink_rest(rewritten: set[str]) -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for src in SRC.iterdir():
        if src.name in rewritten:
            continue
        dest = DST / src.name
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                shutil.rmtree(dest)
        os.symlink(src, dest)


def _verify(rows: list[dict], med: float) -> list[dict]:
    after = []
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["outlier"]:
            by_shard[r["shard"]].append(r)
    for shard, items in by_shard.items():
        with safe_open(str(DST / shard), framework="pt") as f:
            for r in items:
                t = f.get_tensor(r["key"])
                sigma = float(t.float().std())
                after.append(
                    {
                        "layer": r["layer"],
                        "sigma_before": r["sigma"],
                        "sigma_after": sigma,
                        "alpha": r["alpha"],
                        "ratio_after": sigma / med,
                    }
                )
    return after


def main() -> None:
    print(f"scan {SRC}", flush=True)
    rows, med = _scan()
    outliers = [r for r in rows if r["outlier"]]
    print(
        f"n={len(rows)} median_σ={med:.6f} threshold={THRESH*med:.6f} "
        f"outliers={len(outliers)}",
        flush=True,
    )
    for r in outliers:
        print(
            f"  L{r['layer']:02d} σ={r['sigma']:.6f} α={r['alpha']:.4f} {r['shard']}",
            flush=True,
        )
    rewritten = {r["shard"] for r in outliers}
    print(f"link {DST} (rewrite {sorted(rewritten)})", flush=True)
    _symlink_rest(rewritten)
    print("rewrite shards...", flush=True)
    written = _rewrite_shards(outliers)
    print("verify...", flush=True)
    after = _verify(rows, med)
    for a in after:
        print(
            f"  L{a['layer']:02d} {a['sigma_before']:.6f} -> {a['sigma_after']:.6f} "
            f"(ratio {a['ratio_after']:.3f})",
            flush=True,
        )
    report = {
        "src": str(SRC),
        "dst": str(DST),
        "median_sigma": med,
        "threshold": THRESH,
        "n_ssm": len(rows),
        "outliers": [
            {k: r[k] for k in ("layer", "sigma", "alpha", "key", "shard")}
            for r in outliers
        ],
        "after": after,
        "rewritten_shards": written,
        "method": "Fernflower/AEON: α = median_σ / σ for σ > 1.5×median",
    }
    text = json.dumps(report, indent=2)
    for dest in (REPORT, SCRATCH):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        print(f"REPORT {dest}", flush=True)
    if any(a["ratio_after"] > THRESH + 0.05 for a in after):
        raise SystemExit("repair did not bring outliers under 1.5× median")
    print("REPAIR_OK", DST, flush=True)


if __name__ == "__main__":
    main()
