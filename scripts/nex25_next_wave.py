#!/usr/bin/env python3
r"""Emit the next search wave's config, seeded from a previous wave's front.

Reads an abliterix Optuna study, takes the best trials on the (refusals, KL)
Pareto front, and writes a new config whose seed_trials are those recipes — so
the next wave starts from what the last one actually learned instead of from a
hand-written guess.

Usage:
    scripts/nex25_next_wave.py \
        --from-checkpoint checkpoints_nex25_mini_v6_160 \
        --config configs/nex25_mini_rocm_v6_160.toml \
        --out configs/nex25_mini_rocm_v7_160.toml \
        --top 6 \
        --standard-only \          # drop the grimjim transforms (A/B losers)
        --search-kernel            # add lin/gauss/cosine as a TPE dimension
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import optuna
from optuna.storages import JournalFileStorage, JournalStorage

optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED_START = "[[optimization.seed_trials]]"
SEED_END = "[kl]"


def completed(study: optuna.Study) -> list[optuna.Trial]:
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]


def pareto(trials: list[optuna.Trial]) -> list[optuna.Trial]:
    pts = [
        (t, t.user_attrs.get("refusals"), t.user_attrs.get("kl_divergence"))
        for t in trials
        if t.user_attrs.get("refusals") is not None
        and t.user_attrs.get("kl_divergence") is not None
    ]
    front = []
    for t, r, k in pts:
        if not any(
            (r2 <= r and k2 <= k and (r2 < r or k2 < k)) for _, r2, k2 in pts
        ):
            front.append(t)
    return sorted(front, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))


def seed_block(trial: optuna.Trial, note: str) -> str:
    lines = ["[[optimization.seed_trials]]", f"# {note}"]
    for key, value in sorted(trial.params.items()):
        rendered = f'"{value}"' if isinstance(value, str) else repr(value)
        lines.append(f'"{key}" = {rendered}')
    return "\n".join(lines) + "\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--checkpoint-dir", default=None, help="new wave's checkpoint_dir")
    ap.add_argument("--standard-only", action="store_true")
    ap.add_argument("--search-kernel", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.from_checkpoint}/*.jsonl"))
    if not files:
        raise SystemExit(f"no journal under {args.from_checkpoint}")
    study = optuna.load_study(
        study_name="abliterix", storage=JournalStorage(JournalFileStorage(files[-1]))
    )
    front = pareto(completed(study))
    if not front:
        raise SystemExit("no completed trial with refusals/KL")
    picks = front[: args.top]
    print("seeding from:")
    for t in picks:
        a = t.user_attrs
        print(f"  t{t.number}: {a['refusals']}/100 @ KL {a['kl_divergence']:.4f} "
              f"({t.params.get('direct_transform', '—')})")

    config_path = Path(args.config)
    src = config_path.read_text(encoding="utf-8")
    start = src.index(SEED_START)
    end = src.index(SEED_END)
    blocks = "".join(
        seed_block(
            t,
            f"from {Path(args.from_checkpoint).name} t{t.number}: "
            f"{t.user_attrs['refusals']}/100 @ KL {t.user_attrs['kl_divergence']:.4f}",
        )
        for t in picks
    )
    src = src[:start] + blocks + src[end:]

    if args.standard_only:
        src = src.replace(
            'search_direct_transform_choices = ["standard", "orba", "biprojected"]',
            'search_direct_transform_choices = ["standard"]',
        )
    if args.search_kernel:
        src = src.replace(
            'search_direct_transform = true',
            'search_direct_transform = true\nsearch_decay_kernel = true\n'
            'search_decay_kernel_choices = ["linear", "gaussian", "cosine"]',
        )
    if args.checkpoint_dir:
        import re

        src = re.sub(r'(?m)^checkpoint_dir = ".*"$', f'checkpoint_dir = "{args.checkpoint_dir}"', src)

    out = Path(args.out)
    out.write_text(src, encoding="utf-8")
    print(f"\nwrote {out} ({len(src.splitlines())} lines, {len(picks)} seeds)")


if __name__ == "__main__":
    main()
