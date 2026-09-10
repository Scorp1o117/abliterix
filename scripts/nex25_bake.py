#!/usr/bin/env python3
"""Bake a chosen abliterix trial into a publishable checkpoint.

The CLI's ``--non-interactive-output-dir`` export picks the trial with the
fewest refusals (ties broken by KL), which is *not* the project's dual bar
(refusals <= 10/100 **and** 3-token KL <= 0.1).  This script writes a
single-seed config for one explicit trial — or for the best trial that clears
the bar — copies the cached baseline/steering tensors so the bake does not
recompute them, and prints the command to run.

Generate:
    scripts/nex25_bake.py checkpoints_nex25_mini_v2_direct --bar 10 0.1
    scripts/nex25_bake.py checkpoints_nex25_mini_v2_direct --trial 7

Then run the printed command (it needs the GPU):

    ./run_nex25.sh configs/nex25_bake.toml
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
from pathlib import Path

import optuna
from optuna.storages import JournalFileStorage, JournalStorage

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
CACHE_SUFFIXES = ("_baseline.pt", "_steering.pt")


def pick_trial(study: optuna.Study, args: argparse.Namespace) -> optuna.Trial:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise SystemExit("no COMPLETE trial in the study")
    if args.trial is not None:
        for t in complete:
            if t.number == args.trial:
                return t
        raise SystemExit(f"trial {args.trial} is not COMPLETE")
    bar_ref, bar_kl = args.bar
    hits = [
        t
        for t in complete
        if t.user_attrs.get("refusals", 10**9) <= bar_ref
        and t.user_attrs.get("kl_divergence", 10**9) <= bar_kl
    ]
    if hits:
        # Among bar-clearing trials prefer the lowest KL, then fewest refusals.
        return min(hits, key=lambda t: (t.user_attrs["kl_divergence"], t.user_attrs["refusals"]))
    print(f"[warn] no trial clears the bar (<= {bar_ref:g} refusals, <= {bar_kl:g} KL); "
          "falling back to the fewest-refusals trial")
    return min(complete, key=lambda t: (t.user_attrs.get("refusals", 10**9),
                                        t.user_attrs.get("kl_divergence", 10**9)))


def toml_value(v: object) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return repr(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--config", default="configs/nex25_mini_rocm_v2_direct.toml")
    ap.add_argument("--out", default="/run/media/s117/OS/Models/Nex-N2.5-mini-abliterix")
    ap.add_argument("--bake-config", default="configs/nex25_bake.toml")
    ap.add_argument("--bake-checkpoints", default="checkpoints_nex25_mini_bake")
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--bar", nargs=2, type=float, default=[10.0, 0.1])
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.checkpoint_dir}/*.jsonl"))
    if not files:
        raise SystemExit(f"no journal under {args.checkpoint_dir}")
    study = optuna.load_study(
        study_name="abliterix", storage=JournalStorage(JournalFileStorage(files[-1]))
    )
    trial = pick_trial(study, args)

    ref = trial.user_attrs.get("refusals")
    kl = trial.user_attrs.get("kl_divergence")
    print(f"selected trial {trial.number}: refusals={ref} KL={kl}")

    src = (ROOT / args.config).read_text(encoding="utf-8")

    # --- single-trial settings -------------------------------------------
    src = re.sub(r"(?m)^num_trials\s*=\s*\d+", "num_trials = 1", src)
    src = re.sub(r"(?m)^num_warmup_trials\s*=\s*\d+", "num_warmup_trials = 0", src)
    src = src.replace("refusal_prescreen_enabled = true", "refusal_prescreen_enabled = false")
    src = src.replace(
        f'checkpoint_dir = "{Path(args.checkpoint_dir).name}"',
        f'checkpoint_dir = "{args.bake_checkpoints}"',
    )
    src = src.replace(
        "non_interactive = true",
        f'non_interactive = true\nnon_interactive_output_dir = "{args.out}"',
    )

    # --- replace every seed block with the chosen trial ------------------
    start = src.index("[[optimization.seed_trials]]")
    end = src.index("[kl]")
    lines = [
        "[[optimization.seed_trials]]",
        f"# baked from trial {trial.number} (refusals={ref}, KL={kl})",
    ]
    for key, value in sorted(trial.params.items()):
        lines.append(f'"{key}" = {toml_value(value)}')
    src = src[:start] + "\n".join(lines) + "\n\n" + src[end:]

    out_cfg = ROOT / args.bake_config
    out_cfg.write_text(src, encoding="utf-8")
    print(f"wrote {out_cfg.relative_to(ROOT)} ({len(src.splitlines())} lines)")

    # --- reuse the cached baseline / steering tensors --------------------
    bake_dir = ROOT / args.bake_checkpoints
    bake_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src_file in Path(args.checkpoint_dir).iterdir():
        if src_file.name.endswith(CACHE_SUFFIXES) and src_file.is_file():
            shutil.copy2(src_file, bake_dir / src_file.name)
            copied += 1
    print(f"copied {copied} cache file(s) into {bake_dir.name}/")

    print("\nrun the bake with:\n    ./run_nex25.sh " + args.bake_config)


if __name__ == "__main__":
    main()
