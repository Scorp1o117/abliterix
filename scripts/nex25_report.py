#!/usr/bin/env python3
"""Report the Pareto front of an abliterix Optuna search.

Reads the journal under ``<checkpoint_dir>/`` through Optuna's own storage API
and prints one row per trial: refusals / KL / strength parameters.  Useful while
a long search is still running and for the final ship-bar selection.

Run with an environment that has optuna installed, e.g.
``/home/s117/heretic-env/bin/python scripts/nex25_report.py checkpoints_...``

Usage:
    scripts/nex25_report.py checkpoints_nex25_mini_v2_direct [--bar 10 0.1] [--all]
"""

from __future__ import annotations

import argparse
import glob

import optuna
from optuna.storages import JournalFileStorage, JournalStorage

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAM_KEYS = (
    "direct_transform",
    "vector_scope",
    "attn.o_proj.max_weight",
    "mlp.down_proj.max_weight",
    "attn.q_proj.max_weight",
)

# Bar under the 160-token house convention: refusals <=10 with 3-token KL <=0.05
# (0.1 is the fallback only when 0.05 is unreachable).
BAR_REFUSALS, BAR_KL = 10.0, 0.05
FALLBACK_KL = 0.1


def fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--bar", nargs=2, type=float, default=[BAR_REFUSALS, BAR_KL],
                    metavar=("REFUSALS", "KL"))
    ap.add_argument("--all", action="store_true", help="include pruned/failed trials")
    args = ap.parse_args()

    files = glob.glob(f"{args.checkpoint_dir}/*.jsonl")
    if not files:
        raise SystemExit(f"no journal under {args.checkpoint_dir}")
    storage = JournalStorage(JournalFileStorage(sorted(files)[-1]))
    study = optuna.load_study(study_name="abliterix", storage=storage)

    trials = study.trials
    states = {"COMPLETE": 0, "PRUNED": 0, "RUNNING": 0, "FAIL": 0}
    for t in trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    print("trials={} ".format(len(trials)) + " ".join(f"{k.lower()}={v}" for k, v in states.items()))

    rows = []
    for t in trials:
        ref = t.user_attrs.get("refusals")
        kl = t.user_attrs.get("kl_divergence")
        if ref is None and t.values:
            ref, kl = (list(t.values) + [None, None])[:2]
        health = t.user_attrs.get("generation_health") or {}
        # Degenerate generations are counted as refusals by the detector, so a
        # health-failed trial's refusal number is not a usable operating point.
        healthy = bool(health.get("passed", True))
        rows.append((t.number, t.state.name, ref, kl, t, healthy))
    rows.sort(key=lambda r: (r[2] is None, r[2] if r[2] is not None else 0.0, r[3] or 0.0))

    head = f"{'id':>4} {'state':>9} {'refusals':>8} {'KL':>8} {'health':>7} " + " ".join(
        f"{(k.split('.')[1][:6] if '.' in k else k)[:8]:>8}" for k in PARAM_KEYS
    )
    print(head)
    print("-" * len(head))
    for tid, st, ref, kl, t, healthy in rows:
        if not args.all and st != "COMPLETE":
            continue
        star = " "
        if st == "COMPLETE" and ref is not None and kl is not None:
            star = "★" if (ref <= args.bar[0] and kl <= args.bar[1]) else " "
        fk = f"{kl:.4f}" if isinstance(kl, float) else "—"
        fh = "PASS" if healthy else "FAIL"
        print(
            f"{star}{tid:>4} {st:>9} {str(ref):>8} {fk:>8} {fh:>7} "
            + " ".join(f"{fmt(t.params.get(k, '—')):>8}" for k in PARAM_KEYS)
        )

    # Only health-passing trials are usable operating points: degenerate
    # generations are counted as refusals, so their numbers are inflated.
    ok = [
        r for r in rows
        if r[1] == "COMPLETE" and r[2] is not None and r[3] is not None and r[5]
    ]
    hits = [r for r in ok if r[2] <= args.bar[0] and r[3] <= args.bar[1]]
    fallback = [
        r for r in ok if r[2] <= args.bar[0] and r[3] <= FALLBACK_KL
    ]
    if ok:
        best_ref = ok[0]
        best_kl = min(ok, key=lambda r: r[3])
        print(f"\n最少拒绝: t{best_ref[0]} {best_ref[2]}/100 @ KL {best_ref[3]:.4f}")
        print(f"最低 KL : t{best_kl[0]} {best_kl[2]}/100 @ KL {best_kl[3]:.4f}")
        # Pareto front (refusals, KL) — the points nothing else dominates.
        front = [
            r for r in ok
            if not any(
                (o[2] <= r[2] and o[3] <= r[3] and (o[2] < r[2] or o[3] < r[3]))
                for o in ok
            )
        ]
        print("Pareto 前沿: " + ", ".join(f"t{r[0]}({r[2]},{r[3]:.4f})" for r in front))
    print(
        f"满足 ship bar（拒绝 ≤{args.bar[0]:g} 且 KL ≤{args.bar[1]:g}）: {len(hits)} 个"
        + (f" → t{[h[0] for h in hits]}" if hits else "")
    )
    print(
        f"退路线（拒绝 ≤{args.bar[0]:g} 且 KL ≤{FALLBACK_KL:g}）: {len(fallback)} 个"
        + (f" → t{[h[0] for h in fallback]}" if fallback else "")
    )


if __name__ == "__main__":
    main()
