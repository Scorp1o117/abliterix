#!/usr/bin/env python3
"""Nex-N2.5-mini leftover peel (stage 2).

Stage 1 bakes the best single-stage recipe found by the v5 search (direct + EGA,
~21/100 refusals @ KL 0.0845 at the faithful 256-token cap). That point sits on
the KL<=0.1 budget edge, so simply turning the strength up is not available;
the standard next move is the leftover peel used on Spark-X2.5 (-1bv/-1ca):

  1. generate responses on the target prompts under the stage-1 model,
  2. collect the prompts that still refuse ("leftover"),
  3. estimate a second mean direction r2 from leftover-vs-benign residuals,
  4. grid-search r2 on top of the stage-1 weights,
  5. bake on dual HIT (refusals <= 10/100 and KL <= 0.1).

Because MoE + direct mode supports only one direction per layer (EGA), the two
stages are sequential rather than a single multi-direction fit: stage 1 is baked
to disk first, and every grid point restores that baked state before applying r2.

Kill criteria (mirroring the Spark script):
  * cos(stage-1 global direction, r2) > COS_KILL  — r2 is not a new direction
  * the first WALL_POINTS grid points stay on the old wall (refusals >= WALL_REFUSALS
    and KL >= WALL_KL) — the peel is not moving the front

Usage:
    python scripts/nex25_peel.py --trial 0            # winner from the v5 journal
    python scripts/nex25_peel.py --trial 0 --grid 6   # first N grid points only
"""

from __future__ import annotations

import argparse
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
from abliterix.core.steering import apply_steering, resolve_global_vector  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringProfile  # noqa: E402
from abliterix.vectors import compute_configured_steering_vectors  # noqa: E402

CONFIG = "configs/nex25_mini_rocm_v5_256cap.toml"
CHECKPOINT = "checkpoints_nex25_mini_v5_256cap"
BASE_MODEL = "/run/media/s117/OS/Models/Nex-N2.5-mini"
STAGE1 = Path("/run/media/s117/OS/Models/Nex-N2.5-mini-abliterix")
PEELED = Path("/run/media/s117/OS/Models/Nex-N2.5-mini-abliterix-peel")
SLUG = "--run--media--s117--OS--Models--Nex-N2--5-mini"
BASELINE = Path(CHECKPOINT) / f"{SLUG}_baseline.pt"
STEERING = Path(CHECKPOINT) / f"{SLUG}_steering.pt"

COS_KILL = 0.85
WALL_REFUSALS = 18
WALL_KL = 0.10
WALL_POINTS_BEFORE_KILL = 3
BAR_REFUSALS = 10
BAR_KL = 0.1

# r2 strength grid: (o_proj max, o_proj min-fraction, down_proj max, down min-fraction).
# Deliberately small relative to the stage-1 profile (o 1.066 / down 3.033) — the
# peel is a targeted correction for the leftover prompts, not a second main sweep.
GRID = [
    (0.25, 0.46, 0.75, 0.35),
    (0.50, 0.46, 1.50, 0.35),
    (0.80, 0.46, 2.40, 0.35),
    (1.10, 0.46, 3.00, 0.35),
    (0.50, 0.25, 1.50, 0.60),
    (0.80, 0.25, 2.40, 0.60),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["nex25_peel", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    # model_id stays on the original base here: apply_trial_artifact() verifies
    # that the trial was optimised for the configured base model. It is switched
    # to the baked stage-1 checkpoint right after that check.
    cfg.model.text_only = True
    cfg.optimization.checkpoint_dir = "checkpoints_nex25_mini_peel"
    # The v5 search ran this model at batch 128 under the same UMA guard, so the
    # peel keeps the config's batch size rather than throttling it (a 4x slower
    # eval makes six grid points take hours).
    cfg.detection.llm_judge = False
    cfg.steering.n_directions = 1
    return cfg


def _out_path(trial: int) -> Path:
    return Path(f"logs/nex25_peel_t{trial}.json")


def _dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _inject_original_baseline(scorer: TrialScorer, baseline_pt: Path) -> None:
    cache = torch.load(baseline_pt, map_location="cpu", weights_only=False)
    scorer.baseline_logprobs = cache["baseline_logprobs"]
    scorer.baseline_mean_length = cache["baseline_mean_length"]
    scorer.baseline_stdev_length = cache["baseline_stdev_length"]
    scorer.baseline_single_token_lp = cache["baseline_single_token_lp"]
    scorer.baseline_refusal_count = cache["baseline_refusal_count"]
    scorer.baseline_continuations = cache.get("baseline_continuations")
    scorer.baseline_continuation_nll = cache.get("baseline_continuation_nll")
    print(f"original baseline: refusals={scorer.baseline_refusal_count}")


def _profiles(o_max: float, o_frac: float, d_max: float, d_frac: float,
              o_pos: float, o_dist: float, d_pos: float, d_dist: float) -> dict:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=o_pos,
            min_weight=max(0.0, float(o_max) * float(o_frac)),
            min_weight_distance=o_dist,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=float(d_max),
            max_weight_position=d_pos,
            min_weight=max(0.0, float(d_max) * float(d_frac)),
            min_weight_distance=d_dist,
        ),
    }


def _score(tag: str, scorer: TrialScorer, engine: SteeringEngine) -> dict:
    kl = float(scorer.measure_kl_divergence(engine))
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "refusals": int(refusals),
        "n": n,
        "kl": kl,
        "hit_bar": bool(int(refusals) <= BAR_REFUSALS and kl <= BAR_KL),
        "on_old_wall": bool(int(refusals) >= WALL_REFUSALS and kl >= WALL_KL),
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  HIT={row['hit_bar']} wall={row['on_old_wall']}",
        flush=True,
    )
    return row


def _copy_sidecars(src: Path, dst: Path) -> None:
    for name in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
        "README.md",
        "processor_config.json",
        "preprocessor_config.json",
        "configuration.json",
    ):
        s = src / name
        if s.is_file() and not (dst / name).exists():
            shutil.copy2(s, dst / name)


def _bake(engine: SteeringEngine, payload: dict, tag: str, report: Path, dst: Path) -> None:
    print(f"HIT {tag} — saving to {dst}", flush=True)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    model = engine.model.merge_and_unload()
    engine.needs_reload = True
    model.save_pretrained(str(dst), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(dst))
    _copy_sidecars(Path(BASE_MODEL), dst)
    payload["baked"] = True
    payload["merged"] = str(dst)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload, report)
    print(f"wrote {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True, help="stage-1 trial number")
    ap.add_argument("--grid", type=int, default=len(GRID), help="number of grid points to try")
    ap.add_argument("--leftover-n", type=int, default=400, help="target prompts to probe")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    report = _out_path(args.trial)
    payload: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stage1_trial": args.trial,
        "stage1_model": str(STAGE1),
        "cos_kill": COS_KILL,
        "bar": {"refusals": BAR_REFUSALS, "kl": BAR_KL},
        "points": [],
        "baked": False,
    }

    if not STAGE1.exists():
        raise SystemExit(
            f"stage-1 model missing: {STAGE1}\n"
            "run scripts/nex25_bake.py first and bake the winning trial."
        )

    cfg = _cfg()
    trial = load_trial(CHECKPOINT, BASE_MODEL, args.trial)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)  # provenance check against the base
    cfg.model.model_id = str(STAGE1)     # now load the baked stage-1 weights
    payload["stage1_attrs"] = {
        "refusals": trial.user_attrs.get("refusals"),
        "kl": trial.user_attrs.get("kl_divergence"),
        "vector_index": artifact.vector_index,
    }
    print(
        f"stage-1 trial {args.trial}: refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')} vector_index={artifact.vector_index}",
        flush=True,
    )
    _dump(payload, report)

    # Positions/distances come from the stage-1 recipe so r2 only varies strength.
    prof = artifact.profiles
    o_prof = prof.get("attn.o_proj")
    d_prof = prof.get("mlp.down_proj")
    o_pos = float(getattr(o_prof, "max_weight_position", 36.0))
    o_dist = float(getattr(o_prof, "min_weight_distance", 14.0))
    d_pos = float(getattr(d_prof, "max_weight_position", 37.0))
    d_dist = float(getattr(d_prof, "min_weight_distance", 12.5))
    print(f"profiles: o_pos={o_pos:.2f} o_dist={o_dist:.2f} d_pos={d_pos:.2f} d_dist={d_dist:.2f}")

    print("loading stage-1 model...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)

    base_row = _score("stage1_baked", scorer, engine)
    payload["points"].append(base_row)
    _dump(payload, report)
    if base_row["hit_bar"]:
        _bake(engine, payload, "stage1_baked", report, PEELED)
        return

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[: args.leftover_n]
    train_b = load_prompt_dataset(cfg, cfg.benign_prompts)[: max(64, args.leftover_n // 2)]
    print(f"generating {len(train_h)} target prompts under stage 1...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover = []
    prefixes = []
    for msg, text in zip(train_h, texts):
        t = text or ""
        if detector.detect_refusal(t):
            leftover.append(msg)
            prefixes.append(t[:120].replace("\n", "\\n"))
    print(f"leftover refusals {len(leftover)}/{len(train_h)}", flush=True)
    payload["leftover_n"] = len(leftover)
    payload["leftover_of"] = len(train_h)
    payload["leftover_prefixes_sample"] = prefixes[:10]
    _dump(payload, report)
    if len(leftover) < 8:
        payload["kill_reason"] = f"too few leftover refusals ({len(leftover)})"
        _dump(payload, report)
        print(f"KILL: {payload['kill_reason']}")
        return

    print("extracting leftover-vs-benign residuals for r2...", flush=True)
    tgt = engine.extract_hidden_states_batched(leftover)
    beni = engine.extract_hidden_states_batched(train_b)
    r2 = compute_configured_steering_vectors(beni, tgt, cfg)
    payload["r2_shape"] = list(r2.shape)
    print(f"r2 vectors {tuple(r2.shape)}", flush=True)

    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    g = resolve_global_vector(cache["vectors"], artifact.vector_index)
    cosines = {}
    if g is not None and r2.ndim == 2:
        g = g.float()
        for layer_i in (20, 30, 36, 37, 39):
            if layer_i < r2.shape[0]:
                cosines[f"layer_{layer_i}"] = float(
                    torch.nn.functional.cosine_similarity(g, r2[layer_i].float(), dim=0)
                )
    payload["cosines"] = cosines
    max_cos = max(cosines.values()) if cosines else 0.0
    payload["max_cos"] = max_cos
    _dump(payload, report)
    print(f"max cos(stage1, r2) = {max_cos:.4f}", flush=True)
    if max_cos > COS_KILL:
        payload["kill_reason"] = f"r2 collinear with stage 1 (cos={max_cos:.4f})"
        _dump(payload, report)
        print(f"KILL: {payload['kill_reason']}")
        return

    wall_streak: list[bool] = []
    for o_max, o_frac, d_max, d_frac in GRID[: args.grid]:
        engine.restore_baseline()
        apply_steering(
            engine,
            r2,
            None,
            _profiles(o_max, o_frac, d_max, d_frac, o_pos, o_dist, d_pos, d_dist),
            cfg,
            benign_states=beni,
        )
        tag = f"peel_o{o_max:.2f}_d{d_max:.2f}_f{o_frac:.2f}/{d_frac:.2f}"
        row = _score(tag, scorer, engine)
        row.update({"r2_o": o_max, "r2_d": d_max})
        payload["points"].append(row)
        _dump(payload, report)
        if row["hit_bar"]:
            _bake(engine, payload, tag, report, PEELED)
            return
        wall_streak.append(bool(row["on_old_wall"]))
        if len(wall_streak) >= WALL_POINTS_BEFORE_KILL and all(
            wall_streak[:WALL_POINTS_BEFORE_KILL]
        ):
            payload["kill_reason"] = "first grid points still on the old wall"
            _dump(payload, report)
            print(f"KILL: {payload['kill_reason']}")
            return

    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload, report)
    print("peel finished, no dual HIT", flush=True)


if __name__ == "__main__":
    main()
