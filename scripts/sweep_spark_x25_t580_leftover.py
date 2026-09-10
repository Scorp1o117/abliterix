#!/usr/bin/env python3
"""V30 T580 leftover peel for Spark-X2.5-4B.

Base is the KL≤0.1 champion (20/100 @ 0.0929). Estimate a second mean
direction only on prompts that still refuse under T580, then LoRA-peel it
on the merged T580 weights.

Kill immediately when:
  - cos(T580 global direction, leftover direction) > 0.85
  - the first 3 grid points stay on the old wall (refusals ≥12 and KL ≥0.11)

Do not bake unless dual HIT (≤10/100 and 3-token KL ≤0.05).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_qwen38_vs_original import _inject_original_baseline  # noqa: E402
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

CONFIG = "configs/spark_x25_4b_lora_v30.toml"
CHECKPOINT = "checkpoints_spark_x25_4b_lora_v30"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
MERGED = Path("/run/media/s117/OS/Models/Spark-X2.5-4B-abliterix")
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
OUT = Path("logs/spark_x25_t580_leftover.json")
TRIAL = 580
COS_KILL = 0.85
WALL_REFUSALS = 12
WALL_KL = 0.11
WALL_POINTS_BEFORE_KILL = 3

# T580 o/qkv geometry (from journal replay).
T580_O_POS = 23.47562572350038
T580_O_DIST = 8.22954498996837
T580_Q_POS = 33.919608857738886
T580_Q_DIST = 17.354790253099896

GRID = [
    (0.4, 0.50, 0.0, 0.0),
    (0.8, 0.50, 0.0, 0.0),
    (1.2, 0.50, 0.0, 0.0),
    (1.2, 0.50, 0.4, 0.3),
    (1.8, 0.50, 0.0, 0.0),
    (2.4, 0.60, 0.0, 0.0),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_t580_leftover", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.n_directions = 1
    return cfg


def _dump(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _profiles(o_max: float, o_min_frac: float, q_max: float, q_min_frac: float) -> dict:
    o_min = max(0.0, float(o_max) * float(o_min_frac))
    q_min = max(0.0, float(q_max) * float(q_min_frac))
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=T580_O_POS,
            min_weight=o_min,
            min_weight_distance=T580_O_DIST,
        ),
        "attn.qkv_proj": SteeringProfile(
            max_weight=float(q_max),
            max_weight_position=T580_Q_POS,
            min_weight=q_min,
            min_weight_distance=T580_Q_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.0,
            max_weight_position=30.11,
            min_weight=0.0,
            min_weight_distance=18.69,
        ),
    }


def _score(tag: str, scorer: TrialScorer, engine: SteeringEngine) -> dict:
    kl = float(scorer.measure_kl_divergence(engine))
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "keyword_refusals": int(refusals),
        "n": n,
        "full_distribution_kl_3token_vs_original": kl,
        "hit_005": bool(int(refusals) <= 10 and kl <= 0.05),
        "hit_01": bool(int(refusals) <= 10 and kl <= 0.1),
        "on_old_wall": bool(int(refusals) >= WALL_REFUSALS and kl >= WALL_KL),
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  "
        f"HIT0.05={row['hit_005']} HIT0.1={row['hit_01']} wall={row['on_old_wall']}",
        flush=True,
    )
    return row


def _copy_spark_sidecars(dst: Path) -> None:
    src = Path(MODEL)
    for name in (
        "configuration_spark.py",
        "modeling_spark.py",
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
        "README.md",
    ):
        s = src / name
        if s.is_file() and not (dst / name).exists():
            shutil.copy2(s, dst / name)


def _bake(engine: SteeringEngine, payload: dict, tag: str) -> None:
    print(f"HIT {tag} — merging and saving {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    if hasattr(engine.model, "merge_and_unload"):
        model = engine.model.merge_and_unload()
        engine.needs_reload = True
    else:
        model = engine.export_merged()
    model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    _copy_spark_sidecars(MERGED)
    payload["named_candidate"] = tag
    payload["merged"] = str(MERGED)
    payload["baked"] = True
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote {MERGED}", flush=True)


def _merge_t580_keep_engine(engine: SteeringEngine, rank: int) -> None:
    print(f"merging T580 LoRA into base, re-wrap rank-{rank} LoRA for r2...", flush=True)
    merged = engine.model.merge_and_unload()
    engine.model = merged
    engine.needs_reload = False
    engine.config.steering.full_norm_lora_rank = rank
    engine.config.steering.n_directions = 1
    engine._init_adapters()


def _finish(payload: dict, reason: str) -> None:
    payload["kill_reason"] = reason
    payload["named_candidate"] = payload.get("named_candidate")
    payload["baked"] = bool(payload.get("baked"))
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"KILL: {reason}", flush=True)


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_trial": TRIAL,
        "checkpoint": CHECKPOINT,
        "model": MODEL,
        "cos_kill": COS_KILL,
        "wall_kill": {
            "refusals_ge": WALL_REFUSALS,
            "kl_ge": WALL_KL,
            "points": WALL_POINTS_BEFORE_KILL,
        },
        "points": [],
        "baked": False,
    }
    print(f"loading trial {TRIAL} from {CHECKPOINT}...", flush=True)
    trial = load_trial(CHECKPOINT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
    cfg.steering.n_directions = 1
    print(
        f"T580 attrs refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')} "
        f"vector_index={artifact.vector_index}",
        flush=True,
    )
    payload["t580_attrs"] = {
        "refusals": trial.user_attrs.get("refusals"),
        "kl_divergence": trial.user_attrs.get("kl_divergence"),
        "vector_index": artifact.vector_index,
    }

    print("loading original Spark...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    apply_steering(
        engine,
        cache["vectors"],
        artifact.vector_index,
        artifact.profiles,
        cfg,
        benign_states=cache.get("benign_states"),
        target_states=cache.get("target_states"),
    )

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    base_row = _score("t580_base", scorer, engine)
    payload["points"].append(base_row)
    _dump(payload)
    if base_row["hit_005"]:
        _bake(engine, payload, "t580_base")
        return

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[:400]
    train_b = load_prompt_dataset(cfg, cfg.benign_prompts)[:200]
    print(f"generating train harmful n={len(train_h)} under T580...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover = []
    leftover_prefix = []
    for msg, text in zip(train_h, texts):
        t = text or ""
        if detector.detect_refusal(t):
            leftover.append(msg)
            leftover_prefix.append(t[:120].replace("\n", "\\n"))
    print(f"leftover refusals {len(leftover)}/{len(train_h)}", flush=True)
    payload["leftover_n"] = len(leftover)
    payload["leftover_of"] = len(train_h)
    payload["leftover_prefixes_sample"] = leftover_prefix[:12]
    _dump(payload)
    if len(leftover) < 8:
        _finish(payload, f"too few leftover refusals ({len(leftover)}) to estimate r2")
        return

    print("extracting leftover vs benign residuals...", flush=True)
    tgt = engine.extract_hidden_states_batched(leftover)
    beni = engine.extract_hidden_states_batched(train_b)
    r2 = compute_configured_steering_vectors(beni, tgt, cfg)
    print(f"r2 vectors {tuple(r2.shape)}", flush=True)

    v1 = cache["vectors"]
    g = resolve_global_vector(v1, artifact.vector_index)
    cosines = {}
    if g is not None and r2.ndim == 2:
        g = g.float()
        for layer_i in (19, 20, 23, 24):
            if layer_i < r2.shape[0]:
                cos = float(
                    torch.nn.functional.cosine_similarity(
                        g, r2[layer_i].float(), dim=0
                    )
                )
                cosines[f"layer_{layer_i}"] = cos
                print(f"cos(T580_global, r2) layer {layer_i} = {cos:.4f}", flush=True)
    payload["cosines"] = cosines
    max_cos = max(cosines.values()) if cosines else 0.0
    payload["max_cos"] = max_cos
    _dump(payload)
    if max_cos > COS_KILL:
        _finish(
            payload,
            f"leftover direction collinear with T580 (max cos={max_cos:.4f} > {COS_KILL})",
        )
        return

    _merge_t580_keep_engine(engine, rank=3)

    wall_streak = []
    for o_max, o_frac, q_max, q_frac in GRID:
        engine.restore_baseline()
        apply_steering(
            engine,
            r2,
            None,
            _profiles(o_max, o_frac, q_max, q_frac),
            cfg,
            benign_states=beni,
        )
        tag = f"t580+r2_o{o_max:.1f}_q{q_max:.1f}"
        row = _score(tag, scorer, engine)
        row["r2_o"] = o_max
        row["r2_q"] = q_max
        payload["points"].append(row)
        _dump(payload)
        if row["hit_005"]:
            _bake(engine, payload, tag)
            return
        wall_streak.append(bool(row["on_old_wall"]))
        if (
            len(wall_streak) >= WALL_POINTS_BEFORE_KILL
            and all(wall_streak[:WALL_POINTS_BEFORE_KILL])
        ):
            _finish(
                payload,
                "first 3 leftover grid points still on old wall "
                f"(refusals>={WALL_REFUSALS} and KL>={WALL_KL})",
            )
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no dual HIT", flush=True)


if __name__ == "__main__":
    main()
