#!/usr/bin/env python3
"""V9 T32 leftover peel for Spark-X2.5-4B, scored vs the original model.

Single-direction mean LoRA bottoms out at ~60/100 @ KL 0.046 (V9 T32) or
~7/100 @ KL 0.20. n_directions=3 stacked the *same* original residual and
exploded KL. Here we apply T32, keep only prompts that still refuse, and
mean-diff that leftover set vs benign — then LoRA-peel r2 on the merged T32
weights. HIT: keyword refusals ≤10/100 AND 3-token full_distribution_kl ≤0.05
vs the original baseline cache.
"""

from __future__ import annotations

import json
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
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringProfile  # noqa: E402
from abliterix.vectors import compute_configured_steering_vectors  # noqa: E402

CONFIG = "configs/spark_x25_4b_lora_v9.toml"
CHECKPOINT = "checkpoints_spark_x25_4b_lora_v9"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
MERGED = Path("/run/media/s117/OS/Models/Spark-X2.5-4B-abliterix")
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-85e2331705c4/implementer")
OUT = Path("logs/spark_x25_t32_leftover.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
TRIAL = 32
# T32 geometry (V9 log): o 5.97/4.53 pos 26.78 dist 13.01, qkv 1.07/0.22.
T32_O_POS = 26.78
T32_O_DIST = 13.01
T32_Q_POS = 27.56
T32_Q_DIST = 17.02

# r2 LoRA grid: small extra strength on leftover direction. down_proj stays 0.
GRID = [
    # o_max, o_min_frac, q_max, q_min_frac
    (0.6, 0.50, 0.0, 0.0),
    (0.6, 0.50, 0.4, 0.3),
    (1.2, 0.50, 0.0, 0.0),
    (1.2, 0.60, 0.5, 0.3),
    (1.8, 0.60, 0.0, 0.0),
    (1.8, 0.70, 0.6, 0.3),
    (2.4, 0.60, 0.0, 0.0),
    (2.4, 0.75, 0.8, 0.3),
    (3.2, 0.70, 0.0, 0.0),
    (3.2, 0.75, 1.0, 0.2),
    (4.0, 0.75, 0.0, 0.0),
    (4.0, 0.75, 1.2, 0.2),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_t32_leftover", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    return cfg


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_t32_leftover.json").write_text(text, encoding="utf-8")


def _profiles(o_max: float, o_min_frac: float, q_max: float, q_min_frac: float) -> dict:
    o_min = max(0.0, float(o_max) * float(o_min_frac))
    q_min = max(0.0, float(q_max) * float(q_min_frac))
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=T32_O_POS,
            min_weight=o_min,
            min_weight_distance=T32_O_DIST,
        ),
        "attn.qkv_proj": SteeringProfile(
            max_weight=float(q_max),
            max_weight_position=T32_Q_POS,
            min_weight=q_min,
            min_weight_distance=T32_Q_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.0,
            max_weight_position=34.80,
            min_weight=0.0,
            min_weight_distance=9.38,
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
        "hit": bool(int(refusals) <= 10 and kl <= 0.05),
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  HIT={row['hit']}",
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
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    HIT_TXT.write_text(
        json.dumps(
            {
                "tag": tag,
                "merged": str(MERGED),
                "points": payload.get("points"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MERGED} and {HIT_TXT}", flush=True)


def _merge_t32_keep_engine(engine: SteeringEngine, rank: int) -> None:
    print(f"merging T32 LoRA into base, re-wrap rank-{rank} LoRA for r2...", flush=True)
    merged = engine.model.merge_and_unload()
    engine.model = merged
    engine.needs_reload = False
    engine.config.steering.full_norm_lora_rank = rank
    engine.config.steering.n_directions = 1
    engine._init_adapters()


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base_trial": TRIAL,
        "checkpoint": CHECKPOINT,
        "model": MODEL,
        "hit_rule": "refusals<=10/100 AND full_distribution_kl<=0.05 vs original",
        "points": [],
    }
    print(f"loading trial {TRIAL} from {CHECKPOINT}...", flush=True)
    trial = load_trial(CHECKPOINT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
    print(
        f"T32 attrs refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')} "
        f"vector_index={artifact.vector_index}",
        flush=True,
    )

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
    t32_row = _score("t32_base", scorer, engine)
    payload["points"].append(t32_row)
    _dump(payload)
    if t32_row["hit"]:
        _bake(engine, payload, "t32_base")
        return

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[:400]
    train_b = load_prompt_dataset(cfg, cfg.benign_prompts)[:200]
    print(f"generating train harmful n={len(train_h)} under T32...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover = [m for m, t in zip(train_h, texts) if detector.detect_refusal(t or "")]
    print(f"leftover refusals {len(leftover)}/{len(train_h)}", flush=True)
    payload["leftover_n"] = len(leftover)
    payload["leftover_of"] = len(train_h)
    _dump(payload)
    if len(leftover) < 8:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        payload["named_candidate"] = None
        _dump(payload)
        print("too few leftover refusals to estimate r2", flush=True)
        return

    print("extracting leftover vs benign residuals...", flush=True)
    tgt = engine.extract_hidden_states_batched(leftover)
    beni = engine.extract_hidden_states_batched(train_b)
    r2 = compute_configured_steering_vectors(beni, tgt, cfg)
    print(f"r2 vectors {tuple(r2.shape)}", flush=True)

    v1 = cache["vectors"]
    if v1.ndim == 2 and r2.ndim == 2:
        i = min(27, v1.shape[0] - 1, r2.shape[0] - 1)
        cos = float(
            torch.nn.functional.cosine_similarity(v1[i].float(), r2[i].float(), dim=0)
        )
        print(f"cos(r_t32, r2) layer~{i} = {cos:.4f}", flush=True)
        payload["cos_r1_r2_layer27"] = cos
        _dump(payload)

    _merge_t32_keep_engine(engine, rank=3)

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
        tag = f"t32+r2_o{o_max:.1f}_q{q_max:.1f}"
        row = _score(tag, scorer, engine)
        row["r2_o"] = o_max
        row["r2_q"] = q_max
        payload["points"].append(row)
        _dump(payload)
        if row["hit"]:
            _bake(engine, payload, tag)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no dual HIT", flush=True)


if __name__ == "__main__":
    main()
