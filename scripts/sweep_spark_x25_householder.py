#!/usr/bin/env python3
"""Householder isometric peel on Spark-X2.5-4B original mean-diff.

Mean LoRA Pareto is stuck: 57/100 @ KL 0.0456 (V9 T32) vs ~7/100 @ 0.20.
Leftover r2/neg/orth peels did not beat it. Householder is a mergeable
norm-preserving reflection — V4 ORBA rank-1 at o≤1.8 never opened Spark,
but isometric edits can sit at higher strength with less KL.
HIT: refusals≤10/100 AND 3-token full_distribution_kl≤0.05 vs original.
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
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/spark_x25_4b_lora_v9.toml"
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
OUT = Path("logs/spark_x25_householder.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
POS = 26.78
DIST = 13.01
QPOS = 27.56
QDIST = 17.02

# Narrow layer band: default dist=13 touches ~layers 14–36. Refusal may
# live in late layers; shrinking dist cuts early-layer KL.
# (o_max, q_max, dist)
GRID = [
    (1.5, 0.0, 4.0),
    (2.0, 0.0, 4.0),
    (2.5, 0.0, 4.0),
    (3.0, 0.0, 4.0),
    (3.5, 0.0, 4.0),
    (4.0, 0.0, 4.0),
    (1.5, 0.0, 6.0),
    (2.0, 0.0, 6.0),
    (2.5, 0.0, 6.0),
    (3.0, 0.0, 6.0),
    (3.5, 0.0, 6.0),
    (2.0, 0.0, 8.0),
    (2.5, 0.0, 8.0),
    (3.0, 0.0, 8.0),
    (3.5, 0.0, 8.0),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_hh", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.HOUSEHOLDER
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_householder.json").write_text(text, encoding="utf-8")


def _band(o_max: float, q_max: float, dist: float) -> dict:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=POS,
            min_weight=max(0.0, float(o_max) * 0.55),
            min_weight_distance=float(dist),
        ),
        "attn.qkv_proj": SteeringProfile(
            max_weight=float(q_max),
            max_weight_position=QPOS,
            min_weight=max(0.0, float(q_max) * 0.3),
            min_weight_distance=float(dist),
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
    print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  HIT={row['hit']}", flush=True)
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
    print(f"HIT {tag} — saving {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    # Direct edits live on base weights; merge_and_unload keeps them and
    # drops zeroed LoRA wrappers.
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
        json.dumps({"tag": tag, "merged": str(MERGED), "points": payload.get("points")}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MERGED} and {HIT_TXT}", flush=True)


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "transform": "householder",
        "hit_rule": "refusals<=10/100 AND full_distribution_kl<=0.05 vs original",
        "points": [],
    }
    print("loading original Spark...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)

    for o_max, q_max, dist in GRID:
        engine.restore_baseline()
        apply_steering(
            engine,
            cache["vectors"],
            None,
            _band(o_max, q_max, dist),
            cfg,
            benign_states=cache.get("benign_states"),
        )
        tag = f"hh_o{o_max:.1f}_d{dist:.0f}"
        row = _score(tag, scorer, engine)
        row.update({"o": o_max, "q": q_max, "dist": dist})
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
