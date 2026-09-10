#!/usr/bin/env python3
"""Householder o=1.0 first stage + leftover second reflection.

HH dense grid: KL≤0.05 bottoms at 63/100 (o=1.35 @ 0.046); refusals≤10
needs o≥2.9 @ KL 0.182. First-stage HH o=1.0 is 79/100 @ 0.0285 — 0.0215
KL still in budget. Extract leftover-refusal mean-diff on that model, then
compose a second Householder on r2 / r2_orth. Mergeable. HIT: ≤10/100 and
3-token KL≤0.05 vs original.
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
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402
from abliterix.vectors import compute_configured_steering_vectors  # noqa: E402

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
OUT = Path("logs/spark_x25_hh_leftover.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
POS = 26.78
DIST = 13.01
STAGE1_O = 1.0

GRID = [
    ("r2", 0.30),
    ("r2", 0.60),
    ("r2", 0.90),
    ("r2", 1.20),
    ("r2", 1.50),
    ("orth", 0.30),
    ("orth", 0.60),
    ("orth", 0.90),
    ("orth", 1.20),
    ("orth", 1.50),
    ("orth", 1.80),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_hh_left", "--config", CONFIG, "--seed", "117"]
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
    (SCRATCH / "spark_x25_hh_leftover.json").write_text(text, encoding="utf-8")


def _band(o_max: float) -> dict:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=POS,
            min_weight=max(0.0, float(o_max) * 0.55),
            min_weight_distance=DIST,
        )
    }


def _orth(r2: torch.Tensor, r1: torch.Tensor) -> torch.Tensor:
    out = r2.float().clone()
    u = r1.float()
    if u.ndim != 2:
        u = u[0]
    n = min(out.shape[0], u.shape[0])
    for i in range(n):
        ui = u[i]
        vi = out[i]
        nu = torch.linalg.vector_norm(ui)
        if float(nu) > 1e-8:
            vi = vi - (torch.dot(vi, ui) / (nu * nu)) * ui
        nv = torch.linalg.vector_norm(vi)
        if float(nv) > 1e-8:
            vi = vi / nv
        out[i] = vi
    return out.to(dtype=r2.dtype)


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


def _apply_hh(engine, vectors, o_max, cfg, benign_states) -> None:
    apply_steering(
        engine,
        vectors,
        None,
        _band(o_max),
        cfg,
        benign_states=benign_states,
    )


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "stage1": f"householder o={STAGE1_O}",
        "hit_rule": "refusals<=10/100 AND full_distribution_kl<=0.05 vs original",
        "points": [],
    }
    print("loading original Spark...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    r1 = cache["vectors"]
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)

    engine.restore_baseline()
    _apply_hh(engine, r1, STAGE1_O, cfg, cache.get("benign_states"))
    row = _score(f"hh_o{STAGE1_O:.1f}_base", scorer, engine)
    payload["points"].append(row)
    _dump(payload)
    if row["hit"]:
        _bake(engine, payload, row["tag"])
        return

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[:400]
    train_b = load_prompt_dataset(cfg, cfg.benign_prompts)[:200]
    print(f"generating train harmful n={len(train_h)} under HH o={STAGE1_O}...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover = [m for m, t in zip(train_h, texts) if detector.detect_refusal(t or "")]
    print(f"leftover refusals {len(leftover)}/{len(train_h)}", flush=True)
    payload["leftover_n"] = len(leftover)
    if len(leftover) < 8:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("too few leftover refusals", flush=True)
        return

    tgt = engine.extract_hidden_states_batched(leftover)
    beni = engine.extract_hidden_states_batched(train_b)
    r2 = compute_configured_steering_vectors(beni, tgt, cfg)
    r2_orth = _orth(r2, r1 if r1.ndim == 2 else r1[0])
    i = min(27, r1.shape[-2] - 1, r2.shape[0] - 1)
    v1 = r1[i] if r1.ndim == 2 else r1[0, i]
    cos = float(torch.nn.functional.cosine_similarity(v1.float(), r2[i].float(), dim=0))
    cos_o = float(torch.nn.functional.cosine_similarity(v1.float(), r2_orth[i].float(), dim=0))
    print(f"cos layer~{i}: r2={cos:.4f} orth={cos_o:.4f}", flush=True)
    payload["cos_r1_r2_layer27"] = cos
    payload["cos_r1_r2orth_layer27"] = cos_o
    _dump(payload)
    variants = {"r2": r2, "orth": r2_orth}

    for variant, o2 in GRID:
        engine.restore_baseline()
        _apply_hh(engine, r1, STAGE1_O, cfg, cache.get("benign_states"))
        _apply_hh(engine, variants[variant], o2, cfg, beni)
        tag = f"hh1.0+hh{variant}_o{o2:.1f}"
        row = _score(tag, scorer, engine)
        row.update({"variant": variant, "o2": o2})
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
