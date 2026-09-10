#!/usr/bin/env python3
"""T32 leftover-vs-opened contrast peel, scored vs original.

Earlier leftover peels used leftover-harmful vs BENIGN (cos 0.78 with r1)
and undid T32. Here both sides are harmful prompts: still-refusing vs
already-opened under T32. That should isolate the remaining first-token
'Sorry' direction instead of more harmfulness. Prefill last-token
residuals, then Householder peel on merged T32.
HIT: refusals≤10/100 AND 3-token KL≤0.05 vs original.
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
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402
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
OUT = Path("logs/spark_x25_opened_contrast.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
TRIAL = 32
POS = 26.78
DIST = 13.01

# Householder peel of leftover-vs-opened on merged T32.
GRID = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_opened", "--config", CONFIG, "--seed", "117"]
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
    (SCRATCH / "spark_x25_opened_contrast.json").write_text(text, encoding="utf-8")


def _hh_band(o_max: float) -> dict:
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


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "contrast": "leftover_refuse vs opened_harmful under T32",
        "hit_rule": "refusals<=10/100 AND full_distribution_kl<=0.05 vs original",
        "points": [],
    }
    trial = load_trial(CHECKPOINT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    apply_trial_artifact(cfg, artifact)
    print("loading original Spark...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    r1 = cache["vectors"]
    apply_steering(
        engine,
        r1,
        artifact.vector_index,
        artifact.profiles,
        cfg,
        benign_states=cache.get("benign_states"),
        target_states=cache.get("target_states"),
    )
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    t32 = _score("t32_base", scorer, engine)
    payload["points"].append(t32)
    _dump(payload)
    if t32["hit"]:
        _bake(engine, payload, "t32_base")
        return

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[:400]
    print(f"generating train harmful n={len(train_h)} under T32...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover, opened = [], []
    for m, t in zip(train_h, texts):
        if detector.detect_refusal(t or ""):
            leftover.append(m)
        else:
            opened.append(m)
    print(f"leftover {len(leftover)}  opened {len(opened)} / {len(train_h)}", flush=True)
    payload["leftover_n"] = len(leftover)
    payload["opened_n"] = len(opened)
    if len(leftover) < 8 or len(opened) < 8:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("too few leftover or opened", flush=True)
        return

    print("extracting leftover vs opened residuals...", flush=True)
    tgt = engine.extract_hidden_states_batched(leftover)
    opn = engine.extract_hidden_states_batched(opened)
    r2 = compute_configured_steering_vectors(opn, tgt, cfg)
    v1 = r1 if r1.ndim == 2 else r1[0]
    r2_orth = _orth(r2, v1)
    i = min(27, v1.shape[0] - 1, r2.shape[0] - 1)
    cos = float(torch.nn.functional.cosine_similarity(v1[i].float(), r2[i].float(), dim=0))
    cos_o = float(torch.nn.functional.cosine_similarity(v1[i].float(), r2_orth[i].float(), dim=0))
    print(f"cos layer~{i}: leftover-opened={cos:.4f}  orth={cos_o:.4f}", flush=True)
    payload["cos_r1_r2_layer27"] = cos
    payload["cos_r1_r2orth_layer27"] = cos_o
    _dump(payload)

    print("merging T32 LoRA, switching to Householder for contrast peel...", flush=True)
    merged = engine.model.merge_and_unload()
    engine.model = merged
    engine.needs_reload = False
    engine.config.steering.steering_mode = SteeringMode.DIRECT
    engine.config.steering.direct_transform = DirectTransform.HOUSEHOLDER
    engine.config.steering.direct_transform_preserve_row_norm = True
    engine._init_adapters()

    variants = {"open": r2, "orth": r2_orth}
    for variant, vec in variants.items():
        for o in GRID:
            engine.restore_baseline()
            apply_steering(
                engine,
                vec,
                None,
                _hh_band(o),
                engine.config,
                benign_states=opn,
            )
            tag = f"t32+hh{variant}_o{o:.1f}"
            row = _score(tag, scorer, engine)
            row.update({"variant": variant, "o": o})
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
