#!/usr/bin/env python3
"""T32 leftover-vs-opened FIRST-TOKEN contrast, scored vs original.

Prefill leftover-vs-opened was a new direction (cos 0.16) but HH peel
jumped KL 0.045→0.08 while only dropping 57→55. Remaining refusals start
with 'I'm sorry' in the first tokens, so extract residuals at the first
generated-token position (teacher-forced continuation), leftover vs opened.
LoRA peel on merged T32 — LoRA was the KL-efficient tool for r1.
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
from abliterix.types import SteeringProfile  # noqa: E402
from abliterix.util import chunk_batches  # noqa: E402
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
OUT = Path("logs/spark_x25_first_token.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
TRIAL = 32
POS = 26.78
DIST = 13.01
GRID = [
    (0.4, 0.0),
    (0.8, 0.0),
    (1.2, 0.0),
    (1.8, 0.0),
    (2.4, 0.0),
    (3.2, 0.0),
    (0.8, 0.3),
    (1.8, 0.5),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_ftok", "--config", CONFIG, "--seed", "117"]
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
    (SCRATCH / "spark_x25_first_token.json").write_text(text, encoding="utf-8")


def _band(o_max: float, q_max: float) -> dict:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=POS,
            min_weight=max(0.0, float(o_max) * 0.55),
            min_weight_distance=DIST,
        ),
        "attn.qkv_proj": SteeringProfile(
            max_weight=float(q_max),
            max_weight_position=27.56,
            min_weight=max(0.0, float(q_max) * 0.3),
            min_weight_distance=17.02,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.0,
            max_weight_position=34.8,
            min_weight=0.0,
            min_weight_distance=9.38,
        ),
    }


def _prefix(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return " "
    return t[:12]


def _extract_at_prefix(engine: SteeringEngine, messages, prefixes: list[str]):
    parts = []
    for m_batch, p_batch in zip(
        chunk_batches(messages, engine.config.inference.batch_size),
        chunk_batches(prefixes, engine.config.inference.batch_size),
    ):
        inputs, _clen = engine._tokenize_with_continuations(m_batch, p_batch)
        fwd = {"output_hidden_states": True, "use_cache": False}
        part = engine._stack_forward_residuals(inputs, fwd, token_offset=-1)
        parts.append(part.cpu() if part.is_cuda else part)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(parts, dim=0)


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
        "contrast": "first-token leftover vs opened under T32",
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
    leftover_m, leftover_p, opened_m, opened_p = [], [], [], []
    for m, t in zip(train_h, texts):
        pref = _prefix(t)
        if detector.detect_refusal(t or ""):
            leftover_m.append(m)
            leftover_p.append(pref)
        else:
            opened_m.append(m)
            opened_p.append(pref)
    print(f"leftover {len(leftover_m)} opened {len(opened_m)}", flush=True)
    print("leftover prefixes:", leftover_p[:8], flush=True)
    print("opened prefixes:", opened_p[:8], flush=True)
    payload["leftover_n"] = len(leftover_m)
    payload["opened_n"] = len(opened_m)
    payload["leftover_prefixes_sample"] = leftover_p[:12]
    payload["opened_prefixes_sample"] = opened_p[:12]
    if len(leftover_m) < 8 or len(opened_m) < 8:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("too few leftover or opened", flush=True)
        return

    print("extracting first-token leftover vs opened residuals...", flush=True)
    tgt = _extract_at_prefix(engine, leftover_m, leftover_p)
    opn = _extract_at_prefix(engine, opened_m, opened_p)
    # compute_configured expects (n, layers, hidden) or (layers, hidden)?
    r2 = compute_configured_steering_vectors(opn, tgt, cfg)
    v1 = r1 if r1.ndim == 2 else r1[0]
    if r2.ndim == 3:
        # (n, layers, hidden) should already be reduced
        print(f"r2 shape unexpected {tuple(r2.shape)}", flush=True)
    r2_orth = _orth(r2, v1)
    i = min(27, v1.shape[0] - 1, r2.shape[0] - 1)
    cos = float(torch.nn.functional.cosine_similarity(v1[i].float(), r2[i].float(), dim=0))
    print(f"cos layer~{i} first-token leftover-opened = {cos:.4f}", flush=True)
    payload["cos_r1_r2_layer27"] = cos
    _dump(payload)

    print("merging T32 LoRA, re-wrap rank-3 for first-token peel...", flush=True)
    merged = engine.model.merge_and_unload()
    engine.model = merged
    engine.needs_reload = False
    engine.config.steering.full_norm_lora_rank = 3
    engine.config.steering.n_directions = 1
    engine._init_adapters()

    for variant, vec in (("ftok", r2), ("orth", r2_orth)):
        for o, q in GRID:
            engine.restore_baseline()
            apply_steering(
                engine,
                vec,
                None,
                _band(o, q),
                engine.config,
                benign_states=opn if opn.ndim == 3 else None,
            )
            tag = f"t32+lora{variant}_o{o:.1f}_q{q:.1f}"
            row = _score(tag, scorer, engine)
            row.update({"variant": variant, "o": o, "q": q})
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
