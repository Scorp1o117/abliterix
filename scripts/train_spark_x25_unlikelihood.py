#!/usr/bin/env python3
"""Unlikelihood LoRA on T32 leftover refusal prefixes, KL-capped vs original.

Residual peels cannot move the remaining first-token 'I'm sorry' without
KL 0.15+. After T32 (57/100 @ 0.0456), train rank-3 LoRA to LOWER
log-prob of each leftover's own refusal prefix, and stop if 3-token KL
vs original exceeds 0.049. HIT: refusals≤10/100 AND KL≤0.05.
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
import torch.nn.functional as F  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.util import chunk_batches  # noqa: E402

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
OUT = Path("logs/spark_x25_unlikelihood.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
TRIAL = 32
STEPS = 32
BATCH = 4
LR = 2e-3
EVAL_EVERY = 8
KL_CAP = 0.049
FROM_ORIGINAL = True


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_ul", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    return cfg


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_unlikelihood.json").write_text(text, encoding="utf-8")


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


def _prefix_logprob(engine: SteeringEngine, messages, prefixes: list[str]) -> torch.Tensor:
    """Mean log-prob of each row's continuation tokens (refusal prefix)."""
    inputs, clen = engine._tokenize_with_continuations(messages, prefixes)
    out = engine.model(**inputs, use_cache=False)
    logits = out.logits
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    labels = inputs["input_ids"][:, 1:]
    mask = inputs["attention_mask"][:, 1:]
    token_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    # Keep only the last clen[i] predicted tokens (continuation).
    losses = []
    for i, ncont in enumerate(clen.tolist()):
        ncont = int(ncont)
        valid = mask[i].nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() < ncont:
            continue
        cont_pos = valid[-ncont:]
        losses.append(token_lp[i, cont_pos].mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def main() -> None:
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "method": "unlikelihood LoRA on T32 leftover prefixes, KL cap 0.049",
        "points": [],
    }
    print("loading original Spark...", flush=True)
    torch.set_grad_enabled(False)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    if not FROM_ORIGINAL:
        trial = load_trial(CHECKPOINT, MODEL, TRIAL)
        artifact = extract_trial_artifact(trial)
        apply_trial_artifact(cfg, artifact)
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
        row = _score("t32_base", scorer, engine)
        payload["points"].append(row)
        _dump(payload)

    train_h = load_prompt_dataset(cfg, cfg.target_prompts)[:200]
    print(f"generating train harmful n={len(train_h)} under T32...", flush=True)
    texts = engine.generate_text_batched(
        train_h,
        skip_special_tokens=True,
        max_new_tokens=cfg.inference.max_gen_tokens,
        min_new_tokens=cfg.inference.min_gen_tokens,
    )
    leftover_m, leftover_p = [], []
    for m, t in zip(train_h, texts):
        if detector.detect_refusal(t or ""):
            pref = (t or "").strip()[:12]
            if pref:
                leftover_m.append(m)
                leftover_p.append(pref)
    print(f"leftover prefixes {len(leftover_m)}", flush=True)
    payload["leftover_n"] = len(leftover_m)
    if len(leftover_m) < 8:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        return

    if not FROM_ORIGINAL:
        print("merging T32, wrapping rank-3 LoRA for unlikelihood...", flush=True)
        merged = engine.model.merge_and_unload()
        engine.model = merged
        engine.needs_reload = False
        engine.config.steering.full_norm_lora_rank = 3
        engine.config.steering.n_directions = 1
        engine._init_adapters()
    else:
        print("training rank-3 LoRA unlikelihood from original...", flush=True)
        # Adapters already on the original; B is zeroed.
    engine.model.eval()
    for w in engine._lora_b_weights:
        w.requires_grad_(True)
    opt = torch.optim.Adam([w for w in engine._lora_b_weights], lr=LR)

    n = len(leftover_m)
    idx = 0
    for step in range(1, STEPS + 1):
        sl = slice(idx, idx + BATCH)
        if sl.start >= n:
            idx = 0
            sl = slice(0, BATCH)
        batch_m = leftover_m[sl]
        batch_p = leftover_p[sl]
        idx = sl.stop
        opt.zero_grad(set_to_none=True)
        with torch.enable_grad():
            # Minimize log p(refusal prefix) = unlikelihood.
            loss = _prefix_logprob(engine, batch_m, batch_p)
            loss.backward()
        opt.step()
        print(f"step {step}/{STEPS} ul_loss={float(loss.detach()):.4f}", flush=True)
        if step % EVAL_EVERY == 0 or step == STEPS:
            with torch.no_grad():
                row = _score(f"ul_step{step}", scorer, engine)
            row["step"] = step
            row["ul_loss"] = float(loss.detach())
            payload["points"].append(row)
            _dump(payload)
            if row["hit"]:
                _bake(engine, payload, row["tag"])
                return
            if row["full_distribution_kl_3token_vs_original"] > KL_CAP:
                print(
                    f"KL {row['full_distribution_kl_3token_vs_original']:.4f} "
                    f"> cap {KL_CAP}, stop",
                    flush=True,
                )
                break

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no dual HIT", flush=True)


if __name__ == "__main__":
    main()
