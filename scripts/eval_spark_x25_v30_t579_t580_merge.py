#!/usr/bin/env python3
"""Merge V30 T580 and T579 mean-LoRA adapters, score vs original 3-token KL.

Does not bake unless dual HIT (refusals ≤10/100 and KL ≤0.05, or ≤10 @ KL≤0.1
if 0.05 is clearly unreachable on this merge).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft.tuners.lora.layer import Linear as PeftLinear

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

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.util import flush_memory  # noqa: E402

CONFIG = "configs/spark_x25_4b_lora_v30.toml"
CHECKPOINT = "checkpoints_spark_x25_4b_lora_v30"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
OUT = Path("logs/spark_x25_v30_t579_t580_merge.json")
TRIALS = (579, 580)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_v30_merge", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    return cfg


def _dump(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  "
        f"HIT0.05={row['hit_005']} HIT0.1={row['hit_01']}",
        flush=True,
    )
    return row


def _apply(engine, cache, artifact, cfg) -> None:
    apply_trial_artifact(cfg, artifact)
    apply_steering(
        engine,
        cache["vectors"],
        artifact.vector_index,
        artifact.profiles,
        cfg,
        benign_states=cache.get("benign_states"),
        target_states=cache.get("target_states"),
    )


def _snapshot_lora(engine) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, mod in engine.model.named_modules():
        if not isinstance(mod, PeftLinear):
            continue
        a = mod.lora_A["default"].weight.detach().float().cpu().clone()
        b = mod.lora_B["default"].weight.detach().float().cpu().clone()
        out[name] = (a, b)
    return out


def _merge_keep_engine(engine: SteeringEngine) -> None:
    print("merging active LoRA into base, re-wrap adapters...", flush=True)
    merged = engine.model.merge_and_unload()
    engine.model = merged
    engine.needs_reload = False
    engine._init_adapters()


def _apply_avg_deltas(
    engine: SteeringEngine,
    snap_a: dict[str, tuple[torch.Tensor, torch.Tensor]],
    snap_b: dict[str, tuple[torch.Tensor, torch.Tensor]],
    scale: float = 0.5,
) -> int:
    n = 0
    names = set(snap_a) & set(snap_b)
    for name, mod in engine.model.named_modules():
        if name not in names or not isinstance(mod, PeftLinear):
            continue
        a1, b1 = snap_a[name]
        a2, b2 = snap_b[name]
        delta = scale * (
            (b1 @ a1).to(torch.float32) + (b2 @ a2).to(torch.float32)
        )
        w = mod.base_layer.weight
        w.data.add_(delta.to(device=w.device, dtype=w.dtype))
        torch.nn.init.zeros_(mod.lora_A["default"].weight)
        torch.nn.init.zeros_(mod.lora_B["default"].weight)
        n += 1
    return n


def _trial_summary(trial) -> dict:
    attrs = trial.user_attrs
    params = attrs.get("parameters") or {}
    return {
        "index": attrs.get("index", trial.number),
        "number": trial.number,
        "refusals": attrs.get("refusals"),
        "kl_divergence": attrs.get("kl_divergence"),
        "vector_index": attrs.get("vector_index"),
        "vector_scope": (attrs.get("steering_recipe") or {})
        .get("steering", {})
        .get("fixed_vector_scope"),
        "o_proj": params.get("attn.o_proj"),
        "qkv_proj": params.get("attn.qkv_proj"),
        "down_proj": params.get("mlp.down_proj"),
    }


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "trials": list(TRIALS),
        "checkpoint": CHECKPOINT,
        "model": MODEL,
        "hit_rule": "refusals<=10/100 AND 3-token KL <=0.05 (also report <=0.1)",
        "points": [],
    }

    loaded = {}
    artifacts = {}
    for tid in TRIALS:
        trial = load_trial(CHECKPOINT, MODEL, tid)
        art = extract_trial_artifact(trial)
        loaded[tid] = trial
        artifacts[tid] = art
        summary = _trial_summary(trial)
        payload[f"trial_{tid}"] = summary
        print(
            f"loaded T{tid}: refusals={summary['refusals']} "
            f"kl={summary['kl_divergence']} vector_index={summary['vector_index']} "
            f"scope={summary['vector_scope']}",
            flush=True,
        )

    apply_trial_artifact(cfg, artifacts[580])
    print("loading Spark + V30 steering cache...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    vectors = cache["vectors"]
    if vectors.ndim == 2:
        i = min(27, vectors.shape[0] - 1)
        # per-layer vs global: compare the two trials' resolved directions later
        payload["steering_vector_shape"] = list(vectors.shape)

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)

    snaps: dict[int, dict] = {}
    for tid in TRIALS:
        engine.restore_baseline()
        _apply(engine, cache, artifacts[tid], cfg)
        snaps[tid] = _snapshot_lora(engine)
        row = _score(f"t{tid}_alone", scorer, engine)
        payload["points"].append(row)
        _dump(payload)

    # Linear average of the two LoRA deltas on the original base.
    engine.restore_baseline()
    n_avg = _apply_avg_deltas(engine, snaps[579], snaps[580], scale=0.5)
    print(f"avg-delta mix wrote {n_avg} modules", flush=True)
    row = _score("t579_t580_avg_delta_0.5", scorer, engine)
    payload["points"].append(row)
    _dump(payload)

    # Sequential: T580 into weights, then T579 LoRA on top.
    engine.needs_reload = True
    engine.restore_baseline()
    _apply(engine, cache, artifacts[580], cfg)
    _merge_keep_engine(engine)
    _apply(engine, cache, artifacts[579], cfg)
    row = _score("t580_then_t579", scorer, engine)
    payload["points"].append(row)
    _dump(payload)

    # Sequential reverse: T579 into weights, then T580 LoRA on top.
    engine.needs_reload = True
    engine.restore_baseline()
    _apply(engine, cache, artifacts[579], cfg)
    _merge_keep_engine(engine)
    _apply(engine, cache, artifacts[580], cfg)
    row = _score("t579_then_t580", scorer, engine)
    payload["points"].append(row)
    _dump(payload)

    payload["named_candidate"] = None
    payload["baked"] = False
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote {OUT}", flush=True)
    for p in payload["points"]:
        print(
            f"  {p['tag']}: {p['keyword_refusals']}/{p['n']} @ "
            f"{p['full_distribution_kl_3token_vs_original']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
