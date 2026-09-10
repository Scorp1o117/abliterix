#!/usr/bin/env python3
"""Householder direct on pocket mean-diff (mergeable isometric peel)."""

from __future__ import annotations

import json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline
from abliterix.scriptlib import setup_io

setup_io()
import torch
from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.eval.detector import RefusalDetector
from abliterix.eval.scorer import TrialScorer
from abliterix.settings import AbliterixConfig
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
VECTORS = Path("checkpoints_qwen38_27b_pocket/--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt")
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_householder_sweep.json")
POS, DIST = 49.611242819678694, 5.105344096840982


def _cfg():
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["hh", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.HOUSEHOLDER
    return cfg


def _band(s):
    return {"attn.o_proj": SteeringProfile(max_weight=s, max_weight_position=POS, min_weight=max(0.0, s * 0.5), min_weight_distance=DIST)}


def _dump(p):
    t = json.dumps(p, indent=2)
    OUT.write_text(t)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_householder_sweep.json").write_text(t)


def main():
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {"started_utc": datetime.now(timezone.utc).isoformat(), "points": []}
    engine = SteeringEngine(cfg)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    hit = None
    for s in (0.35, 0.70, 1.00):
        engine.restore_baseline()
        apply_steering(engine, cache["vectors"], 50.79667354766495, _band(s), cfg, benign_states=cache.get("benign_states"))
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {"tag": f"hh_{s:.2f}", "keyword_refusals": int(refusals), "n": n, "full_distribution_kl_3token_vs_original": kl, "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1)}
        print(f"SCORE {row['tag']}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            hit = s
            payload["named_candidate"] = row["tag"]
            break
    if hit is None:
        payload["named_candidate"] = None
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("sweep finished, no in-budget point", flush=True)
        return
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    payload["merged"] = str(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
