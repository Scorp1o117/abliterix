#!/usr/bin/env python3
"""Mergeable mixes aiming at 8-12/100 @ KL<=0.1.

Pocket LoRA / ORBA t24-band sit on the same Pareto (50@0.087, 34@0.101).
This sweep tries biprojected, stronger ORBA, and t24 LoRA + extra ORBA peel.
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

from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline  # noqa: E402
from abliterix.scriptlib import extract_trial_artifact, load_trial, setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
CKPT = "checkpoints_qwen38_27b_pocket"
MODEL = "/run/media/s117/OS/Models/Qwen3.8-27B"
VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_mergeable_mix_sweep.json")
T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982
T24_VINDEX = 50.79667354766495


def _cfg(transform: DirectTransform) -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_mix", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = transform
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _t24_band(o_max: float, down_max: float = 0.37) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=T24_POS,
            min_weight=max(0.0, float(o_max) * 0.5),
            min_weight_distance=T24_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=float(down_max),
            max_weight_position=37.89,
            min_weight=max(0.0, float(down_max) * 0.6),
            min_weight_distance=30.46,
        ),
    }


def _flat(o_max: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=32.0,
            min_weight=float(o_max),
            min_weight_distance=40.0,
        )
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_mergeable_mix_sweep.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg(DirectTransform.BIPROJECTED)
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "points": [],
    }
    print("loading engine (direct)...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    vectors = cache["vectors"]
    benign = cache.get("benign_states")
    trial = load_trial(CKPT, MODEL, 24)
    t24 = extract_trial_artifact(trial)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    jobs = [
        ("biprojected_o7.5", DirectTransform.BIPROJECTED, _t24_band(7.5), T24_VINDEX, False),
        ("biprojected_o9.0", DirectTransform.BIPROJECTED, _t24_band(9.0), T24_VINDEX, False),
        ("orba_o8.0", DirectTransform.ORBA, _t24_band(8.0), T24_VINDEX, True),
        ("orba_o9.0", DirectTransform.ORBA, _t24_band(9.0), T24_VINDEX, True),
        ("orba_flat_o1.8", DirectTransform.ORBA, _flat(1.8), None, True),
        ("orba_flat_o2.6", DirectTransform.ORBA, _flat(2.6), None, True),
    ]

    hit = None
    for tag, transform, profiles, vindex, need_benign in jobs:
        engine.restore_baseline()
        cfg.steering.direct_transform = transform
        apply_steering(
            engine,
            vectors,
            vindex,
            profiles,
            cfg,
            benign_states=benign if need_benign else None,
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": tag,
            "keyword_refusals": int(refusals),
            "n": int(n),
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(
            f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
            flush=True,
        )
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            hit = tag
            payload["named_candidate"] = tag
            break

    if hit is None:
        payload["named_candidate"] = None
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("sweep finished, no in-budget point", flush=True)
        return

    print(f"HIT {hit} — saving merged...", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in ("chat_template.jinja", "generation_config.json"):
        src = Path(MODEL) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    payload["merged"] = str(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
