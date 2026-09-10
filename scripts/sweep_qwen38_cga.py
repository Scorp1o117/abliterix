#!/usr/bin/env python3
"""Adaptive + concept-gated angular on pocket mean-diff.

Ungated mean-diff angular (all-layer or t24-band) follows the same Pareto
as LoRA: ~50 refusals at KL 0.12, 9 at 0.46. Concept-gated angular only
rotates tokens the scorer labels harmful, so the *benign* 3-token KL
meter can stay low while harmful refusals drop.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_qwen38_vs_original import (  # noqa: E402
    POCKET_BASELINE,
    _inject_original_baseline,
)
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.svf import train_concept_scorers  # noqa: E402
from abliterix.types import SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
POCKET_VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT = Path("logs/qwen38_cga_sweep.json")
SCORER_CACHE = Path(
    "checkpoints_qwen38_27b_cga/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_concept_scorers.pt"
)

T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982


def _load_cfg(mode: SteeringMode) -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_cga", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = mode
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.angular_overrotation = False
    cfg.steering.concept_gate_angular_overrotation = False
    cfg.steering.concept_gate_positive_alignment_only = True
    return cfg


def _band(strength: float, position: float, distance: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=float(position),
            min_weight=0.0,
            min_weight_distance=float(distance),
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_sweep.json").write_text(text, encoding="utf-8")


def _score(scorer: TrialScorer, engine: SteeringEngine, tag: str, extra: dict) -> dict:
    kl = float(scorer.measure_kl_divergence(engine))
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        **extra,
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
        flush=True,
    )
    return row


def _ensure_scorers(engine: SteeringEngine, cache: dict, cfg: AbliterixConfig):
    if engine._concept_scorers:
        return
    SCORER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SCORER_CACHE.is_file():
        blob = torch.load(SCORER_CACHE, map_location="cpu", weights_only=False)
        engine._concept_scorers = blob["scorers"] if "scorers" in blob else blob
        print(f"loaded concept scorers {SCORER_CACHE}", flush=True)
        return
    benign = cache["benign_states"]
    target = cache["target_states"]
    print(
        f"training concept scorers on {tuple(benign.shape)} residuals...",
        flush=True,
    )
    device = next(engine.model.parameters()).device
    engine._concept_scorers = train_concept_scorers(
        benign,
        target,
        hidden_dim=int(benign.shape[2]),
        n_epochs=cfg.steering.svf_scorer_epochs,
        lr=cfg.steering.svf_scorer_lr,
        hidden_dim_scorer=cfg.steering.svf_scorer_hidden,
        device=device,
        seed=117,
    )
    torch.save({"scorers": engine._concept_scorers}, SCORER_CACHE)
    print(
        f"trained scorers n={len(engine._concept_scorers)} → {SCORER_CACHE}",
        flush=True,
    )


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _load_cfg(SteeringMode.ADAPTIVE_ANGULAR)
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "vectors": str(POCKET_VECTORS),
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "apply": "adaptive angular then concept-gated angular on mean-diff",
        "points": [],
    }
    print("loading engine (adaptive angular, no LoRA)...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    cache = torch.load(POCKET_VECTORS, map_location="cpu", weights_only=False)
    vectors = cache["vectors"]
    print(f"mean-diff vectors {tuple(vectors.shape)}", flush=True)

    jobs = [
        ("adaptive", 1.00, T24_POS, T24_DIST, "token", None),
        ("adaptive", 0.70, T24_POS, 12.0, "token", None),
        ("cga", 0.70, T24_POS, T24_DIST, "prompt", 0.50),
        ("cga", 1.00, T24_POS, T24_DIST, "prompt", 0.50),
        ("cga", 1.00, T24_POS, T24_DIST, "prompt", 0.70),
        ("cga", 1.00, T24_POS, 12.0, "prompt", 0.50),
        ("cga", 1.00, T24_POS, T24_DIST, "global_prompt", 0.60),
    ]

    for kind, strength, pos, dist, scope, thresh in jobs:
        engine.restore_baseline()
        if kind == "adaptive":
            cfg.steering.steering_mode = SteeringMode.ADAPTIVE_ANGULAR
            engine._concept_scorers = None
        else:
            cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
            cfg.steering.concept_gate_scope = scope
            cfg.steering.concept_gate_threshold = float(thresh)
            _ensure_scorers(engine, cache, cfg)
        apply_steering(engine, vectors, None, _band(strength, pos, dist), cfg)
        n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
        tag = (
            f"{kind}_{scope}_{strength:.2f}"
            if kind == "adaptive"
            else f"cga_{scope}_t{thresh:.2f}_{strength:.2f}_d{dist:.0f}"
        )
        print(
            f"hooks={n_hooks} {kind} scope={scope} thresh={thresh} "
            f"strength={strength} dist={dist}",
            flush=True,
        )
        row = _score(
            scorer,
            engine,
            tag,
            {
                "kind": kind,
                "scope": scope,
                "threshold": thresh,
                "strength": strength,
                "position": pos,
                "distance": dist,
                "n_hooks": n_hooks,
            },
        )
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {tag}", flush=True)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
