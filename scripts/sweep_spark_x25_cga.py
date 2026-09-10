#!/usr/bin/env python3
"""Concept-gated angular on Spark-X2.5-4B V9 mean-diff.

SOM 2x2 LoRA: 17/30 ~57est @ KL 0.1385, 2/30 @ 0.82 — worse than mean.
CGA only rotates tokens the scorer labels harmful, so the benign 3-token
KL meter can stay low while harmful refusals drop. Runtime-only; a dual
HIT is recorded for distill, not baked as merged weights.
HIT: ≤10/100 AND 3-token KL≤0.05 vs original.
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

from eval_qwen38_vs_original import _inject_original_baseline  # noqa: E402
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

CONFIG = "configs/spark_x25_4b_lora_v9.toml"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
BASELINE = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
STEERING = Path(
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_steering.pt"
)
SCORER_CACHE = Path(
    "checkpoints_spark_x25_4b_cga_v25/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_concept_scorers.pt"
)
SCRATCH = Path("/tmp/grok-goal-85e2331705c4/implementer")
OUT = Path("logs/spark_x25_cga_v25.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
POS = 26.78
DIST = 13.01

# (scope, strength, threshold, position, distance)
GRID = [
    ("prompt", 1.00, 0.20, POS, DIST),
    ("prompt", 1.00, 0.15, POS, DIST),
    ("prompt", 1.00, 0.30, POS, DIST),
    ("prompt", 1.40, 0.20, POS, DIST),
    ("prompt", 1.80, 0.20, POS, DIST),
    ("prompt", 1.40, 0.15, POS, DIST),
    ("prompt", 1.00, 0.20, 18.0, 40.0),
    ("prompt", 1.40, 0.15, 18.0, 40.0),
    ("token", 1.00, 0.50, POS, DIST),
    ("token", 1.40, 0.30, POS, DIST),
    ("global_prompt", 1.00, 0.50, POS, DIST),
    ("global_prompt", 1.40, 0.30, POS, DIST),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_cga", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.concept_gate_angular_overrotation = True
    cfg.steering.concept_gate_positive_alignment_only = True
    cfg.steering.concept_gate_global_decision_layer = 24
    cfg.steering.svf_scorer_epochs = 20
    cfg.steering.svf_scorer_lr = 0.001
    cfg.steering.svf_scorer_hidden = 128
    return cfg


def _band(strength: float, position: float, distance: float) -> dict[str, SteeringProfile]:
    # Wide-band (distance>=30) keeps min_weight == strength so every layer fires.
    min_w = float(strength) if distance >= 30.0 else max(0.0, float(strength) * 0.55)
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=float(position),
            min_weight=min_w,
            min_weight_distance=float(distance),
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_cga_v25.json").write_text(text, encoding="utf-8")


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
        "needs_distill": True,
    }
    print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  HIT={row['hit']}", flush=True)
    return row


def _ensure_scorers(engine: SteeringEngine, cache: dict, cfg: AbliterixConfig) -> None:
    if getattr(engine, "_concept_scorers", None):
        return
    SCORER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if SCORER_CACHE.is_file():
        blob = torch.load(SCORER_CACHE, map_location="cpu", weights_only=False)
        engine._concept_scorers = blob["scorers"] if "scorers" in blob else blob
        print(f"loaded concept scorers {SCORER_CACHE}", flush=True)
        return
    benign = cache["benign_states"]
    target = cache["target_states"]
    print(f"training concept scorers on {tuple(benign.shape)} residuals...", flush=True)
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
    print(f"trained scorers n={len(engine._concept_scorers)} → {SCORER_CACHE}", flush=True)


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "vectors": str(STEERING),
        "apply": "concept-gated angular on V9 mean-diff; runtime-only, distill on HIT",
        "points": [],
    }
    print("loading original Spark (CGA, no LoRA)...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    _ensure_scorers(engine, cache, cfg)
    vectors = cache["vectors"]
    print(f"mean-diff vectors {tuple(vectors.shape)}", flush=True)

    for scope, strength, thresh, pos, dist in GRID:
        engine.restore_baseline()
        cfg.steering.concept_gate_scope = scope
        cfg.steering.concept_gate_threshold = float(thresh)
        apply_steering(engine, vectors, None, _band(strength, pos, dist), cfg)
        n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
        tag = f"cga_{scope}_t{thresh:.2f}_s{strength:.2f}_d{dist:.0f}"
        print(
            f"hooks={n_hooks} scope={scope} thresh={thresh} "
            f"strength={strength} dist={dist}",
            flush=True,
        )
        row = _score(tag, scorer, engine)
        row.update(
            {
                "scope": scope,
                "threshold": thresh,
                "strength": strength,
                "position": pos,
                "distance": dist,
                "n_hooks": n_hooks,
            }
        )
        payload["points"].append(row)
        _dump(payload)
        if row["hit"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            HIT_TXT.write_text(
                json.dumps(
                    {
                        "tag": tag,
                        "needs_distill": True,
                        "runtime_only": True,
                        "merged": None,
                        "points": payload["points"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"HIT {tag} — runtime CGA, wrote {HIT_TXT} (distill next)", flush=True)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no dual HIT", flush=True)


if __name__ == "__main__":
    main()
