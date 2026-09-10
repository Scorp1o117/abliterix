#!/usr/bin/env python3
"""CGA V29: add qkv to the 12@0.047 all-layer global_prompt plateau.

Leftovers are compliance-theater / 'I can't', not Sorry-prefix. Prefix
retry and lower thresh did not move them. Add a small all-layer qkv
rotation beside o=1.35 (0.0026 KL budget remaining).
Runtime-only; dual HIT is distilled next.
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
OUT = Path("logs/spark_x25_cga_v29.json")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"
POS = 26.78
DIST = 13.01

# (o_strength, qkv_strength, threshold)
GRID = [
    (1.35, 0.15, 0.50),
    (1.35, 0.30, 0.50),
    (1.35, 0.45, 0.50),
    (1.35, 0.60, 0.50),
    (1.30, 0.30, 0.50),
    (1.32, 0.20, 0.50),
    (1.28, 0.40, 0.50),
    (1.35, 0.30, 0.45),
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_cga_v29", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.concept_gate_angular_overrotation = True
    cfg.steering.concept_gate_scope = "global_prompt"
    cfg.steering.concept_gate_positive_alignment_only = True
    cfg.steering.concept_gate_global_decision_layer = 24
    cfg.steering.svf_scorer_epochs = 20
    cfg.steering.svf_scorer_lr = 0.001
    cfg.steering.svf_scorer_hidden = 128
    return cfg


def _band(o_s: float, q_s: float) -> dict[str, SteeringProfile]:
    band = {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_s),
            max_weight_position=18.0,
            min_weight=float(o_s),
            min_weight_distance=40.0,
        ),
    }
    if q_s > 0:
        band["attn.qkv_proj"] = SteeringProfile(
            max_weight=float(q_s),
            max_weight_position=18.0,
            min_weight=float(q_s),
            min_weight_distance=40.0,
        )
    return band


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_cga_v29.json").write_text(text, encoding="utf-8")


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
        "apply": "CGA V29 all-layer o+qkv on 12@0.047 plateau; runtime-only, distill on HIT",
        "points": [],
    }
    print("loading original Spark (CGA V29 o+qkv plateau)...", flush=True)
    engine = SteeringEngine(cfg)
    cache = torch.load(STEERING, map_location="cpu", weights_only=False)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    _ensure_scorers(engine, cache, cfg)
    vectors = cache["vectors"]
    print(f"mean-diff vectors {tuple(vectors.shape)}", flush=True)

    for o_s, q_s, thresh in GRID:
        engine.restore_baseline()
        cfg.steering.concept_gate_scope = "global_prompt"
        cfg.steering.concept_gate_threshold = float(thresh)
        cfg.steering.concept_gate_positive_alignment_only = True
        cfg.steering.concept_gate_intervention_geometry = "angular"
        cfg.steering.concept_gate_global_decision_layer = 24
        apply_steering(engine, vectors, None, _band(o_s, q_s), cfg)
        n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
        tag = f"cga_gp_t{thresh:.2f}_o{o_s:.2f}_q{q_s:.2f}_d40_L24"
        print(
            f"hooks={n_hooks} t={thresh} o={o_s} q={q_s}",
            flush=True,
        )
        row = _score(tag, scorer, engine)
        row.update(
            {
                "scope": "global_prompt",
                "threshold": thresh,
                "o_strength": o_s,
                "qkv_strength": q_s,
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
