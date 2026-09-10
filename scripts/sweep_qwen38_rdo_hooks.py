#!/usr/bin/env python3
"""Apply cached RDO (then pocket mean-diff) as residual-stream hooks.

RDO was trained with f_ablate: h ← h − (h·r̂)r̂ at every decoder block.
The o_proj-only direct sweep stayed 100/100 @ KL ≤ 0.006 because that is
not the paper apply. This sweep installs the matching hooks:

  linear_projection  — paper f_ablate at scale α
  angular            — norm-preserving 90°·strength rotation

If RDO hooks miss, pocket mean-diff is applied the same way (activation
steering, not another mean+LoRA TPE).
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
from abliterix.core.steering import (  # noqa: E402
    _make_linear_projection_hook,
    apply_steering,
)
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_rdo.toml"
RDO_VECTORS = Path(
    "checkpoints_qwen38_27b_rdo/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
POCKET_VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT = Path("logs/qwen38_rdo_hooks_sweep.json")

# Paper α=1 is full residual projection at every layer.
RDO_LINEAR = (0.20, 0.50, 1.00)
RDO_ANGULAR = (0.50, 1.00)
MEAN_ANGULAR = (0.30, 0.50, 0.70, 1.00)


def _load_cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_rdo_hooks", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.angular_overrotation = False
    return cfg


def _flat(strength: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=32.0,
            min_weight=float(strength),
            min_weight_distance=40.0,
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_rdo_hooks_sweep.json").write_text(text, encoding="utf-8")


def _apply_linear(engine: SteeringEngine, vectors: torch.Tensor, fraction: float) -> int:
    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []
    installed = 0
    layers = engine.transformer_layers
    for layer_idx, layer in enumerate(layers):
        v = vectors[layer_idx + 1]
        hook = _make_linear_projection_hook(v, float(fraction), adaptive=False)
        engine._angular_hooks.append(layer.register_forward_hook(hook))
        installed += 1
    if installed == 0:
        raise RuntimeError("no decoder layers for linear residual hooks")
    return installed


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


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _load_cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rdo_vectors": str(RDO_VECTORS),
        "pocket_vectors": str(POCKET_VECTORS),
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "apply": "decoder_block residual hooks (linear_projection then angular)",
        "points": [],
    }
    print("loading engine (angular, no LoRA)...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    rdo = torch.load(RDO_VECTORS, map_location="cpu", weights_only=False)["vectors"]
    print(f"RDO vectors {tuple(rdo.shape)}", flush=True)

    jobs: list[tuple[str, str, float, torch.Tensor]] = []
    for a in RDO_LINEAR:
        jobs.append(("rdo", "linear", float(a), rdo))
    for a in RDO_ANGULAR:
        jobs.append(("rdo", "angular", float(a), rdo))

    pocket = None
    for source, geometry, strength, vectors in jobs:
        engine.restore_baseline()
        if geometry == "linear":
            n_hooks = _apply_linear(engine, vectors, strength)
            print(f"linear hooks={n_hooks} α={strength}", flush=True)
        else:
            apply_steering(engine, vectors, None, _flat(strength), cfg)
            n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
            print(f"angular hooks={n_hooks} strength={strength}", flush=True)
        tag = f"{source}_{geometry}_{strength:.2f}"
        row = _score(
            scorer,
            engine,
            tag,
            {
                "source": source,
                "geometry": geometry,
                "strength": strength,
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

    # Mean-diff residual angular: different apply path than pocket LoRA TPE.
    pocket = torch.load(POCKET_VECTORS, map_location="cpu", weights_only=False)["vectors"]
    print(f"pocket mean-diff vectors {tuple(pocket.shape)}", flush=True)
    for strength in MEAN_ANGULAR:
        engine.restore_baseline()
        apply_steering(engine, pocket, None, _flat(strength), cfg)
        n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
        tag = f"meandiff_angular_{strength:.2f}"
        print(f"angular hooks={n_hooks} strength={strength} source=meandiff", flush=True)
        row = _score(
            scorer,
            engine,
            tag,
            {
                "source": "meandiff",
                "geometry": "angular",
                "strength": strength,
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
