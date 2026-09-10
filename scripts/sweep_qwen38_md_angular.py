#!/usr/bin/env python3
"""Mean-diff residual angular with a narrow layer band, not all 64 layers.

Flat all-layer mean-diff angular already scored 23/100 @ 0.327 at strength
0.30 — too much KL. Pocket t24 LoRA only peaks around layer 49.6 with
min_weight_distance 5.1. This sweep uses that envelope (and a slightly
wider one) plus a couple of weak flat points near KL 0.1.
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
POCKET_VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT = Path("logs/qwen38_md_angular_sweep.json")

# t24 LoRA envelope: o_proj peak ~49.6, distance 5.1 (layers ~44–55).
T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982


def _load_cfg(*, adaptive: bool = False) -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_md_angular", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = (
        SteeringMode.ADAPTIVE_ANGULAR if adaptive else SteeringMode.ANGULAR
    )
    cfg.steering.runtime_hook_site = "decoder_block"
    cfg.steering.angular_overrotation = False
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
    (SCRATCH / "qwen38_md_angular_sweep.json").write_text(text, encoding="utf-8")


def _apply_linear_band(
    engine: SteeringEngine,
    vectors: torch.Tensor,
    strength: float,
    position: float,
    distance: float,
) -> int:
    if not hasattr(engine, "_angular_hooks"):
        engine._angular_hooks = []
    installed = 0
    for layer_idx, layer in enumerate(engine.transformer_layers):
        dist = abs(layer_idx - position)
        if dist > distance:
            continue
        t = dist / distance if distance > 0 else 0.0
        frac = float(strength) * (1.0 - t)  # linear decay to 0 at edge
        if frac <= 0:
            continue
        v = vectors[layer_idx + 1]
        hook = _make_linear_projection_hook(v, frac, adaptive=False)
        engine._angular_hooks.append(layer.register_forward_hook(hook))
        installed += 1
    if installed == 0:
        raise RuntimeError("linear band installed no hooks")
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
    cfg = _load_cfg(adaptive=False)
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "vectors": str(POCKET_VECTORS),
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "apply": "mean-diff decoder_block angular/linear, t24 band then weak flat",
        "points": [],
    }
    print("loading engine (angular, no LoRA)...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    vectors = torch.load(POCKET_VECTORS, map_location="cpu", weights_only=False)["vectors"]
    print(f"mean-diff vectors {tuple(vectors.shape)}", flush=True)

    jobs = []
    for s in (0.40, 0.70, 1.00):
        jobs.append(("angular_t24band", s, T24_POS, T24_DIST, "angular"))
    for s in (0.40, 0.70):
        jobs.append(("angular_wide12", s, T24_POS, 12.0, "angular"))
    for s in (0.50, 1.00):
        jobs.append(("linear_t24band", s, T24_POS, T24_DIST, "linear"))
    for s in (0.08, 0.14):
        jobs.append(("angular_flat", s, 32.0, 40.0, "angular"))

    for name, strength, pos, dist, geometry in jobs:
        engine.restore_baseline()
        cfg.steering.steering_mode = SteeringMode.ANGULAR
        if geometry == "linear":
            n_hooks = _apply_linear_band(engine, vectors, strength, pos, dist)
        else:
            apply_steering(engine, vectors, None, _band(strength, pos, dist), cfg)
            n_hooks = len(getattr(engine, "_angular_hooks", []) or [])
        tag = f"md_{name}_{strength:.2f}"
        print(
            f"{geometry} hooks={n_hooks} strength={strength} pos={pos:.1f} dist={dist}",
            flush=True,
        )
        row = _score(
            scorer,
            engine,
            tag,
            {
                "source": "meandiff",
                "geometry": geometry,
                "band": name,
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
