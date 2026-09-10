#!/usr/bin/env python3
"""Map remaining methods that are not another mean+LoRA TPE pocket.

Phase A: lerp the stored full-weight ARA matrix toward / past the original
(``W <- (1-s) W0 + s W_ara``). Scale 1.0 already scored 61/100 @ 0.1208.

Phase B: apply pocket t24, re-extract a residual refusal direction on the
steered model, then add a small direct-weight peel on o_proj and score
each point vs the original 3-token baseline.

Every point is keyword refusals + teacher-forced 3-token
``full_distribution_kl`` vs original. Writes JSON as it goes.
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
    _hf_layers,
    _layer_modules,
    apply_ara_delta,
)
from abliterix.scriptlib import (  # noqa: E402
    extract_trial_artifact,
    load_trial,
    setup_io,
)

setup_io()

import torch  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import (  # noqa: E402
    _apply_direct_steering,
    apply_steering,
    resolve_global_vector,
)
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringProfile  # noqa: E402
from abliterix.util import flush_memory, slugify_model_name  # noqa: E402
from abliterix.vectors import compute_configured_steering_vectors  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
CKPT = "checkpoints_qwen38_27b_pocket"
MODEL = "/run/media/s117/OS/Models/Qwen3.8-27B"
TRIAL = 24
ARA_DELTA = Path("exports/qwen38_ara_fixed_delta.pt")
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
OUT_LOGS = Path("logs/qwen38_next_sweep.json")
ARA_SCALES = (0.70, 0.85, 1.15, 1.35, 1.60)
RESIDUAL_STRENGTHS = (0.40, 0.80, 1.20, 1.80, 2.50)


def _load_cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["sweep_qwen38_next", "--config", CONFIG, "--seed", "117"]
    return AbliterixConfig()


def _selfcheck_lerp() -> None:
    a = torch.tensor([0.0, 10.0])
    b = torch.tensor([10.0, 0.0])
    c = a.clone()
    c.lerp_(b, 0.3)
    expect = torch.tensor([3.0, 7.0])
    if not torch.allclose(c, expect, atol=1e-6):
        raise AssertionError(f"lerp self-check failed: {c} != {expect}")
    print("lerp self-check ok", flush=True)


def _snapshot_modules(hf_model, weight_keys) -> dict[str, torch.Tensor]:
    layers = _hf_layers(hf_model)
    snap: dict[str, torch.Tensor] = {}
    for key in weight_keys:
        layer_s, rest = key.split(".", 1)
        component, idx_s = rest.rsplit(".", 1)
        mods = _layer_modules(layers[int(layer_s)], component)
        dest = mods[int(idx_s)].weight
        snap[key] = dest.detach().to("cpu").clone()
    return snap


def _restore_modules(hf_model, snap: dict[str, torch.Tensor]) -> None:
    layers = _hf_layers(hf_model)
    for key, tensor in snap.items():
        layer_s, rest = key.split(".", 1)
        component, idx_s = rest.rsplit(".", 1)
        dest = _layer_modules(layers[int(layer_s)], component)[int(idx_s)].weight
        dest.data.copy_(tensor.to(device=dest.device, dtype=dest.dtype))


def _score(tag: str, scorer: TrialScorer, engine: SteeringEngine) -> dict:
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
    return row


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT_LOGS.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_next_sweep.json").write_text(text, encoding="utf-8")


def _residual_profiles(t24_profiles: dict, strength: float) -> dict[str, SteeringProfile]:
    o = t24_profiles["attn.o_proj"]
    down = t24_profiles.get("mlp.down_proj")
    ratio = (o.min_weight / o.max_weight) if o.max_weight else 0.2
    out = {
        "attn.o_proj": SteeringProfile(
            max_weight=float(strength),
            max_weight_position=o.max_weight_position,
            min_weight=float(max(0.05, strength * ratio)),
            min_weight_distance=max(o.min_weight_distance, 8.0),
        )
    }
    if down is not None:
        out["mlp.down_proj"] = SteeringProfile(
            max_weight=0.0,
            max_weight_position=down.max_weight_position,
            min_weight=0.0,
            min_weight_distance=down.min_weight_distance,
        )
    return out


def main() -> None:
    _selfcheck_lerp()
    torch.set_grad_enabled(False)
    cfg = _load_cfg()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False

    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "base": MODEL,
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original",
        "detector": "keyword RefusalDetector",
        "eval_set": "harmful_1000 train[900:]",
        "points": [],
    }

    print("loading engine...", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    if not POCKET_BASELINE.is_file():
        raise FileNotFoundError(POCKET_BASELINE)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    trial = load_trial(CKPT, MODEL, TRIAL)
    artifact = extract_trial_artifact(trial)
    slug = slugify_model_name(MODEL)
    cache = torch.load(
        Path(CKPT) / f"{slug}_steering.pt",
        map_location="cpu",
        weights_only=False,
    )
    t24_vectors = cache["vectors"]
    print(
        f"t24 recorded refusals={trial.user_attrs.get('refusals')} "
        f"kl={trial.user_attrs.get('kl_divergence')}",
        flush=True,
    )

    ara_blob = torch.load(ARA_DELTA, map_location="cpu", weights_only=False)
    ara_keys = list(ara_blob["weights"].keys())
    print(f"snapshot {len(ara_keys)} original modules for ARA restore", flush=True)
    orig_snap = _snapshot_modules(engine.model, ara_keys)

    print("=== phase A: ARA scale sweep ===", flush=True)
    for scale in ARA_SCALES:
        _restore_modules(engine.model, orig_snap)
        apply_ara_delta(engine.model, ARA_DELTA, scale=scale)
        row = _score(f"ara_scale_{scale:.2f}", scorer, engine)
        row["method"] = "ara_lerp"
        row["scale"] = scale
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = row["tag"]
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {row['tag']}", flush=True)
            return
    _restore_modules(engine.model, orig_snap)

    print("=== t24 sanity ===", flush=True)
    engine.restore_baseline()
    apply_steering(engine, t24_vectors, artifact.vector_index, artifact.profiles, cfg)
    row = _score("pocket_t24", scorer, engine)
    row["method"] = "t24_sanity"
    payload["points"].append(row)
    _dump(payload)

    print("=== phase B: residual extract on t24 ===", flush=True)
    benign = load_prompt_dataset(cfg, cfg.benign_prompts)
    target = load_prompt_dataset(cfg, cfg.target_prompts)
    print(f"extracting residual states n_benign={len(benign)} n_target={len(target)}", flush=True)
    benign_states = engine.extract_hidden_states_batched(benign)
    target_states = engine.extract_hidden_states_batched(target)
    residual_vectors = compute_configured_steering_vectors(
        benign_states, target_states, cfg
    )
    del benign, target, benign_states, target_states
    flush_memory()
    print(f"residual vectors shape={tuple(residual_vectors.shape)}", flush=True)

    for strength in RESIDUAL_STRENGTHS:
        engine.restore_baseline()
        _restore_modules(engine.model, orig_snap)
        apply_steering(
            engine, t24_vectors, artifact.vector_index, artifact.profiles, cfg
        )
        profiles = _residual_profiles(artifact.profiles, strength)
        _apply_direct_steering(
            engine,
            residual_vectors,
            resolve_global_vector(residual_vectors, None),
            profiles,
            cfg,
            None,
        )
        row = _score(f"t24_residual_o{strength:.2f}", scorer, engine)
        row["method"] = "t24_residual_direct"
        row["residual_o_proj"] = strength
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = row["tag"]
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {row['tag']}", flush=True)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
