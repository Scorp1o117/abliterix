# Score a candidate against the ORIGINAL Qwen3.8-27B, not against a
# peeled merge. Uses the same teacher-forced 3-token KL as search
# (KL(candidate || original) on original's continuations).
#
# Trial on the original base (v3):
#   python scripts/eval_qwen38_vs_original.py \
#     --config configs/qwen38_27b_rocm_v3.toml \
#     --checkpoint checkpoints_qwen38_27b_v3 --trial 12
#
# Merged folder vs original baseline cache:
#   python scripts/eval_qwen38_vs_original.py \
#     --config configs/qwen38_27b_rocm.toml \
#     --merged /path/to/merged \
#     --original-baseline checkpoints_qwen38_27b/..._baseline.pt
#
# ARA / other full-weight delta.pt (keys "{layer}.{component}.{idx}"):
#   python scripts/eval_qwen38_vs_original.py \
#     --config configs/qwen38_27b_rocm_pocket.toml \
#     --delta exports/qwen38_ara_fixed_delta.pt \
#     --delta-scale 0.85 \
#     --original-baseline checkpoints_qwen38_27b_pocket/..._baseline.pt

from __future__ import annotations

import argparse
import os
import sys
from contextlib import suppress
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.nn import Module, ModuleList

from abliterix.core.engine import SteeringEngine
from abliterix.core.steering import apply_steering
from abliterix.eval.detector import RefusalDetector
from abliterix.eval.scorer import TrialScorer, _safe_kl_divergence
from abliterix.scriptlib import extract_trial_artifact, load_trial
from abliterix.settings import AbliterixConfig
from abliterix.util import slugify_model_name

ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
POCKET_BASELINE = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_baseline.pt"
)


def _load_cfg(path: str) -> AbliterixConfig:
    os.environ["AX_CONFIG"] = path
    sys.argv = ["abliterix", "--config", path, "--seed", "117"]
    return AbliterixConfig()


def _inject_original_baseline(scorer: TrialScorer, baseline_pt: Path) -> None:
    cache = torch.load(baseline_pt, map_location="cpu", weights_only=False)
    scorer.baseline_logprobs = cache["baseline_logprobs"]
    scorer.baseline_mean_length = cache["baseline_mean_length"]
    scorer.baseline_stdev_length = cache["baseline_stdev_length"]
    scorer.baseline_single_token_lp = cache["baseline_single_token_lp"]
    scorer.baseline_refusal_count = cache["baseline_refusal_count"]
    scorer.baseline_continuations = cache.get("baseline_continuations")
    scorer.baseline_continuation_nll = cache.get("baseline_continuation_nll")
    print(
        f"original baseline: refusals={scorer.baseline_refusal_count}  "
        f"from {baseline_pt}"
    )


def _default_original_baseline() -> Path:
    slug = slugify_model_name(ORIGINAL)
    return Path("checkpoints_qwen38_27b") / f"{slug}_baseline.pt"


def _hf_layers(model) -> ModuleList:
    inner = model
    # SteeringEngine wraps the HF model in PEFT; ARA keys index decoder layers.
    with suppress(Exception):
        from peft import PeftModel

        if isinstance(inner, PeftModel):
            inner = inner.base_model.model
    with suppress(Exception):
        return inner.model.language_model.layers
    with suppress(Exception):
        return inner.model.layers
    with suppress(Exception):
        return inner.language_model.layers
    with suppress(Exception):
        return inner.backbone.layers
    raise AttributeError(f"cannot find transformer layers on {type(model)}")


def _layer_modules(layer, component: str) -> list[Module]:
    found: list[Module] = []

    def try_add(mod):
        if isinstance(mod, Module):
            found.append(mod)

    if component == "attn.o_proj":
        with suppress(Exception):
            try_add(layer.self_attn.o_proj)
        with suppress(Exception):
            try_add(layer.linear_attn.out_proj)
        with suppress(Exception):
            try_add(layer.attention.o_proj)
        with suppress(Exception):
            try_add(layer.attention.dense)
    elif component == "mlp.down_proj":
        with suppress(Exception):
            try_add(layer.mlp.down_proj)
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add(expert.down_proj)
        with suppress(Exception):
            try_add(layer.mlp.shared_experts.down_proj)
        with suppress(Exception):
            try_add(layer.mlp.shared_expert.down_proj)
    else:
        raise ValueError(f"unsupported ARA component {component}")
    return found


def apply_ara_delta(hf_model, delta_path: Path, scale: float = 1.0) -> int:
    """Copy post-ARA weights onto the live model.

    ``scale=1`` replaces each module with the stored ARA matrix. Other
    scales lerp the *currently loaded* weight toward that matrix
    (``W <- (1-s) W + s W_ara``), so the model must be at the original
    baseline before a scale other than 1 is applied.
    """
    blob = torch.load(delta_path, map_location="cpu", weights_only=False)
    weights = blob["weights"]
    layers = _hf_layers(hf_model)
    applied = 0
    for key, tensor in weights.items():
        layer_s, rest = key.split(".", 1)
        component, idx_s = rest.rsplit(".", 1)
        layer = layers[int(layer_s)]
        mods = _layer_modules(layer, component)
        idx = int(idx_s)
        if idx >= len(mods):
            raise KeyError(f"{key}: only {len(mods)} modules for {component}")
        dest = mods[idx].weight
        src = tensor.to(device=dest.device, dtype=dest.dtype)
        if tuple(src.shape) != tuple(dest.shape):
            raise ValueError(f"{key}: {tuple(src.shape)} != {tuple(dest.shape)}")
        if abs(float(scale) - 1.0) < 1e-8:
            dest.data.copy_(src)
        else:
            dest.data.lerp_(src, float(scale))
        applied += 1
    print(f"applied ARA delta {delta_path} n={applied} scale={scale}", flush=True)
    return applied


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--merged")
    ap.add_argument("--delta", type=Path)
    ap.add_argument(
        "--delta-scale",
        type=float,
        default=1.0,
        help="Lerp original→ARA (1.0 = full stored ARA weights). Requires --delta.",
    )
    ap.add_argument("--original-baseline", type=Path)
    args = ap.parse_args()
    if args.delta_scale != 1.0 and not args.delta:
        ap.error("--delta-scale requires --delta")

    modes = [bool(args.merged), args.trial is not None, bool(args.delta)]
    if sum(modes) != 1:
        ap.error("specify exactly one of --trial, --merged, or --delta")
    if args.trial is not None and not args.checkpoint:
        ap.error("--trial requires --checkpoint")

    cfg = _load_cfg(args.config)
    if args.merged:
        cfg.model.model_id = args.merged

    print(f"candidate model_id={cfg.model.model_id}")
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)

    if args.merged or args.delta:
        if args.delta:
            apply_ara_delta(engine.model, args.delta, scale=args.delta_scale)
        scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
        base_pt = args.original_baseline or (
            POCKET_BASELINE if args.delta and POCKET_BASELINE.is_file()
            else _default_original_baseline()
        )
        if not base_pt.is_file():
            raise FileNotFoundError(
                f"original baseline cache missing: {base_pt} "
                "(run a search on the original model first)"
            )
        _inject_original_baseline(scorer, base_pt)
    else:
        vs_original = bool(args.original_baseline)
        if cfg.model.model_id != ORIGINAL and not vs_original:
            raise SystemExit(
                f"trial eval must load the original base, got {cfg.model.model_id} "
                "(pass --original-baseline to score a repaired/derived trial vs original)"
            )
        if vs_original:
            scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
            _inject_original_baseline(scorer, args.original_baseline)
        else:
            scorer = TrialScorer(cfg, engine, detector)
        trial = load_trial(args.checkpoint, cfg.model.model_id, args.trial)
        artifact = extract_trial_artifact(trial)
        slug = slugify_model_name(cfg.model.model_id)
        cache = torch.load(
            Path(args.checkpoint) / f"{slug}_steering.pt",
            map_location="cpu",
            weights_only=False,
        )
        apply_steering(
            engine,
            cache["vectors"],
            artifact.vector_index,
            artifact.profiles,
            cfg,
        )
        print(f"applied trial {args.trial} on {cfg.model.model_id}")

    kl = scorer.measure_kl_divergence(engine)
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    print()
    print("=== vs ORIGINAL ===")
    print(f"KL(candidate || original)  {kl:.4f} nats/token")
    print(f"refusals                   {refusals}/{n}")
    print(f"original refusals          {scorer.baseline_refusal_count}/{n}")
    print(f"KL target                  {cfg.kl.target}")


if __name__ == "__main__":
    main()
