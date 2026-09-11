#!/usr/bin/env python3
"""Full-weight ARA on Nex-N2.5-mini (trohrbaugh-style, via heretic-ara-lora).

ARA optimises the *weights themselves* with LBFGS against a triple objective
(preserve good behaviour, steer bad behaviour, overcorrect), which is a different
family from abliterix's rank-1 direction ablation — and on Qwen3.8-27B the same
recipe reached 0/100 @ KL 0.0535 where mean+LoRA TPE never entered the box.

The heretic-ara-lora library explicitly handles Qwen3.5-MoE hybrid layers
(GatedDeltaNet linear attention + fused experts), so Nex is in scope.

    python scripts/nex25_ara_probe.py --probe        # inspect components only
    python scripts/nex25_ara_probe.py --go           # run ARA and save

Runs under the muse-glimmer-env python (torch 2.13 + ROCm 10), same as run_nex25.sh.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, "/home/s117/heretic-ara-lora/src")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.linalg import vector_norm  # noqa: E402
from torch.optim import LBFGS  # noqa: E402

from heretic.config import DatasetSpecification, Settings  # noqa: E402
from heretic.model import ARAParameters, Model  # noqa: E402
from heretic.utils import load_prompts  # noqa: E402

BASE = "/run/media/s117/OS/Models/Nex-N2.5-mini"
OUT = Path("/run/media/s117/OS/Models/Nex-N2.5-mini-ara")
DELTA = ROOT / "exports" / "nex25_ara_delta.pt"

# Nex has 40 layers; Qwen3.8-27B used start=26/end=56 (the latter half), so the
# analogous band here is 16..40. Knobs start from the published Qwen3.8 recipe.
ARA = ARAParameters(
    start_layer_index=16,
    end_layer_index=40,
    preserve_good_behavior_weight=0.94,
    steer_bad_behavior_weight=0.0009,
    overcorrect_relative_weight=0.50,
    neighbor_count=10,
)

TARGET_COMPONENTS = ["attn.o_proj", "mlp.down_proj"]


def _settings(n_good: int, n_bad: int) -> Settings:
    ds = str(ROOT / "datasets")
    # heretic's Settings is a pydantic-settings model with a CLI source, so it
    # would try to parse our own flags; hide them while it is constructed.
    saved_argv = sys.argv
    sys.argv = ["nex25_ara"]
    try:
        return _settings_inner(ds, n_good, n_bad)
    finally:
        sys.argv = saved_argv


def _settings_inner(ds: str, n_good: int, n_bad: int) -> Settings:
    return Settings.model_validate(
        {
            "model": BASE,
            "dtypes": ["bfloat16"],
            "quantization": "none",
            "device_map": "auto",
            "trust_remote_code": True,
            "enable_thinking": False,
            "skip_prefix_check": True,
            "use_ara": True,
            "use_ara_lora": False,
            "target_components": TARGET_COMPONENTS,
            "row_normalization": "full",
            "batch_size": 32,
            "max_batch_size": 64,
            "max_response_length": 160,
            "n_trials": 1,
            "non_interactive": True,
            "good_prompts": DatasetSpecification(
                dataset=f"{ds}/good_1000", split=f"train[:{n_good}]", column="prompt"
            ),
            "bad_prompts": DatasetSpecification(
                dataset=f"{ds}/harmful_1000", split=f"train[:{n_bad}]", column="prompt"
            ),
        }
    )


def probe(model: Model) -> None:
    n = len(model.get_layers()) if hasattr(model, "get_layers") else None
    print(f"layers detected: {n}", flush=True)
    seen: dict[str, int] = {}
    for idx in range(0, n or 40):
        try:
            mods = model.get_layer_modules(idx)
        except Exception as error:  # noqa: BLE001
            print(f"  layer {idx}: <error {type(error).__name__}: {error}>", flush=True)
            continue
        for component, modules in mods.items():
            seen[component] = seen.get(component, 0) + len(modules)
        if idx in (0, 1, 3, 7, 16, 20, 39):
            print(f"  layer {idx}: {sorted(mods)}", flush=True)
    print(f"component totals across layers: {seen}", flush=True)


def apply_ara_f32(model: Model, good_io, bad_io, params: ARAParameters) -> None:
    n_layers = params.end_layer_index - params.start_layer_index
    done = 0
    for layer_index in range(params.start_layer_index, params.end_layer_index):
        for component, modules in model.get_layer_modules(layer_index).items():
            if component not in TARGET_COMPONENTS:
                continue
            if layer_index >= len(good_io) or component not in good_io[layer_index]:
                print(f"  skip {layer_index} {component}: no I/O", flush=True)
                continue
            for module_index, module in enumerate(modules):
                if module_index not in good_io[layer_index][component]:
                    continue
                W = module.weight
                work = W.detach().to(torch.float32).clone().requires_grad_(True)
                row_norms = vector_norm(work.detach(), dim=1, keepdim=True)
                g_in, g_out = good_io[layer_index][component][module_index]
                b_in, b_out = bad_io[layer_index][component][module_index]
                # The I/O hook stores everything on CPU; move both sides of each
                # pair to the working device (the original template only moved the
                # inputs, which fails on a CUDA/HIP device).
                g_in = g_in.to(work.device, dtype=torch.float32)
                b_in = b_in.to(work.device, dtype=torch.float32)
                g_out = g_out.to(work.device, dtype=torch.float32)
                b_out = b_out.to(work.device, dtype=torch.float32)

                def get_matrix():  # noqa: ANN202
                    return work

                def objective():  # noqa: ANN202
                    Wm = get_matrix()
                    g_pred = g_in @ Wm.T
                    b_pred = b_in @ Wm.T
                    good_loss = F.mse_loss(g_pred, g_out)
                    bad_loss = F.mse_loss(b_pred, b_out)
                    norm_loss = F.mse_loss(
                        vector_norm(Wm, dim=1, keepdim=True), row_norms
                    )
                    return (
                        params.preserve_good_behavior_weight * good_loss
                        + params.steer_bad_behavior_weight * bad_loss
                        + params.overcorrect_relative_weight * norm_loss
                    )

                def closure():  # noqa: ANN202
                    optimizer.zero_grad()
                    loss = objective()
                    loss.backward()
                    return loss

                optimizer = LBFGS([work], lr=1.0, max_iter=20, history_size=10)
                last = None
                for _ in range(4):
                    last = optimizer.step(closure)
                    if last is not None and float(last) < 1e-9:
                        break
                with torch.no_grad():
                    module.weight.copy_(work.to(W.dtype))
                done += 1
                if done % 10 == 0:
                    print(
                        f"  {done} modules done | layer {layer_index} {component}"
                        f" loss={float(last):.6e}",
                        flush=True,
                    )
    print(f"ARA finished: {done} modules optimised", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="inspect components, do not optimise")
    ap.add_argument("--go", action="store_true", help="run ARA and save the delta")
    ap.add_argument("--good", type=int, default=200)
    ap.add_argument("--bad", type=int, default=200)
    ap.add_argument("--start-layer", type=int, default=ARA.start_layer_index)
    ap.add_argument("--end-layer", type=int, default=ARA.end_layer_index)
    args = ap.parse_args()
    if not (args.probe or args.go):
        raise SystemExit("pass --probe or --go")

    os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
    ARA.start_layer_index = args.start_layer
    ARA.end_layer_index = args.end_layer

    settings = _settings(args.good, args.bad)
    print(f"ARA params: {ARA}", flush=True)
    model = Model(settings)
    model.response_prefix = ""

    if args.probe:
        probe(model)
        return

    print("Loading prompts...", flush=True)
    good = load_prompts(settings, settings.good_prompts)
    bad = load_prompts(settings, settings.bad_prompts)
    print(f"good={len(good)} bad={len(bad)}", flush=True)
    print("Module I/O (good)...", flush=True)
    good_io = model.get_module_io_batched(good)
    print("Module I/O (bad)...", flush=True)
    bad_io = model.get_module_io_batched(bad)
    print("Abliterating (full-weight ARA, float32 LBFGS)...", flush=True)
    apply_ara_f32(model, good_io, bad_io, ARA)
    del good_io, bad_io

    DELTA.parent.mkdir(parents=True, exist_ok=True)
    delta = {}
    for layer_index in range(ARA.start_layer_index, ARA.end_layer_index):
        for component, modules in model.get_layer_modules(layer_index).items():
            if component not in TARGET_COMPONENTS:
                continue
            for module_index, module in enumerate(modules):
                delta[f"{layer_index}.{component}.{module_index}"] = (
                    module.weight.detach().to("cpu").contiguous()
                )
    torch.save({"base": BASE, "ara": dict(ARA.__dict__), "weights": delta}, DELTA)
    print(f"DELTA_OK {DELTA} n={len(delta)}", flush=True)

    # Save the text-only decoder, mirroring the validated abliterix bake so the
    # two artefacts are directly comparable. The conditional wrapper keeps the
    # vision tower, which the published model does not carry.
    hf = model.model
    target = hf
    if not hasattr(target, "language_model"):
        inner = getattr(hf, "model", None)
        if inner is not None and hasattr(inner, "language_model"):
            target = inner.language_model
    print(f"saving text decoder ({type(target).__name__}) -> {OUT}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    target.save_pretrained(str(OUT), safe_serialization=True, max_shard_size="4GB")
    model.tokenizer.save_pretrained(str(OUT))
    for name in ("chat_template.jinja", "generation_config.json"):
        src = Path(BASE) / name
        if src.is_file():
            import shutil as _sh
            _sh.copy2(src, OUT / name)
    print("SAVE_OK", OUT, flush=True)


if __name__ == "__main__":
    main()
