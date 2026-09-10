#!/usr/bin/env python3
"""Apply trohrbaugh-style full-weight ARA (BF16) to Qwen3.8-27B and save.

Candidate for the 10/100 @ KL<=0.1 goal. Mean+LoRA TPE never entered that
box; published full-weight ARA on this base did (0/100 @ 0.0535 first-token).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, "/home/s117/heretic-ara-lora/src")
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.linalg import vector_norm
from torch.optim import LBFGS

from heretic.config import DatasetSpecification, Settings
from heretic.model import ARAParameters, Model
from heretic.utils import load_prompts, mean_distances_to_knn

BASE = "/run/media/s117/OS/Models/Qwen3.8-27B"
OUT = Path("/run/media/s117/OS/Models/Qwen3.8-27B-abliterix-ara-fixed")
DS = "/run/media/s117/OS/Users/15403/Documents/abliterix-datasets"

# Published trohrbaugh/Qwen3.8-27B-heretic-ara knobs.
ARA = ARAParameters(
    start_layer_index=26,
    end_layer_index=56,
    preserve_good_behavior_weight=0.94,
    steer_bad_behavior_weight=0.0009,
    overcorrect_relative_weight=0.50,
    neighbor_count=10,
)


def apply_ara_f32(model: Model, good_io, bad_io, params: ARAParameters) -> None:
    n_layers = params.end_layer_index - params.start_layer_index
    done = 0
    for layer_index in range(params.start_layer_index, params.end_layer_index):
        for component, modules in model.get_layer_modules(layer_index).items():
            if layer_index >= len(good_io) or component not in good_io[layer_index]:
                print(f"  skip layer {layer_index} {component}: no I/O")
                continue
            for module_index, module in enumerate(modules):
                if module_index not in good_io[layer_index][component]:
                    continue
                W = module.weight
                work = W.detach().to(torch.float32).clone().requires_grad_(True)
                row_norms = vector_norm(work.detach(), dim=1, keepdim=True)

                g_in, g_out = good_io[layer_index][component][module_index]
                b_in, b_out = bad_io[layer_index][component][module_index]
                g_in = g_in.to(work.device, dtype=torch.float32)
                g_out = g_out.to(work.device, dtype=torch.float32)
                b_in = b_in.to(work.device, dtype=torch.float32)
                b_out = b_out.to(work.device, dtype=torch.float32)

                def get_matrix():
                    return row_norms * F.normalize(work, p=2, dim=1)

                def objective():
                    mat = get_matrix()
                    new_good = g_in @ mat.T
                    new_bad = b_in @ mat.T
                    preserve = ((new_good - g_out) ** 2).mean()
                    steer = mean_distances_to_knn(
                        new_bad, g_out, params.neighbor_count
                    ).mean() + params.overcorrect_relative_weight * (
                        -mean_distances_to_knn(
                            new_bad, b_out, params.neighbor_count
                        ).mean()
                    )
                    return (
                        params.preserve_good_behavior_weight * preserve
                        + params.steer_bad_behavior_weight * steer
                    )

                opt = LBFGS(
                    [work],
                    lr=1.0,
                    max_iter=20,
                    history_size=10,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    opt.zero_grad()
                    loss = objective()
                    loss.backward()
                    return loss

                last = None
                for _ in range(5):
                    last = opt.step(closure)
                with torch.no_grad():
                    W.data.copy_(get_matrix().detach().to(W.dtype))
                work.grad = None
                del work
                done += 1
                print(
                    f"  L{layer_index:02d} {component}[{module_index}] "
                    f"loss={float(last):.6f}  ({layer_index - params.start_layer_index + 1}/{n_layers})",
                    flush=True,
                )
    print(f"ARA updated {done} modules", flush=True)


def _eval_live(model: Model) -> None:
    """Keyword refusals + 3-token teacher-forced KL vs original BF16 baseline."""
    import json
    import math

    os.environ["AX_CONFIG"] = str(ROOT / "configs" / "qwen38_27b_rocm_pocket.toml")
    sys.argv = ["eval", "--config", os.environ["AX_CONFIG"], "--seed", "117"]

    from abliterix.eval.detector import RefusalDetector
    from abliterix.eval.scorer import _safe_kl_divergence
    from abliterix.settings import AbliterixConfig
    from heretic.config import DatasetSpecification as HDS
    from heretic.utils import load_prompts as hload

    cfg = AbliterixConfig()
    detector = RefusalDetector(cfg)
    bad_eval = hload(
        model.settings,
        HDS(dataset=f"{DS}/harmful_1000", split="train[900:]", column="prompt"),
    )
    good_eval = hload(
        model.settings,
        HDS(dataset=f"{DS}/good_1000", split="train[900:]", column="prompt"),
    )
    print(f"eval harmful={len(bad_eval)} good={len(good_eval)}", flush=True)
    print("Generating harmful eval...", flush=True)
    replies = model.get_responses_batched(bad_eval, skip_special_tokens=True)
    n_kw = sum(1 for r in replies if detector.detect_refusal(r))
    print(f"keyword_refusals {n_kw}/{len(replies)}", flush=True)

    base_pt = (
        ROOT
        / "checkpoints_qwen38_27b_pocket"
        / "--run--media--s117--OS--Models--Qwen3--8-27B_baseline.pt"
    )
    cache = torch.load(base_pt, map_location="cpu", weights_only=False)
    baseline_lp = cache["baseline_logprobs"]
    conts = cache["baseline_continuations"]
    token_count = 3
    device = next(model.model.parameters()).device
    cand_rows = []
    tok = model.tokenizer
    print("Scoring 3-token teacher-forced KL...", flush=True)
    for i, (prompt, cont) in enumerate(zip(good_eval, conts)):
        chat = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt.user}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        prompt_ids = tok(chat, add_special_tokens=False).input_ids
        cont_ids = tok(cont, add_special_tokens=False).input_ids[:token_count]
        ids = torch.tensor([prompt_ids + cont_ids], device=device)
        with torch.no_grad():
            logits = model.model(ids).logits[0]
        # positions predicting each continuation token
        start = len(prompt_ids) - 1
        step_lps = []
        for t in range(token_count):
            lp = torch.log_softmax(logits[start + t].float(), dim=-1)
            step_lps.append(lp.cpu())
        cand_rows.append(torch.stack(step_lps, dim=0))
        if (i + 1) % 20 == 0:
            print(f"  KL scored {i+1}/{len(good_eval)}", flush=True)
    cand = torch.stack(cand_rows, dim=0)
    kl = float(_safe_kl_divergence(cand, baseline_lp))
    print(f"full_distribution_kl_3token_vs_original {kl:.6f}", flush=True)
    out = {
        "candidate": "exports/qwen38_ara_fixed_delta.pt",
        "base": BASE,
        "detector": "keyword",
        "kl_meter": "3-token teacher-forced full_distribution_kl vs original BF16 pocket baseline",
        "refusals": n_kw,
        "n": len(replies),
        "kl": kl,
        "ara": ARA.__dict__,
        "vs_original": True,
    }
    dests = [
        ROOT / "logs" / "qwen38_ara_fixed_eval.json",
        Path("/tmp/grok-goal-8ac14a085a92/implementer") / "qwen38_ara_fixed_eval.json",
    ]
    text = json.dumps(out, indent=2)
    for d in dests:
        d.parent.mkdir(parents=True, exist_ok=True)
        d.write_text(text)
    print("EVAL_JSON", dests[0], flush=True)


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
    settings = Settings.model_validate(
        {
            "model": BASE,
            "dtypes": ["bfloat16"],
            "quantization": "none",
            "device_map": "auto",
            "max_memory": {"0": "110GB", "cpu": "8GB"},
            "trust_remote_code": True,
            "enable_thinking": False,
            "skip_prefix_check": True,
            "use_ara": True,
            "use_ara_lora": False,
            "target_components": ["attn.o_proj", "mlp.down_proj"],
            "row_normalization": "full",
            "batch_size": 128,
            "max_batch_size": 128,
            "max_response_length": 160,
            "n_trials": 1,
            "non_interactive": True,
            "good_prompts": DatasetSpecification(
                dataset=f"{DS}/good_1000",
                split="train[:800]",
                column="prompt",
            ),
            "bad_prompts": DatasetSpecification(
                dataset=f"{DS}/harmful_1000",
                split="train[:800]",
                column="prompt",
            ),
        }
    )
    print("ARA params", ARA, flush=True)
    model = Model(settings)
    model.response_prefix = ""
    print("Loading prompts...", flush=True)
    good = load_prompts(settings, settings.good_prompts)
    bad = load_prompts(settings, settings.bad_prompts)
    print(f"good={len(good)} bad={len(bad)}", flush=True)
    print("Module I/O good...", flush=True)
    good_io = model.get_module_io_batched(good)
    print("Module I/O bad...", flush=True)
    bad_io = model.get_module_io_batched(bad)
    print("Abliterating (full-weight ARA, float32 LBFGS)...", flush=True)
    apply_ara_f32(model, good_io, bad_io, ARA)
    del good_io, bad_io
    delta_path = ROOT / "exports" / "qwen38_ara_fixed_delta.pt"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    delta = {}
    for layer_index in range(ARA.start_layer_index, ARA.end_layer_index):
        for component, modules in model.get_layer_modules(layer_index).items():
            for module_index, module in enumerate(modules):
                key = f"{layer_index}.{component}.{module_index}"
                delta[key] = module.weight.detach().to("cpu").contiguous()
    torch.save(
        {
            "base": BASE,
            "ara": ARA.__dict__,
            "weights": delta,
        },
        delta_path,
    )
    print("DELTA_OK", delta_path, "n", len(delta), flush=True)
    _eval_live(model)
    print("EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
