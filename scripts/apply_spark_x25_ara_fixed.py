#!/usr/bin/env python3
"""Full-weight ARA on Spark-X2.5-4B (mergeable, not ARA-LoRA).

trohrbaugh knobs, o_proj + down_proj only, layers 8–32 of 36 (skip shallow
like Huihui; cover mid-front where Spark refusal actually lives).
Score keyword refusals + 3-token KL vs the original V9 baseline.
Bake merged weights only on dual HIT (≤10/100 and KL≤0.05).

Usage:
  python scripts/apply_spark_x25_ara_fixed.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.linalg import vector_norm  # noqa: E402
from torch.nn import Module, ModuleList  # noqa: E402
from torch.optim import LBFGS  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import ChatMessage, SteeringMode  # noqa: E402
from eval_qwen38_vs_original import _inject_original_baseline  # noqa: E402

CONFIG = "configs/spark_x25_4b_rocm.toml"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
MERGED = Path("/run/media/s117/OS/Models/Spark-X2.5-4B-abliterix")
BASELINE = ROOT / (
    "checkpoints_spark_x25_4b_lora_v9/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
DELTA = ROOT / "exports" / "spark_x25_ara_fixed_delta.pt"
OUT = ROOT / "logs" / "spark_x25_ara_fixed_eval.json"
SCRATCH = Path("/tmp/grok-goal-85e2331705c4/implementer")
HIT_TXT = SCRATCH / "spark_x25_hit.txt"

# 36 layers. Qwen ARA used 26–56 / 64. Spark refusal sits mid-front
# (narrow late Householder 21–32 did not open), so start earlier.
START_LAYER = 8
END_LAYER = 32  # exclusive
PRESERVE = 0.94
STEER = 0.0009
OVERCORRECT = 0.50
NEIGHBORS = 10
COMPONENTS = ("attn.o_proj", "mlp.down_proj")
LBFGS_ROUNDS = 5
LBFGS_MAX_ITER = 20


def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    distances = torch.cdist(a, b)
    nearest, _ = distances.topk(k, dim=1, largest=False)
    return nearest.mean(1)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_ara", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 64
    cfg.inference.max_batch_size = 64
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    cfg.model.model_id = MODEL
    return cfg


def _hf_layers(model) -> ModuleList:
    inner = model
    with suppress(Exception):
        from peft import PeftModel

        if isinstance(inner, PeftModel):
            inner = inner.base_model.model
    for path in (
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.layers,
    ):
        with suppress(Exception):
            layers = path(inner)
            if isinstance(layers, ModuleList):
                return layers
    raise AttributeError(f"cannot find decoder layers on {type(model)}")


def _layer_modules(layer, component: str) -> list[Module]:
    found: list[Module] = []

    def try_add(mod):
        if isinstance(mod, Module):
            found.append(mod)

    if component == "attn.o_proj":
        with suppress(Exception):
            try_add(layer.self_attn.out_proj)  # Spark
        with suppress(Exception):
            try_add(layer.self_attn.o_proj)
        with suppress(Exception):
            try_add(layer.linear_attn.out_proj)
    elif component == "mlp.down_proj":
        with suppress(Exception):
            try_add(layer.mlp.down_proj)
    else:
        raise ValueError(component)
    return found


def _iter_targets(layers: ModuleList):
    for layer_index in range(START_LAYER, END_LAYER):
        layer = layers[layer_index]
        for component in COMPONENTS:
            for module_index, module in enumerate(_layer_modules(layer, component)):
                yield layer_index, component, module_index, module


def _last_token_rows(hidden: Tensor, mask: Tensor) -> Tensor:
    """Gather the last non-pad position of each row."""
    # last 1 in attention_mask
    idx = mask.size(1) - 1 - mask.flip(1).argmax(1)
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def collect_io(
    engine: SteeringEngine,
    messages: list[ChatMessage],
    targets: list[tuple[int, str, int, Module]],
) -> dict[tuple[int, str, int], tuple[Tensor, Tensor]]:
    keys = [(i, c, m) for i, c, m, _ in targets]
    acc_in: dict[tuple[int, str, int], list[Tensor]] = {k: [] for k in keys}
    acc_out: dict[tuple[int, str, int], list[Tensor]] = {k: [] for k in keys}
    bs = int(engine.config.inference.batch_size)

    def hook_factory(key):
        def hook(module, inputs, outputs):
            hs_in = inputs[0]
            hs_out = outputs[0] if isinstance(outputs, tuple) else outputs
            mask = hook.mask  # type: ignore[attr-defined]
            acc_in[key].append(_last_token_rows(hs_in, mask).detach().float().cpu())
            acc_out[key].append(_last_token_rows(hs_out, mask).detach().float().cpu())

        return hook

    handles = []
    hook_objs = []
    for layer_i, comp, mod_i, module in targets:
        h = hook_factory((layer_i, comp, mod_i))
        hook_objs.append(h)
        handles.append(module.register_forward_hook(h))

    engine.model.eval()
    n = len(messages)
    for start in range(0, n, bs):
        batch = messages[start : start + bs]
        enc = engine._tokenize(batch)
        mask = enc["attention_mask"]
        for h in hook_objs:
            h.mask = mask
        with torch.no_grad():
            engine.model(
                input_ids=enc["input_ids"],
                attention_mask=mask,
                use_cache=False,
            )
        if (start // bs + 1) % 4 == 0 or start + bs >= n:
            print(f"  io {min(start + bs, n)}/{n}", flush=True)

    for h in handles:
        h.remove()

    out = {}
    for key in keys:
        if not acc_in[key]:
            raise RuntimeError(f"no I/O captured for {key}")
        out[key] = (torch.cat(acc_in[key], 0), torch.cat(acc_out[key], 0))
    return out


def apply_ara(
    targets: list[tuple[int, str, int, Module]],
    good_io: dict,
    bad_io: dict,
) -> int:
    done = 0
    n = len(targets)
    for layer_i, comp, mod_i, module in targets:
        key = (layer_i, comp, mod_i)
        W = module.weight
        work = W.detach().to(torch.float32).clone().requires_grad_(True)
        row_norms = vector_norm(work.detach(), dim=1, keepdim=True)
        g_in, g_out = good_io[key]
        b_in, b_out = bad_io[key]
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
            steer = mean_distances_to_knn(new_bad, g_out, NEIGHBORS).mean() + (
                OVERCORRECT * (-mean_distances_to_knn(new_bad, b_out, NEIGHBORS).mean())
            )
            return PRESERVE * preserve + STEER * steer

        opt = LBFGS(
            [work],
            lr=1.0,
            max_iter=LBFGS_MAX_ITER,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        last = None
        with torch.enable_grad():
            for _ in range(LBFGS_ROUNDS):
                last = opt.step(closure)
        with torch.no_grad():
            W.data.copy_(get_matrix().detach().to(W.dtype))
        work.grad = None
        del work
        done += 1
        print(
            f"  L{layer_i:02d} {comp}[{mod_i}] loss={float(last):.6f}  ({done}/{n})",
            flush=True,
        )
    return done


def _copy_spark_sidecars(dst: Path) -> None:
    src = Path(MODEL)
    for name in (
        "configuration_spark.py",
        "modeling_spark.py",
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
        "README.md",
    ):
        s = src / name
        if s.is_file() and not (dst / name).exists():
            shutil.copy2(s, dst / name)


def _bake(engine: SteeringEngine, payload: dict) -> None:
    print(f"HIT — saving {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(
        str(MERGED), safe_serialization=True, max_shard_size="4GB"
    )
    engine.tokenizer.save_pretrained(str(MERGED))
    _copy_spark_sidecars(MERGED)
    payload["merged"] = str(MERGED)
    HIT_TXT.parent.mkdir(parents=True, exist_ok=True)
    HIT_TXT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MERGED} and {HIT_TXT}", flush=True)


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "spark_x25_ara_fixed_eval.json").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
    os.environ.setdefault("ABLITERIX_MAX_SEQ", "4096")
    cfg = _cfg()
    print(
        f"ARA Spark layers [{START_LAYER}, {END_LAYER})  "
        f"preserve={PRESERVE} steer={STEER} overcorrect={OVERCORRECT}",
        flush=True,
    )
    print("loading original Spark...", flush=True)
    engine = SteeringEngine(cfg)
    layers = _hf_layers(engine.model)
    print(f"layers={len(layers)}", flush=True)
    targets = list(_iter_targets(layers))
    print(f"target modules={len(targets)}", flush=True)
    if not targets:
        raise SystemExit("no o_proj/down_proj targets — Spark mapping failed")
    for layer_i, comp, mod_i, module in targets[:4]:
        print(
            f"  sample L{layer_i} {comp}[{mod_i}] {tuple(module.weight.shape)}",
            flush=True,
        )

    good = load_prompt_dataset(cfg, cfg.benign_prompts)
    bad = load_prompt_dataset(cfg, cfg.target_prompts)
    print(f"good={len(good)} bad={len(bad)}", flush=True)

    print("Module I/O good...", flush=True)
    good_io = collect_io(engine, good, targets)
    print("Module I/O bad...", flush=True)
    bad_io = collect_io(engine, bad, targets)

    print("Abliterating (full-weight ARA, float32 LBFGS)...", flush=True)
    n_upd = apply_ara(targets, good_io, bad_io)
    del good_io, bad_io
    print(f"ARA updated {n_upd} modules", flush=True)

    delta = {}
    for layer_i, comp, mod_i, module in targets:
        delta[f"{layer_i}.{comp}.{mod_i}"] = (
            module.weight.detach().to("cpu").contiguous()
        )
    DELTA.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base": MODEL,
            "start_layer": START_LAYER,
            "end_layer": END_LAYER,
            "preserve": PRESERVE,
            "steer": STEER,
            "overcorrect": OVERCORRECT,
            "neighbors": NEIGHBORS,
            "weights": delta,
        },
        DELTA,
    )
    print(f"DELTA_OK {DELTA} n={len(delta)}", flush=True)

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)
    kl = float(scorer.measure_kl_divergence(engine))
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    hit = bool(int(refusals) <= 10 and kl <= 0.05)
    payload = {
        "named_candidate": "Spark-X2.5-4B-ara-fixed",
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "hit": hit,
        "delta": str(DELTA),
        "start_layer": START_LAYER,
        "end_layer": END_LAYER,
        "n_modules": n_upd,
        "preserve": PRESERVE,
        "steer": STEER,
        "overcorrect": OVERCORRECT,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(f"SCORE ara_fixed: {refusals}/{n} @ {kl:.4f}  HIT={hit}", flush=True)
    _dump(payload)
    if hit:
        _bake(engine, payload)
    print("EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
