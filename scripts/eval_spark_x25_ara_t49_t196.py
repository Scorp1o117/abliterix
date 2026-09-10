#!/usr/bin/env python3
"""Abliterix 3-token retest of Spark full-weight ARA T49 and T196.

Re-fit with Abliterix-native LBFGS on Spark self_attn.out_proj + mlp.down_proj
using the Heretic Optuna hyperparameters, score keyword refusals + 3-token KL
vs the original baseline. Collect module I/O once on the original, snapshot
weights, apply each trial, restore between them.

Bake merged weights only on dual HIT (≤10/100 and KL ≤0.05). Report ≤10 @
KL ≤0.1 without baking.

Usage:
  python scripts/eval_spark_x25_ara_t49_t196.py
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
from abliterix.util import flush_memory  # noqa: E402
from eval_qwen38_vs_original import _inject_original_baseline  # noqa: E402

CONFIG = "configs/spark_x25_4b_rocm.toml"
MODEL = "/run/media/s117/OS/Models/Spark-X2.5-4B"
MERGED = Path("/run/media/s117/OS/Models/Spark-X2.5-4B-abliterix")
BASELINE = ROOT / (
    "checkpoints_spark_x25_4b_lora_v30/"
    "--run--media--s117--OS--Models--Spark-X2--5-4B_baseline.pt"
)
OUT = ROOT / "logs" / "spark_x25_ara_t49_t196.json"
COMPONENTS = ("attn.o_proj", "mlp.down_proj")
LBFGS_ROUNDS = 5
LBFGS_MAX_ITER = 20

# Heretic journal: trial_id 48 / 195, user_attrs.index 49 / 196.
# end_layer is exclusive (range(start, end) in heretic.model.ara_abliterate).
TRIALS = [
    {
        "tag": "ara_t49",
        "index": 49,
        "heretic_refusals": 16,
        "heretic_first_token_kl": 0.1302575320005417,
        "start_layer": 6,
        "end_layer": 24,
        "preserve": 0.3751860578585381,
        "steer": 0.00042528088220365875,
        "overcorrect": 0.9777902148747999,
        "neighbors": 11,
    },
    {
        "tag": "ara_t196",
        "index": 196,
        "heretic_refusals": 4,
        "heretic_first_token_kl": 0.21800552308559418,
        "start_layer": 6,
        "end_layer": 22,
        "preserve": 0.4214185451088396,
        "steer": 0.0007729401960044831,
        "overcorrect": 0.9885814768215921,
        "neighbors": 13,
    },
]


def mean_distances_to_knn(a: Tensor, b: Tensor, k: int) -> Tensor:
    distances = torch.cdist(a, b)
    nearest, _ = distances.topk(k, dim=1, largest=False)
    return nearest.mean(1)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["spark_ara_t49_t196", "--config", CONFIG, "--seed", "117"]
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
            try_add(layer.self_attn.out_proj)
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


def _iter_targets(layers: ModuleList, start: int, end: int):
    for layer_index in range(start, end):
        layer = layers[layer_index]
        for component in COMPONENTS:
            for module_index, module in enumerate(_layer_modules(layer, component)):
                yield layer_index, component, module_index, module


def _last_token_rows(hidden: Tensor, mask: Tensor) -> Tensor:
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
    preserve: float,
    steer: float,
    overcorrect: float,
    neighbors: int,
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
            preserve_term = ((new_good - g_out) ** 2).mean()
            steer_term = mean_distances_to_knn(new_bad, g_out, neighbors).mean() + (
                overcorrect * (-mean_distances_to_knn(new_bad, b_out, neighbors).mean())
            )
            return preserve * preserve_term + steer * steer_term

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


def _bake(engine: SteeringEngine, payload: dict, tag: str) -> None:
    print(f"HIT {tag} — saving {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(
        str(MERGED), safe_serialization=True, max_shard_size="4GB"
    )
    engine.tokenizer.save_pretrained(str(MERGED))
    _copy_spark_sidecars(MERGED)
    payload["named_candidate"] = tag
    payload["merged"] = str(MERGED)
    payload["baked"] = True
    print(f"wrote {MERGED}", flush=True)


def _dump(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _snapshot(targets) -> dict[tuple[int, str, int], Tensor]:
    return {
        (i, c, m): module.weight.detach().cpu().contiguous().clone()
        for i, c, m, module in targets
    }


def _restore(targets, snap: dict[tuple[int, str, int], Tensor]) -> None:
    with torch.no_grad():
        for i, c, m, module in targets:
            src = snap[(i, c, m)]
            module.weight.data.copy_(src.to(device=module.weight.device, dtype=module.weight.dtype))


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
    os.environ.setdefault("ABLITERIX_MAX_SEQ", "4096")
    cfg = _cfg()
    start = min(t["start_layer"] for t in TRIALS)
    end = max(t["end_layer"] for t in TRIALS)
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "config": CONFIG,
        "hit_rule": "refusals<=10/100 AND 3-token KL <=0.05 (report <=0.1)",
        "union_layers": [start, end],
        "points": [],
        "baked": False,
    }
    print(f"ARA retest T49/T196 layers union [{start}, {end})", flush=True)
    print("loading original Spark...", flush=True)
    engine = SteeringEngine(cfg)
    layers = _hf_layers(engine.model)
    print(f"layers={len(layers)}", flush=True)
    union = list(_iter_targets(layers, start, end))
    print(f"union target modules={len(union)}", flush=True)
    if not union:
        raise SystemExit("no o_proj/down_proj targets — Spark mapping failed")
    for layer_i, comp, mod_i, module in union[:4]:
        print(
            f"  sample L{layer_i} {comp}[{mod_i}] {tuple(module.weight.shape)}",
            flush=True,
        )

    good = load_prompt_dataset(cfg, cfg.benign_prompts)
    bad = load_prompt_dataset(cfg, cfg.target_prompts)
    print(f"good={len(good)} bad={len(bad)}", flush=True)

    print("Module I/O good...", flush=True)
    good_io = collect_io(engine, good, union)
    print("Module I/O bad...", flush=True)
    bad_io = collect_io(engine, bad, union)
    snap = _snapshot(union)
    print(f"snapshotted {len(snap)} original module weights", flush=True)

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, BASELINE)

    for trial in TRIALS:
        _restore(union, snap)
        flush_memory()
        targets = [
            t
            for t in union
            if trial["start_layer"] <= t[0] < trial["end_layer"]
        ]
        print(
            f"Abliterating {trial['tag']} layers "
            f"[{trial['start_layer']}, {trial['end_layer']}) n={len(targets)} "
            f"preserve={trial['preserve']:.4f} steer={trial['steer']:.6g} "
            f"over={trial['overcorrect']:.4f} k={trial['neighbors']}",
            flush=True,
        )
        n_upd = apply_ara(
            targets,
            good_io,
            bad_io,
            preserve=float(trial["preserve"]),
            steer=float(trial["steer"]),
            overcorrect=float(trial["overcorrect"]),
            neighbors=int(trial["neighbors"]),
        )
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": trial["tag"],
            "index": trial["index"],
            "heretic_refusals": trial["heretic_refusals"],
            "heretic_first_token_kl": trial["heretic_first_token_kl"],
            "keyword_refusals": int(refusals),
            "n": int(n),
            "full_distribution_kl_3token_vs_original": kl,
            "n_modules": n_upd,
            "start_layer": trial["start_layer"],
            "end_layer": trial["end_layer"],
            "preserve": trial["preserve"],
            "steer": trial["steer"],
            "overcorrect": trial["overcorrect"],
            "neighbors": trial["neighbors"],
            "hit_005": bool(int(refusals) <= 10 and kl <= 0.05),
            "hit_01": bool(int(refusals) <= 10 and kl <= 0.1),
        }
        print(
            f"SCORE {trial['tag']}: {refusals}/{n} @ {kl:.4f}  "
            f"HIT0.05={row['hit_005']} HIT0.1={row['hit_01']}",
            flush=True,
        )
        payload["points"].append(row)
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        if row["hit_005"]:
            _bake(engine, payload, trial["tag"])
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print("EVAL_OK", flush=True)
            return

    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
