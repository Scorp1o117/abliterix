#!/usr/bin/env python3
"""Mergeable interpolant: (1-α) t24 + α ORBA-o9, target KL=0.10.

t24 is 50/100 @ 0.0869; ORBA o9.0 was 21/100 @ 0.131. Linear KL mix
α≈0.30 sits on the 0.10 boundary. Sweep α including overshoot past o9.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline  # noqa: E402
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import DirectTransform, SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
T24_DIR = Path("/run/media/s117/OS/Models/Qwen3.8-27B-t24")
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
VECTORS = Path(
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_t24_orba_lerp.json")
T24_POS = 49.611242819678694
T24_DIST = 5.105344096840982
ALPHAS = (0.25, 0.35, 0.50, 0.80, 1.00, 1.25, 1.40)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["t24_orba_lerp", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    # gfx1151 UMA is ~122G: 27B BF16 + two CPU clones of o_proj/down_proj
    # + batch-128 KL activations previously tripped the kernel OOM killer.
    cfg.inference.batch_size = 16
    cfg.inference.max_batch_size = 16
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.DIRECT
    cfg.steering.direct_transform = DirectTransform.ORBA
    cfg.steering.direct_transform_preserve_row_norm = True
    return cfg


def _band(o_max: float) -> dict[str, SteeringProfile]:
    return {
        "attn.o_proj": SteeringProfile(
            max_weight=float(o_max),
            max_weight_position=T24_POS,
            min_weight=max(0.0, float(o_max) * 0.5),
            min_weight_distance=T24_DIST,
        ),
        "mlp.down_proj": SteeringProfile(
            max_weight=0.37,
            max_weight_position=37.89,
            min_weight=0.25,
            min_weight_distance=30.46,
        ),
    }


def _dump(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_t24_orba_lerp.json").write_text(text, encoding="utf-8")


def _is_steer_weight(name: str) -> bool:
    if "lora_" in name or not name.endswith("weight"):
        return False
    return any(s in name for s in (".o_proj.", ".out_proj.", ".down_proj."))


def _layer_suffix(name: str) -> str | None:
    marker = ".layers."
    if marker not in name:
        return None
    suf = name[name.rindex("layers.") :]
    return suf.replace(".base_layer.weight", ".weight")


def _load_t24_weights(param_names: list[str]) -> dict[str, torch.Tensor]:
    index = json.loads((T24_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    t24_keys = [k for k in weight_map if _is_steer_weight(k)]
    t24_by_suffix = {}
    for key in t24_keys:
        suf = _layer_suffix(key)
        if suf:
            t24_by_suffix[suf] = key
    mapping: dict[str, str] = {}
    for name in param_names:
        suf = _layer_suffix(name)
        if suf and suf in t24_by_suffix:
            mapping[name] = t24_by_suffix[suf]
    print(
        f"matched t24 tensors {len(mapping)}/{len(param_names)} "
        f"(t24 keys {len(t24_keys)})",
        flush=True,
    )
    if len(mapping) < 80:
        sample = param_names[:5]
        raise RuntimeError(f"too few t24 matches: {len(mapping)} sample={sample}")
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for name, key in mapping.items():
        by_shard.setdefault(weight_map[key], []).append((name, key))
    out: dict[str, torch.Tensor] = {}
    for shard, pairs in by_shard.items():
        path = T24_DIR / shard
        with safe_open(str(path), framework="pt", device="cpu") as fh:
            for name, key in pairs:
                out[name] = fh.get_tensor(key).clone()
    return out


def _snapshot(engine, names: list[str]) -> dict[str, torch.Tensor]:
    want = set(names)
    snap = {}
    for n, p in engine.model.named_parameters():
        if n in want:
            snap[n] = p.detach().cpu().contiguous().clone()
    return snap


def _assign_lerp(
    engine,
    t24_w: dict[str, torch.Tensor],
    delta: dict[str, torch.Tensor],
    alpha: float,
) -> None:
    """W ← t24 + α (orba − t24). Mix one tensor at a time in bf16."""
    with torch.no_grad():
        for n, p in engine.model.named_parameters():
            if n not in t24_w:
                continue
            mixed = t24_w[n].to(dtype=p.dtype) + float(alpha) * delta[n].to(dtype=p.dtype)
            p.data.copy_(mixed.to(device=p.device, non_blocking=False))
            del mixed


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "t24": str(T24_DIR),
        "strong": "ORBA o9.0 t24-band",
        "alphas": list(ALPHAS),
        "points": [],
    }
    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    names = [n for n, _p in engine.model.named_parameters() if _is_steer_weight(n)]
    print(f"live steer weights {len(names)}", flush=True)
    t24_w = _load_t24_weights(names)

    cache = torch.load(VECTORS, map_location="cpu", weights_only=False)
    apply_steering(
        engine,
        cache["vectors"],
        50.79667354766495,
        _band(9.0),
        cfg,
        benign_states=cache.get("benign_states"),
    )
    orba_w = _snapshot(engine, list(t24_w))
    print(f"orba snapshot {len(orba_w)}", flush=True)
    delta = {}
    for n, tw in t24_w.items():
        if n in orba_w:
            delta[n] = (orba_w[n].to(torch.float32) - tw.to(torch.float32)).to(tw.dtype)
    del orba_w
    print(f"delta tensors {len(delta)} (orba copy dropped)", flush=True)
    engine.restore_baseline()

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    hit = None
    best_kl10 = None
    for a in ALPHAS:
        _assign_lerp(engine, t24_w, delta, a)
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": f"lerp_t24_orba9_a{a:.2f}",
            "alpha": a,
            "keyword_refusals": int(refusals),
            "n": int(n),
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(
            f"SCORE {row['tag']}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
            flush=True,
        )
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            hit = a
            payload["named_candidate"] = row["tag"]
            break
        if best_kl10 is None or abs(kl - 0.10) < abs(best_kl10["full_distribution_kl_3token_vs_original"] - 0.10):
            best_kl10 = row
        engine.restore_baseline()

    if hit is None:
        payload["named_candidate"] = None
        payload["closest_kl_0_10"] = best_kl10
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("sweep finished, no in-budget point", flush=True)
        if best_kl10 is None:
            return
        # Still bake the KL≈0.10 interpolant so --merged can score it.
        a = float(best_kl10["alpha"])
        print(f"baking closest-to-0.10 interpolant α={a}", flush=True)
        _assign_lerp(engine, t24_w, delta, a)
        hit_tag = best_kl10["tag"]
    else:
        hit_tag = payload["named_candidate"]

    print(f"saving interpolant {hit_tag} → {MERGED}", flush=True)
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json"):
        src = Path(ORIGINAL) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    payload["merged"] = str(MERGED)
    payload["merged_tag"] = hit_tag
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
