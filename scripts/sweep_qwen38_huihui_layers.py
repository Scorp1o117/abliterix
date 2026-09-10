#!/usr/bin/env python3
"""Huihui-style layer skip + small pair set. In-memory, no merge.

From @support_huihui:
  * 32 dataset pairs beat larger sets on Qwen3.8-27B
  * first 15 layers left intact
  * later layers may also be skippable (our bands: 56–63 are dead)

We estimate r from 32+32 house train prompts (layer 38 mean-diff),
then project only layers [skip_lo, skip_hi).
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

from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline  # noqa: E402
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringMode  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_huihui_layers.json")
N_PAIRS = 32
R_LAYER = 38

# (tag, skip_before, skip_from_layer, strength)
# skip_before = don't touch layers [0, skip_before)
# skip_from_layer = don't touch layers [skip_from_layer, 64)
JOBS = [
    ("skip0-15_s1.00", 15, 64, 1.00),
    ("skip0-15_s0.85", 15, 64, 0.85),
    ("skip0-15_56-63_s1.00", 15, 56, 1.00),
    ("skip0-15_56-63_s1.15", 15, 56, 1.15),
    ("skip0-15_32-39_56-63_s1.00", 15, 56, 1.00),  # extra skip 32-39 applied in code
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["huihui_layers", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    return cfg


def _layer_idx(name: str) -> int | None:
    if ".layers." not in name:
        return None
    part = name.split(".layers.")[1]
    try:
        return int(part.split(".")[0])
    except ValueError:
        return None


def _is_writer(name: str) -> bool:
    if "visual." in name or "lora_" in name:
        return False
    if not name.endswith("weight"):
        return False
    return any(s in name for s in (".o_proj.weight", ".out_proj.weight", ".down_proj.weight"))


def _project(weight: Tensor, r: Tensor, strength: float) -> None:
    W = weight.data.to(torch.float32)
    v = r.to(device=W.device, dtype=torch.float32)
    v = v / v.norm().clamp(min=1e-8)
    if v.numel() == W.shape[0]:
        W_new = W - strength * v.unsqueeze(1) * (v @ W).unsqueeze(0)
    elif v.numel() == W.shape[1]:
        W_new = W - strength * (W @ v).unsqueeze(1) * v.unsqueeze(0)
    else:
        return
    weight.data.copy_(W_new.to(weight.dtype))


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_huihui_layers.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "save_pretrained": False,
        "n_pairs": N_PAIRS,
        "r_layer": R_LAYER,
        "points": [],
    }
    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    harm = load_prompt_dataset(cfg, cfg.target_prompts)[:N_PAIRS]
    beni = load_prompt_dataset(cfg, cfg.benign_prompts)[:N_PAIRS]
    print(f"extract 32+32 for r at layer {R_LAYER}...", flush=True)
    h_st = engine.extract_hidden_states_batched(harm)
    b_st = engine.extract_hidden_states_batched(beni)
    idx = R_LAYER + 1
    r = (h_st[:, idx].float().mean(0) - b_st[:, idx].float().mean(0))
    r = r / r.norm().clamp(min=1e-8)
    del h_st, b_st
    print(f"r dim={r.numel()} |r|_max={float(r.abs().max()):.4f}", flush=True)

    writers = [(n, p) for n, p in engine.model.named_parameters() if p.ndim == 2 and _is_writer(n)]
    originals = {n: p.detach().clone() for n, p in writers}
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    jobs = [
        ("huihui_skip0-15_s1.00", 15, 64, 1.00, set()),
        ("huihui_skip0-15_s0.85", 15, 64, 0.85, set()),
        ("huihui_skip0-15_56-63_s1.00", 15, 56, 1.00, set()),
        ("huihui_skip0-15_56-63_s1.15", 15, 56, 1.15, set()),
        ("huihui_skip0-15_32-39_56-63_s1.00", 15, 56, 1.00, set(range(32, 40))),
    ]
    for tag, lo, hi, s, extra_skip in jobs:
        n_edit = 0
        with torch.no_grad():
            for n, p in writers:
                p.data.copy_(originals[n])
                li = _layer_idx(n)
                if li is None or li < lo or li >= hi or li in extra_skip:
                    continue
                _project(p, r, s)
                n_edit += 1
        print(f"{tag} edited {n_edit} tensors", flush=True)
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": tag,
            "keyword_refusals": int(refusals),
            "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
            "n_edit": n_edit,
            "strength": s,
        }
        print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = tag
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {tag} (not saving)", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
