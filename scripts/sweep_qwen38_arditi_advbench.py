#!/usr/bin/env python3
"""Arditi r from AdvBench vs Alpaca (OrcaRouter's datasets), eval on our holdout.

In-memory, no 52G write.
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
from datasets import load_dataset  # noqa: E402
from torch import Tensor  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import ChatMessage, SteeringMode  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_arditi_advbench.json")
LAYER = 38
N_ALPACA = 800
STRENGTHS = (1.00, 1.12, 1.20)


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["arditi_adv", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    return cfg


def massive_masked_direction(harm: Tensor, benign: Tensor, q: float = 0.99) -> Tensor:
    both = torch.cat([harm, benign], dim=0).float()
    peak = both.abs().amax(dim=0)
    keep = peak <= torch.quantile(peak, q)
    h = harm.float().clone()
    b = benign.float().clone()
    h[:, ~keep] = 0
    b[:, ~keep] = 0
    r = h.mean(0) - b.mean(0)
    r = r / r.norm().clamp(min=1e-8)
    print(f"r kept {int(keep.sum())}/{keep.numel()} |r|_max={float(r.abs().max()):.4f}", flush=True)
    return r


def _is_writer(name: str) -> bool:
    if "visual." in name or "vision" in name or "lora_" in name:
        return False
    if not name.endswith("weight"):
        return False
    return any(s in name for s in (".o_proj.weight", ".out_proj.weight", ".down_proj.weight", "embed_tokens.weight"))


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
    (SCRATCH / "qwen38_arditi_advbench.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "save_pretrained": False,
        "r_source": "mlabonne/harmful_behaviors vs tatsu-lab/alpaca[:800] layer 38",
        "points": [],
    }
    harm_ds = load_dataset("mlabonne/harmful_behaviors", split="train")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harm_msgs = [
        ChatMessage(system="You are a helpful assistant.", user=str(row["text"]))
        for row in harm_ds
        if str(row.get("text") or "").strip()
    ]
    beni_msgs = []
    for row in alpaca:
        if len(beni_msgs) >= N_ALPACA:
            break
        inst = str(row.get("instruction") or "").strip()
        extra = str(row.get("input") or "").strip()
        user = f"{inst}\n{extra}".strip() if extra else inst
        if user:
            beni_msgs.append(ChatMessage(system="You are a helpful assistant.", user=user))
    print(f"extract n_harm={len(harm_msgs)} n_benign={len(beni_msgs)}", flush=True)

    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    print("extracting hidden states...", flush=True)
    h_states = engine.extract_hidden_states_batched(harm_msgs)
    b_states = engine.extract_hidden_states_batched(beni_msgs)
    idx = LAYER + 1
    r = massive_masked_direction(h_states[:, idx, :], b_states[:, idx, :])
    del h_states, b_states

    writers = [(n, p) for n, p in engine.model.named_parameters() if p.ndim == 2 and _is_writer(n)]
    print(f"writers {len(writers)}", flush=True)
    originals = {n: p.detach().clone() for n, p in writers}

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    for s in STRENGTHS:
        for n, p in writers:
            p.data.copy_(originals[n])
            _project(p, r, s)
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": f"advbench_s{s:.2f}",
            "keyword_refusals": int(refusals),
            "n": n,
            "full_distribution_kl_3token_vs_original": kl,
            "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        }
        print(f"SCORE {row['tag']}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
        payload["points"].append(row)
        _dump(payload)
        if row["in_budget"]:
            payload["named_candidate"] = row["tag"]
            payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
            _dump(payload)
            print(f"HIT {row['tag']} (not saving)", flush=True)
            return
    payload["named_candidate"] = None
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print("sweep finished, no in-budget point", flush=True)


if __name__ == "__main__":
    main()
