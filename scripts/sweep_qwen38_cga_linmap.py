#!/usr/bin/env python3
"""Bake CGA's residual change as a mergeable low-rank map.

Fit per-layer A such that h @ A ≈ (h_cga − h_orig), with extra weight on
benign rows targeting 0. Then W' = (I + s Aᵀ) W on residual writers.
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
from torch import Tensor  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import SteeringMode, SteeringProfile  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
ARTIFACT = ROOT / "artifacts" / "qwen38_cga_global_t035_s160.json"
POCKET_VECTORS = ROOT / (
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCORER_CACHE = ROOT / (
    "checkpoints_qwen38_27b_cga/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_concept_scorers.pt"
)
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_cga_linmap.json")
MAPS = Path("checkpoints_qwen38_27b_cga/cga_linmap_A_l24-56_r4.pt")
N = 160
RANK = 4
LAM = 8.0
BENIGN_W = 3.0
STRENGTHS = (0.5, 0.85, 1.0, 1.3)
LAYER_LO, LAYER_HI = 24, 56


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["cga_linmap", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.runtime_hook_site = "decoder_block"
    return cfg


def _apply_cga(engine: SteeringEngine, cfg: AbliterixConfig) -> None:
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.concept_gate_scope = art["concept_gate_scope"]
    cfg.steering.concept_gate_threshold = float(art["concept_gate_threshold"])
    cfg.steering.concept_gate_global_decision_layer = int(art["concept_gate_global_decision_layer"])
    cfg.steering.concept_gate_angular_overrotation = bool(art["concept_gate_angular_overrotation"])
    cfg.steering.concept_gate_positive_alignment_only = bool(art["concept_gate_positive_alignment_only"])
    blob = torch.load(SCORER_CACHE, map_location="cpu", weights_only=False)
    engine._concept_scorers = blob["scorers"] if isinstance(blob, dict) and "scorers" in blob else blob
    vectors = torch.load(POCKET_VECTORS, map_location="cpu", weights_only=False)["vectors"]
    p = art["profile"]["attn.o_proj"]
    apply_steering(
        engine,
        vectors,
        None,
        {
            "attn.o_proj": SteeringProfile(
                max_weight=float(p["max_weight"]),
                max_weight_position=float(p["max_weight_position"]),
                min_weight=float(p["min_weight"]),
                min_weight_distance=float(p["min_weight_distance"]),
            )
        },
        cfg,
    )
    print(f"CGA hooks={len(getattr(engine, '_angular_hooks', []) or [])}", flush=True)


def fit_factors(X: Tensor, Y: Tensor, rank: int, lam: float) -> dict[str, Tensor]:
    """Ridge A: X @ A ≈ Y on CPU via the n×n dual, then rank-k factors.

    Avoids a 5120×5120 GPU SVD (that wedged amdgpu / gnome-shell).
    A = Xᵀ (XXᵀ + λI)⁻¹ Y, then randomized truncated SVD of A.
    """
    X = X.cpu().contiguous().float()
    Y = Y.cpu().contiguous().float()
    n, d = X.shape
    k = min(rank, n, d)
    G = X @ X.T
    G.diagonal().add_(float(lam))
    C = torch.linalg.solve(G, Y)
    q = min(k + 6, n, d)
    omega = torch.randn(d, q, dtype=X.dtype)
    ym = X.T @ (C @ omega)
    qf, _ = torch.linalg.qr(ym, mode="reduced")
    B = (qf.T @ X.T) @ C
    ub, s, vh = torch.linalg.svd(B, full_matrices=False)
    U = qf @ ub[:, :k]
    return {
        "U": U[:, :k].contiguous(),
        "S": s[:k].contiguous(),
        "Vh": vh[:k].contiguous(),
    }


def apply_factors(W: Tensor, fac: dict[str, Tensor], scale: float) -> Tensor:
    """W ← W + s Aᵀ W with A = (U diag(S) Vh), all on CPU."""
    U = fac["U"]
    S = fac["S"]
    Vh = fac["Vh"]
    mid = U.T @ W
    mid = S.unsqueeze(1) * mid
    return W + float(scale) * (Vh.T @ mid)


def _is_writer(name: str) -> bool:
    if "visual." in name or "lora_" in name:
        return False
    if not name.endswith("weight"):
        return False
    return any(s in name for s in (".o_proj.weight", ".out_proj.weight", ".down_proj.weight"))


def _layer_idx(name: str) -> int | None:
    if ".layers." not in name:
        return None
    part = name.split(".layers.")[1]
    try:
        return int(part.split(".")[0])
    except ValueError:
        return None


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_linmap.json").write_text(text, encoding="utf-8")


def main() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "save_pretrained": False,
        "rank": RANK,
        "n": N,
        "points": [],
    }
    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    device = next(engine.model.parameters()).device
    maps: dict[int, Tensor] = {}
    if MAPS.is_file():
        blob = torch.load(MAPS, map_location="cpu", weights_only=False)
        maps = {int(k): v for k, v in blob["maps"].items()}
        print(f"loaded A maps {MAPS} n={len(maps)}", flush=True)
    else:
        harm = load_prompt_dataset(cfg, cfg.target_prompts)[:N]
        beni = load_prompt_dataset(cfg, cfg.benign_prompts)[:N]
        print(f"extract orig n={N}+{N}", flush=True)
        orig_h = engine.extract_hidden_states_batched(harm).cpu().float()
        orig_b = engine.extract_hidden_states_batched(beni).cpu().float()
        _apply_cga(engine, cfg)
        print("extract CGA states...", flush=True)
        cga_h = engine.extract_hidden_states_batched(harm).cpu().float()
        cga_b = engine.extract_hidden_states_batched(beni).cpu().float()
        engine.restore_baseline()
        print(
            f"delta_h rms={float((cga_h - orig_h).pow(2).mean().sqrt()):.5f} "
            f"delta_b rms={float((cga_b - orig_b).pow(2).mean().sqrt()):.5f}",
            flush=True,
        )
        for layer in range(LAYER_LO, LAYER_HI + 1):
            idx = layer + 1
            Xh = orig_h[:, idx]
            Xb = orig_b[:, idx]
            Yh = cga_h[:, idx] - orig_h[:, idx]
            Yb = torch.zeros_like(Xb)
            X = torch.cat([Xb * BENIGN_W, Xh], dim=0)
            Y = torch.cat([Yb, Yh], dim=0)
            maps[layer] = fit_factors(X, Y, RANK, LAM)
            fro = float((maps[layer]["S"] ** 2).sum().sqrt())
            print(f"  A[{layer}] fro≈{fro:.4f} (CPU rank-{RANK})", flush=True)
        del orig_h, orig_b, cga_h, cga_b
        MAPS.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"maps": maps, "rank": RANK, "lam": LAM, "n": N}, MAPS)
        print(f"wrote {MAPS}", flush=True)

    writers = [(n, p) for n, p in engine.model.named_parameters() if p.ndim == 2 and _is_writer(n)]
    originals = {n: p.detach().clone() for n, p in writers}
    print(f"writers {len(writers)} maps {len(maps)}", flush=True)

    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)

    for s in STRENGTHS:
        with torch.no_grad():
            for n, p in writers:
                p.data.copy_(originals[n])
                li = _layer_idx(n)
                if li is None or li not in maps:
                    continue
                fac = maps[li]
                W = p.data.detach().cpu().float()
                if fac["U"].shape[0] != W.shape[0]:
                    continue
                W_new = apply_factors(W, fac, s)
                p.data.copy_(W_new.to(device=p.device, dtype=p.dtype))
                del W, W_new
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
        n = len(scorer.target_msgs)
        row = {
            "tag": f"cga_lin_s{s:.2f}_r{RANK}",
            "scale": s,
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
