# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cliff-head attention ablation for reasoning models.

Implements the inverse of the safety-head finding from
`Bao et al., 2025 <https://arxiv.org/abs/2510.06036>`_ — *Refusal Falls Off
a Cliff: How Safety Alignment Fails in Reasoning Models*.

The paper finds that in reasoning models (R1, o-style, Qwen-Thinking,
Kimi-Thinking) a sparse set of attention heads carries the refusal
signal; ablating ~3 % of them sharply moves refusal behaviour. The paper
ablates *anti*-refusal heads to recover safety; we ablate the *pro*-refusal
heads to remove safety, taking the same mechanism in the opposite
direction.

Identification heuristic
------------------------
For each ``(layer, head)`` we score the alignment between the head's
output sub-space and the per-layer refusal direction. The output of
head ``h`` lives in the column range ``[h * head_dim, (h + 1) * head_dim)``
of ``o_proj.weight`` — projecting the refusal direction through those
columns measures how much that head's output contributes to the refusal
axis. The top fraction (default 3 %) is ablated by scaling the
corresponding o_proj columns toward zero.

This is a *static* alignment heuristic, not the activation-patching probe
the paper uses; we trade a small loss in localisation precision for ~10 ⁴×
speedup, which is acceptable for an automated abliteration sweep.

bitsandbytes 4-bit
------------------
``Params4bit`` stores packed uint8 data — in-place float column scaling
would corrupt it. Identify dequantises read-only; apply dequantises →
edits → requantises and replaces ``base.weight`` (original Parameter kept
for :func:`restore_cliff_head_ablation`).

Reversibility
-------------
Original weight slices / Parameters are cached on the engine in
``engine._cliff_head_originals`` so :func:`restore` can roll them back —
the same mechanism direct-mode steering already uses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class HeadScore:
    layer: int
    head: int
    score: float


# ---------------------------------------------------------------------------
# Model-shape discovery / weight helpers
# ---------------------------------------------------------------------------


def _get_head_dim(engine) -> tuple[int, int]:
    """Return ``(num_attention_heads, head_dim)`` for the engine's model.

    Prefer ``v_head_dim`` when present (MLA ``dense`` / o_proj columns are
    ``num_heads * v_head_dim``). Falls back to ``head_dim`` or
    ``hidden_size // num_heads``. Raise on missing attributes — cliff-head
    cannot proceed without head-level addressing.
    """
    cfg = getattr(engine.model, "config", None)
    if cfg is None:
        raise RuntimeError("Engine has no loaded HF model — cannot read head shape.")
    text_cfg = getattr(cfg, "text_config", cfg)

    num_heads = getattr(text_cfg, "num_attention_heads", None)
    if num_heads is None:
        raise RuntimeError(
            "Model config exposes no num_attention_heads — cliff-head ablation "
            "needs head-level addressing. Disable cliff_head_ablation for this "
            "architecture."
        )

    # o_proj / dense columns track value head width on MLA hybrids (Bailing).
    v_head_dim = getattr(text_cfg, "v_head_dim", None)
    head_dim = getattr(text_cfg, "head_dim", None)
    if v_head_dim is not None:
        head_dim = int(v_head_dim)
    elif head_dim is None:
        hidden = getattr(text_cfg, "hidden_size", None)
        if hidden is None:
            raise RuntimeError(
                "Cannot infer head_dim: neither v_head_dim, head_dim, nor "
                "hidden_size is set on the model config."
            )
        head_dim = hidden // num_heads
    return int(num_heads), int(head_dim)


def _resolve_head_dim(
    in_features: int,
    num_heads: int,
    config_head_dim: int,
    *,
    alt_dims: Iterable[int] | None = None,
) -> int | None:
    """Pick a column width that tiles ``o_proj.in_features``.

    Prefer the primary config value when ``in_features == num_heads * dim``.
    Otherwise try any alternate dims from the model config (e.g. switch
    between ``v_head_dim`` and ``head_dim`` on hybrids). Return ``None``
    when no configured dim matches — never invent a width from
    ``in_features // num_heads`` alone (that would mis-slice GQA / odd
    projections).
    """
    candidates: list[int] = [config_head_dim]
    if alt_dims is not None:
        for d in alt_dims:
            d_int = int(d)
            if d_int not in candidates:
                candidates.append(d_int)
    for dim in candidates:
        if dim > 0 and in_features == num_heads * dim:
            return dim
    return None


def _weight_as_float32(weight) -> Tensor:
    """Return a float32 view of ``weight`` for scoring / float edits.

    Dequantises bitsandbytes 4-bit / int8 packed storage. Does **not**
    write back — callers that need a persistent edit must requantise.
    """
    qs = getattr(weight, "quant_state", None)
    if qs is not None:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "bitsandbytes is required to dequantize 4-bit o_proj weights "
                "for cliff-head ablation."
            ) from exc
        return bnb.functional.dequantize_4bit(  # ty:ignore[possibly-missing-attribute]
            weight.data, qs
        ).to(torch.float32)

    CB = getattr(weight, "CB", None)
    if CB is not None:
        scb = getattr(weight, "SCB", None)
        if scb is None:
            raise RuntimeError("int8 weight has CB but no SCB scale buffer.")
        return CB.float() * scb.float().unsqueeze(1) / 127.0

    return weight.detach().to(torch.float32)


def _is_quantized_weight(weight) -> bool:
    return (
        getattr(weight, "quant_state", None) is not None
        or getattr(weight, "CB", None) is not None
    )


def _requantize_4bit(W_float: Tensor, *, quant_type: str, device) -> torch.nn.Parameter:
    """Pack a float matrix back into a ``Params4bit`` on ``device``."""
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Params4bit
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "bitsandbytes is required to requantize cliff-head o_proj edits."
        ) from exc

    # Match common HF bnb load dtype; float16 is fine for NF4 packing input.
    src = W_float.detach().to(device=device, dtype=torch.float16).contiguous()
    q, st = bnb.functional.quantize_4bit(src, quant_type=quant_type)
    param = Params4bit(q, requires_grad=False, quant_type=quant_type)
    param.quant_state = st
    return param


def _refusal_vector_at_layer(refusal_vector: Tensor, layer_idx: int) -> Tensor:
    """Pick the per-layer refusal direction, handling 2-D and 3-D inputs.

    Accepts either ``(layers+1, hidden)`` (single-direction) or
    ``(n_dirs, layers+1, hidden)`` (multi-direction / harmfulness pair),
    in which case the first slot (refusal slot by convention) is used.

    Index ``layer_idx + 1`` accounts for the embedding being at position 0
    in residual-stream tensors.
    """
    if refusal_vector.ndim == 3:
        return refusal_vector[0, layer_idx + 1, :]
    return refusal_vector[layer_idx + 1, :]


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def identify_safety_heads(
    engine,
    refusal_vector: Tensor,
    *,
    top_k_frac: float = 0.03,
    min_heads: int = 1,
) -> list[HeadScore]:
    """Score every ``(layer, head)`` by alignment with the refusal direction.

    Parameters
    ----------
    engine : SteeringEngine
        Must have an HF model loaded (``engine.model is not None``).
    refusal_vector : Tensor
        Either ``(layers+1, hidden)`` or ``(n_dirs, layers+1, hidden)``.
        Per-layer slices index from 1 (embedding at 0).
    top_k_frac : float
        Fraction of all heads to flag. Bao et al. report ~3 % is enough
        to flip refusal behaviour.
    min_heads : int
        Floor on the returned list size so very small models still get at
        least one ablation.

    Returns
    -------
    list[HeadScore]
        Sorted by score descending (most refusal-aligned first), truncated
        to the requested fraction.
    """
    num_heads, config_head_dim = _get_head_dim(engine)
    n_layers = engine.get_n_layers()
    cfg = getattr(engine.model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    alt_dims: list[int] = []
    if text_cfg is not None:
        for name in ("head_dim", "v_head_dim"):
            val = getattr(text_cfg, name, None)
            if val is not None:
                alt_dims.append(int(val))

    scores: list[HeadScore] = []
    for layer_idx in range(n_layers):
        modules = engine.steerable_modules(layer_idx)
        o_proj_list = modules.get("attn.o_proj", [])
        if not o_proj_list:
            continue
        o_proj = o_proj_list[0]

        base = o_proj.base_layer if hasattr(o_proj, "base_layer") else o_proj
        weight = base.weight

        try:
            W32 = _weight_as_float32(weight)
        except RuntimeError:
            # Unreadable quant layout — skip layer rather than crash the run.
            continue

        if W32.ndim != 2:
            continue

        out_features, in_features = W32.shape
        head_dim = _resolve_head_dim(
            in_features, num_heads, config_head_dim, alt_dims=alt_dims
        )
        if head_dim is None:
            continue

        v_layer = _refusal_vector_at_layer(refusal_vector, layer_idx)
        # Align device with the dequant matrix (may be CPU for some paths).
        v = v_layer.to(device=W32.device, dtype=torch.float32)

        # Project the refusal direction through each head's column block.
        # head_cols shape: (out_features, head_dim).
        # contrib = || head_cols^T @ v_layer || / ||v_layer||  (norm in float32)
        v_norm = torch.linalg.vector_norm(v).clamp(min=1e-8)
        for head in range(num_heads):
            head_cols = W32[:, head * head_dim : (head + 1) * head_dim]
            contrib = (torch.linalg.vector_norm(head_cols.T @ v) / v_norm).item()
            scores.append(HeadScore(layer=layer_idx, head=head, score=contrib))

    if not scores:
        return []

    scores.sort(key=lambda s: s.score, reverse=True)
    k = max(min_heads, int(round(top_k_frac * len(scores))))
    return scores[: min(k, len(scores))]


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def apply_cliff_head_ablation(
    engine,
    head_list: Iterable[HeadScore],
    *,
    strength: float = 1.0,
) -> int:
    """Scale toward zero the ``o_proj`` columns of the listed heads.

    Parameters
    ----------
    engine : SteeringEngine
    head_list : Iterable[HeadScore]
        Output of :func:`identify_safety_heads` (or any iterable of
        ``HeadScore`` instances).
    strength : float
        Multiplicative ablation strength. ``1.0`` zeroes the columns fully;
        ``0.5`` halves them (useful when the heuristic over-flags heads);
        ``0.0`` is a no-op.

    Returns
    -------
    int
        Number of ``(layer, head)`` pairs actually modified.

    Side effects
    ------------
    Caches original weight slices (float path) or original Parameters
    (bnb path) in ``engine._cliff_head_originals`` so
    :func:`restore_cliff_head_ablation` can roll back.

    For bitsandbytes 4-bit ``o_proj`` weights the module is dequantised,
    columns are scaled, then re-quantised and reassigned — packed uint8
    storage is never edited in place.
    """
    if strength <= 0.0:
        return 0

    num_heads, config_head_dim = _get_head_dim(engine)
    cfg = getattr(engine.model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    alt_dims: list[int] = []
    if text_cfg is not None:
        for name in ("head_dim", "v_head_dim"):
            val = getattr(text_cfg, name, None)
            if val is not None:
                alt_dims.append(int(val))

    if not hasattr(engine, "_cliff_head_originals"):
        engine._cliff_head_originals = {}

    # Group by layer so we touch each o_proj module once per layer.
    by_layer: dict[int, list[int]] = defaultdict(list)
    for entry in head_list:
        by_layer[entry.layer].append(entry.head)

    n_modified = 0
    touched_quant_modules = False

    for layer_idx, heads in by_layer.items():
        modules = engine.steerable_modules(layer_idx)
        for o_proj in modules.get("attn.o_proj", []):
            base = o_proj.base_layer if hasattr(o_proj, "base_layer") else o_proj
            weight = base.weight

            try:
                W32 = _weight_as_float32(weight)
            except RuntimeError:
                continue

            if W32.ndim != 2:
                continue

            in_features = W32.shape[1]
            head_dim = _resolve_head_dim(
                in_features, num_heads, config_head_dim, alt_dims=alt_dims
            )
            if head_dim is None:
                continue

            quantized = _is_quantized_weight(weight)

            if quantized:
                # Keep the original Parameter for restore; edit a float copy.
                mod_key = ("bnb_module", id(base))
                if mod_key not in engine._cliff_head_originals:
                    engine._cliff_head_originals[mod_key] = ("bnb", base, weight)

                for head in heads:
                    lo = head * head_dim
                    hi = lo + head_dim
                    W32[:, lo:hi] *= 1.0 - strength
                    n_modified += 1

                qs = getattr(weight, "quant_state", None)
                quant_type = getattr(qs, "quant_type", None) or "nf4"
                device = weight.data.device
                base.weight = _requantize_4bit(
                    W32, quant_type=quant_type, device=device
                )
                touched_quant_modules = True
            else:
                data = weight.data
                for head in heads:
                    lo = head * head_dim
                    hi = lo + head_dim
                    # Cache the slice keyed by (weight, head) so restore is O(1).
                    key = (id(weight), head)
                    if key not in engine._cliff_head_originals:
                        engine._cliff_head_originals[key] = (
                            weight,
                            head,
                            data[:, lo:hi].clone(),
                        )
                    data[:, lo:hi] *= 1.0 - strength
                    n_modified += 1

    # Stale dequant cache would serve pre-cliff matrices to LoRA writers.
    if touched_quant_modules and hasattr(engine, "_dequant_cache"):
        cache = engine._dequant_cache
        if isinstance(cache, dict):
            cache.clear()
        if hasattr(engine, "_dequant_cache_bytes"):
            engine._dequant_cache_bytes = 0

    return n_modified


def restore_cliff_head_ablation(engine) -> int:
    """Restore every cached ``o_proj`` column slice / Parameter.

    Returns the number of cache entries restored. Safe to call when no
    ablation has been applied — returns 0 in that case.
    """
    cache = getattr(engine, "_cliff_head_originals", None)
    if not cache:
        return 0

    n = 0
    for entry in cache.values():
        # bnb path: ("bnb", base_module, original_Params4bit)
        if (
            isinstance(entry, tuple)
            and len(entry) == 3
            and entry[0] == "bnb"
        ):
            _, base, original_param = entry
            base.weight = original_param
            n += 1
            continue

        # float path: (weight, head, original_slice)
        weight, head, original = entry
        head_dim = original.shape[1]
        lo = head * head_dim
        hi = lo + head_dim
        weight.data[:, lo:hi] = original.to(weight.dtype)
        n += 1

    cache.clear()
    return n


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def run_cliff_head_ablation(
    engine,
    refusal_vector: Tensor,
    *,
    top_k_frac: float = 0.03,
    strength: float = 1.0,
    min_heads: int = 1,
) -> tuple[int, list[HeadScore]]:
    """End-to-end: identify + ablate, returning ``(n_modified, head_list)``.

    Convenience wrapper around :func:`identify_safety_heads` followed by
    :func:`apply_cliff_head_ablation`. Returns the ablated head list so
    the caller can log it.
    """
    heads = identify_safety_heads(
        engine,
        refusal_vector,
        top_k_frac=top_k_frac,
        min_heads=min_heads,
    )
    n_modified = apply_cliff_head_ablation(engine, heads, strength=strength)
    return n_modified, heads
