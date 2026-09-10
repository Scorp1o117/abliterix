# Abliterix — Qwen4Exp / Qwen3.8-Flash-Next load and residual helpers.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qwen4Exp-specific load, placement, and residual-capture helpers.

Flash-Next widens the decoder residual to ``hc_count * hidden_size`` (4×2560 =
10240) and stores a ~95 GiB n-gram / PLE table as ``nn.Embedding`` shards.
Abliteration must:

* never clone those shards onto the UMA pool as a dense device tensor
* never ``torch.stack`` a 10240/2560 mix onto 2560 projection weights
* capture refusal directions in the 2560-d attn/MLP write-back space
* refuse in-process BF16 merge of the ~336G checkpoint on 128G hosts
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Linear, Module

# Substrings that identify the hashed n-gram / PLE embedding table.  Matching
# is done on dotted module/parameter names from ``named_parameters``.
_NGRAM_MARKERS = (
    "ple.ple_embedding.ngram_embedding",
    "ngram_embedding.shard_",
    "ngram_embedding.weight",
)

_FUSED_EXPERT_MARKERS = (
    "mlp.experts.down_proj",
    "mlp.experts.gate_up_proj",
    "ffn_down_exps",
    "ffn_gate_up_exps",
)

# Linear write-backs whose *output* lives in residual/hidden space (2560).
_WRITEBACK_PATHS = (
    "self_attn.o_proj",
    "linear_attn.out_proj",
    "mlp.down_proj",
    "mlp.shared_expert.down_proj",
)

# Host-side estimate used when safetensors index is not readable.
_NGRAM_BF16_GIB = 95.4
_FUSED_EXPERT_BF16_GIB = 241.6
_FLASH_NEXT_TOTAL_BF16_GIB = 335.3


def is_ngram_parameter_name(name: str) -> bool:
    """Return True if *name* is a PLE / n-gram embedding weight."""
    dotted = name.replace("-", ".")
    return any(marker in dotted for marker in _NGRAM_MARKERS)


def is_fused_expert_parameter_name(name: str) -> bool:
    """Return True if *name* is a fused 3-D routed-expert parameter."""
    dotted = name.replace("-", ".")
    if dotted.endswith(".weight") and "shared_expert" in dotted:
        return False
    return any(marker in dotted for marker in _FUSED_EXPERT_MARKERS)


def is_qwen4_exp_config(config: Any | None) -> bool:
    """Detect ``qwen4_exp`` from a HF config, text_config, or dict."""
    if config is None:
        return False
    if isinstance(config, dict):
        mt = str(config.get("model_type") or "")
        arches = config.get("architectures") or []
        text = config.get("text_config") or {}
        text_mt = str(text.get("model_type") or "") if isinstance(text, dict) else ""
        return (
            mt in {"qwen4_exp", "qwen4_exp_text"}
            or text_mt in {"qwen4_exp", "qwen4_exp_text"}
            or any("Qwen4Exp" in str(a) for a in arches)
        )
    mt = str(getattr(config, "model_type", "") or "")
    text = getattr(config, "text_config", None)
    text_mt = str(getattr(text, "model_type", "") or "") if text is not None else ""
    arches = getattr(config, "architectures", None) or []
    return (
        mt in {"qwen4_exp", "qwen4_exp_text"}
        or text_mt in {"qwen4_exp", "qwen4_exp_text"}
        or any("Qwen4Exp" in str(a) for a in arches)
    )


def model_type_from_id(model_id: str | os.PathLike[str] | None) -> str | None:
    """Read ``model_type`` from a local ``config.json`` when present."""
    if not model_id:
        return None
    path = Path(model_id)
    cfg_path = path / "config.json" if path.is_dir() else None
    if cfg_path is None or not cfg_path.is_file():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    mt = data.get("model_type")
    return str(mt) if mt else None


def writeback_hidden_size(config: Any | None) -> int | None:
    """Return the projection write-back width (``hidden_size``), not HC width."""
    if config is None:
        return None
    text = getattr(config, "text_config", None)
    for obj in (text, config):
        if obj is None:
            continue
        hidden = getattr(obj, "hidden_size", None)
        if isinstance(hidden, int) and hidden > 0:
            return hidden
        if isinstance(obj, dict):
            value = obj.get("hidden_size")
            if isinstance(value, int) and value > 0:
                return value
    return None


def gated_residual_stream_width(config: Any | None) -> int | None:
    """Return ``hc_count * hidden_size`` when gated residual is active."""
    if config is None:
        return None
    text = getattr(config, "text_config", None)
    hc = None
    hidden = None
    for obj in (text, config):
        if obj is None:
            continue
        if hc is None:
            hc = getattr(obj, "hc_count", None)
            if hc is None and isinstance(obj, dict):
                hc = obj.get("hc_count")
        if hidden is None:
            hidden = getattr(obj, "hidden_size", None)
            if hidden is None and isinstance(obj, dict):
                hidden = obj.get("hidden_size")
    if isinstance(hc, int) and hc > 1 and isinstance(hidden, int) and hidden > 0:
        return hc * hidden
    return None


def uses_writeback_residual_capture(model: Any | None) -> bool:
    """True when decoder-layer outputs are wider than projection write-backs."""
    if model is None:
        return False
    config = getattr(model, "config", None)
    if is_qwen4_exp_config(config):
        return True
    stream = gated_residual_stream_width(config)
    hidden = writeback_hidden_size(config)
    return bool(stream and hidden and stream != hidden)


def iter_writeback_modules(layer: Module) -> Iterator[Linear]:
    """Yield Linear modules that write  the per-token residual (o/out/down)."""
    for path in _WRITEBACK_PATHS:
        obj: Any = layer
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, Linear):
            yield obj


def ngram_parameter_names(model: Module) -> list[str]:
    """Return dotted names of n-gram / PLE embedding parameters."""
    return [name for name, _ in model.named_parameters() if is_ngram_parameter_name(name)]


def iter_bnb_linear_targets(model: Module) -> list[tuple[str, Linear]]:
    """Linears bitsandbytes may convert — never the n-gram embedding table.

    The n-gram table is ``nn.Embedding``, so it would be skipped anyway; the
    filter is kept so a future Linear-shaped shard cannot be quantized in
    place (random-access gathers break under NF4).
    """
    out: list[tuple[str, Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, Linear):
            continue
        if is_ngram_parameter_name(name) or is_ngram_parameter_name(name + ".weight"):
            continue
        out.append((name, mod))
    return out


def move_parameters_skipping_ngram(
    model: Module,
    *,
    device: torch.device | str = "cuda",
) -> tuple[int, int]:
    """Move unique storages to *device*, leaving n-gram / PLE weights on host.

    Returns ``(moved, skipped)`` tensor counts.
    """
    moved = 0
    skipped = 0
    relocated: dict[int, Tensor] = {}
    named: list[tuple[str, Tensor]] = list(model.named_parameters()) + list(
        model.named_buffers()
    )
    with torch.no_grad():
        for name, tensor in named:
            if not isinstance(tensor, torch.Tensor):
                continue
            if is_ngram_parameter_name(name):
                skipped += 1
                continue
            if tensor.device.type == str(device).split(":", 1)[0] or (
                isinstance(device, torch.device) and tensor.device.type == device.type
            ):
                continue
            storage = tensor.untyped_storage()
            if storage.size() == 0:
                continue
            key = storage.data_ptr()
            if key in relocated:
                tensor.data = relocated[key]
                moved += 1
                continue
            gpu = tensor.data.to(device)
            relocated[key] = gpu
            tensor.data = gpu
            moved += 1
    return moved, skipped


def stack_residual_rows(
    hidden_states: Iterable[Tensor],
    token_offset: int,
    *,
    writeback_dim: int | None,
) -> Tensor:
    """Stack last-token rows, refusing a 10240/2560 mix.

    ``torch.stack`` on mixed widths either errors opaquely or — if a caller
    sliced wrong — would silently feed a 10240 vector into 2560 ``o_proj``.
    Fail closed whenever last-dims disagree or disagree with *writeback_dim*.
    """
    slices = [hs[:, token_offset, :] for hs in hidden_states]
    if not slices:
        raise RuntimeError("No hidden states to stack for residual capture.")
    dims = {int(row.shape[-1]) for row in slices}
    if len(dims) != 1:
        raise RuntimeError(
            "Cannot torch.stack residual rows with mixed last-dims "
            f"{sorted(dims)}; gated-residual models must use 2560-d write-back "
            "capture, not the 10240-d four-branch layer output."
        )
    dim = next(iter(dims))
    if writeback_dim is not None and dim != writeback_dim:
        raise RuntimeError(
            f"Residual last-dim {dim} is not write-back width {writeback_dim}; "
            "refusing to apply a 10240-d layer output to 2560-d projections."
        )
    return torch.stack(slices, dim=1)


def collect_writeback_hidden_states(
    layers: list[Module] | Any,
    forward: Callable[[], Any],
    *,
    writeback_dim: int,
) -> tuple[list[Tensor], Any]:
    """Run *forward* with hooks on 2560-d write-backs; return per-layer tensors.

    Each layer contributes its **last** firing write-back (MLP down after
    attn out), shape ``[batch, seq, writeback_dim]``.
    """
    layer_list = list(layers)
    buckets: dict[int, Tensor] = {}

    def _hook_for(index: int):
        def _hook(_module: Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(tensor, torch.Tensor):
                return
            if tensor.shape[-1] != writeback_dim:
                return
            buckets[index] = tensor

        return _hook

    handles = []
    empty: list[int] = []
    for index, layer in enumerate(layer_list):
        modules = list(iter_writeback_modules(layer))
        if not modules:
            empty.append(index)
            continue
        for module in modules:
            handles.append(module.register_forward_hook(_hook_for(index)))
    if empty:
        for handle in handles:
            handle.remove()
        raise RuntimeError(
            f"No {writeback_dim}-d write-back Linear on layers {empty}; "
            "refusing 10240-d decoder-layer residuals."
        )
    try:
        outputs = forward()
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index in range(len(layer_list)) if index not in buckets]
    if missing:
        raise RuntimeError(
            f"Write-back hooks did not fire on layers {missing} "
            f"(expected last-dim {writeback_dim})."
        )
    return [buckets[index] for index in range(len(layer_list))], outputs


def assemble_writeback_hidden_sequence(
    per_layer: list[Tensor],
    outputs: Any,
    *,
    writeback_dim: int,
) -> list[Tensor]:
    """Build a 2560-d hidden sequence from write-backs and the HC mixer.

    Real Qwen4Exp records decoder-layer outputs at ``hc_count * hidden_size``
    (10240). ``hidden_states[0]`` is therefore 10240 after the residual is
    repeated into four branches — it must **not** be used as an embedding
    row. The HC mixer (last recorded state, if it has ``writeback_dim``)
    occupies Heretic's unused index-0 slot so the stack is
    ``n_layers + 1`` when the mixer is present; per-layer write-backs follow.
    """
    hidden = list(getattr(outputs, "hidden_states", None) or [])
    mixer = None
    for hs in reversed(hidden):
        if (
            isinstance(hs, torch.Tensor)
            and hs.ndim >= 2
            and int(hs.shape[-1]) == writeback_dim
        ):
            mixer = hs
            break
    sequence: list[Tensor] = []
    if mixer is not None:
        sequence.append(mixer)
    sequence.extend(per_layer)
    if not sequence:
        raise RuntimeError(
            f"No {writeback_dim}-d write-back or mixer rows; "
            "refusing to stack 10240-d decoder-layer hidden_states."
        )
    bad = [int(t.shape[-1]) for t in sequence if int(t.shape[-1]) != writeback_dim]
    if bad:
        raise RuntimeError(
            f"Write-back sequence last-dims {bad} are not {writeback_dim}; "
            "refusing a 10240/2560 mix."
        )
    return sequence


def ngram_module_paths_from_index(
    model_id: str | os.PathLike[str] | None,
) -> list[str]:
    """Module paths of PLE n-gram shards from ``model.safetensors.index.json``."""
    if not model_id:
        return []
    index_path = Path(model_id) / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    weight_map = payload.get("weight_map") or {}
    modules: list[str] = []
    seen: set[str] = set()
    for key in weight_map:
        if not is_ngram_parameter_name(str(key)):
            continue
        name = str(key)
        if name.endswith(".weight"):
            name = name[: -len(".weight")]
        if name not in seen:
            seen.add(name)
            modules.append(name)
    return modules


def override_device_map_ngram_cpu(
    device_map: Any,
    ngram_modules: Iterable[str],
) -> Any:
    """Force n-gram / PLE modules onto CPU in an accelerate device_map dict."""
    names = [n for n in ngram_modules if n]
    if not isinstance(device_map, dict):
        return device_map
    out = dict(device_map)
    for key in list(out):
        if is_ngram_parameter_name(str(key)) or "ngram_embedding" in str(key):
            out[key] = "cpu"
    for name in names:
        out[name] = "cpu"
    return out


def pin_ngram_parameters_to_host(model: Module) -> int:
    """Move n-gram/PLE parameters to CPU if they landed on an accelerator.

    Returns the number of parameters moved. Already-host tensors are left
    as-is (mmap-backed embeddings stay file-backed).
    """
    moved = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_ngram_parameter_name(name):
                continue
            if param.device.type == "cpu":
                continue
            param.data = param.data.to("cpu")
            moved += 1
        for name, module in model.named_modules():
            if "ngram_embedding" not in name:
                continue
            hook = getattr(module, "_hf_hook", None)
            if hook is None:
                continue
            exec_dev = getattr(hook, "execution_device", None)
            if exec_dev not in (None, "cpu", torch.device("cpu")):
                hook.execution_device = "cpu"
    return moved


def _infer_auto_device_map(
    model_id: str,
    max_memory: dict[Any, Any] | None,
) -> dict[str, Any] | None:
    """Best-effort empty-weights ``infer_auto_device_map``; None if unavailable."""
    try:
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig, AutoModelForImageTextToText
    except Exception:
        return None
    try:
        config = AutoConfig.from_pretrained(model_id)
        with init_empty_weights():
            empty = AutoModelForImageTextToText.from_config(config)
        return infer_auto_device_map(
            empty,
            max_memory=max_memory,
            no_split_module_classes=list(
                getattr(empty, "_no_split_modules", None) or []
            ),
        )
    except Exception:
        return None


def device_map_keeping_ngram_on_host(
    model_id: str,
    device_map: Any,
    max_memory: dict[Any, Any] | None = None,
) -> Any:
    """Return a device_map that pins PLE n-gram modules to CPU.

    For ``device_map='auto'`` this tries ``infer_auto_device_map`` on a meta
    model so shards cannot each land on the accelerator (128×~800 MiB).
    """
    names = ngram_module_paths_from_index(model_id)
    if device_map == "auto" or device_map is None:
        inferred = _infer_auto_device_map(model_id, max_memory)
        if inferred is not None:
            return override_device_map_ngram_cpu(inferred, names)
        return device_map
    if isinstance(device_map, dict):
        return override_device_map_ngram_cpu(device_map, names)
    return device_map


def full_precision_merge_refusal(model_id: str | None, config: Any | None) -> str | None:
    """Return an error string if in-process BF16 merge must not run here."""
    if is_qwen4_exp_config(config):
        return (
            "Refusing in-process full-precision merge of Qwen4Exp / "
            "Qwen3.8-Flash-Next: reloading the ~336G BF16 checkpoint on a "
            "128G UMA host would OOM. Export the LoRA adapter with "
            "export_adapter() and merge on a machine that can hold the base."
        )
    mt = model_type_from_id(model_id) if model_id else None
    if mt in {"qwen4_exp", "qwen4_exp_text"}:
        return (
            "Refusing in-process full-precision merge of Qwen4Exp / "
            "Qwen3.8-Flash-Next (~336G BF16). Use export_adapter() instead."
        )
    return None


def estimate_flash_next_resident_gib(
    model_id: str | os.PathLike[str] | None = None,
    *,
    skip_ngram: bool = True,
    quantize_fused_experts: bool = False,
) -> float:
    """Rough resident GiB if this host materialised Flash-Next weights."""
    ngram = _NGRAM_BF16_GIB
    experts = _FUSED_EXPERT_BF16_GIB
    rest = max(0.0, _FLASH_NEXT_TOTAL_BF16_GIB - ngram - experts)
    if model_id:
        index_path = Path(model_id) / "model.safetensors.index.json"
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
                total = float((payload.get("metadata") or {}).get("total_size") or 0)
                if total > 0:
                    rest = max(
                        0.0,
                        total / 1024**3 - ngram - experts,
                    )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    if skip_ngram:
        ngram = 0.0
    if quantize_fused_experts:
        experts = experts * 0.5 / 2.0  # NF4 ~0.5 byte vs BF16 2 bytes
    return ngram + experts + rest


def preflight_qwen4exp_load(
    model_id: str,
    *,
    mem_available_bytes: int | None = None,
    allow_env: str = "ABLITERIX_QWEN4EXP_ALLOW_LOAD",
    skip_ngram: bool = True,
    quantize_fused_experts: bool = False,
) -> None:
    """Refuse a load that would clone PLE + fused BF16 experts into 128G UMA.

    Set ``ABLITERIX_QWEN4EXP_ALLOW_LOAD=1`` only on a host that can hold the
    resident estimate (or a properly quantized expert checkpoint).
    """
    mt = model_type_from_id(model_id)
    if mt not in {"qwen4_exp", "qwen4_exp_text"}:
        return
    if os.environ.get(allow_env, "") in {"1", "true", "yes"}:
        return
    need_gib = estimate_flash_next_resident_gib(
        model_id,
        skip_ngram=skip_ngram,
        quantize_fused_experts=quantize_fused_experts,
    )
    avail = mem_available_bytes
    if avail is None:
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
                        break
        except OSError:
            avail = None
    avail_gib = (avail / 1024**3) if avail else 0.0
    # Keep ~20G for the desktop / uma_guard min_free.
    if avail is None or need_gib + 20.0 > avail_gib:
        raise RuntimeError(
            "Refusing to load Qwen3.8-Flash-Next on this host: fused 3-D "
            f"experts remain BF16 (~{need_gib:.0f} GiB resident even with "
            "n-gram skipped; bitsandbytes does not convert 3-D Parameters) "
            f"and MemAvailable is {avail_gib:.0f} GiB. Use a GGUF / expert-"
            "quantized checkpoint, or set ABLITERIX_QWEN4EXP_ALLOW_LOAD=1 on "
            "a machine that can hold the weights. LoRA adapters stay "
            "mergeable via export_adapter()."
        )


def transformers_has_qwen4_exp() -> bool:
    """True when this process can import ``transformers.models.qwen4_exp``."""
    try:
        import transformers.models.qwen4_exp  # noqa: F401
    except Exception:
        return False
    return True
