# Qwen3.5 MoE ROCm / bnb 4-bit changes

This branch adds the local fixes and configuration used to run Abliterix on
Qwen3.5 MoE derivatives such as Agents-A1 and Qwen-AgentWorld-35B-A3B on
Windows ROCm / HIP with bitsandbytes 4-bit loading.

Branch:

```text
codex/qwen35moe-rocm-bnb4bit
```

## Summary

- Add Windows ROCm + bnb 4-bit example configs for Agents-A1 and Qwen-AgentWorld.
- Add `inference.min_batch_size` so auto batch-size probing can start above 1.
- Add `inference.skip_common_response_prefix` to skip the common prefix scan.
- Treat Qwen3.5 MoE models as text-only CausalLM models even when multimodal
  metadata is present.
- Improve model-load failure logging so dtype/quantization failures are visible.
- Make LLM judge parsing tolerate fenced JSON, extra surrounding text, and dict
  labels.
- Add an interactive option to save only the LoRA adapter instead of exporting a
  full merged model.

## Changed files

### `src/abliterix/settings.py`

Adds:

```toml
[inference]
min_batch_size = 1
skip_common_response_prefix = false
```

`min_batch_size` is the lower bound for automatic batch-size tuning. This is
useful when a model is known to handle larger batches and probing from 1 wastes
time.

`skip_common_response_prefix` skips the prefix-detection stage. This is useful
for local experimentation where that stage is not needed or is slow.

### `src/abliterix/cli.py`

- Auto batch-size probing now starts at `config.inference.min_batch_size`.
- The positional `model_id` inference no longer triggers when `--config` is
  present.
- Common response prefix detection is skipped when
  `config.inference.skip_common_response_prefix` is true.

### `src/abliterix/core/engine.py`

Qwen3.5 MoE models can expose multimodal-related config fields while still being
used as text-only language models in Abliterix. This branch makes
`model_type == "qwen3_5_moe"` resolve to `AutoModelForCausalLM`.

The load failure log now includes the exception type and repr:

```text
Failed (RuntimeError: RuntimeError(...))
```

This makes quantization / dtype / CUDA-HIP failures much easier to diagnose.

### `src/abliterix/eval/detector.py`

LLM judge outputs are often not perfectly clean JSON. The parser now handles:

- JSON wrapped in triple-backtick fences,
- valid JSON surrounded by extra explanatory text,
- labels returned as dict objects such as `{ "label": "R" }`.

This reduces false judge failures with local or OpenAI-compatible endpoints.

### `src/abliterix/interactive.py`

Adds an interactive action:

```text
Save LoRA adapter only
```

This calls `save_pretrained()` on the PEFT model and saves tokenizer files, but
does not merge/export full base weights. This is important for large 35B MoE
models where full in-process merge/export can exceed system RAM.

## Added configs

### `configs/agents_a1_rocm_bnb4bit.toml`

Example local config for Agents-A1 on Windows ROCm + bnb 4-bit.

Important fields:

```toml
[model]
quant_method = "bnb_4bit"
dtype_fallback_order = ["float16"]
device_map = "auto"
max_memory = {0 = "92GB", "cpu" = "24GB"}

[inference]
batch_size = 128
min_batch_size = 1
max_batch_size = 192
skip_common_response_prefix = true
```

### `configs/qwen_agentworld_35b_a3b_rocm_bnb4bit.toml`

Example local config for Qwen-AgentWorld-35B-A3B after unpacking fused MoE
experts into a bnb-compatible layout.

The local path in the sample points to:

```text
C:/Models/Qwen-AgentWorld-35B-A3B-unpacked-bnb
```

Adjust paths before use on another machine.

## Operational notes

For Qwen3.5 MoE derivatives, requesting bnb 4-bit is not enough. Verify that it
actually applies to expert modules:

- VRAM should drop significantly compared with BF16/FP16.
- Expert linear modules should become bitsandbytes `Linear4bit` modules.
- If fused expert tensors remain as raw parameters, bnb may appear enabled while
  most MoE weights are still effectively full precision.

When exporting final models, prefer saving the LoRA adapter first and merge
offline/streaming if system RAM is limited.

## Known local context

These changes were tested during local processing of:

- Agents-A1
- Qwen-AgentWorld-35B-A3B

The branch is intended as a reproducible local workflow branch, not necessarily
as an upstream-ready PR without cleanup.
