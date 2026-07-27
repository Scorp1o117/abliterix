# Laguna-S-2.1 Abliteration 项目交接文档

> 最后更新: 2026-07-25

---

## 1. 硬件环境

| 项目 | 规格 |
|------|------|
| CPU/APU | AMD AI MAX+ 395 Strix Halo |
| 统一内存 | 128GB (CPU + GPU 共享) |
| GPU 计算框架 | ROCm (HIP) |
| Swap | 已禁用 |
| 操作系统 | Linux (Ubuntu, GNOME) |
| 主机名 | S117-ROG-Flow-X |

**内存预算**: `max_memory = 120GB`，留 8GB 给系统。模型 4-bit 量化后约 60GB，剩余约 60GB 给激活/中间张量/LoRA 适配器。

---

## 2. 模型信息

| 项目 | 值 |
|------|-----|
| 模型 | poolside/Laguna-S-2.1 |
| 参数量 | 118B (MoE) |
| 架构 | 48 layers, 256 routed experts + 1 shared expert, top-10/token |
| hidden_size | 3072 |
| 上下文长度 | 1M |
| BF16 大小 | ~236GB (无法直接加载) |
| 量化方式 | bitsandbytes 4-bit (NF4) |
| 量化后大小 | ~60GB |

### 模型路径

| 用途 | 路径 |
|------|------|
| 原始模型 | `/run/media/s117/OS/Models/Laguna-S-2.1/` |
| Patch 后模型 (bnb 4-bit) | `/run/media/s117/OS/Models/Laguna-S-2.1-bnb/` |

### 模型 Patch (在模型目录中)

1. **modeling_laguna.py** — 将 fused 3D expert 权重 (`LagunaExperts` 的 `gate_up_proj`/`down_proj` 3D 参数) 拆分为 `nn.ModuleList` 的 `LagunaMLP`，使 bnb 4-bit 量化能逐 expert 应用。
2. **chat_template.jinja** — 第一行添加 `{%- set enable_thinking = false -%}` 禁用 thinking。

---

## 3. 项目路径

| 项目 | 路径 |
|------|------|
| abliterix 源码 | `/home/s117/heretic-abliterix/` |
| 配置文件 | `configs/laguna_s_2.1_rocm_bnb4bit.toml` |
| Python 环境 | `/home/s117/heretic-env/` |
| Checkpoint 目录 | `checkpoints_laguna_s_2.1_rocm_bnb4bit/` |
| 数据集目录 | `/home/s117/heretic-abliterix/datasets/` |
| 外部数据集源 | `/run/media/s117/OS/Users/15403/Documents/abliterix-datasets/` |

### 数据集

| 用途 | 数据集 | 数量 | 列名 |
|------|--------|------|------|
| Steering (benign) | `good_1000` | 1000 | `prompt` |
| Steering (harmful) | `harmful_1000` | 1000 | `prompt` |
| Eval (benign) | `good_500` | 100 ([:100]) | `prompt` |
| Eval (harmful) | `harmful_500` | 100 ([:100]) | `prompt` |

---

## 4. 代码修改清单

### 4.1 engine.py (`src/abliterix/core/engine.py`)

#### 4.1.1 bnb compute_dtype 固定 bf16
- **位置**: `_build_quant_config` 方法, ~line 810
- **内容**: `compute_dtype = torch.bfloat16` 硬编码，不再跟随加载 dtype
- **原因**: 如果 compute_dtype 跟随 dtype_fallback_order 的 fp16，bnb 量化不生效

#### 4.1.2 加载后 fp16→bf16 转换
- **位置**: 模型加载后, ~lines 433-448
- **内容**: 遍历所有非量化参数 (embed_tokens, norm, lm_head 等)，将 fp16 转为 bf16
- **原因**: fp16 的 hidden state norms 会在深层爆炸 (L0:35 → L47:inf → L48:NaN)；bf16 的指数范围更大，不会溢出

#### 4.1.3 Dequant cache 4GB 限制
- **位置**: ~lines 209-211
- **内容**: `_dequant_cache_max_bytes = 4 * 1024**3`，steering.py 中写入缓存前检查大小
- **原因**: 无限制时 dequant cache 存了全部 12288 个 expert 的 fp32 副本 (~154GB)，直接 OOM

#### 4.1.4 Per-expert LoRA 注册 — 已恢复
- **位置**: `steerable_modules` 方法, ~lines 947-1011
- **内容**: 4 处 per-expert LoRA 注册已取消注释：
  - `layer.mlp.experts` (Qwen3 / Laguna MoE)
  - `layer.block_sparse_moe.experts` (Phi-3.5-MoE)
  - `layer.moe.experts` (Granite MoE)
  - `layer.mixer.experts` (NemotronH)
- **原因**: 之前误判为 per-expert LoRA 导致模型不稳定而禁用，实际根因是 interactive.py 的参数缺失 bug。恢复后每个 expert 的 down_proj 可独立 steer，精度远高于只 steer shared_expert。
- **Laguna 模型匹配**: `layer.mlp.experts` → 256 experts/layer × 48 layers = 12288 per-expert modules + 48 shared_expert + 48 attn.o_proj = **~12384 LoRA modules**

### 4.2 scorer.py (`src/abliterix/eval/scorer.py`)

#### 4.2.1 Baseline 缓存
- **内容**: 添加 `_baseline_cache_key()`, `_baseline_cache_path()`, `_save_baseline_cache()`, `_load_baseline_cache()` 方法
- **缓存路径**: `<checkpoint_dir>/<slug>_baseline.pt`
- **缓存键**: model_id + eval 数据集配置 + max_gen_tokens + kl_token_count + seed
- **效果**: 重启时跳过 baseline 计算 (~5-10 min)

#### 4.2.2 Baseline 张量移至 CPU
- **内容**: `_capture_baseline` 末尾将 `baseline_logprobs`、`baseline_single_token_lp`、`baseline_continuation_nll` 移至 CPU
- **原因**: 这些张量 (~240MB) 常驻 GPU 但只在 KL 计算时短暂需要；代码已有 `.to(device)` 处理设备转移
- **额外**: 末尾调用 `flush_memory()` 清理中间张量

### 4.3 cli.py (`src/abliterix/cli.py`)

#### 4.3.1 Prefix 缓存
- **内容**: 检查 `<checkpoint_dir>/<slug>_prefix.json`，命中则跳过 `_detect_response_prefix`，否则计算后保存

#### 4.3.2 Steering 数据缓存
- **内容**: 检查 `<checkpoint_dir>/<slug>_steering.pt`，命中则跳过 residual extraction + steering vector 计算 + expert profiling
- **缓存键**: model_id + prompt 配置 + steering 参数 + seed
- **缓存大小**: ~1.2GB (主要是 residual states)

#### 4.3.3 传入 benign_states/target_states 到交互菜单
- **内容**: `show_interactive_results()` 调用新增 `benign_states=benign_states, target_states=target_states`
- **原因**: 交互菜单需要这些数据来正确执行 discriminative layer selection

### 4.4 stage_evaluator.py (`src/abliterix/eval/stage_evaluator.py`)

- **Top-1 disagreement 阈值**: `low_margin_thresh` 从 0.5 改为 1.0 nat
- **主指标**: 改为 `high_margin_rate` (只统计高置信度 top-1 翻转)
- **新增属性**: `validation_top1_total_rate`
- **移除**: thinking leak 检测代码 (thinking 已关闭，无关)

### 4.5 screening.py (`src/abliterix/eval/screening.py`)

- **新增**: `digit_ratio` 健康检查指标
- **计算**: `sum(1 for ch in resp if ch.isdigit()) / len(resp)`
- **失败阈值**: `digit_ratio > 0.15` (超过 15% 数字 = 退化输出)

### 4.6 optimizer.py (`src/abliterix/optimizer.py`)

- **新增**: trial 间调用 `flush_memory()` (在 `finally` 块末尾)
- **原因**: PyTorch GPU 内存分配器缓存碎片不还给系统，多 trial 后累积 OOM
- **移除**: thinking leak 检测代码

### 4.7 interactive.py (`src/abliterix/interactive.py`) — **关键修复**

#### 4.7.1 apply_steering 参数补全
- **Bug**: 交互菜单的 `apply_steering` 调用缺少 `config`、`safety_experts`、`routing_config`、`benign_states`、`target_states`
- **后果**:
  - 无 `benign_states`/`target_states` → discriminative layer selection 退化为全部 48 层 → 非 discriminative 层被 steer → 模型输出乱码
  - 无 `safety_experts`/`routing_config` → MoE expert steering (router suppression + EGA) 不执行
- **修复**: 从 `trial.user_attrs["moe_parameters"]` 重建 `ExpertRoutingConfig`，传入所有参数
- **影响范围**: trial 选择后的 steering 应用、`_run_benchmarks` 中的 re-apply、interactive 内的 `run_search` 调用

#### 4.7.2 show_interactive_results 签名更新
- 新增参数: `benign_states=None, target_states=None`

### 4.8 配置文件 (`configs/laguna_s_2.1_rocm_bnb4bit.toml`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `seed` | 117 | 固定随机种子 |
| `system_prompt` | "You are a helpful assistant." | 顶层设置 |
| `dtype_fallback_order` | ["float16"] | 加载用 fp16 保证 bnb 量化生效 |
| `max_memory` | {0 = "120GB"} | 留 8GB 给系统 |
| `batch_size` | 16 | 当前值 (8→16，80GB 占用有余量) |
| `discriminative_layer_selection` | true | 只 steer discriminative 层 |
| `disabled_components` | q_proj, k_proj, v_proj | 减少搜索空间 |
| `refusal_prescreen_enabled` | true | 30 prompts 预筛 |
| `validation_kl_enabled` | true | 验证集 KL |
| `generation_health_enabled` | true | 健康检查 |
| `thinking_leak_detection_enabled` | true | 代码已移除，配置无害 |
| `num_trials` | 50 | Optuna 搜索轮数 |
| `num_warmup_trials` | 10 | TPE 预热 |

---

## 5. Bug 修复时间线

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | 加载阶段 OOM | `dtype_fallback_order=["bfloat16"]` 导致 bnb 量化不生效 | 保持 `["float16"]` 加载，硬编码 `compute_dtype=bf16` |
| 2 | Hidden state NaN | 非量化参数 fp16 导致深层 norm 爆炸 | 加载后 fp16→bf16 转换 |
| 3 | Steering 阶段 OOM | dequant cache 存全部 expert fp32 (~154GB) | 4GB 缓存上限 |
| 4 | 系统 OOM kill | max_memory=120GB 只留 8GB 给系统 | 降 batch_size；用户关 swap |
| 5 | Top-1 disagreement 无区分度 | 低 margin 翻转噪声 (threshold=0.5) | threshold→1.0，主指标改 high_margin_rate |
| 6 | **Chat 输出乱码** | **interactive.py 缺参数 → 全层 steer** | **补全 config/safety_experts/routing/benign_states/target_states** |
| 7 | 多 trial 后渐进 OOM | GPU 内存碎片未释放 | trial 间 `flush_memory()` |
| 8 | Per-expert LoRA 误禁用 | 误判 #6 为 per-expert LoRA 导致 | 恢复 per-expert LoRA |

---

## 6. 当前状态 (2026-07-25)

### 运行中
- Per-expert LoRA 已恢复 (~12384 modules)
- batch_size = 16
- 优化正在运行，内存约 80-100GB
- interactive.py 修复已生效 (chat 输出正常)

### 已验证
- ✅ Baseline 模型正常 (59.6±16.1 words, 无 NaN)
- ✅ Shared-expert-only 模式下 chat 输出正常 (Trial 6: 10/100 refusals, KL=0.0055)
- ✅ 模型自认为 GPT 是原始行为，非 abliteration 副作用

### 待验证
- ⏳ Per-expert LoRA 模式的优化结果 (refusals / KL / chat 质量)
- ⏳ batch_size=16 是否稳定 (无 OOM)
- ⏳ 最终模型导出和 benchmark

---

## 7. 启动命令

```bash
cd /home/s117/heretic-abliterix
/home/s117/heretic-env/bin/abliterix --config configs/laguna_s_2.1_rocm_bnb4bit.toml
```

### 缓存文件 (checkpoint 目录)

| 文件 | 大小 | 可否删除 | 说明 |
|------|------|----------|------|
| `*_baseline.pt` | ~60MB | ✅ 可保留 | 基线 logprobs，模型不变则有效 |
| `*_prefix.json` | ~72B | ✅ 可保留 | 响应前缀 |
| `*_steering.pt` | ~1.2GB | ⚠️ 改 LoRA 结构后需删 | 残差状态 + steering 向量 + expert profiling |
| `*.jsonl` | 变长 | ⚠️ 改 LoRA 结构后需删 | Optuna trial 记录 |

---

## 8. 关键设计决策

1. **fp16 加载 + bf16 计算**: bnb 4-bit 量化要求加载时 dtype 与量化配置匹配 (fp16)，但计算用 bf16 避免 fp16 溢出。两者解耦。

2. **Per-expert LoRA + discriminative layer selection**: 注册全部 expert 的 LoRA，但只 steer discriminative 层的 expert。非 discriminative 层被跳过，避免 coherence damage。

3. **Dequant cache 4GB 限制**: 防止 dequant 全部 expert 权重导致 OOM。超出限制后不再缓存，每次重新 dequant (慢但安全)。

4. **Baseline 张量在 CPU**: 节省 ~240MB GPU 内存。KL 计算时按需 `.to(device)`。

5. **flush_memory 策略**: 每个 trial 结束后 + baseline 计算后调用 `gc.collect()` + `torch.cuda.empty_cache()`，防止碎片累积。

6. **Thinking 禁用**: chat_template 硬编码 `enable_thinking=false`，代码中也传 `enable_thinking=False`。thinking leak 检测已移除。

---

## 9. 注意事项

- **不要随意改配置**: 用户偏好手动控制配置变更
- **不要输出 think 标签**: 会被终端截断
- **seed 固定 117**: 保证可复现性
- **模型自认为 GPT**: 训练数据污染，非 bug
- **Swap 已关**: 用户主动关闭，不要建议开启
- **max_memory=120GB**: 用户确认的值，不要擅自修改
