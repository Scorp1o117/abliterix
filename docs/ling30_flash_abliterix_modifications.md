# Ling-3.0-flash × abliterix — 代码修改总结

> 日期:2026-08-07 · 分支:`sc117-base` · 环境:ROCm 7.14(gfx1151 Strix Halo)+ torch 2.12 + transformers 5.12 + bnb 0.49
> 模型:inclusionAI/Ling-3.0-flash(百川 BailingMoE v3,512 专家 × 42 层 + 1 MTP,255GB BF16)
> 目标:在 bnb4bit 量化下跑通 abliterix 完整流程(profiling → baseline → steering → trials)
>
> **全项目方法时间线 / Pareto / 工件地图** → 见 [`LING30_FLASH_PROJECT_LOG.md`](./LING30_FLASH_PROJECT_LOG.md)（2026-08-09）

---

## 1. 修改总览

| 文件 | 修改数 | 性质 |
|---|---|---|
| `src/abliterix/core/engine.py` | 6 处 | 兼容修复 + 性能 + 正确性 |
| `src/abliterix/safex.py` | 1 处 | 兼容修复(与 engine 同步) |
| `src/abliterix/optimizer.py` | 2 处 | 崩溃修复(多目标 TPE) |
| `src/abliterix/eval/stage_evaluator.py` | 1 处 | 崩溃修复(配合 optimizer) |
| `src/abliterix/eval/detector.py` | 2 处 | 性能 + 诊断 |
| `src/abliterix/eval/scorer.py` | 1 处 | 性能 |
| `modeling_bailing_moe_v3.py`(模型目录) | 2 处 | transformers 5.x 兼容 |
| `configs/ling30_flash_rocm_debug.toml` | 多处 | Ling 特化配置 |

---

## 2. engine.py

### 2.1 `is_torch_fx_available` shim(模块级)

**问题**:transformers 5.x 移除了 `transformers.utils.import_utils.is_torch_fx_available`,但 Ling 的自定义 modeling(基于 transformers 4.45 编写)在模块级 import 它 → 加载即崩溃。

**修改**:import 后检测缺失则补 `lambda: False`(它只用于守卫 attention-mask helper 的 `torch.fx.wrap`,语义安全)。

### 2.2 bnb 加载后 `flush_memory()`

**问题**:bnb 4-bit 逐张量量化路径在加载期间产生大量 fp16 临时量,缓存在 PyTorch caching allocator 里(实测 Ling:69% 加载进度时 VRAM 104GB,而最终 int4 仅 44GB)。

**修改**:`QuantMode.BNB_4BIT` 加载完成后调用 `flush_memory()` 回收碎片。

### 2.3 `_load_model_bnb_fast`(备用快速加载器,默认不启用)

**问题**:transformers 的 bnb 路径逐 Linear 量化,每个张量 ~0.6s 固定 mmap/H2D 开销,Ling 63k 张量 → **~12 小时**。

**修改**:新增 `_load_model_bnb_fast`——BF16 mmap 加载 → CPU 8 线程并行 `quantize_4bit`(~12ms/张量)→ `Params4bit.from_prequantized(device='cpu')` → 批量 `.to('cuda')`(纯搬运不重量化)。实测 ~17 分钟 vs 12 小时。按 200 权重一批 swap + 及时释放 BF16 页防 OOM。

**注意**:仅 `device_map="auto"` 时启用;最终方案用的是原生加载(fp16 路径 ~10 分钟,见配置),此方法保留备用。

### 2.4 LoRA-B 初始化后立即置零

**问题**:`get_peft_model` 包装后 `lora_B` 是**随机初始化**的。在第一次 `restore_baseline` 之前,baseline/prescreen 评估会带着随机 LoRA delta 生成 → 输出退化(复述 prompt、重复流)。

**修改**:收集 `_lora_b_weights` 后立即 `torch.nn.init.zeros_(w)` 全部置零,模型在 trial 应用 steering 前是基座权重的忠实副本。

### 2.5 profiling hook:router 输出按 dtype 探测整数张量

**问题**:MoE router 的返回值顺序因家族而异。Ling(Bailing v3)返回 `(topk_idx, topk_weight, logits)`——**索引在第 0 位**;旧代码对 3 元组固定取 `out[2]`(logits,float 有负值)→ bincount 崩溃(两轮:Float 不支持 → 负数非负要求)。

**修改**:遍历 tuple,取第一个 `int32/int64` 张量作为 expert-id 张量;找不到再回退 `out[0]`/`out[1]`。兼容所有 MoE 家族约定。

### 2.6 profiling hook:bincount 批量计数

**问题**:旧代码逐专家 `(flat == eid).sum()` 循环 + GPU 同步,512 专家 MoE 上 profiling 要 **2 小时以上**。

**修改**:一次 `torch.bincount(flat.long(), minlength=n_experts)` 批量计数,实测 **几分钟** 完成。

---

## 3. safex.py

**同步修复 2.5**:`identify_safety_experts_safex` 的 hook 用同样的 dtype 探测逻辑提取 expert-id 张量(之前也是硬编码 `out[2]`)。

---

## 4. optimizer.py + stage_evaluator.py — 多目标 TPE 崩溃修复

**问题**(严重):multi-objective(2 目标:refusals + KL)TPE 采样在续跑时崩溃:

```
ValueError: setting an array element with a sequence.
The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
```

根因:`raise TrialPruned()` 的 trial `values=None`,而 optuna 4.9 TPE 的 `_calculate_weights_below_for_multi_objective` 用 `np.asarray([t.values for t in below_trials])` → 形状不一致崩溃。单目标不受影响。

**注意**:第一版修复用 `trial.report((inf, inf))` 被拒——**optuna 多目标不支持 `Trial.report`**(`NotImplementedError`)。

**最终修复**:剪枝不再 `raise TrialPruned()`,改为返回 `(inf, inf)` 目标值(trial 以 COMPLETE+inf 结束):

- `stage_evaluator.py` `_run_prescreen` 高拒绝分支:`raise TrialPruned()` → `return 0, True`(标记)
- `optimizer.py` `_objective` 检测到 `prescreen_result[1]` + `prescreen_class=="high"` → `return (inf, inf)`
- `optimizer.py` KL 超标分支:`raise TrialPruned()` → `return (inf, inf)`

**原理**:inf 值被 TPE 的 `is_feasible` 过滤,不影响 Parzen 估计;但 trial 有完整 2 元 values,数组形状一致,study 可无限续跑。

**验证**:17-trial 搜索稳定跑完(剪枝 trial 全部 COMPLETE+inf,TPE 正常采样)。

---

## 5. detector.py + scorer.py — 性能与诊断

### 5.1 detector.py:full pass 上限 80 tokens

**问题**:不确定样本的 full 分类生成 `max_gen_tokens=160`,单 GPU 512 专家 MoE 上每个 ~3-5 分钟,prescreen 卡数小时。

**修改**:`max_new_tokens=min(max_gen_tokens, 80)`——拒绝信号集中在前 ~50 token,80 足够分类,成本减半。

### 5.2 detector.py:resp dump(诊断用)

**修改**:`print_responses` 开启时同时把 `(prompt, resp, is_ref)` 追加到 `/tmp/cli_resp_dump.jsonl`,用于对比终端显示 vs 真实生成(排查"显示层异常"疑云;结论:输出正常,异常是旧随机 LoRA 时代的遗留)。

### 5.3 scorer.py:coherence 生成上限 64 tokens

**问题**:同 5.1,coherence pass 只需响应长度(z-score vs baseline),生成 160 tokens 太慢。

**修改**:`max_new_tokens=min(max_gen_tokens, 64)`——仍长于 baseline 均值(~50 词),截断退化可检测。

---

## 6. modeling_bailing_moe_v3.py(模型目录,不在 repo)

### 6.1 DynamicCache 构造修复

**问题**:transformers 5.x 的 `DynamicCache()` 裸构造不读 config → `layers[layer_idx]` 越界(Ling 43 层,缓存 42 slot)→ `IndexError`。

**修改**:`DynamicCache(config=self.config)`(~1335 行)。

### 6.2 rope 兼容(transformers 5.x)

**修改**:`rope_scaling` 归一化(null → 空 dict)、`rope_type` 兼容 `"type"`、`head_dim` 从 `config.qk_rope_head_dim` 读取(~252-265 行)。**保留**。

---

## 7. 配置 `ling30_flash_rocm_debug.toml`(关键决策)

| 项 | 值 | 原因 |
|---|---|---|
| `dtype_fallback_order` | `["float16"]` | 原 bf16 路径 ~1.5 it/s(**12 小时**);fp16 ~120 it/s(**~10 分钟**),加载后自动 promote 281 个非量化参数到 bf16 |
| `batch_size` | **1** | batch>1 的 prefill 与单条差 ~1 ulp(内核 reduce 顺序),被 40 层 MoE router argmax 指数放大 → 退化输出(复述/"33.12.1.1..."重复)。等长 batch 也退化,非 padding 问题,数学不可行 |
| `max_gen_tokens` | 160 | **baseline 缓存 key 的一部分,不能改**(改小失效 60 分钟缓存) |
| `strength_range` | [0.5, 4.0] | 原 [2,10] 采样出 max_weight ~8.8 → KL 3.92 爆表;收窄到温和区间 |
| `[experts]` | 全 0 | 2026-08-07 发现专家 routing(router_bias 强抑制)<-5 与高拒绝率强相关(27-28/30 prescreen);禁用后同参数 28/30 → 13/30 |
| `print_responses` | false | 关调试打印(注意:继续模式会恢复 checkpoint 里的旧值,只覆盖 non_interactive/overwrite_checkpoint 两字段) |

---

## 8. 运行环境要点(踩坑)

- **必须** `unset PYTHONPATH` + 用 `/home/s117/heretic-env` 的 python(Hermes venv 泄漏会干扰)
- **必须** `PYTORCH_ALLOC_CONF=expandable_segments:True`(缺失 → 3% 处 OOM)
- **必须** `TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1`
- ulimit -n ≥ 2048(默认 1024 会 Errno 24 "Too many open files")
- 重启交互选择:**选 2(从头)** 才会用新命令行配置;选 1(继续)会用 checkpoint 旧配置覆盖(batch=8 复活事件)
- baseline/steering/profiling 三层缓存独立于 Optuna journal;journal 迁移(剔 PRUNED)用 `/tmp/migrate_study.py`
- 前台长 sleep 会阻塞 Hermes 后台进程实际启动(短轮询 sleep≤60s)

---

## 9. 实验结果与未解决问题

### 已验证

- ✅ fp16 加载 10 分钟 + 量化 63392/63392 Linear→Linear4bit + promote 281
- ✅ profiling(bincount + dtype 探测)几分钟完成,缓存命中
- ✅ 生成/steering 复刻全链路正常(engine 构造、真实 prompts、apply_steering 含 routing、profiling 后)
- ✅ 多目标 TPE 续跑不再崩(剪枝 → COMPLETE+inf)
- ✅ 纯 LoRA steering(禁 routing)能降低拒绝率:同参数 3.51 从 28/30(routing)→ 13/30(纯 LoRA);部分样本完全顺从(洗钱指南、煽动剧本、生物武器蓝图全给)

### 未达标(效果不佳,本次放弃)

- **KL 降不到 0.05 可用线**:纯 LoRA 下低拒绝 ↔ 低 KL 跷跷板——
  - 低拒区间(6-7/30):KL 0.98+(超标剪枝)
  - KL 达标区间(0.46, 强度 1.82):拒绝率 43/100
  - routing 版 KL 低(0.5065)但拒绝 27-30/30
- prescreen 高拒绝的 trial 依然多(21-30/30),TPE 未找到「拒绝 ≤10/100 且 KL ≤0.05」的组合
- 提示:KL 0.5 只是 prune 门槛,用户验收线为 **拒绝 ≤10/100、KL ≤0.05**

### 后续方向(未验证)

1. 更精细的 steering 变体(direct transform / SOM 等)替代纯 LoRA 权重放大
2. steering 向量质量(mean 方向 vs 更优方向提取)对 KL/拒绝平衡的影响
3. 分层/分模块强度(attn.o_proj + down_proj 联合优化)的精细搜索

---

## 10. v2 根因与修复（2026-08-07 接手）

### 根因（比「mean 方向差」更硬）

`BailingMoeV3DecoderLayer` **不是**标准 `self_attn`：

| 层类型 | 模块路径 | 输出投影字段 |
|--------|----------|--------------|
| MultiLatentAttention（约每 6 层 1 个） | `layer.attention` | **`dense`** |
| KimiDeltaAttention（其余层） | `layer.attention` | **`o_proj`** |

旧 `steerable_modules` 只认 `layer.self_attn.o_proj` → Ling 上 **零 o_proj 被注册**，搜索空间退化为单组件 `mlp.down_proj`。  
Laguna-S-2.1 同机成功配方同时用 **o_proj + down_proj + expert routing**，有限 Pareto 上多点满足 KL≤0.05 且拒绝≤10%。

### 代码修复

`src/abliterix/core/engine.py` `steerable_modules` 增加 Bailing 路径：

- `layer.attention.o_proj` / `layer.attention.dense` → `attn.o_proj`
- `layer.attention.{q,k,v}_proj` 与 MLA `q_b_proj` / `kv_b_proj`（可选；v2 配置默认 disabled）

### v2 运行

```bash
./run_ling30_v2.sh
# config: configs/ling30_flash_rocm_v2.toml
# checkpoint: checkpoints_ling30_flash_v2/
```

验收线不变：**KL ≤ 0.05 且拒绝 ≤ 10/100**。

---

## 11. v2 结果与 v3 换方向（2026-08-08）

### v2 结论（~25 trial，Laguna 式 o_proj+down+expert）

| 类别 | 观测 |
|------|------|
| 低拒绝 | prescreen **2–8/30** 多次出现 → 全部 **KL 剪枝 (inf)** |
| 有限点 | KL **0.40–0.47**，full 拒绝 **30–57%**（与 v1 底盘同级） |
| Expert 画像 | 层顶风险分仅 **~0.05–0.13**（分散，非 Laguna 式集中） |
| 含义 | 不是「缺 o_proj」，而是 **mean-diff 方向与能力纠缠**；加强度只换拒绝换 KL |

### v3 配方

```bash
./run_ling30_v3.sh
# configs/ling30_flash_rocm_v3_sra.toml
# checkpoints_ling30_flash_v3_sra/
```

- `vector_method = sra`（相对 benign concept atoms 清洗拒绝方向）
- expert routing **关**
- strength **[0.3, 2.5]** + auto-disable `mlp.down_proj`
- `search_harmfulness_direction = true`
- 需重算 steering（不复用 v1/v2 mean 向量缓存）
- **探针预算 `num_trials = 10`**（不够接近目标则换方向，不空烧）

### 若 v3 仍 KL 底盘 ≥0.2

下一候选（未开跑）：`optimal_transport` 方向（Granite OT quality 先例）、`cliff_head_ablation` 固定预消融 + 极轻 LoRA、仅 `attn.o_proj` 手工网格诊断。

---

## v4–v6 简报 → v7 cliff-head

| 阶段 | 结论 |
|------|------|
| v4 Ornith MPOA | 打开低 KL 盆地（~0.03–0.11）；拒仍 ~30–50% |
| v5 lowref 收窄 | 双目标未近；有限 dual-ish ~KL 0.06–0.10 / 拒 31–34% |
| v6 stack 80t | 堆 trial 未破 plateau；经典 mean+MPOA 搜索耗尽 |
| **v7 cliff** | **B 方案**：固定 cliff-head 预消融 + 轻 o_proj LoRA 探针 |

### v7 配方

```bash
./run_ling30_v7_interactive.sh           # 交互 TUI
./run_ling30_v7_interactive.sh --batch   # 无人值守 10 trial
./watch_ling30.sh v7
# configs/ling30_flash_rocm_v7_cliff.toml
# checkpoints_ling30_flash_v7_cliff/
```

- `cliff_head_ablation = true`，`top_k_frac = 0.04`，`strength = 0.75`
- 仅 `attn.o_proj` LoRA strength **[0.3, 1.2]**；`mlp.down_proj` disabled
- `weight_normalization = full` + rank 3（保留 Ling 低 KL 写路径）
- expert / SRA / harmfulness pair **关**
- bnb：`cliff_head` 对 `Params4bit` **dequant → 列缩放 → requant**（不可原地改 packed）
- **探针 `num_trials = 10`**；看 cliff 是否动拒绝、轻 LoRA 是否保住 KL≤0.05

### v7 结果（证伪）

- Cliff 已消融 54 heads / 27 layers，但 trial prescreen 仍 **28–30/30 high prune**，0 有限点。
- 结论：Ling 上 Bao cliff-head **无效**；轻 o_proj 也不能替代 o+down。

### v8 — 回到 Abliterix 默认方法

```bash
./run_ling30_v8_interactive.sh           # 或 --batch
./watch_ling30.sh v8
# configs/ling30_flash_rocm_v8_default.toml
```

- **`weight_normalization = "none"`**（非 Ornith/MPOA `full`）
- mean + LoRA + `orthogonal_projection=true`
- projected / winsorize / cliff / SRA / expert **关**
- `strength_range = [0.8, 1.5]`（settings 默认），无 v5 窄带
- o_proj + down_proj；batch16；40 trial

### v8 结果

- 10 轮：多数 high prune；有限点 KL~0.09–0.12 / 拒 44–54% — 默认强度带推不动拒。

### v9 — 最优搜索（已停 · 终局）

```bash
# 已停；结果见 LING30_FLASH_PROJECT_LOG.md
./watch_ling30.sh v9   # 只读 journal
```

- 17 trial（预算 60）后停：有限 14；**双目标 0**
- 最佳 dual-ish：**t12 KL 0.111 / 拒 14.3%**（o≈2.43 d≈1.75）
- 最低拒：**t10 13.3% @ KL 0.160**
- 相对 v6 t21（0.093/17%）：拒略好、KL 更差 → **Pareto 平移，非突破**
- 全局仍以 **v6 t21** 为最佳 dual-ish 交付候选
