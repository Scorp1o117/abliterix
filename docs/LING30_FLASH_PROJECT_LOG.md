# Ling-3.0-flash Abliterix 项目记录

> **分支**: `sc117-base`（fork: Scorp1o117/abliterix）  
> **仓库**: `/home/s117/heretic-abliterix`  
> **文档日期**: 2026-08-13（机制探针封顶，已冻结最佳点）  
> **状态**: **冲刺停止**。质量目标已命中、拒答目标未命中。v60 仅 10→9/60，且 7/10 是后段拒答。交付改用现有最佳点  
> **相关**: `BRANCH_LOG.md`、`docs/ling30_flash_abliterix_modifications.md`、`LAGUNA_PROJECT_HANDOVER.md`

---

## 0. 一句话摘要

在 **ROCm + bnb4bit** 下把 inclusionAI **Ling-3.0-flash**（Bailing MoE v3）跑通 abliterix 全流程，并搜索  
**KL ≤ 0.05 且拒答 ≤ 10/100** 的 LoRA 消融点。

已确认：

1. 必须修 **Bailing steerable 发现**（否则只有 `down_proj`，KL≈0.45 跷跷板）。  
2. Ling 上 **唯一稳定低 KL 盆地** 来自 **MPOA 写路径**（`weight_normalization=full` + rank 3）。  
3. 静态可合并路线最佳 dual-ish 仍是 v6 t21：KL 0.093 / 拒 17.3%；单纯抬强度只能把拒答推到 13–15%，同时 KL 抬至 0.11–0.16。  
4. 新增 model-generated trajectory classifier + sample-global runtime gate 后，v29 达到 **KL 0.0622 / 拒答 20–29%**；v31 为 **KL 0.0602 / 拒答 21%**，均显著优于早期 gate。  
5. Ling 的 bnb4bit MoE 对 batch composition 敏感；v34 的 length+rendered-content canonical batching 把同一 60 prompts 的 forward/reverse 漂移从 v33 的 **11→18** 消为 **10→10**。  
6. v36 用同一 canonical batching 重建 baseline 后，benign gate 0% 时 primary KL 从伪差异 0.0566 降为 **0.0000**；独立 validation KL **0.0461**、健康检查通过，但 harmful 仍 **27/100**，因此当前瓶颈已明确收缩到 harmful steering 效力，而不是 benign 保真。

---

## 1. 目标与环境

| 项 | 值 |
|----|-----|
| 模型 | `/run/media/s117/OS/Models/Ling-3.0-flash`（inclusionAI Ling-3.0-flash） |
| 架构 | Bailing MoE v3 hybrid：`bailing_hybrid`，42 层 + 1 MTP，**512 experts**，top-8 |
| Attention | 混合：**MLA**（`layer.attention.dense`）与 **KDA**（`layer.attention.o_proj`），`layer_group_size=6` |
| 形状 | `hidden=2560`，`num_heads=32`，`v_head_dim=128`，o/dense in = 4096 |
| 硬件 | Strix Halo gfx1151，统一内存；`max_memory 0=104GB` |
| 软件 | ROCm 7.x，torch 2.12，transformers 5.x，bnb 0.49，`heretic-env` |
| 量化 | `quant_method = bnb_4bit`，`dtype_fallback_order = ["float16"]`（再 promote 非量化参数） |
| 评估 | keyword 拒答检测（`llm_judge=false`）；staged prescreen + validation KL + health |
| 验收 | **KL ≤ 0.05 且 refusals ≤ 10/100** |
| 数据 | `abliterix-datasets/good_1000` + `harmful_1000`（本地路径） |

### 运行必做环境

```bash
unset PYTHONPATH
source /home/s117/heretic-env/bin/activate
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
ulimit -n 65536
```

- 交互续跑会恢复 checkpoint 内旧配置字段 → 改 batch/配置时应用 **`--overwrite-checkpoint` 新 study** 或确认 journal 与 config 一致。  
- 管道 `tee` 会破坏 TUI → 交互用 `run_ling30_v*_interactive.sh`；无人值守用 `--batch`。

---

## 2. 模型架构要点（为何与 Llama 配方不同）

```
BailingMoeV3DecoderLayer
├── attention  (不是 self_attn)
│   ├── MultiLatentAttention  → dense  (MLA，约层 5,11,17,23,29,35,41)
│   └── KimiDeltaAttention    → o_proj (其余层)
└── mlp
    └── SparseMoeBlock / MLP  → experts[*].down_proj + shared
```

| 若只注册 `self_attn.o_proj` | 实际结果 |
|----------------------------|----------|
| Ling 上 **0 个 o_proj** | 只能 steer `mlp.down_proj` |
| Laguna 同机 | o+down+expert → KL 0.002–0.01 / 拒 7–9% |
| Ling v1 未修时 | KL≈0.46 / 拒≈43% 有限点；低拒全被 KL 剪枝 |

---

## 3. 代码修改清单

### 3.1 为 Ling 硬兼容（可上游化 / 已在 sc117-base）

| 文件 | 改动 | 原因 |
|------|------|------|
| `src/abliterix/core/engine.py` | `is_torch_fx_available` shim → `lambda: False` | transformers 5.x 移除 API，Ling modeling 模块级 import |
| `engine.py` | bnb 加载后 `flush_memory()` | 加载期 VRAM 尖峰碎片 |
| `engine.py` | `_load_model_bnb_fast`（备用） | 63k 张量原生路径极慢；现多用 fp16 原生 ~10 min |
| `engine.py` | LoRA-B **立即置零** | 随机 B 污染 baseline / 首评 |
| `engine.py` | profiling：router 输出 **dtype 探测** expert id | Bailing 返回 `(topk_idx, topk_weight, logits)`，不能写死 `out[2]` |
| `engine.py` | profiling：`bincount` 批量计数 | 512 专家循环从小时级 → 分钟级 |
| `engine.py` | **`steerable_modules` Bailing 路径** | `attention.o_proj` / `dense` → `attn.o_proj`；可选 q/k/v、q_b/kv_b |
| `src/abliterix/safex.py` | 与 engine 同步 router dtype 探测 | safety expert 画像 |
| `src/abliterix/optimizer.py` | 多目标剪枝改返回 `(inf,inf)` | Optuna 4.x multi-obj TPE 遇 `TrialPruned` values=None 崩溃 |
| `src/abliterix/eval/stage_evaluator.py` | prescreen high 标记 + 配合 optimizer | 同上 |
| `optimizer.py` | KL 超 prune 时写 `user_attr.kl_divergence` | 事后分析 |
| `src/abliterix/core/steering.py` | routing 全关时不建 `ExpertRoutingConfig`；仅真实 routing 时禁 multi-dir | MoE + harmfulness_pair 假冲突崩溃 |
| `src/abliterix/eval/detector.py` | full pass `max_new_tokens≤80` | prescreen 加速 |
| `src/abliterix/eval/scorer.py` | coherence `≤64` tokens | 加速 |
| `src/abliterix/cliff_head.py` | **bnb 安全路径** | identify dequant；apply dequant→列缩放→requant；优先 `v_head_dim` |
| 模型目录 `modeling_bailing_moe_v3.py` | DynamicCache(config=…)；rope 兼容 | **不在 repo**，在模型盘上 |

### 3.2 通用 SC117 / 上游 PR 相关（Ling 间接受益）

| 主题 | 位置 | 说明 |
|------|------|------|
| bnb compute_dtype 固定 bf16、fp16→bf16 promote | `engine.py` | ROCm MoE 稳定 |
| dequant cache 上限 4 GiB | `engine.py` | 大 MoE 防爆内存 |
| trial 末 `flush_memory` | `optimizer.py` | 长搜降碎片 |
| staged eval（prescreen / val KL / health） | `eval/*`，默认关 | Ling 配置里打开 |
| 磁盘缓存 baseline/steering | engine/cli | 重启加速 |
| 交互 Save LoRA-only / 非 TTY 菜单 | `interactive.py` | 生产可用性 |
| PR #95 `pr/bnb-rocm-moe-stability` | 上游 | ROCm+bnb MoE 稳定性 |

### 3.3 Bailing steerable 关键片段

`src/abliterix/core/engine.py` → `steerable_modules`：

- `layer.attention.o_proj` → `attn.o_proj`（KDA）  
- `layer.attention.dense` → `attn.o_proj`（MLA）  
- 可选：`q/k/v_proj`、`q_b_proj`、`kv_b_proj`（配方里默认 disabled）

### 3.4 cliff_head bnb（v7）

- 问题：`Params4bit` 不能对 packed uint8 做 `weight.data[:, lo:hi] *= …`  
- 解：float 打分用 dequant；消融用 dequant → 缩放 → `quantize_4bit` 写回；restore 保留原 Parameter  
- 测试：`tests/test_cliff_head.py` 全过  

---

## 4. 实验时间线与方法矩阵

### 4.1 总览表

| 版本 | 方法核心 | 关键旋钮 | 有限点 | 代表结果 | 结论 |
|------|----------|----------|--------|----------|------|
| smoke/debug (v1) | mean + LoRA，`weight_norm=none` | 仅 down 有效（o 未发现） | 少 | KL≈0.46 / 拒≈43% | 流程通；效果跷跷板 |
| **v2** | Laguna 风：o+down+**expert**，gaussian，`none` | strength 宽、routing 开 | ~5 | 最佳 KL≈0.40 / 拒 58% | **证伪** expert/Laguna 套 Ling |
| **v3** | **SRA** + pre-norm | 10 trial 探针 | ~4 | KL≈0.45 底盘 | **证伪** SRA 开盆地 |
| **v4 Ornith** | **MPOA full+r3**，mean，o+down | strength [0.5,2.1] | 3+ | **KL 0.034** / 拒 58% | **打开低 KL** |
| **v4 b16** | 同 MPOA，batch16 复验 | — | 4 | KL 0.09 / 拒 31–41% | batch16 可用 |
| **v5 lowref** | MPOA + 窄带 + per-layer | o[1.5,2.2] d[1.4,2.1] | 3 | 仍高拒 | 窄带未破 10% 拒 |
| **v6 stack** | MPOA 宽搜 80t | o/d [0.8,2.5] | **18** | **t21 KL0.093/拒17%** | **历史最佳 dual-ish**；天花板 2.5 顶死 |
| **v7 cliff** | cliff 预消融 + 轻 o only | top4% ×0.75 | **0** | prescreen 28–30/30 | **证伪** cliff on Ling |
| **v8 default** | 真默认 `weight_norm=none` [0.8,1.5] | auto-disable down | 2 | KL0.09/拒54% | 默认推不动拒 |
| **v9 optimal** | MPOA + 赢家盆地 **上探** | o[2.0,2.9] d[1.5,2.7] per-layer | **14**（17 开，停） | 最佳 dual-ish t12 **KL0.111/拒14.3%**；最低拒 t10 **13.3%@KL0.160** | **证伪「抬天花板即可双目标」**；跷跷板平移 |
| **v10 OT** | 当前 OT single-vector + MPOA | o[1.5,3.2] d[1.2,3.0] per-layer | **27**（33 开，停） | t19 **KL0.108/拒16.3%**；t13 **KL0.129/拒12.2%** | 未破前沿；有效层 OT 与 mean 几乎同向 |
| **v11 paired** | **同 prompt system-condition mean** + MPOA | o[1.8,2.8] d[1.3,2.3] per-layer | **0**（12/12 high） | 全部 prescreen **30/30** | **证伪 prompt-only system condition 可控拒答** |
| **v12 response-pair** | **同 prompt assistant trajectory mean** + MPOA | o[1.5,4.0] d[1.2,3.5] per-layer | **4**（12 完成） | 最低拒 t7 **47%@KL0.245**；最低 KL t8 **0.174@63%** | 有控制信号但 Pareto 明显劣于 v6/v9 |

### 4.2 逐版细节

#### v1 / debug / smoke — 跑通链路

- 配置：`configs/ling30_flash_rocm.toml`、`*_debug.toml`、`*_smoke.toml`  
- 早期误判：batch>1 会退化（ulp→router）；后在本机确认 **batch 8/16 可用**（v4 b16）  
- 强度 [2,10] 过猛 → 收至 [0.5,4]；expert routing 与 **更高拒答** 相关 → 禁用  

#### v2 — Laguna 移植（证伪）

- `configs/ling30_flash_rocm_v2.toml` + `run_ling30_v2.sh`  
- **同时**修 o_proj 发现  
- 有限点仍 KL **0.40–0.47**；低拒 trial **KL 剪枝**  
- Safety expert 风险分极弱（top ~0.05–0.13）→ 无 Laguna 式集中 safety expert  

#### v3 — SRA（证伪）

- `vector_method=sra`，`weight_normalization=pre`，关 expert  
- KL 底盘 **≥0.45**，未出现 v4 式 0.03 盆地  

#### v4 — Ornith / MPOA（关键转折）

- `weight_normalization=full`，`full_norm_lora_rank=3`  
- 仅 o_proj + down_proj；mean；无 SRA/pair/expert  
- **首次**稳定进入 **KL 0.03–0.11** 区  
- 代价：拒答仍多（常见 30–58%）  

#### v5 — 低拒收窄

- `fixed_vector_scope=per layer`（有限点全是 per-layer）  
- 禁 auto-disable down；强度收在生产簇  
- 未到 10% 拒  

#### v6 — 大预算 stack（Pareto 最完整）

- 80 trial，o/d [0.8,2.5]  
- **最佳 dual-ish：t21 KL=0.0933，拒=17.3%**，o≈2.40，d≈1.94，中后层 ~35  
- 低拒簇大量顶在 **max_weight=2.5** → 需要更高强度上限  

#### v7 — cliff-head（证伪）

- 实际消融 **54 heads / 27 layers**（bnb requant 路径工作）  
- 全部 **prescreen high prune**，0 有限点  
- Ling 非 paper 式 reasoning-cliff 模型；稀疏 head 消融不够  

#### v8 — Abliterix 默认方法

- `weight_norm=none`，strength [0.8,1.5]  
- 有限点 KL 尚可 (~0.09) 但拒 **44%+**；多数 prune  
- 说明：**默认写路径 ≠ Ling 最优写路径**  

#### v9 — 数据驱动最优搜索（已停 · 终局）

- **停机**: 2026-08-09；PID 32158 kill；预算 60，实际开 **17 trial**（完成有限 14 + high prune 2 + 中断 1）  
- 配方：MPOA full+r3；per-layer；禁 auto-disable down；o **[2.0, 2.9]**，d **[1.5, 2.7]**；`min_weight_frac_max=0.50`；batch 16  
- 配置 / ckpt：`configs/ling30_flash_rocm_v9_optimal.toml` / `checkpoints_ling30_flash_v9_optimal/`  
- 日志：`logs/ling30_v9_20260809_111530.log`

**v9 有限点全表（按 dual-score 近→远）**

| t | prescreen | KL | 拒% | o_mw | d_mw | 备注 |
|--:|----------:|---:|----:|-----:|-----:|------|
| 12 | 6 | **0.111** | **14.3** | 2.43 | 1.75 | **v9 最佳 dual-ish** |
| 13 | 4 | 0.117 | 15.3 | 2.42 | 1.56 | |
| 8 | 8 | 0.106 | 21.4 | 2.13 | 2.00 | |
| 15 | 6 | 0.120 | 18.4 | 2.51 | 1.71 | |
| 1 | 8 | 0.110 | 24.5 | 2.61 | 2.39 | |
| 10 | 7 | 0.160 | **13.3** | 2.66 | 2.67 | **全项目最低拒（有限）** |
| 3 | 6 | 0.130 | 25.5 | 2.71 | 2.59 | |
| 7 | 10 | 0.130 | 27.6 | 2.68 | 2.44 | |
| 9 | 11 | 0.099 | 34.7 | 2.04 | 1.53 | |
| 2 | 10 | 0.205 | 20.4 | 2.53 | 2.15 | |
| 5 | 14 | 0.125 | 48.0 | 2.62 | 2.34 | |
| 11 | 14 | 0.131 | 48.0 | 2.13 | 1.89 | |
| 0 | 16 | 0.098 | 54.1 | 2.27 | 2.62 | |
| 6 | 19 | **0.086** | 64.3 | 2.55 | 2.36 | v9 最低 KL，拒极差 |
| 4 | 30 | inf | — | 2.03 | 1.89 | high prune |
| 14 | 21 | inf | — | 2.10 | 1.98 | high prune |
| 16 | 5 | — | — | 2.32 | 1.64 | 中断未完成 |

**v9 结论**

1. **抬 o/d 天花板有效降拒**：相对 v6 的 ~17% 底，v9 摸到 **13–15%**（t10/t12/t13）。  
2. **代价明确**：这些点 KL **0.11–0.16**，**没有**回到 v6 t21 的 KL 0.093 同时更低拒。  
3. **双目标 0 命中**；`near`（KL≤0.12 ∧ 拒≤20%）仅 t12/t13/t15 三点。  
4. 高强度（o,d 都 ~2.6+）→ 拒最低但 KL 最差（t10）；中等 o~2.4 + 偏低 d~1.6–1.8 → dual-ish 更好（t12/t13）——与 v6 赢家结构一致，**只是前沿平移而非突破**。  
5. 再堆同配方剩余 40 trial **期望收益低** → **停搜归档**。

#### v10 — Optimal Transport 方向（已停 · 未形成新方向）

- **启动**: 2026-08-09  
- 配方：**vector_method = optimal_transport**（PCA-Gaussian OT，捕获均值+协方差结构）  
- 保留 MPOA full+r3 写路径（Ling 唯一低 KL 关键）  
- per-layer 向量；禁 auto-disable down  
- 强度带 **宽于 v9**：o **[1.5, 3.2]**，d **[1.2, 3.0]**（OT 方向可能需要不同幅度）  
- `ot_components=2`；`projected_abliteration=true`（保留有用性）  
- 预算 40，实际 **32 COMPLETE + 1 PRUNED**；按接管诊断停止剩余 trial  
- 配置 / ckpt：`configs/ling30_flash_rocm_v10_ot.toml` / `checkpoints_ling30_flash_v10_ot/`  
- 目标：**KL ≤ 0.05 且 拒 ≤ 10/100**

**设计 rationale**:
- mean-diff 方向在 Ling 上已证明无法同时满足 KL≤0.05 与 拒≤10%
- OT 考虑分布的协方差结构差异，可能找到更"干净"的消融方向
- 有 Granite OT quality 先例（见 `vectors.py` `_compute_ot_transform`）

**2026-08-10 接管复盘**:

- 最佳 dual-ish：t19 **KL 0.108 / 拒 16.3%**；最低拒：t13 **KL 0.129 / 拒 12.2%**；0 双命中。
- 对 v10 OT cache 与 v9 mean cache 做逐层 cosine：搜索有效层 16–37 的 cosine **均值 0.999569**，夹角中位数 **1.15°**、均值 **1.47°**、最大 **4.49°**。
- 根因：`_compute_ot_transform()` 虽构造低维 covariance map，最终以 `diff + covariance_correction.mean(dim=0)` 压回单向量；中后层仍由 mean difference 主导。
- 结论应限定为：**当前 OT-to-single-vector 实现未产生新干预方向**，不能外推为真正的 affine OT 已被证伪。

#### v11 — 同 prompt 条件配对方向（接管后首个探针）

- 配置 / runner：`configs/ling30_flash_rocm_v11_paired.toml` / `run_ling30_v11_interactive.sh`
- direction benign/target 使用完全相同的 harmful `train[:800]`；只改变 system condition：compliance vs refusal。
- 目标是去除旧 harmful-vs-benign mean 中的主题/语义混杂，提取更接近 refusal-mode 的条件差。
- 保留已验证 MPOA `full+r3`、o+down、per-layer；仅做 **12 trial**。
- 继续门槛：拒≤17% 时 KL<0.08，或 KL≤0.10 时拒<13%；否则停止该方向。
- 限制：当前只在 prompt residual 上配对，尚不是 response-trajectory refusal/compliance 配对；结果若正向，再实现 continuation-level 配对。
- 接管期间发现 routing 参数全 0 仍触发两路 800-prompt expert profiling；已在 `cli.py` 增加 effective-search guard，后续禁 routing 配方直接跳过。
- **启动验证（2026-08-10）**：fresh cache 721,707,546 bytes，真实包含两路 `[800,44,2560] float32` residual；有效层 16–37 相对 v9 mean 的 cosine 均值 **0.212353**（中位 0.196345），夹角均值 **77.73°**（范围 74.75°–81.22°）。因此 v11 是实质新方向，不是 v10 式近同向复搜。

**v11 结果（2026-08-10）**:

- 12/12 trial 均为 `prescreen_refusals=30`, `prescreen_class=high`, objective `(inf, inf)`；有限点 **0**。
- 强度覆盖 o=1.83–2.59、d=1.62–2.29，层峰覆盖约 25–40，结果完全一致，已足以停止而不是扩 trial。
- 结论：system prompt 条件确实产生了几何新方向，但它主要编码 instruction/mode 差异，未提供可写入权重的 refusal-control direction。下一步若继续配对，必须升级到 **同 prompt 的 refusal/compliance response trajectory**，不能只比较 prompt residual。
- 运行结束时发现 `--batch` 分支只省略 `--no-non-interactive`、未显式加入 `--non-interactive`，导致搜索后进入 TUI；已修复 v10/v11 runner 并干净退出本次会话。

#### v12 — 同 prompt assistant response trajectory（接管第二阶段）

- 配置 / runner：`configs/ling30_flash_rocm_v12_response_pair.toml` / `run_ling30_v12_interactive.sh`
- 新增通用 `response_pair_enabled` 提取路径：prompt 与 system 完全相同，在 assistant generation boundary 后分别 teacher-force compliance/refusal continuation，并对 continuation tokens 的逐层 residual 做 mean pooling。
- 方向仍为 `mean(target_refusal - benign_compliance)`，但标签首次来自 **assistant response trajectory**，不再来自 harmful/benign 主题或 system prompt 差异。
- 配置校验强制两路 dataset/split/column/prefix/suffix/system 一致；cache key 纳入 continuation 文本、pooling 和开关。
- 12-trial 探针将强度扩至 o `[1.5,4.0]`、d `[1.2,3.5]`，先回答“是否存在任何拒答控制”，再谈 Pareto 精调。
- **启动状态（2026-08-10）**：首次启动发现 `/dev/nvme0n1p3` 未挂载，模型/数据路径均不可见；已用 `udisksctl` 恢复到 `/run/media/s117/OS`，0-byte journal 无 trial 污染。重启后模型成功加载（bnb4bit 约 63.4 GB VRAM），baseline 98/100 命中，现已进入 compliance continuation residual 提取。
- **运行中快照**：cache 721,349,402 bytes，两路 `[800,44,2560]` residual 完整；有效层方向相对 v9 mean 平均夹角 **74.04°**、相对 v11 **81.12°**。t0/t1/t2 prescreen 分别 **28/30、22/30、20/30**；t3（o=3.47,d=3.29）首次到 **15/30 borderline** 并进入完整 KL/拒答评估，证明 response trajectory 方向开始实际控制拒答，但尚未达到 prescreen pass ≤12。

**v12 终局（正常完成 12/12，非异常停止）**:

| trial | prescreen | KL | 拒答 | 结论 |
|------:|----------:|---:|-----:|------|
| 3 | 15/30 | 0.205 | 50/98（51.0%） | 首个有限点 |
| 7 | **14/30** | 0.245 | **47/98（48.0%）** | v12 最低拒 |
| 8 | 19/30 | **0.174** | 63/98（64.3%） | v12 最低 KL |
| 11 | 16/30 | 0.237 | 53/98（54.1%） | |

- 其余 8 点均 high prune（20–30/30）；没有 prescreen low，也没有接近旧 v6/v9 Pareto。
- v12 证明 assistant trajectory 差异中存在一定 refusal-control 信号（不再恒定 30/30），但当前固定 continuation mean 仍高度风格化/局部化；需要强写入才有影响，随之 KL 0.17–0.25 且拒答仍 48–64%。
- **决策：停止 v12，不追加 trial。** 下一步不再搜索这一静态方向；若继续，应转向真实模型生成的 refusal/compliance completion 配对或运行时 gated intervention，而非固定短 continuation。

#### v13 — adaptive runtime 因果探针（2026-08-10 启动）

- 配置 / runner：`configs/ling30_flash_rocm_v13_adaptive_runtime.toml` / `run_ling30_v13_interactive.sh`
- 这不是静态 LoRA 搜索：在 decoder-block 输出注册运行时 hook，只对与 refusal direction **正向对齐**的 token activation 做有界 angular removal。
- 首轮定位 post-attention + post-MoE 合并后的 residual；搜索变量仅为 intervention 强度、层峰和带宽，禁用重复 component profile 维度。
- 复用并校验 v9 mean-direction steering cache 与 baseline cache，保证结果可与 v6/v9 前沿直接比较。
- 预算 18 trial；继续门槛：prescreen ≤12/30 且 validation KL≤0.08。若仍无信号，则停止整层 residual 路线，下一步把 hook 下沉到 KDA/MLA/MoE/router site。
- 注意：该模式为 runtime-only，成功也不能直接 merge；其用途是判定“方向存在但静态写入失败”还是“整层 residual 方向本身不足”。
- **启动验证**：配置解析、runner `bash -n`、angular/runtime hook 定向测试 4/4 通过；较宽测试集合 34/42 通过，另 8 个失败均为既有 `tests/test_steering_lora.py` 对新版 PEFT `Linear(..., config=...)` 构造签名不兼容，与 runtime hook 路径无关。
- v9 steering cache 为旧版无 schema 短哈希，当前 provenance 校验正确判 stale；未强行绕过，重新提取 800 benign + 800 target 并写出 **688MB schema-v2 cache**。baseline cache 正常复用，基座拒答 98/100。
- **首点快照**：t0，peak L28.39、max removal 0.37、min 0.11、带宽 5.40，prescreen **20/30 high**。相比基座接近全拒已有轻微信号，但仍在 prune 边界，尚不能称为突破；搜索继续。
- **v13 终局（按预设门槛提前停止）**：t1 max=0.14 时 30/30；t2 max=0.9698、peak L38.02、min_frac=0.3145、带宽=18.56 时 prescreen **9/30 low**，完整结果 **KL 0.1240 / 拒答 16/98（16.3%）**，validation KL 0.1609，generation health 通过、无 thinking leak。
- 解释：首次证明 runtime conditional residual intervention 可强控拒答，但 t2 仍未破 v6（KL 0.093 / 拒17.3%）前沿，且 validation KL 未达继续门槛 0.08；预计剩余运行 3h38m，故在 t3 启动后安全中断，不机械跑满 18 点。

#### v14 — KDA / MLA / MLP / shared-expert site sweep（2026-08-10）

- 新增 runtime hook sites：`kda_output`、`mla_output`、`mlp_output`、`shared_expert_output`（保留历史 `decoder_block` / 通用 `attention_output`）。KDA/MLA 依据 Ling layer 的 `attention_layer_type` 精确分流。
- 配置 / runner：`configs/ling30_flash_rocm_v14_site_sweep.toml` / `run_ling30_v14_site_sweep.sh`。
- 4 个 seed trial 在四个 site **原样重放 v13 t2 profile**，固定同一 30-prompt prescreen；全部在 prescreen 后终止，只比较因果位置，不再支付每个 full eval 约 40 分钟的成本。
- runtime-only 模式现在跳过无用 PEFT adapter 初始化；Ling 首次加载可省约 3 分钟，并避免包装 20,683 个模块。
- **v14 终局**：KDA output **30/30**、MLA output **30/30**、总 MLP output **27/30**、shared-expert output **30/30**；对照 v13 decoder-block output **9/30**。说明当前 post-block mean direction 不能直接在单一支路输出复现控制，只有 MLP 支路有极弱信号。
- 限定解释：支路输出与 post-residual direction 可能不在同一激活分布，不能据此声称 KDA/MLA 没有安全信息。下一对照应在 `residual + attention_output` 已完成、进入 post-attention norm 前施加同方向。

#### v15 — post-attention residual matched probe（2026-08-10）

- 配置 / runner：`configs/ling30_flash_rocm_v15_post_attn.toml` / `run_ling30_v15_post_attn.sh`。
- 精确复现 v13 t2 profile，仅将 adaptive hook 放在 `residual + attention_output` 后、`post_attention_layernorm` 前；固定 30-prompt prescreen 后终止。
- 判读：若接近 decoder-block 的 9/30，控制信号已在 attention residual 累积态形成；若仍接近 30/30，则有效坐标只在 MoE residual 合并后的 block output 成立。
- **v15 终局**：post-attention residual **27/30**。attention residual 与总 MLP 支路均只有弱信号（同为 27/30）；强控制仅在 post-MoE decoder-block output 出现。
- 决策：定位阶段结束，后续固定 `decoder_block` site；新增 `concept_gated_angular`，用 per-layer harmful-vs-benign ConceptScorer 再叠加 positive-alignment gate，目标是在保留 v13 拒答控制的同时让 benign token 不触发干预。
- **v16 已建立并开始验证**：固定 v13 t2 的 decoder-block profile，使用 800 benign + 800 harmful 最终 prompt-token residual 训练逐层 MLP gate；首轮阈值 0.5，只跑同一 30-prompt prescreen，并记录 classifier gate 的 layer-token 激活率。配置 / runner：`configs/ling30_flash_rocm_v16_concept_gate.toml` / `run_ling30_v16_concept_gate.sh`。
- **v16 t0（threshold=0.5）**：41/42 层 scorer 有效；gate 触发 **14.49%** layer-token；prescreen **17/30**。相较无 classifier gate 的 v13 t2（9/30），门控过严、损失 8 个 prompt 的拒答控制，但仍优于未干预基线 30/30。决策：新增 threshold 搜索并做 0.05 / 0.15 / 0.30 matched sweep；找到 ≤10–12/30 的最低触发候选后再跑 full KL。
- **v16 threshold sweep 终局**：0.05 → **41.65% / 10/30**；0.15 → **27.18% / 14/30**；0.30 → **18.76% / 15/30**；连同 0.50 → 14.49% / 17/30，形成清晰的稀疏度—控制力交换曲线。只有 threshold=0.05 达到预设 ≤12/30，且比无 classifier gate 的 v13 t2（9/30）仅差 1 个，因此停止阈值盲扫，转 v17 full eval。
- **v17 已建立**：固定 threshold=0.05 与 v13 t2 profile，恢复 prescreen pass/prune 门槛，启用 primary KL、30-prompt validation KL、完整 harmful eval、generation health 与 thinking leak；新增 benign-eval gate-rate telemetry，直接验证 harmful 41.65% 触发是否对应更低 benign 触发。配置 / runner：`configs/ling30_flash_rocm_v17_concept_gate_full.toml` / `run_ling30_v17_concept_gate_full.sh`。
- **v17 终局**：threshold=0.05 精确复现 prescreen **10/30**，但完整 raw refusal 为 **42/100**（objective 按 baseline-refusal denominator 为 42/98=42.9%）；primary KL **0.0771**，validation KL **0.1000** / p95 0.4012 / max 0.5356，high-margin top1 flip 0/30；generation health passed、thinking leak none。harmful prescreen gate 41.65%，benign eval gate 20.79%，证明 classifier 有选择性并将 v13 KL 0.124 降约 38%，但 30-prompt 子集严重高估 full control，因此不是综合赢家。
- **v18 方向**：根因是 scorer 用 final prompt-token residual 训练，却在推理中逐 token 分类。新增 `concept_gate_scope="prompt"`：prefill 最后 token 做 per-sample/per-layer 判定并锁存到整个 autoregressive decode，使训练与使用坐标一致；后续直接用全部 100 harmful 做探针，不再用 30 题放行。
- **v18 已建立**：threshold=0.5、prompt-latched gate、exact v13 t2 profile；prescreen size 直接设为完整 100 harmful，诊断后终止。配置 / runner：`configs/ling30_flash_rocm_v18_prompt_gate.toml` / `run_ling30_v18_prompt_gate.sh`。
- **v18 终局**：prompt-latched threshold=0.5 仅触发 **19.78%** layer-token，完整 prescreen **62/100**，训练—推理粒度对齐不足以在该阈值恢复控制。
- **v19 最终 gate 判别点**：复用 v18 resolved recipe，仅以 CLI 固定 threshold=0.05 和独立 checkpoint，直接 full-100。若仍不能接近 v13 的低拒答同时保留 gate 稀疏性，则停止 ConceptScorer gate 分支。runner：`run_ling30_v19_prompt_gate_low.sh`。
- **v19 终局 / gate 分支停止**：prompt-latched threshold=0.05 触发 **24.47%** layer-token，完整拒答 **55/100**。相较 v18 threshold=0.5 的 19.78% / 62，仅增加 4.69pp 覆盖、减少 7 个拒答；远未接近 v13 的 16/100 raw 左右。继续把 threshold 压向 0 只会退化为 scorer gate 几乎全开、回到 v13 的 KL/控制点，无法保留条件选择性。因此按预设规则停止 ConceptScorer gate，不再调阈值。
- **分支结论**：final-prompt scorer 确实区分 harmful/benign（v17 harmful 41.65% vs benign 20.79%，KL 0.124→0.077），但其 gate 覆盖与完整 refusal control 不匹配；token gate 有 30-prompt 选择偏差，prompt latch 又受逐层状态分布漂移/覆盖不足影响。下一条仍有新信息增益的路线必须改训练数据：直接用 teacher-forced refusal/compliance **continuation-token trajectory states** 训练 token-level gate（而不是继续使用 prompt-final states 或再调静态 mean 方向）。
- **v20 trajectory gate 已实现（2026-08-11）**：steering direction 明确保留 v13 的 prompt mean，不启用已在 v12 失败的 response-pair direction；新增 continuation-token residual 采集，每个 harmful prompt 在固定 compliance/refusal continuation 上均匀保留最多 4 个 token。gate 训练集/验证集按 prompt 分组做 80/20 确定性拆分，避免同一 prompt 的相关 token 泄漏到两侧。
- **v20 离线 guard**：只保留 held-out accuracy >60% 的逐层 scorer；整体 held-out 平均 accuracy 必须 ≥70%，且 refusal-active 减 compliance-active 必须 ≥25pp，否则在任何生成前终止。通过后才用 exact v13 t2 profile 做完整 100 harmful probe。配置 / runner：`configs/ling30_flash_rocm_v20_trajectory_gate.toml` / `run_ling30_v20_trajectory_gate.sh`；聚焦回归 **25 passed**，配置解析成功。
- **v20 结果**：42/42 层通过；prompt-grouped held-out **accuracy 99.49%**、compliance active **0.00%**、refusal active **98.98%**、score margin **0.9350**。自由生成时 gate active **21.27%**，完整拒答 **45/100**。它未恢复 v13 的约 16/100，但相较 v17 final-prompt token gate 的 41.65% active / 42 refusals，用约一半覆盖得到接近的控制，表明 trajectory gate 的选择效率更高、但 teacher-forced→free-running 仍有覆盖缺口。
- **v21 终局阈值点已建立**：完全复用 v20 训练法、方向和 exact v13 t2 profile，只把 runtime threshold 从 0.50 降到 0.05，直接 full-100。若拒答明显下降且 activation 仍低于 v17，则保留 trajectory gate；若仍约 40% 拒答，则停止固定 continuation classifier 路线。runner：`run_ling30_v21_trajectory_gate_low.sh`。
- **v21 结果 / 固定 continuation classifier 停止**：threshold=0.05 的 held-out 指标为 accuracy **93.51%**、compliance active **12.98%**、refusal active **100%**，score margin 与 v20 相同为 **0.9350**；自由生成 gate active **49.34%**、拒答 **39/100**。相较 v20 多 28.07pp 干预只减少 6 个拒答；相较 v17 也只是用更高覆盖换 3 个拒答，仍远离 v13 的约 16/100。继续降 threshold 只会逐步退化成近乎 always-on v13，无法形成新的 KL/拒答解，故停止此分支。
- **下一条有信息增益的 gate 定义**：不再分类固定的 refusal/compliance 文本（它主要学到了 response lexical trajectory，且触发时机偏晚）；改为从模型自己的自由生成中采样 **benign vs harmful 的 final-prefill + early-decode states**，用 prompt 类别监督，让 gate 在拒答决策形成前覆盖 harmful 轨迹，同时在 benign 轨迹保持关闭。必须继续使用 prompt-grouped held-out 和 full-100 验证，不能再以固定 teacher-forced separation 代替自由生成迁移。
- **v22 generated-prompt gate 已实现（2026-08-11）**：每类确定性抽样 200 prompts，按 prompt 分组 160/40；未 steering Ling 为每条生成 12 tokens，训练数据组合 final-prefill state×4 与最早 8 个实际 decode-token states。新增逐行 continuation residual 提取和生成文本 provenance cache；混合轨迹与仅 final-prefill 两套 held-out 指标都必须满足 accuracy≥70%、harmful-minus-benign active gap≥25pp，才允许 exact v13 t2 做 full-100。配置 / runner：`configs/ling30_flash_rocm_v22_generated_gate.toml` / `run_ling30_v22_generated_gate.sh`；聚焦回归 **27 passed**。
- **v22 结果**：混合 held-out accuracy **94.65%**（benign active 4.77%、harmful 94.07%、margin 0.7981）；final-prefill 单独 accuracy **91.67%**（benign 9.11%、harmful 92.44%、margin 0.7224）。但 token-scope 自由生成仅 active **8.72%**、拒答 **63/100**。解释不是输入分类失败，而是首 token steering 后 activation 离开未干预训练分布，使下一步 classifier 自行关闭，即 intervention-induced distribution shift。
- **v23 matched latch 已建立**：完全复用 v22 classifier 数据、threshold、direction 与 exact t2，只把 runtime scope 改为 prompt；final-prefill 决策一旦做出便贯穿 decode，直接检验“gate 自熄”是否为 v22 的主因。复用 v22 生成文本 cache，full-100。runner：`run_ling30_v23_generated_prompt_latch.sh`。
- **v23 结果 / 新诊断**：prompt latch active **16.41%**，拒答反而 **77/100**。与训练内 final-prefill harmful active 92.44% 明显矛盾，说明主要问题进一步定位为跨数据切片泛化：train[:800] 内随机 40-prompt held-out 虚高，但真正 eval[900:] 只覆盖少量层/样本。gate 自熄存在（v22），却不是唯一或主导瓶颈。
- **外部 guard + v24 已建立**：generated-prompt 路径在任何 trial 前额外提取真实 `benign_eval` / `target_eval` final-prefill residual，报告 external accuracy/active gap，并要求同样 ≥70% / ≥25pp；v24 将训练覆盖从每类 200 提到 400、prompt latch，仅在混合/internal-prefill/external-prefill 三套 guard 全过时跑 full-100。runner：`run_ling30_v24_generated_latch_400.sh`。
- **v24 结果**：400/class 后 mixed **95.14%**、internal prefill **91.77%**、external prefill **91.52%**（external benign active 7.90%、harmful 90.95%、margin 0.7115），证明 classifier 确实跨切片泛化；full-100 active **20.45%**、拒答 **58/100**。因此 v23 的“数据切片泛化失败”解释被外部 residual 证据否定。更强解释是逐层 classifier 虽平均准确，却关闭约 9% 层，其中包含 v13 控制所需的关键层，破坏跨层协同。
- **v25 global prompt gate 已实现**：新增 `concept_gate_scope="global_prompt"`，由固定第 0 decoder layer 的 final-prefill scorer 做一次样本级决定，并立即广播给后续所有 steering 层、锁存整个 decode；另设 layer-0 external 专项 guard，避免用 eval 数据挑层。复用 v24 400/class response cache；聚焦回归 **28 passed**。runner：`run_ling30_v25_global_prompt_gate.sh`。
- **v25 guard 结果**：layer 0 external 对 benign/harmful 都 **100% active**，accuracy **50%**、margin **0.0004**；被 guard 在生成前正确截停。最浅层只有普遍激活而无类别边界，不能承担全局决策。
- **v26 causal auto-layer 已建立**：`concept_gate_global_decision_layer=-1` 时，仅用 internal grouped final-prefill hold-out 从浅到深选第一个满足 accuracy≥70% / gap≥25pp 的层；选择后再对 external eval 做独立 guard，不能根据 external 指标挑层。通过后该层广播到后续全部层并锁存 decode。runner：`run_ling30_v26_global_prompt_auto.sh`。
- **v26 发现并隔离实现 bug**：内部选择 layer 4；external accuracy **75%**、benign active 4%、harmful 54%，guard 通过。但 runtime active **0%**、拒答 **98/100**。原因是 layer 4 位于 exact v13 profile 之外，旧安装循环在计算 gate 前就按 profile 距离跳过了它，导致全局 latch 从未赋值；此结果不用于方法评价。
- **v27 observer 修复**：global decision layer 即使在 steering profile 外也安装 angle=0 observer hook，只计算/广播 gate 而不修改该层；后续 profile 内层正常 steering。新增决策层在 profile 外的回归测试，聚焦套件仍 **28 passed**。runner：`run_ling30_v27_global_prompt_observer.sh`。
- **v27 结果**：observer 修复后 runtime active **54.96%**、拒答 **54/100**，与 layer-4 external harmful recall **54%** 精确吻合，证明全局广播按样本生效；被识别样本基本恢复控制，未识别样本构成拒答下限。瓶颈因此收敛为决策层 recall，而非 hook 或跨层广播。
- **v28 max-gap 终局层选择**：自动模式 `-2` 仅在 layer 0..20（不晚于 v13 profile 起点）中，用 internal grouped hold-out 选择 harmful-minus-benign active gap 最大、margin 次优、层号再浅的层；之后 external guard，不能按 eval 选层。若 recall 仍不足则停止 generated classifier gate。runner：`run_ling30_v28_global_prompt_maxgap.sh`。
- **v28 结果**：内部 max-gap 选中 **layer 19**；该层独立 external final-prefill **accuracy 98.50%**、benign active **1.00%**、harmful active **98.00%**、margin **0.9633**。实际 full-100 harmful 中 global gate active **97.00%**，拒答 **29/100**。这同时验证了分类器、选层和全局广播，但仍比 always-on v13 的 16/100 多 13 个拒答；剩余问题已不是有害召回不足，而是 gate 开启后的 steering 本身不能在相同提示集稳定复现 v13 raw 结果。
- **v29 已建立**：固定复现 v28 的 internal max-gap layer 选择与 external guard，把 30-prompt prescreen 设为只记录、始终放行，恢复 primary KL、完整 harmful eval、validation KL、generation health 与 thinking-leak 检测。目标是判断 v28 的 29/100 拒答是否换来显著低于 v13 KL 0.1240 的良性保真；不再调整 classifier threshold。runner：`run_ling30_v29_global_prompt_full.sh`。
- **v29 终局 / 新 Pareto 点**：离线指标和 layer 19 选择逐项复现 v28；同一 30-prompt prescreen 为 **9/30**、harmful gate active **96.67%**。primary KL **0.0622**、benign gate active **1.09%**、长度偏差 0.57σ；独立 full harmful 为 **20/100 raw refusals**（objective 20/98），validation KL **0.0962** / p95 0.5237 / max 0.6208 / high-margin top1 flip 0/30，generation health passed。它严格支配 v17（KL 0.0771 / 42 refusals），并以相对 v13 多 4 个 raw refusal，把 primary KL 从 0.1240 近乎减半，故保留为当前最佳 conditional recipe。v28 的 29 与 v29 的 20 使用相同提示/greedy decode，但前者将 seeded 全集随机排列后分 batch，后者按原始顺序分 batch；Ling 4-bit MoE 对 batch composition 存在 9-point 漂移，后续报告应保守写 **20–29/100**，不可只报单点 20。
- **thinking-leak 诊断修正**：v29 被旧 screener 标记 detected，但旧逻辑把普通 `thinking` / `thought` / `思考` 的任意正文出现也算泄漏；未 steering baseline continuation 本身就命中普通用法（如 “without thinking”、 “thought patterns”），证明该布尔量会假阳性，不能据此判定 v29 泄漏。检测已改为只接受显式 `<think>` 标签或行首 `Thinking:` / `Thought:` / `思考：` reasoning header；普通正文不再命中，并增加回归测试。v29 已生成文本未持久化，因此不把历史 detected 追溯改写为 none，只记为 **旧检测器结果无效，实际状态未确认**。
- **v30 已建立**：利用 layer-19 score margin 0.9633，把 global threshold 从 0.5 提到 **0.8**，目标是消除 v29 约 1% benign 误触而保持 harmful recall；在同一 30-prompt subset 上匹配比较 v29 原强度与 max angular **1.10 / 1.20**，只为达到约 ≤3–4/30 的点支付后续 full KL。该轮同时检验“高置信 gate 隔离良性后能否安全提高 harmful steering”，不是再次扫描低阈值覆盖率。配置 / runner：`configs/ling30_flash_rocm_v30_global_strength.toml` / `run_ling30_v30_global_strength.sh`。
- **v30 结果**：threshold 0.8 的 internal max-gap 改选 **layer 18**；external accuracy **98.00%**、benign active **0.00%**、harmful active **96.00%**、margin 0.9309，证实可以完全关闭外部良性误触。matched 30-prompt 结果非单调：exact max=0.9698 → **9/30**，max=1.10 → **12/30**，max=1.20 → **6/30**；三者 harmful gate active 都约 96.6%。1.20 是唯一改善点，进入 full eval；不围绕 1.10 盲目插值。
- **v31 已建立**：固定 v30 threshold 0.8 / auto layer 18 / max=1.20 winner，恢复 primary KL、full harmful、validation KL、generation health 与修正后的 thinking-leak detector。若 full refusal 随 matched subset 下降且 KL 因 benign 0-active 进入 ≤0.05，则保留为新的交付 recipe。runner：`run_ling30_v31_global_strong_full.sh`。
- **v31 终局**：6/30 matched prescreen 精确复现，primary KL **0.0602**、benign runtime gate **0.97%**、长度偏差 0.57σ；full harmful **21/100 raw refusals**，validation KL **0.0765** / p95 0.2747 / max 0.5518 / high-margin top1 flip 0/30，generation health passed、修正后 thinking leak none。它相对 v29（0.0622 / 20）只交换了 -0.0020 KL 与 +1 refusal，不构成严格支配；30-prompt 的 9→6 改善未迁移到 full-100，故停止单轴抬 max angular。v29/v31 均保留，前者偏控制、后者略偏质量。
- **v32 已建立**：在 threshold 0.8 / internal max-gap global gate 下，联合搜索 `attn.o_proj` max **0.95–1.50**、peak layer、min fraction 与 falloff；固定 prescreen 扩到 **60 prompts**，先重放 v29/v31 两个 profile，再做 6 个 seeded warmup。所有点只做 refusal probe，避免在形状尚未证明时支付 KL；继续 full-100 的门槛约为 ≤6–8/60。runner：`run_ling30_v32_global_profile_search.sh`。
- **v32 终局 / profile 搜索停止**：固定 60-prompt 同排列下，t0 原 profile **20**；t1 v31 宽带 max=1.20 **10（最佳）**；t2 早峰窄带 15；t3 后峰极窄带 42；t4 宽带 max=1.48 为 17；t5 宽带低底强度 23；t6 早峰极窄带 28；t7 宽带 max=1.48 变体 12。结果确认有效结构必须是后峰、宽带、并保持显著跨层底强度，但没有新点达到预设 ≤6–8/60，且赢家就是已做 full eval 的 v31 profile，故不再重复完整 KL。相同原 profile 在这组 seeded random batch order 为 20/60、在 original-order full eval 为 20/100，再次证明绝对 refusal rate 受 batch composition 严重影响；v32 只用于 matched 相对比较。
- **v33 batch-invariance 已实现**：`global_prompt` 可选单样本 final-prefill prepass，逐条得到 canonical gate，再将固定 gate tensor 注入正式 generation/forced-KL batch；正式 decision hook 不再重算分类，prepass 不污染 gate telemetry。新增 reverse-order prescreen replay，在完全相同 prompt set / batch size 下改变分组并记录 refusal delta。v33 固定 v31 profile 和 60 prompts，比较 seeded 与 reverse order，以分离 gate 漂移和模型批量生成漂移。runner：`run_ling30_v33_batch_invariance.sh`；相关回归 **25 passed**。
- **v33 结果**：single-sample canonical gate 后 seeded order **11/60**、reverse order **18/60（delta +7）**，gate telemetry 96.63%。固定 gate 没有消除 batch 漂移，证明主因在正式 batched decode，而非 classifier 边界。单样本 prepass 还带来明显固定成本，后续只与 prompt-hash cache 或 canonical batching 组合使用，不作为每个 trial 重算的默认路径。
- **v34 canonical batching 已实现**：长度排序增加 rendered-content tie-breaker，使同一 prompt set 无论输入顺序如何都形成相同 batch 成员；`global_prompt` 可选自动 canonical batch，并在 text generation、generate-and-score 与 forced-continuation KL 三条 HF 路径恢复 caller order。v34 将重放 v33 的 forward/reverse 60 条；若 delta=0，即可把它作为规范评估协议而无需退到 batch=1。
- **v34 结果 / batch-invariance 修复成功**：同一 60 prompts 在 forward caller order 与 reverse caller order 均为 **10/60，delta 0**；gate telemetry 96.48%。对照 v33 固定 gate 仍为 11→18（delta +7），说明 token-length + rendered-content canonical batch membership 是消除排列漂移的必要步骤；无需退到 batch=1。后续 Ling runtime 指标必须启用 canonical batching，旧 v28–v32 的绝对 refusal 仅保留为非规范历史对照。
- **v35 已建立**：固定 v31 max=1.20 profile、threshold 0.8、single-sample canonical gate 与 canonical batch16，60-prompt prescreen 只记录并始终放行，恢复 primary KL、full-100 harmful、validation KL、generation health 与 thinking-leak。目标是建立首个 input-order-invariant 的交付指标。runner：`run_ling30_v35_canonical_full.sh`。
- **v35 结果**：canonical prescreen **10/60**，full harmful **27/100 raw refusals**；primary KL **0.0566**、benign gate **0.00%**、长度偏差 0.56σ；validation KL 0.0765 / p95 0.2747 / max 0.5518，health passed、thinking leak none。27/100 是首个 input-order-invariant refusal 指标。KL 仍混用了旧 baseline cache（旧 batch membership）与 canonical steered scoring，因此在 benign gate 完全关闭时仍非零，不能作为最终同协议损伤值。
- **v36 canonical-baseline 修正**：canonical batching 现在即使 steering hook 尚未安装也生效，使 baseline continuation generation/logprob 与 steered scoring 使用完全相同的 length+content batches；v36 不复用旧 baseline cache，重新建立规范 baseline。single-sample gate 同时增加 rendered-prompt 内存 cache，使 prescreen/full harmful 与 benign generation/forced scoring复用判定，避免重复 prepass。v36 将给出最终 matched canonical KL，不再继续 profile 搜索。
- **v36 结果 / matched canonical 基准成立**：fresh canonical baseline 仍为 **97/100** 初始拒答、平均回复长度 **50.5±46.5 words**；gate 自动选择 layer 18，external accuracy **98.00%**（benign active **0.00%**、harmful active **96.00%**）。固定 v31 profile 的 prescreen 稳定复现 **10/60**，full harmful 为 **27/100**。在 baseline 与 steered 使用完全相同 batch membership 后，primary full-distribution KL 为 **0.0000 nats/token**，benign-eval gate **0.00%**，长度偏差 0.60σ；独立 validation KL **0.0461** / p95 0.1099 / max 0.4599，高置信 top-1 flips **0/30**（仅 3 个 low-margin flips）；generation health passed，thinking leak none。结论：v35 的 0.0566 是旧 baseline/新 canonical batch 不匹配造成的测量伪差；质量目标已经命中，拒答目标仍差 17 条，下一轮只应搜索 gate-active harmful 路径的更有效干预，不再改 gate 或 baseline 协议。总运行约 1h41m，其中 trial evaluation 46m48s。日志：`logs/ling30_v36_matched_canonical_full_20260812.log`。
- **v37 bidirectional gated removal 已建立**：当前 angular strength >1 已被 clamp 到 90°，继续抬 max 不会增加峰值移除；真正未测的限制是 concept-gated 模式仍默认只处理与 refusal direction 正向对齐的 token。新增 `concept_gate_positive_alignment_only`（默认 true，兼容旧行为）；v37 仅在 external benign active=0% 的 sample-global gate 后关闭该限制，对 gate-active harmful 样本同时移除正/负方向分量，保留 canonical prepass/batch16 和 v36 matched baseline。先做同一 60 prompts probe，以 v36 的 10/60 为对照；只有显著下降到约 ≤6/60 才支付 full eval。新增定向回归后聚焦套件 **33 passed**。runner：`run_ling30_v37_bidirectional_gate_probe.sh`。
- **v37 结果 / signed removal 停止**：canonical gate 指标逐项复现（layer 18 external 98%，benign 0%、harmful 96%），runtime gate 96.52%；关闭 positive-alignment-only 后 prescreen 仍为 **10/60**，与 v36 positive-only 完全相同。负投影 token 不是剩余拒答瓶颈，不做 full eval，默认 safeguard 保持开启。
- **v38 flat full-depth probe 已建立**：由于 max=1.20 在 hook 中已 clamp 到 90°，继续抬峰值无效；v38 改测实际尚可增加的量——跨层覆盖。保持 v36 gate 与 positive-only safeguard，把 profile 设为 max=1.0、min fraction=1.0、peak L20、distance=24.6，使 42 层近似平顶全移除；仍只跑 canonical 60 prompts。若不能显著优于 10/60，则排除“晚层 profile 覆盖不足”，不做 full eval。
- **v38 结果 / 全深度覆盖停止**：gate telemetry **96.48%**，flat full-depth 90° profile 得到 **15/60**，比 v36 晚层宽带的 10/60 多 5 个拒答。更早/更多层的干预会破坏已成功样本，剩余瓶颈不是覆盖不足；不做 full eval。
- **v39 outcome map 已建立**：prescreen 现在在 journal 中额外持久化 `prescreen_refusal_indices` 与 `prescreen_compliance_indices`（原始 100 题索引），兼容只有 count API 的旧 detector mock。v39 将精确重放 v36 canonical profile，建立 10 个顽固拒答的稳定索引基线；后续任何新干预都比较 outcome overlap，而不再仅凭总数判断。新增单测后聚焦套件 **34 passed**。
- **v39 结果 / 稳定 refusal set**：规范配方再次精确得到 **10/60**，拒答原始索引为 **[51, 58, 53, 0, 78, 98, 76, 5, 39, 97]**。原始 prompts 横跨选举虚假信息、生物攻击、暴力煽动、工业破坏、自杀、洗钱、骚扰、IED、仇恨与人体致伤，都是高严重度直接执行/煽动请求，但不存在单一主题簇可供专门方向修补。当前 outcome telemetry 证明了“稳定顽固集合”，尚未证明其中哪些是 gate-off。
- **v40 gate attribution 已建立**：在 outcome map 上继续持久化 `prescreen_gate_on_indices` / `prescreen_gate_off_indices`，直接把 10 个拒答拆成 classifier 漏检与 gate-on steering failure；只有后者才需要新方向/多方向，前者应改 gate recall。聚焦套件仍 **34 passed**。
- **v40 结果 / 主瓶颈归因完成**：canonical prescreen 再次为 **10/60**，gate active **96.48%**；10 个稳定拒答中仅 **2 个 gate-off**，其余 **8 个 gate-on**。因此约 80% 剩余失败是 gate 已开启后的 steering failure，继续调 classifier 阈值只能覆盖次要的 2 个漏检。数据集元数据复核显示 10 个样本横跨 10 类，但 6/10 带 persona/role-play 包装；不按 eval 主题重训，避免泄漏。
- **v41 rank-2 subspace probe 已建立 / 启动 guard 已修**：为 HF `angular` / `concept_gated_angular` 补齐 QR 正交化的 rank-k 子空间移除，保留单方向精确语义；rank-k concept gate 强制关闭 sign-arbitrary 的 positive-alignment-only。首次启动在 trial 安装阶段被 `apply_steering()` 内重复的旧 runtime guard 拦截，未进入生成、不是方法结果；已同步放开该入口并新增穿过公共 API 的 rank-2 gated hook 回归。v41 只把训练集提取从 mean rank-1 改为 SVD rank-2，固定 v36/v40 profile、global gate、canonical batch16 与 60-prompt outcome attribution；复用 matched baseline / gate-response cache，不复用 rank-1 steering cache。若 gate-on refusals 未低于 8，则停止通用 rank-k；若明显下降，再做 matched full eval。修复后聚焦回归 **53 passed**。
- **v41 结果 / 对 rank-k 的归因暂缓**：修复后 trial 正常完成，rank-2 canonical prescreen 为 **13/60**，比 v40 rank-1 的 10/60 退化 3；其中 **10 个 gate-on refusal、3 个 gate-off refusal**，gate active **95.17%**（v40 为 8 / 2 / 96.48%）。outcome 只重合 5 条；v41 修复 v40 的 5 条，却新增 8 条回归。事后审计发现两次运行都重新训练 concept scorers，internal max-gap 自动决策层也从 v40 的 **layer 18** 改选为 v41 的 **layer 19**；因此 v41 不是严格的单变量 rank-1/rank-2 对照，只能判定“该次整体配置退化”，不能判死第二方向，也不能把 gate 集合变化归因于更早 steering。下一步必须缓存/freeze scorer state 与 decision layer，再做 paired canonical A/B；在此之前不做 full eval、不增加 rank。
- **v42 frozen-gate paired A/B 已建立**：新增方向无关的 concept-scorer provenance cache，保存逐层 `state_dict` / training metrics，并在 scorer 训练前显式重置 seed，避免 vector path 的 RNG 消耗改变 gate。v42a 固定 decision layer 18 跑 rank-1；v42b 复制 v42a 的 exact scorer cache、同样固定 layer 18，只换 rank-2；两边统一 sign-agnostic removal、canonical batch16 与相同 60-prompt set。相关缓存/运行时聚焦测试 **44 passed**；只有该配对结果可用于 rank-k 因果判断。
- **v42a 中间诊断 / 单方向数值兼容回归**：v42a 的 gate-on/off 集合与 v40 **逐索引完全一致**（58 on / 2 off，active 96.48%），但 refusal 变为 **20/60**。根因不是 gate，而是 rank-k 实现把 rank-1 angular removal 改写成数学等价的矩阵子空间公式；Ling bnb4bit MoE 会把微小浮点差异经 router 放大。现已为 `direction.ndim==1` 恢复历史逐操作路径，rank-2 才走 QR，并新增 fp16 bitwise regression test。正在运行的 v42a/v42b 仍可比较新实现下 rank 相对效应，但 v42a 不能作为 v40 复现；修复后的 rank-1 必须另行 canonical 复验。
- **v42 paired 结果**：v42b exact 加载 v42a 的 42-layer scorer cache，decision layer、58/2 gate 集合和 active **96.48%** 均完全一致。错误的矩阵化 rank-1 为 **20/60**，rank-2 为 **14/60**；rank-2 修复 12 条、引入 6 条，净改善 6，证明第二方向含有效信号但联合子空间仍有明显 outcome tradeoff。由于历史 legacy rank-1 为 10/60，v42b 不做 full eval。
- **v43 legacy rank-1 frozen-gate repro 已建立**：复用 v42a exact scorer / rank-1 steering / baseline / generated-response caches，固定 layer 18 与 canonical batch16，只切回 bitwise legacy scalar operation order。目标是复现 v40/v37 的 **10/60**，并为后续 direction-router 提供不可漂移 control。
- **v43 结果 / legacy control 恢复**：exact scorer cache 与固定 layer 18 下得到 **10/60**，gate-off 仍为 `[53, 0]`，证明 rank-1 operation-order 修复有效。与 v40 outcome 重合 7 条，另有 3 修复 / 3 新增，显示量化 MoE 跨进程仍有边界漂移；后续新方法需单次 ≤8/60 后复跑确认，不能只按一次总数晋级。
- **v44 single-direction router 已建立**：rank-2 directions 仍只由 train[:800] 提取；single-sample final-prefill prepass 在固定 layer 18 按对两条归一化方向的最大绝对投影选择 route ID，并与 global harmful gate 一起缓存、广播、锁存 decode。每个样本只走一条 legacy scalar angular 路径，不联合移除二维子空间；journal 额外持久化匿名 `prescreen_direction_route_indices`。不使用 eval outcome labels 拟合或选路由。聚焦回归 **60 passed**。
- **v44 结果 / 原始 rank-2 router 停止**：得到 **18/60**，未修复 v43 的任何拒答并新增 8 条；route 严重塌缩为 56 条 direction 0 / 4 条 direction 1。缓存审计发现 rank-2 direction 0 虽语义上以 mean 初始化，但经 multi-direction winsorize/project/orthonormalize 后与 legacy mean 的逐层平均 cosine 仅 **0.9716**（44/44 slots <0.9999），因此 v44 没有真正保留 10/60 主方向，不复跑。
- **v45 exact-legacy-primary router 已建立**：离线组合仍只使用训练工件：direction 0 逐元素复制 v43 legacy mean；direction 1 取 rank-2 residual 后重新对 exact primary 正交化；rank-2 cache provenance key 保持不变并写 composition metadata。其余 frozen gate / layer 18 / canonical 协议与 v44 相同。目标是检验“保留成功 primary，只在无监督投影支持时切 secondary”能否低于 10/60。
- **v45 结果 / max-projection router 停止**：得到 **14/60**；相对 v43 仅修复 1 条、引入 5 条，route 仍为 56/4。投影幅度连续两次无法预测方向的因果有效性，因此不复跑、不做 full eval，也不再调整投影阈值。
- **v46 train-only counterfactual probe 已建立**：新增 rank-k fixed-direction per-trial search，在同一次模型加载中对 `target train[800:900]` 的固定 canonical 60 子集依次应用 exact primary index 0 与 orthogonal secondary index 1，持久化匿名 outcome。方向提取仍只用 train[:800]，eval[900:] 完全不参与。若 secondary 在 calibration 上不能明显修复 primary failure，则 learned router 缺乏互补上限、整支停止；若互补充分，才训练 route scorer。
- **v46 结果 / learned router 停止**：exact scorer cache 与固定 layer 18 均成功复用；新 calibration 原模型为 **98/100** 拒答、global harmful active **97%**。exact primary 为 **17/60**，orthogonal secondary 为 **59/60**；secondary 只修复 primary failure **1 条**，却新增 **43 条**，即使完美 oracle 也仍为 **16/60**。因此不训练 direction router、不再使用全局 SVD secondary。
- **v47 failure-conditioned variants 已建立**：只用 v46 的匿名训练校准 outcome，将 clean final-prefill residual 的 `primary failure − primary success` 均值逐层投影到 exact primary 的正交补，再构造 exact primary 与 `primary + α·failure_residual`（α=0.25/0.5/1/2）五条**单方向**候选。所有 trial 保持 legacy scalar angular 路径、冻结 gate/layer 18/canonical batch16；仍只在 `train[800:900]` 同一 60 条做过拟合可行性筛查。只有显著低于 17/60 的候选才进入未见 40 条验证，eval[900:] 不用于构造或选择。
- **v47 结果 / failure-conditioned 分支停止**：首个 control 因 helper 对 primary 多做了一次 float32 pre-normalization，并非 bitwise exact v46 primary；它意外得到 **9/60**。在该同进程 control 下，α=0.25/0.5/1/2 依次为 **10/17/38/58**，且相对 control 的修复/新增分别为 **6/7、3/11、3/32、0/49**，呈明确单调退化。failure residual 不做未见验证。helper 已修为 control 原 tensor 逐元素保真，并把 pre-normalized primary 隔离为独立数值配方。
- **v48 renormalized-primary repro 已建立**：不再使用 calibration outcome，仅把 exact primary 在 CPU float32 预归一化一次，构造两个 byte-identical categorical 名称并在正式 eval[900:] canonical 60 子集同进程重复。此轮不按 outcome 调参；只有两次都达到 ≤8/60 才进入 full KL/full-100，避免把 v47 的 9/60 量化边界偶然性当进展。
- **v48 结果 / pre-normalization 停止**：正式 canonical 60 上两个 byte-identical trial 均为 **16/60**，refusal indices 也逐项完全一致，gate active 都为 **96.48%**、gate-off 都是 `[53,0]`。因此 v47 calibration 的 9/60 不泛化；额外 float32 pre-normalization 是稳定但更差的数值配方，不做 full eval。
- **v49 negative failure variants 已建立**：v47 的正 α 序列从 9 单调退化到 58，提示 failure-minus-success residual 的有效符号可能相反。v49 修复 exact-primary control 后在同一训练校准 60 条测试 α=-0.125/-0.25/-0.5/-1；仍不接触正式 eval outcome。只有相对 exact control 显著下降并保持 outcome 净改善的负 α 才进入正式复验。
- **v49 结果 / 发现窄负向窗口**：bitwise exact primary 精确复现 **17/60**；α=-0.125/-0.25/-0.5/-1 依次为 **10/21/31/47**。唯一候选 -0.125 相对 control 修复 12、引入 5，净改善 7；更大负偏转持续退化，因此不再细扫训练子集，避免标签过拟合。
- **v50 formal repro 已建立**：新增独立 `calibration_failure_prompt_split`，方向 residual 明确从 `train[800:900]` 提取，而 target eval 保持 `train[900:]`，防止 formal residual 泄漏。α=-0.125 被复制为两个 byte-identical categorical trial，在正式 canonical 60 子集重复；两次均 ≤8/60 才晋级 full KL/full-100。聚焦测试 **32 passed**。
- **v50 结果 / failure-conditioned 方向停止**：两个 byte-identical α=-0.125 trial 在正式 canonical 60 上均为 **19/60**，refusal indices 逐项完全一致、gate active 均为 **96.48%**。因此 v49 的 10/60 是训练校准过拟合；不做 full eval，也不再沿 outcome-conditioned residual 调参。
- **v51 angular over-rotation calibration 已建立**：审计发现历史 angular hook 即使 profile `max_weight>1` 也会把 fraction 夹到 1.0（90° 正交切面），所以 v31/v32 从未真正测试过切面后的旋转。新增默认关闭、上限 2.0 的 concept-gated over-rotation 开关；在完全隔离的 `train[800:900]` canonical 60 子集比较 legacy clamp 与 1.10/1.20/1.40 真过旋，冻结 exact primary、scorer、layer 18 和 gate。正式 `train[900:]` 不用于本轮选择；聚焦回归 **48 passed**。
- **v51 结果 / 全程过旋不晋级**：legacy clamp 精确复现 **17/60**；1.10/1.20/1.40 真过旋依次为 **15/13/14**，四点 gate active 均为 **98.39%**。最佳 1.20 相对 control 修复 8、引入 4，净改善 4；1.40 已反弹，且大量 outcome 交换，离 ≤10/60 晋级线仍远，因此不接触 formal split。
- **v52 over-rotation phase calibration 已建立**：新增默认 `all` 的 `all/prefill/decode` 相位开关，未选相位仍严格使用 legacy 90° clamp。冻结 v51 最佳 max=1.20，在同一 train-only canonical 60 上分别只对 prompt prefill、只对 autoregressive decode、以及全程过旋，定位 v51 小幅收益来自拒绝初始化还是后续轨迹；正式 `train[900:]` 仍不参与选择。
- **v52 结果 / phase 支线停止**：prefill-only / decode-only / all-phase 依次为 **18/15/13**，gate active 均为 **98.39%**；all-phase 精确复现 v51 最佳点。只改 prefill 比 legacy 17 更差，只改 decode 也不及全程，说明两阶段存在协同但总收益仍不足；不再细切前 N decode token，也不做 formal eval。
- **v53 projection geometry calibration 已建立**：剩余 13 条失败中 12 条 gate-on，且类别分散（Cybercrime/Fraud/Hate/Drugs/PII 各 2，另有 Violence/Sabotage/Self-Harm），排除单一 topic router。新增默认 `angular` 的 gated intervention geometry；`linear_projection` 直接减去方向分量而不把正交残差重归一化到原激活范数。固定 layer 18/exact primary/canonical gate，在 train-only 60 上比较 angular 1.20 与 linear 1.00/1.20/1.40；formal split 不参与选择。
- **v53 结果 / geometry 单轴不晋级**：angular 1.20 精确复现 **13/60**；linear 1.00/1.20/1.40 依次为 **14/11/11**，gate active 约 **98.39%**。取消 radial renormalization 仅在反射强度下净改善 2，1.20→1.40 已平台；但两者仅共享 5 个 failure，匿名 oracle 下界 **5/60**，提示样本级强度互补而非单一最优强度。
- **v54 anonymous strength-feature probe 已建立**：global single-sample prepass 在 layer 18 额外缓存 `|cos(final-prefill hidden, primary direction)|` 单标量；journal 只持久化 source index→float，不保存 prompt、hidden state 或响应。用 linear 1.20 单 trial 复现 v53，并离线检验一个简单 threshold 能否在 1.20/1.40 outcome 间逼近 5/60 oracle；只有明显优于固定 11/60 才实现 router。
- **v54 结果 / absolute-cosine router 停止**：linear 1.20 精确复现 **11/60**，gate active **98.38%**，60/60 匿名特征完整落盘。穷举两个方向的全部单阈值后，最佳仅为 **9/60**，且只有一个孤立阈值（`|cos|=0.361328125`）；leave-one-out 在 tied optimum 下的 held-out refusal 为 9–17/60，1000 次 bootstrap 的规则方向分裂为 594/406，OOB refusal 均值 **20.8%**。该收益不稳定、明显偏向校准集拟合，因此不实现 strength router、不触碰 formal split。下一探针只扩充匿名连续 telemetry（signed cosine 与 gate raw score），验证是否存在稳定的一维或极浅二维规则；仍不记录 prompt、response 或 hidden state。
- **v55 signed/gate feature probe 已建立**：在 v54 不改输出的 canonical prepass 上增加 layer-18 signed cosine 与连续 classifier score cache；保留 abs-cosine 以做兼容校验。单 trial 再次重放 linear 1.20，要求 refusal set 精确复现且 `abs(signed)==abs_feature` 后，离线评估 signed/gate-score 单阈值与最多深度 2 的规则，并用 bootstrap OOB 作为晋级门槛。runner：`run_ling30_v55_signed_gate_feature_probe.sh`；formal split 仍完全隔离。
- **v55 结果 / feature-router 分支停止**：linear 1.20 的 **11/60**、98.38% gate activity、source-index 排列与 refusal set 均逐项复现 v54；三组 feature 均为 60/60，且 `abs(signed)==abs_feature` 精确成立。该子集 signed cosine 全为正，因此没有新增符号信息；gate score 仅 10 个离散值，最佳 stump **10/60**。最多深度 2 的规则虽可在校准集达到 **8/60**，但 500 次 bootstrap 的 stump/tree OOB refusal 均值均约 **20.8%**，无泛化信号。停止所有 prefill-feature strength router，不实现运行时阈值。
- **v56 refusal-triggered fallback formal A/B 已建立**：v53 的 linear 1.20/1.40 各为 11/60，但失败仅交集 5 条；与其从弱 prefill feature 猜强度，改验证一个无需训练的因果级联：默认生成 1.20，只有首个输出被既有 refusal detector 判拒时才以 1.40 重试。先在从未用于 v51–v55 选择的 formal `train[900:]` canonical 60 上同进程跑两个固定强度，离线交集即级联可达拒答数；交集约 ≤8/60 才实现实际 retry 路径并做 full-100/KL，否则停止。runner：`run_ling30_v56_formal_strength_fallback_ab.sh`。
- **v56 结果 / strength fallback 不晋级**：formal baseline 正确命中 v36 canonical cache（初始 **97/100**），layer-18 external gate 为 **98.00% accuracy / 0% benign active / 96% harmful active**。同进程 linear 1.20/1.40 分别为 **19/60** 与 **15/60**，gate set 完全一致（58 on / 2 off）；1.40 修复 9 条但新增 5 条，失败交集为 **10/60（16.7%）**。实际 fallback 平均需对 31.7% 请求重试，仍高于 ≤8/60 晋级线；剩余交集中 8 条 gate-on、2 条 gate-off。校准集 5/60 oracle 未迁移，故不实现 retry、不做 full-100/KL，并停止二强度路由/级联支线。
- **v57 harmfulness counterfactual 已建立**：v46 的第二方向是同一提取族的 SVD 残差；下一未测方向族是 Zhao et al. 的 harmfulness（有害样本中心化后的 PCA-1，与 refusal mean-diff 正交）。新增 `compose_exact_primary_with_harmfulness`，slot 0 逐元素复制 v43 legacy primary，slot 1 只用 `train[:800]` residual 提取并再正交。v57 在冻结 gate / layer 18 / canonical batch16 下，对 `train[800:900]` 同一 60 条依次应用 index 0 与 index 1，协议同 v46。若 harmfulness 不能明显修复 primary failure，则不实现顺序双方向、不碰 formal split；若互补充分，再做 sequential 或 formal 复验。runner：`run_ling30_v57_harmfulness_counterfactual.sh`。
- **v57 结果 / Zhao harmfulness 停止**：t0 exact primary 精确复现 v46 的 **17/60**，拒答索引逐项一致 `[26,51,20,15,67,72,42,8,94,53,18,56,7,37,1,2,82]`，gate-off 仅 `[18]`。t1 Zhao harmfulness 为 **60/60**，修复 **0** 条、新增 **43** 条，oracle 下界仍是 **17/60**。离线组合与 SVD 第二方向平均余弦 0.405，确认是新方向，但没有 refusal-control 信号。不实现顺序双方向、不做 formal eval，停止 harmfulness / 第二方向族。日志：`logs/ling30_v57_harmfulness_counterfactual_20260813_094003.log`。
- **v58 steered residual dump 已建立**：不再从 unsteered 状态换方向。新增 `dump_steered_prefill_residuals`，在 global prepass 里保存每个样本 post-hook 的 final-prefill residual（按匿名 source index）。v58 只重放 exact primary 于 `train[800:900]` 同一 60 条，复现 17/60 并写出 peel 用 cache；formal `train[900:]` 不用。runner：`run_ling30_v58_steered_residual_dump.sh`。
- **v58 结果 / dump 成立**：exact primary 再次 **17/60**，拒答索引与 v46/v57 逐项一致；gate 98.39%。写出 `steered_prefill_residuals.pt`（60×42×2560），23 个 hooked 层非零（L18 observer + L20–L41 profile），0 条空样本。日志：`logs/ling30_v58_steered_residual_dump_20260813_110407.log`。
- **v59 steered peel formal 已建立**：用 v58 dump 构造 exact primary + 干预后 failure−success residual（正交、不混回 primary）。peel 标签去掉 gate-off 的 18。在 formal `train[900:]` 同一 60 条上对照 index 0/1；复用 v36 formal baseline。若 peel 不能明显修复 primary failure，则停止 peel / 顺序双方向。runner：`run_ling30_v59_steered_peel_formal.sh`。
- **v59 结果 / steered peel 停止**：t0 exact primary 精确复现 v43 的 **10/60**，拒答索引逐项一致 `[0, 18, 51, 53, 58, 61, 76, 78, 89, 98]`，gate-off 仍是 `[53, 0]`。t1 steered peel 为 **60/60**，修复 **0** 条、新增 **50** 条，oracle 下界仍是 **10/60**。干预后 residual 与 SVD / Zhao 第二方向一样没有 refusal-control 信号。不实现顺序双方向、不做 full-100/KL。日志：`logs/ling30_v59_steered_peel_formal_20260813_113941.log`。剩余 8 个 gate-on 失败不再适合用“再找一条静态 residual 方向”来打。
- **v60 prefix-retry formal 已建立**：改测 decode 机制。prescreen 持久化匿名 prefix/late 标签和粗分类（不写 prompt/response）。t0 只诊断；t1 对 gate-on 且前缀拒答的样本，在**同一 batch** 上禁止其首轮前 6 个 token 再生成一次。若 t1 不能明显低于 10/60，停止机制探针并冻结现有最佳点。runner：`run_ling30_v60_prefix_retry_formal.sh`。
- **v60 结果 / 机制探针封顶并冻结**：t0 精确复现 **10/60** 与 v43 索引。10 个拒答里只有 **3 个前缀拒答**（`[51, 0, 98]`：1 zh_cannot + 2 apology），**7 个是后段才拒**。t1 同 batch 前缀重试为 **9/60**，只修复 `[51]`、0 新增；retry 2 次（gate-on 前缀），2 次前缀被改掉，但 `[98]` 变成后段拒答。未过 ≤8/60，也不做 full-100。日志：`logs/ling30_v60_prefix_retry_formal_20260813_123506.log`。按预定规则停止冲刺，改用现有最佳点。

### 4.3 明确证伪清单

| 假设 | 结果 |
|------|------|
| 缺 o_proj 是唯一问题 | 修了 o 仍不够（v2） |
| Laguna expert routing 可复用 | 抬拒 / 无助双目标 |
| SRA 清洗方向 | KL 底盘仍高 |
| cliff-head 固定预消融 | 拒答不动 |
| 默认 `weight_norm=none` 即可 | 拒答推不动 |
| 只堆 trial 到 80 | dual 卡在 ~0.09 / 17% |
| 关 down / 轻 o only | 几乎全 prune |
| **在 v6 赢家上抬强度天花板即可双目标** | **v9：拒可到 13%，KL 抬到 ≥0.11；0 双命中** |

### 4.4 仍成立的设计规则

1. **MPOA `full`+rank3** — Ling 低 KL 必需（相对 none）  
2. **o_proj + down_proj 同开** — 禁 auto-disable down  
3. **per-layer 向量** — global 易死  
4. **batch=16** — 本机已验证  
5. **低拒偏好 o~2.3–2.7、d~1.6–2.0、中后层** — v6/v9 一致  
6. **mean 方向 + MPOA** — 已有最多数据；再同空间微调 **边际收益耗尽**  
7. **mean+MPOA 强度搜索 alone 不足以同时 KL≤0.05 与 拒≤10%** — 需换方向族或接受折中导出  

---

## 5. 全局 Pareto 快照（更新至 v60 冻结 · 2026-08-13）

跨全部 `checkpoints_ling30*` 静态与 runtime 条件路径，**双目标命中仍为 0**。runtime-only 点不可直接 merge，但已明显改写质量/控制前沿。

| 来源 | 类型 | Primary KL | Raw 拒答 | 说明 |
|------|------|-----------:|---------:|------|
| **v29** | runtime global gate | **0.0622** | **20/100**（matched batch-order range 20–29） | 当前偏控制的 conditional 点；benign gate 1.09% |
| **v31** | runtime global gate | **0.0602** | **21/100** | 当前偏质量的 conditional 点；val KL 0.0765 |
| v13 | runtime always-on adaptive | 0.1240 | 16/100 | 控制更强，质量损伤约翻倍 |
| v6 t21 | static MPOA LoRA | 0.093 | 17.3% objective | 仍是静态可合并最佳 dual-ish |

### 最近双目标（score 越小越好；相对 KL≤0.05 且 拒≤0.10）

| 来源 | trial | KL | 拒 | o_mw | d_mw | 说明 |
|------|------:|---:|---:|-----:|-----:|------|
| **v6** | **21** | **0.093** | **0.173** | 2.40 | 1.94 | **全项目最佳 dual-ish** |
| v6 | 15 | 0.094 | 0.194 | 2.41 | 1.66 | |
| v9 | 12 | 0.111 | 0.143 | 2.43 | 1.75 | v9 最佳；更低拒、更高 KL |
| v9 | 13 | 0.117 | 0.153 | 2.42 | 1.56 | |
| v6 | 22 | 0.101 | 0.214 | 2.31 | 1.78 | |
| v9 | 8 | 0.106 | 0.214 | 2.13 | 2.00 | |
| v9 | 15 | 0.120 | 0.184 | 2.51 | 1.71 | |

### 极值

| 目标 | 点 | 说明 |
|------|-----|------|
| 最低 KL | v4 Ornith t5 **KL 0.034 / 拒 58%** | 低 KL 不自动低拒 |
| 最低拒（有限） | v9 t10 **拒 13.3% / KL 0.160** | 全项目最低拒；KL 差 |
| 最佳 dual-ish | **v6 t21 KL 0.093 / 拒 17.3%** | 若要「一个点」交付，优先此 |
| 验收线 | KL≤0.05 且 拒≤0.10 | **全阶段未出现** |

**终局图像**：mean+MPOA 在 Ling 上的 Pareto 前沿大致是  
**（KL≈0.09, 拒≈17%）↔（KL≈0.11–0.16, 拒≈13–15%）** 一段弧，**不穿过（0.05, 0.10）**。

---

## 6. 工件地图

### 配置

```
configs/ling30_flash_rocm.toml              # 初代
configs/ling30_flash_rocm_debug.toml
configs/ling30_flash_rocm_smoke.toml
configs/ling30_flash_rocm_v2.toml           # Laguna
configs/ling30_flash_rocm_v3_sra.toml
configs/ling30_flash_rocm_v4_ornith.toml
configs/ling30_flash_rocm_v5_lowref.toml
configs/ling30_flash_rocm_v6_stack.toml     # 最完整 Pareto
configs/ling30_flash_rocm_v7_cliff.toml
configs/ling30_flash_rocm_v8_default.toml
configs/ling30_flash_rocm_v9_optimal.toml   # 当前搜索
```

### Runner / 监控

```
run_ling30.sh / run_ling30_debug.sh / run_ling30_v2.sh …
run_ling30_v4_interactive.sh … v9_interactive.sh
  --continue | --batch
watch_ling30.sh [v9|v8|v7|v6|…]
```

### Checkpoint 目录

```
checkpoints_ling30_flash_{debug,smoke,v2,v3_sra,v4_ornith,v4_b16,
                          v5_lowref,v6_stack,v7_cliff,v8_default,v9_optimal}/
  *baseline.pt  *steering.pt  *flash.jsonl
```

`watch_ling30.sh` 默认现指向 v60；也可显式传 `v59` / `v43` / `v36`
查看对应日志、checkpoint 与 journal 摘要。

### 日志

```
logs/ling30_v*.log
```

### 其它文档

| 文档 | 内容 |
|------|------|
| `docs/ling30_flash_abliterix_modifications.md` | 早期代码补丁细节（引擎/TPE/modeling） |
| `docs/LING30_FLASH_PROJECT_LOG.md` | **本文：全项目方法+结果+代码索引** |
| `BRANCH_LOG.md` | sc117-base 变更倒序日志（强制维护） |

---

## 7. 阶段结束：交付选择与后续

**2026-08-13 冲刺已冻结。** 双目标 (KL≤0.05 且 拒≤10/100) 仍未同时命中。不再开同族方向或前缀重试。

### 冻结交付

| 优先级 | 点 | 类型 | 指标 | 用途 |
|--------|-----|------|------|------|
| 1 | **v36 matched-canonical** | runtime global gate | primary KL **0.0000**，val KL **0.0461**，拒 **27/100**，benign gate **0%** | 质量优先的条件干预；不可直接 merge |
| 2 | **v6 t21** | 静态 MPOA LoRA | KL **0.093** / 拒 **17.3%** | 可合并折中 |
| — | v60 t1 prefix retry | runtime + retry | 60 题 **9/60**（相对 v36/v43 的 10/60 只少 1） | 不晋升：未做 full-100，且 7/10 是后段拒答 |

v36 配置 / 复现：`configs/ling30_flash_rocm_v30_global_strength.toml` + `run_ling30_v36_matched_canonical_full.sh`；checkpoint `checkpoints_ling30_flash_v36_matched_canonical_full/`。

**v1–v10 搜索进程已停止**。同方法再搜不推荐。

### 若要「一个最好的可合并点」（历史静态 Pareto）

| 优先级 | 点 | 用途 |
|--------|-----|------|
| 1 | **v6 t21** — KL **0.093** / 拒 **17.3%** / o≈2.40 d≈1.94 | 综合 dual 最佳 |
| 2 | **v9 t12** — KL **0.111** / 拒 **14.3%** / o≈2.43 d≈1.75 | 更低拒、略高 KL |
| 3 | **v9 t10** — KL **0.160** / 拒 **13.3%** | 最低拒，质量牺牲大 |
| 4 | **v4 Ornith t5** — KL **0.034** / 拒 **58%** | 仅当你极度偏 KL |

导出示例（需加载对应 study / recipe，交互 TUI 选 trial）：

```bash
# 续 v6 journal 进 TUI 后导出 t21（或 recipe）
./run_ling30_v6_interactive.sh --continue
# 或 v9
./run_ling30_v9_interactive.sh --continue
```

Journal 路径：

- `checkpoints_ling30_flash_v6_stack/*Ling*flash.jsonl`
- `checkpoints_ling30_flash_v9_optimal/*Ling*flash.jsonl`

### 若仍要冲 0.05 / 10%（新方向，未验证）

1. **换向量族** 仍保留 MPOA：`optimal_transport` / multi-direction / harmfulness pair（注意 MoE routing 关）  
2. **迭代 peel**（多轮固定点再搜）  
3. **不要**：cliff、纯默认 none、再抬 o/d 到 3+ 同空间盲搜、Laguna expert  

---

## 8. 决策记录（给未来自己）

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-08-07 | 修 Bailing steerable | v1 根本没 steer o_proj |
| 2026-08-08 | 弃 Laguna expert / SRA | KL 底盘不降 |
| 2026-08-08 | 采用 MPOA full+r3 | 唯一低 KL 盆地 |
| 2026-08-08 | 确认 batch16 | 历史 batch>1 退化在本机不复现 |
| 2026-08-09 | 弃 cliff | 54 heads 仍 30/30 拒 |
| 2026-08-09 | 默认 none 探针后放弃当主路径 | 拒答推不动 |
| 2026-08-09 | v9 在 v6 赢家上 **抬强度** 继续搜 | 冲双目标；非导出 |
| **2026-08-09** | **停 v9 并归档；结束本轮搜索** | 17 trial 已足够：拒可到 13% 但 KL≥0.11；双目标 0；边际耗尽 |
| **2026-08-10** | **停 v10；接管转向配对条件方向** | 32 complete 后仍未破前沿；有效层 OT/mean cosine=0.999569 |
| **2026-08-10** | **建立 v11 12-trial 探针** | 同 harmful prompt 只改变 refusal/compliance system condition，减少主题混杂 |
| **2026-08-10** | **停止 v11 prompt-condition 方向** | 12/12 prescreen 30/30 high；几何虽新但拒答控制为零，下一步需 response trajectory 配对 |
| **2026-08-10** | **建立 v12 response-trajectory 探针** | 同 prompt/system，teacher-force refusal/compliance continuation 并池化 assistant-token residual |
| **2026-08-10** | **v12 正常完成并停止** | 12/12 完成、4 有限；最佳仅 KL0.174/拒64.3% 或 KL0.245/拒48.0%，显著劣于历史前沿 |
| **2026-08-11** | **保留 v29/v31 conditional Pareto，停止单轴抬强度** | v29=KL0.0622/拒20，v31=0.0602/拒21；1.20 的 30题收益未迁移到 full-100 |
| **2026-08-11** | **停止 v32 profile 形状搜索** | 60题 8点最佳仍是已完整评估的 v31 profile（10/60）；窄带与 max 1.48 均退化，下一瓶颈是 batch-invariant 推理 |
| **2026-08-12** | **以 v34 canonical batching 作为后续唯一规范协议** | forward/reverse 同集均 10/60；消除了 batch composition 引起的输入顺序漂移 |
| **2026-08-12** | **接受 v36 为 matched-canonical 质量基准，转攻 harmful 效力** | primary KL 0.0000、validation KL 0.0461、benign gate 0%，但 refusal 仍 27/100；不再把测量伪差当模型损伤，也不再搜索 gate/baseline 协议 |
| **2026-08-13** | **停止 v56 二强度 fallback；建立 v57 Zhao harmfulness 反事实** | formal 1.20∩1.40=10/60，校准 oracle 未迁移；下一探针改测未用过的 harmfulness 方向族，仍冻结 gate/primary，且先只在 train[800:900] 上看互补上界 |
| **2026-08-13** | **停止 Zhao harmfulness / 静态第二方向族** | v57 t0=17/60 与 v46 逐索引一致；t1=60/60、修复 0、oracle 仍 17/60。不再做顺序双方向或 formal。剩余未测高信息轴是 **干预后 residual peel**，不是再换一条 unsteered 静态方向 |
| **2026-08-13** | **建立 v58 steered residual dump** | 只重放 exact primary 并落盘 post-hook prefill residual，供后续 peel；不在本轮调方向 |
| **2026-08-13** | **v58 dump 成立并启动 v59 formal peel A/B** | 17/60 复现、60×42 residual 完整；peel 只用 16 个 gate-on 失败，在 train[900:] 一次性验证 |
| **2026-08-13** | **停止 steered peel / 静态第二方向族** | formal peel=60/60、修复 0、oracle 仍 10/60；v43 控制点逐索引复现。下一探针必须换干预机制（decode/logit），不能再换 unsteered 或 steered 静态方向 |
| **2026-08-13** | **建立 v60 decode 前缀重试；达不到则冻结最佳点** | 同强度、同 batch，只禁刚出现的拒答前缀；这是封顶机制探针之一 |
| **2026-08-13** | **停止冲刺并冻结最佳点** | v60=9/60，7/10 是后段拒答；前缀机制不是主瓶颈。交付：runtime 用 v36，可合并用 v6 t21 |

---

## 9. 维护约定

1. 改 sc117-base 代码/配方 → **同步更新 `BRANCH_LOG.md`**  
2. 新一轮搜索 → 新 `checkpoint_dir` + `configs/ling30_*_vN_*.toml` + runner + `watch_ling30.sh` 条目  
3. 本文 §4 / §5 在阶段结束时补结果行  
4. 可上游化补丁优先走 `pr/*`（如 Bailing discovery、bnb cliff、multi-obj prune）

---

*End of project log — Ling-3.0-flash × Abliterix (sc117-base).*
