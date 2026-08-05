# sc117-base 分支日志

> **维护约定（强制）**  
> 凡在本分支（`sc117-base` 及其从它分出的 SC117 工作分支）上的代码/配置/文档变更，**必须在同一提交或紧随其后的提交中更新本文件**。  
> 未更新本日志的改动视为未完成。  
> 上游 PR 用干净分支（如 `pr/*`）时，合回 `sc117-base` 后也要在此记一笔。

| 字段 | 当前值 |
|------|--------|
| 分支名 | `sc117-base` |
| 远端 | `origin` → https://github.com/Scorp1o117/abliterix |
| 上游 | `upstream` → https://github.com/wuwangzhang1216/abliterix |
| 最近对齐上游 | **v1.12.1**（`76a7a31`，merge `09c6e53`） |
| 本日志最近更新 | 2026-07-27 (PR #95 CI) |

---

## 1. 分支定位

`sc117-base` 是 **Scorp1o117 自用生产分支**，在上游 Abliterix 之上叠加：

1. ROCm + bitsandbytes 大 MoE（Qwen3.5-MoE / Agents-A1 / Laguna-S-2.1 等）实战修复  
2. 可选的 **trial 分级筛查 + 生成健康检测** 管线  
3. 重启加速缓存（prefix / steering / baseline）  
4. 机台相关 TOML 配方与项目交接文档  

**不是**「整支替换上游」的 fork 默认分支；可上游化的小补丁走独立 `pr/*` 分支提 PR（例如 #95）。

---

## 2. 与上游的区别（总览）

| 类别 | 上游 `master` | `sc117-base` |
|------|----------------|--------------|
| 默认评估路径 | KL + compliance 目标 | **相同**（staged 全默认 **关**） |
| bnb-4bit `compute_dtype` | 可跟随 load dtype | **固定 `bfloat16`** |
| load 后非量化参数 | 无强制 promote | float16 加载时 **fp16→bf16 promote** |
| dequant 缓存 | 无字节上限（大 MoE 易爆内存） | **默认上限 4 GiB** |
| `qwen3_5_moe` 加载 | 可能走 ImageTextToText | **强制 CausalLM（文本 abliteration）** |
| Optuna trial 间内存 | 依赖既有 unload/cleanup | 额外 **`flush_memory()`** |
| 交互菜单 | 无「只存 LoRA」 | **Save LoRA adapter only** |
| 管道 `tee` 菜单 | questionary 易挂 | **非 TTY 数字菜单回退** |
| 拒答 prescreen / 健康筛 | 无 | **可选** StageEvaluator + TrialScreener |
| 磁盘缓存 | 无 | **可选** prefix / steering / baseline |
| 配方与笔记 | 通用 configs | + Laguna/Agents ROCm 配方、`LAGUNA_PROJECT_HANDOVER.md` 等 |

---

## 3. SC117 特别功能（详表）

### 3.1 ROCm / bnb / 大 MoE 稳定性

| 功能 | 位置 | 说明 |
|------|------|------|
| bnb compute_dtype 固定 bf16 | `core/engine.py` `_build_quant_config` | 避免 load=fp16 时假量化 / 专家未量化 OOM |
| 非量化参数 fp16→bf16 | `core/engine.py` 加载成功后 | 深层 residual 防 inf/NaN |
| dequant cache 上限 | `core/engine.py` + `core/steering.py` | `_dequant_cache_max_bytes = 4GB`，写入前检查 |
| qwen3_5_moe → CausalLM | `core/engine.py` `resolve_model_class` | Agents-A1 / Qwen3.5-MoE 文本路径 |
| trial 末 flush | `optimizer.py` `_objective_safe` finally | 长搜索降碎片 |

### 3.2 Trial 筛查与健康测试（默认关闭）

配置节：`[optimization]`（见 `settings.py` OptimizationConfig）。

| Flag | 默认 | 作用 |
|------|------|------|
| `refusal_prescreen_enabled` | `false` | 固定子集拒答预筛；high → `TrialPruned`（在 **完整 KL 前**） |
| `refusal_prescreen_size` | `30` | 预筛题量 |
| `refusal_prescreen_pass_max` | `8` | ≤ 此为 low，继续全量 |
| `refusal_prescreen_prune_min` | `19` | ≥ 此为 high，直接剪枝 |
| `prescreen_estimation_enabled` | `true` | borderline 可用估计跳过全量拒答 |
| `validation_kl_enabled` | `false` | hold-out benign 上 validation KL + top-1 指标 |
| `generation_health_enabled` | `false` | ngram 重复、token 频率、digit_ratio 等 |
| `thinking_leak_detection_enabled` | `false` | 检测 think 标记泄漏 |

实现：

- `eval/stage_evaluator.py` — `StageEvaluator` 管线  
- `eval/screening.py` — `TrialScreener`  
- `optimizer.py` — `use_staged` 时接入；flag 全关 = **纯上游路径**

**Laguna 配方**（`configs/laguna_s_2.1_rocm_bnb4bit.toml`）上述筛查 flag 为 **true**。

### 3.3 重启缓存

| 缓存 | 文件约定 | 触发点 |
|------|----------|--------|
| response prefix | `<checkpoint_dir>/<slug>_prefix.json` | `cli.py` |
| steering 残差/向量 | `<checkpoint_dir>/<slug>_steering.pt` | `cli.py` |
| baseline KL 等 | `<checkpoint_dir>/<slug>_baseline.pt` | `eval/scorer.py` |

缓存键含 model_id / 数据集 / 方法 / seed 等；不匹配则重算。

### 3.4 交互与 UX

| 功能 | 位置 |
|------|------|
| Save LoRA adapter only | `interactive.py` |
| 非 TTY 数字菜单 | `util.py` `_stdin_is_tty` / `_print_choices` |

### 3.5 配方与文档（自用，一般不进上游）

| 路径 | 说明 |
|------|------|
| `configs/laguna_s_2.1_rocm_bnb4bit.toml` | Laguna-S-2.1 ROCm bnb4bit |
| `configs/agents_a1_*.toml` | Agents-A1 相关 |
| `configs/qwen_agentworld_35b_a3b_rocm_bnb4bit.toml` | Qwen AgentWorld |
| `configs/lfm2.5_2.6b_rocm*.toml`, `run_lfm2.5.sh` | LFM2.5-2.6B ROCm 冒烟/完整搜索 |
| `docs/qwen35moe-rocm-bnb4bit-changes.md` | Qwen3.5 MoE 变更说明 |
| `LAGUNA_PROJECT_HANDOVER.md` | Laguna 项目交接 |
| `diag_*.py`, `run-agents-a1.sh` | 诊断 / 运行辅助 |
| **`BRANCH_LOG.md`（本文件）** | 分支差异与变更日志 |

### 3.6 不在本仓库的配套

| 项 | 说明 |
|----|------|
| `Laguna-S-2.1-bnb` modeling 拆 3D expert | 模型目录 patch，供 bnb 逐 expert 量化 |
| Uncensored GGUF / APEX / HF 发布 | 下游产物，见模型卡 README |

---

## 4. 已向上游贡献 / 待贡献

| 状态 | 内容 |
|------|------|
| PR 已开 | https://github.com/wuwangzhang1216/abliterix/pull/95 — bnb/dequant/flush/Save LoRA/非 TTY（基于 upstream 干净分支 `pr/bnb-rocm-moe-stability`） |
| 仅 fork | staged 筛查、磁盘缓存、机台 toml、handover |
| 上游已自有、勿回退 | interactive `_restore_selected_trial` 全参 steering、多轮 `flush_memory` 等（merge 时以上游为准） |

---

## 5. 变更记录（按时间倒序）

格式建议：

```text
### YYYY-MM-DD — 简短标题
- **类型**: fix | feat | chore | merge | docs
- **摘要**: …
- **涉及**: path1, path2
- **与上游关系**: 仅 fork / 已 cherry-pick 到 pr/* / 已合入上游 #N
```

---

### 2026-08-05 — fix: 导出兼容官方 generation config 缺 do_sample + TUI 管道渲染

- **类型**: fix（导出路径 + 运行脚本）
- **摘要**: ① LiquidAI/LFM2.5-2.6B 官方 `generation_config.json` 带 `temperature=0.1` 但无 `do_sample`，transformers 5.x 在 `save_pretrained` 时严格校验报 `GenerationConfig is invalid`，合并导出失败。新增 `interactive._sanitize_generation_config()`：`do_sample` 非 True 时把仅采样字段（temperature/top_p/top_k/min_p/typical_p）置 None 再保存（greedy 本就是实际默认行为，非静默改采样）；`_save_model_locally` 调用。上游 master 无此处理（已查）。② `run_lfm2.5.sh` 的 `2>&1 | tee` 管道在交互 TUI 下使 rich/questionary 渲染错乱（导出输入路径界面花屏）。改为 `[ -t 1 ]` 检测：TTY 直接跑（无管道），非 TTY（cron/nohup）才 tee。
- **涉及**: `src/abliterix/interactive.py`, `run_lfm2.5.sh`
- **与上游关系**: ① 可上游化（对任意带此类官方 config 的模型有益）；② 仅 fork 脚本

### 2026-08-05 — fix: validation KL 改同前缀 teacher-forcing（此前自由生成导致 KL 虚高 50 倍）

- **类型**: fix（fork 独有筛查指标）
- **摘要**: `_run_validation_kl` 用 `generate_and_score_batched` 让 steered 模型**自由生成**并捕获逐步 logprob，再对比 baseline 在固定 continuation 上的 logprob——一旦 steering 改变第一个 token（成功的 trial 正是如此），两侧前缀分叉，KL 爆炸式虚高。LFM2.5-2.6B 实测 `validation_kl_mean` 0.93~4.24（成功 trial 反而更高），而正确同前缀的优化目标 `kl_divergence` 仅 0.009~0.3；Agents-A1 两指标一致（~0.015）佐证。修法：steered 侧改用 `score_continuation_logprobs_batched` 在 baseline 固定续写（`baseline_continuations[validation_indices]`）上打分，两侧前缀逐 token 一致；vLLM 路径同样处理（带 adapter_path）。top-1 disagreement 部分本就用 `_logprobs_forward_pass` 同前缀，未动。baseline 续写缺失时跳过并提示。注：当前运行中的 trial 进程不受影响（旧代码），下次启动生效。
- **涉及**: `src/abliterix/eval/stage_evaluator.py`
- **与上游关系**: 仅 fork（`_run_validation_kl` 为 fork 独有）

### 2026-08-05 — fix: 固定全局 seed（TOML 顶层标量字段不生效）

- **类型**: fix / chore（运行脚本 + 配方）
- **摘要**: 实测 pydantic-settings `TomlConfigSettingsSource` 在此堆栈**只读嵌套 section**，顶层标量（`seed`/`non_interactive`/`overwrite_checkpoint`/`system_prompt`）被静默丢弃（`src()` 返回键仅含 section 名）。有效通道是 CLI flag（`--seed 117`）与 `AX_` 环境变量（`AX_SEED=117`）。此前 LFM2.5 完整搜索每次随机种子 → trial 轨迹不可复现、checkpoint 缓存失去意义。修法：`run_lfm2.5.sh` 显式 `--seed 117`（与 `refusal_prescreen_seed` 一致，`optimization.sampler_seed` 未设时继承全局 seed，TPE 探索轨迹随之固定）；config 删除无效的 `seed = 117` 行。另：交互模式由"不传 `--non-interactive`"实现（默认 False），TOML 里的 `non_interactive` 键本身无效。
- **涉及**: `run_lfm2.5.sh`, `configs/lfm2.5_2.6b_rocm.toml`
- **与上游关系**: 仅 fork（机台脚本 + 配方）；上游若想修 TOML 顶层字段失效需另查 pydantic-settings 配置

### 2026-08-05 — feat: 单卡 BF16 快速加载（绕开 accelerate 逐张量 H2D copy）

- **类型**: perf（引擎加载路径）
- **摘要**: `device_map='auto'` 下 accelerate 把每个 safetensors mmap 懒加载张量逐个 copy 到 GPU；ROCm 上 file-backed mmap 页首次触碰有 ~1.2s/张量固定开销（DMA page pin），266 张量的 LFM2.5-2.6B（5.4GB）加载要 5 分半，而 70GB GGUF（llama.cpp 顺序 mmap 预读）只要 5 分钟。新增 `_load_model_fast()`：CPU 懒加载（~0.4s）→ 全参数 `clone()` 物化 mmap 页（~0.8s，共享 storage 去重保持 tie）→ 整体 `.to('cuda')`（~0.6s），引擎初始化含 smoke test 共 **3.9s**（原 5:25，约 84×）。仅当 `device_map='auto'` 且无 `max_memory` 且 `quant_method=none` 且非 FP8 时启用；显式 device map / 多卡 offload / 量化模型走原路径。`__init__` 与 `restore_baseline` reload 两处共用。
- **涉及**: `src/abliterix/core/engine.py`
- **与上游关系**: 仅 fork（ROCm 单卡优化；上游若合并需确认非 ROCm 平台无此开销时仍安全——条件限定 device_map='auto' 单卡即可，通用有益）

### 2026-08-05 — fix: run_lfm2.5.sh 显式传 --non-interactive/--overwrite-checkpoint

- **类型**: chore（运行脚本）
- **摘要**: 顶层 TOML 布尔（`non_interactive`/`overwrite_checkpoint`）被 `CliSettingsSource(cli_implicit_flags=True)` 未传参时注入的默认 `False` 覆盖，导致脚本在管道（无 TTY）下弹出 checkpoint 恢复交互菜单、`input()` EOF 崩溃。嵌套字段（如 `optimization.refusal_prescreen_enabled`）不受影响，仅顶层布尔中招。修法：runner 显式传两个 flag（CLI 优先于 TOML）。缓存（baseline/steering/prefix）为独立 key 文件，不受 journal 覆盖影响。
- **涉及**: `run_lfm2.5.sh`
- **与上游关系**: 仅 fork（机台脚本）

### 2026-08-05 — fix: validation KL 多 token 形状 + scorer 测试 fixture

- **类型**: fix / test（仅 fork 筛查代码与测试）
- **摘要**: ① `stage_evaluator._run_validation_kl` 假设引擎已把多 token logprobs 平均成 `(batch, vocab)`，实际 `kl_token_count>1` 时返回 `(batch, step, vocab)`，`sum(dim=-1).tolist()` 产生嵌套 list，`statistics.mean` TypeError 崩溃（LFM2.5-2.6B smoke token_count=3 实测触发）。改为 sum vocab 后对 step 维求平均，与 `_safe_kl_divergence` 语义一致；单 token 路径不变。② `test_scorer.py` 4 个失败单测修复：fake engine 补 `_logprobs_forward_pass` stub（fork 加 Ornith top-1 时漏更新 fixture）、SimpleNamespace 外壳透传该方法、4 个测试各挂 `tmp_path` 隔离 baseline 缓存键（此前共享 `checkpoints/--dummy--model_baseline.pt` 互相串扰）。test_scorer.py 现 25 passed 全绿。
- **涉及**: `src/abliterix/eval/stage_evaluator.py`, `tests/test_scorer.py`
- **与上游关系**: 仅 fork；`_run_validation_kl` 与缓存为 fork 独有

### 2026-08-05 — fix: baseline cache save 缺 baseline_continuation_nll 属性

- **类型**: fix（仅 fork 缓存代码）
- **摘要**: `_capture_baseline` 的 HF 后端 + `kl.token_count > 1` 分支只设 `baseline_continuations`，不设 `baseline_continuation_nll`；末尾 `hasattr("baseline_continuations")` 保护因此跳过初始化，`_save_baseline_cache` 存缓存时 AttributeError 崩溃（LFM2.5-2.6B smoke 实测触发）。改为对两个属性分别做 `hasattr` 初始化。上游 master 无此崩溃点（`_save_baseline_cache` 为 fork 独有，上游读该属性处均有 getattr 保护）。
- **涉及**: `src/abliterix/eval/scorer.py`
- **与上游关系**: 仅 fork；如上游化缓存功能需一并带上

### 2026-08-05 — LFM2.5-2.6B (LiquidAI) ROCm 配方 + 运行脚本

- **类型**: feat / chore
- **摘要**: 新增 LFM2.5-2.6B（混合 22 短卷积块 + 8 GQA 注意力）abliteration 配方：`configs/lfm2.5_2.6b_rocm.toml`（60 trials 完整搜索）与 `configs/lfm2.5_2.6b_rocm_smoke.toml`（8 trials 冒烟验证）；新增 `run_lfm2.5.sh` 运行脚本（含 `unset PYTHONPATH` 防 Hermes venv 泄漏）。已验证引擎 `steerable_modules()` 原生注册 LFM2 三个投影点（`conv.out_proj` → attn.o_proj、`self_attn.out_proj` → attn.o_proj、`feed_forward.w2` → mlp.down_proj），无需改引擎代码。筛查 flag（prescreen/validation KL/generation health/thinking leak）全开，配方沿用 agents_a1_full 风格。模型权重放 `/run/media/s117/OS/Models/LFM2.5-2.6B`（BF16，~5.4GB，无量化）。
- **涉及**: `configs/lfm2.5_2.6b_rocm.toml`, `configs/lfm2.5_2.6b_rocm_smoke.toml`, `run_lfm2.5.sh`
- **与上游关系**: 仅 fork（机台配方）

### 2026-08-05 — PR #95 修 test_export_contract

- **类型**: test（上游 PR 分支）
- **摘要**: 作者指出全量 pytest 因 needs_reload 未初始化失败；按他的补丁修好 happy-path。
- **与上游关系**: https://github.com/wuwangzhang1216/abliterix/pull/95

### 2026-07-30 — PR #95 empty-adapter guard + 作者第三轮补丁

- **类型**: fix（上游 PR 分支）
- **摘要**: merge 后禁止空 LoRA 导出；DEQUANT_CACHE_MAX_BYTES 共享；_build_quant_config 去掉死参数；非 TTY Ctrl-C 返回 None；测试加强。
- **提交**: `f540863` on `pr/bnb-rocm-moe-stability`
- **与上游关系**: https://github.com/wuwangzhang1216/abliterix/pull/95

### 2026-07-29 — PR #95 native bf16 判定

- **类型**: fix（上游 PR 分支）
- **摘要**: 按作者第二轮意见：`_bf16_compute_supported()`（ROCm 或 sm≥8），compute 与 promote 共用；测试改 patch 该 helper。
- **提交**: 见 `pr/bnb-rocm-moe-stability` 最新 tip
- **与上游关系**: https://github.com/wuwangzhang1216/abliterix/pull/95

### 2026-07-29 — PR #95 按作者 review 修改

- **类型**: fix（上游 PR 分支 `pr/bnb-rocm-moe-stability`）
- **摘要**: 采纳 wuwangzhang1216 审查意见：export_adapter、model.text_only、bf16 硬件回退、promote 去 gate、_cache_dequant、非 TTY 全覆盖、行为单测。
- **提交**: `862f4f6`；PR 评论已回复。
- **与上游关系**: https://github.com/wuwangzhang1216/abliterix/pull/95

### 2026-07-27 — PR #95 CI/tests 补全

- **类型**: docs / test（上游 PR 分支）
- **摘要**: 在 `pr/bnb-rocm-moe-stability` 上 `ruff format` + 新增 `tests/test_bnb_moe_load_fixes.py`；本地 ruff/ty 通过；聚焦 pytest 26 passed。PR 评论已贴硬件背景与检查结果。
- **涉及**: `pr/bnb-rocm-moe-stability` @ `405dfe8`；PR https://github.com/wuwangzhang1216/abliterix/pull/95
- **与上游关系**: 已 push 到 PR 分支（fork）

### 2026-07-27 — 建立分支日志

- **类型**: docs  
- **摘要**: 新增本文件，约定此后每次改动同步更新。  
- **涉及**: `BRANCH_LOG.md`  
- **与上游关系**: 仅 fork  

### 2026-07-27 — 恢复 staged trial 筛查接线

- **类型**: fix  
- **摘要**: merge 上游后 optimizer 曾只走 compliance 路径；重新接入 prescreen / validation KL / generation health / thinking leak；默认 flag 仍为关。  
- **涉及**: `optimizer.py`, `eval/stage_evaluator.py`  
- **提交**: `31336a3`  
- **与上游关系**: 仅 fork  

### 2026-07-27 — 合入上游 v1.12.1

- **类型**: merge  
- **摘要**: `merge upstream/master` 到 sc117-base；interactive/optimizer 清理与 damage metric 取上游；保留 bnb/dequant/缓存/配方。  
- **涉及**: 全树大量上游变更 + 冲突解决  
- **提交**: `09c6e53`  
- **与上游关系**: 已对齐 `76a7a31` (v1.12.1)  

### 2026-07-26 前后 — Laguna / ROCm 生产补丁首批提交

- **类型**: fix / feat / chore  
- **摘要**:  
  - bnb 稳定性、dequant 上限、interactive 参数补全、非 TTY 菜单  
  - 磁盘缓存 + staged eval 模块与 settings  
  - Laguna/Agents 配方与 handover  
- **提交**: `63778d9`, `28ff4c5`, `453858b`  
- **与上游关系**: 核心修复另拆 `pr/bnb-rocm-moe-stability` → PR #95  

### 更早 — Qwen3.5 MoE ROCm 配方

- **类型**: feat / docs  
- **摘要**: Qwen3.5-MoE / Agents 加载策略、LoRA-only 保存雏形、相关 configs 与文档。  
- **提交**: `04f318e`, `10c0e5f`, `5b8e903`  

---

## 6. 维护检查清单（改代码前勾一下）

1. [ ] 改动是否只影响 SC117 配方/缓存/筛查？→ 记「仅 fork」  
2. [ ] 是否通用 bugfix？→ 考虑 `pr/*` + 上游 PR，并在本日志 §4/§5 记链接  
3. [ ] 是否合入/rebase 上游？→ 更新「最近对齐上游」表项与 §5 merge 条目  
4. [ ] **本文件 §5 是否已追加条目？** → 没有则不要 push  

---

## 7. 常用命令

```bash
# 对齐上游（在 sc117-base 上）
git fetch upstream
git merge upstream/master   # 或 rebase；冲突解决后更新本日志

# 推送自用分支
git push origin sc117-base

# 干净上游 PR 分支示例
git checkout -B pr/some-fix upstream/master
# … 只挑通用补丁 …
git push -u origin pr/some-fix
gh pr create --repo wuwangzhang1216/abliterix --base master --head Scorp1o117:pr/some-fix
```
