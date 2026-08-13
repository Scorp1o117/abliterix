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
| 最近对齐上游 | **v1.12.1**（`76a7a31`，merge `09c6e53`）；上游 `master` 已前进到 `05c3e87`，尚未再 merge |
| 本日志最近更新 | 2026-08-13 (commit Ling campaign library + frozen deliverables) |

---

## 0. 2026-08-13 — Ling campaign library + frozen deliverables

Committed the in-tree Ling-3.0-flash work that had been sitting unstaged
on `sc117-base`.

**Library (replay + export):** Bailing `layer.attention` steerable
discovery (`o_proj` / `dense` / MLA `*_b_proj`); MoE router expert-id
dtype probe + `bincount` (engine + SAFEx); wrap-time `lora_B` zero-init;
transformers 5.x `is_torch_fx_available` shim; multi-objective TPE prune
returns `(inf, inf)` instead of `TrialPruned`; optional concept-gated
angular / canonical batch / prefix-retry runtime (not mergeable);
`apply_trial_artifact` fills missing steering fields from defaults so
v6 t21 can export.

**Frozen deliverables:** `configs/ling30_flash_rocm_v6_stack.toml` +
`run_ling30_v6_interactive.sh` (mergeable best: KL 0.093 / 17.3%);
`scripts/merge_ling30_v6_t21_lora.py`; project logs. Remaining
`configs/ling30_*` / `run_ling30_v*.sh` kept as lab replay, not for
upstream. `logs/` and `checkpoints_ling30_*` stay untracked.

**Upstream PRs (separate `pr/*` from `upstream/master`, not this
branch):** Bailing steerable discovery; MoE router dtype + bincount.
Do not fold those into #95.

Follow-up commit on this branch adds the frozen v6 recipe, merge
script, project logs, and the remaining Ling lab configs/runners.

---

## 0b. 2026-08-12 — Ling rank-k angular runtime support / v41 probe

- `core/steering.py`: angular removal now accepts either one direction or a
  QR-orthonormalized rank-k subspace while preserving activation norm; the
  per-layer runtime path now indexes stacked vectors correctly.
- `settings.py`: HF `angular` and sign-agnostic `concept_gated_angular` permit
  rank-k recipes; adaptive/spherical/vector-field restrictions remain.
- Tests cover full rank-2 projection removal, identity at zero strength, and
  configuration guards. A duplicate public-API runtime guard found by the
  first v41 startup was removed before generation, with an end-to-end hook
  installation regression added; focused suite: 53 passed.
- Added `run_ling30_v41_rank2_subspace_probe.sh` and v41 watcher entry. The
  experiment uses only train-derived directions and keeps the v36/v40
  canonical 60-prompt protocol; no eval prompt is used for fitting.
- v41 completed at 13/60 (10 gate-on, 3 gate-off; 95.17% active), worse than
  v40 rank-1 at 10/60 (8 gate-on, 2 gate-off; 96.48% active). It fixed five
  prior failures but introduced eight new ones. Post-run audit found the
  concept scorers were retrained and max-gap auto-selection changed decision
  layer 18→19, so this is not a strict direction-only A/B. Full evaluation is
  withheld pending a frozen-scorer, fixed-layer paired rerun.
- Added direction-independent concept-scorer caching with explicit training
  seeding. The cache stores exact per-layer weights and ignores steering rank
  while tracking its prompt/trajectory/training provenance. Added v42a/v42b
  runners for a fixed-layer-18, shared-gate rank-1/rank-2 canonical pair.
- v42a reproduced v40's gate decisions exactly but moved refusal 10→20 because
  the generic rank-k matrix formula changed rank-1 floating-point operation
  order. Restored the historical scalar path verbatim for rank 1 and added an
  fp16 bitwise regression; only rank-k now uses QR subspace math.
- Under the exact frozen gate, v42b rank-2 improved the matrix-formula control
  20→14 by fixing 12 outcomes while regressing 6. It remains worse than the
  historical scalar rank-1 result, so v43 was added to reproduce that 10/60
  baseline with exact scorer and operation-order caches before routing work.
- v43 restored the scalar frozen-gate control to 10/60. Added v44's
  train-derived rank-2 direction router: a batch-of-one final-prefill prepass
  chooses one direction by maximum absolute projection, caches the anonymous
  route ID with the gate, and applies only that legacy scalar path through
  decode. Route telemetry is persisted without prompt text.
- v44 regressed to 18/60 and routed 56/60 samples to direction 0. Cache audit
  showed that multi-direction post-processing had moved that nominal mean
  direction to only 0.9716 average cosine with the exact legacy vector. Added
  v45 cache composition: exact legacy primary plus a train-derived secondary
  reorthogonalized against it, preserving rank-2 provenance metadata.
- v45 still regressed to 14/60 (one fix, five new failures), stopping
  max-projection routing. Added fixed rank-k direction selection/search and a
  v46 two-trial counterfactual probe on train[800:900] only, so primary vs
  secondary outcome complementarity can be measured in one model load before
  investing in a learned router.
- v46 found 17/60 refusals for the exact primary and 59/60 for the orthogonal
  secondary. The secondary fixed only one primary failure and introduced 43;
  even an oracle router bottoms out at 16/60, so learned routing is stopped.
- Added training-only failure-conditioned vector variants: anonymous refusal
  and compliance row indices define a primary-orthogonal residual direction,
  then categorical trials evaluate exact-primary-plus-alpha blends while each
  remains on the legacy scalar angular path. Added v47 with five canonical
  matched variants and focused configuration/vector tests (29 passed).
- v47's pre-normalised control reached 9/60, but adding the orthogonal
  failure residual monotonically regressed across alpha 0.25/0.5/1/2 to
  10/17/38/58. Outcome overlap likewise showed net regressions, so the
  failure-conditioned branch is stopped. Fixed its control to preserve the
  source tensor bitwise and isolated the accidental pre-normalisation as a
  repeated v48 formal-prescreen probe (31 focused tests passed).
- v48 repeated the byte-identical pre-normalised primary twice on the formal
  canonical subset; both runs were exactly 16/60 with identical indices and
  96.48% gate activity, worse than the legacy 10/60 control. Stopped that
  numerical recipe. Added v49 to test the opposite sign of the train-only
  failure residual because v47's positive-alpha regression was monotonic.
- v49 reproduced the exact primary at 17/60 and found a narrow alpha=-0.125
  improvement to 10/60 (12 fixes, 5 regressions); larger negative blends
  regressed to 21/31/47. Added an independent calibration prompt split so v50
  can derive the candidate from train[800:900] while evaluating train[900:]
  twice with byte-identical tensors and no formal-residual leakage (32 tests).
- v50 produced 19/60 refusals in both byte-identical formal repetitions, with
  identical refusal indices and 96.48% gate activity. This rejects the v49
  gain as calibration overfit and stops the failure-conditioned direction
  branch.
- Added an opt-in concept-gated angular over-rotation path. Legacy behavior
  still clamps at the 90-degree tangent; the new mode permits fractions up to
  2.0 while preserving activation norm. Optuna can compare the mode per trial,
  and focused runtime/configuration coverage passes 48 tests. Added v51 to
  compare the legacy clamp with true 1.10/1.20/1.40 gated over-rotation only on
  train[800:900], with fixed primary direction, scorer, layer, and canonical
  batching before any formal evaluation.
- v51 reproduced the legacy control at 17/60 and yielded 15/13/14 for true
  1.10/1.20/1.40 over-rotation, all at the identical 98.39% gate activity.
  The best 1.20 point fixed eight control failures but introduced four; 1.40
  rebounded, so no formal evaluation was run.
- Added an `all`/`prefill`/`decode` over-rotation phase selector while retaining
  the legacy 90-degree clamp outside the chosen phase. Added v52 to isolate the
  train-only max-1.20 gain between prompt-state initialization and subsequent
  autoregressive trajectory, with an all-phase reproduction as control.
- v52 yielded 18/15/13 for prefill-only/decode-only/all-phase over-rotation,
  all at 98.39% gate activity; the all-phase point exactly reproduced v51.
  Neither isolated phase improved on the combined intervention, so token-window
  phase slicing is stopped and no formal evaluation was run.
- Anonymous metadata audit found the 13 remaining all-phase failures span eight
  safety categories and 12/13 are gate-on, ruling out a single topic-specific
  direction or classifier recall as the main bottleneck. Added an opt-in
  `linear_projection` gated geometry that subtracts the direction component
  without angular radial renormalization; default remains norm-preserving
  angular. Added v53 to compare angular 1.20 against linear 1.00/1.20/1.40 on
  the isolated train-only canonical subset.
- v53 reproduced angular 1.20 at 13/60; linear projection 1.00/1.20/1.40 gave
  14/11/11 at essentially identical gate activity. The linear strengths share
  only five failures (oracle 5/60), so a sample-level strength route may be
  useful even though the scalar geometry axis plateaued.
- Added anonymous layer-18 absolute-cosine telemetry to the canonical global
  prepass. Only source-index-to-float mappings reach the journal; prompt text
  and hidden states remain internal. Added v54 to reproduce linear 1.20 and
  test offline whether a one-threshold 1.20/1.40 router can approach the
  train-only oracle before adding any runtime routing machinery.
- v54 exactly reproduced linear 1.20 at 11/60 with 98.38% gate activity and
  complete anonymous scalar telemetry. Exhaustive one-threshold routing reached
  only 9/60 at a single isolated cut; bootstrap rule orientation split 594/406
  and mean out-of-bag refusal was 20.8%. The absolute-cosine router is stopped
  as unstable calibration fit; no formal split was touched.
- Added v55 anonymous telemetry for the signed primary-direction cosine and
  continuous global-gate classifier score alongside the existing absolute
  cosine. Canonical prepass caches and prescreen journals contain only
  source-index-to-scalar mappings. Focused gate/evaluator tests pass 26 cases;
  v55 replays linear 1.20 before any shallow offline routing analysis.
- v55 exactly reproduced v54's 11/60 refusal set and 98.38% gate activity.
  Signed cosine added no information because all 60 signs were positive; gate
  score stumps reached only 10/60. A depth-2 rule fit 8/60 in sample, but both
  stump and tree bootstrap OOB refusal stayed near 20.8%, so all prefill-feature
  strength routing is stopped.
- Added v56 formal fixed A/B for a refusal-triggered 1.20-to-1.40 fallback.
  It uses no prompt feature or fitted threshold: the existing output refusal
  detector alone decides whether a second generation is needed. Two fixed
  trials on untouched train[900:] measure the formal failure intersection
  before any runtime retry implementation.
- Corrected the v56 cache wiring before trial execution: the first launch
  detected the calibration-split baseline as stale and was interrupted during
  recomputation. v56 now hard-links the validated v36 formal canonical
  baseline while retaining v55 steering/scorer caches, avoiding both a stale
  comparison and an unsafe write through the calibration baseline hard link.
- v56 formal A/B completed with identical 58/2 gate sets: linear 1.20 and 1.40
  yielded 19/60 and 15/60 refusals. Their failure intersection was 10/60;
  1.40 fixed nine 1.20 failures but introduced five. Because the predefined
  fallback gate was at most 8/60 and the calibration oracle did not transfer,
  no runtime retry path or full evaluation was implemented.
- Added `compose_exact_primary_with_harmfulness`: slot 0 stays a bitwise copy
  of the validated legacy primary; slot 1 is Zhao PCA-1 of centered train
  target residuals, orthogonalised to that primary (helpfulness projection
  applies to the secondary only). Focused harmfulness tests cover exactness,
  orthogonality, and refusal-slot non-recomputation.
- Added v57 train-only counterfactual: exact primary vs Zhao harmfulness on
  the same frozen gate / layer 18 / canonical `train[800:900]` 60 as v46.
  This is a new direction family, not the SVD residual already rejected.
  Formal `train[900:]` is unused until the secondary clearly complements the
  17/60 primary control. Runner: `run_ling30_v57_harmfulness_counterfactual.sh`.
- v57 completed: exact primary reproduced v46's 17/60 with identical refusal
  indices; Zhao harmfulness was 60/60, fixed 0 primary failures, and introduced
  43. The oracle lower bound remains 17/60. Stopped the harmfulness / second
  static-direction family without sequential removal or formal evaluation.
- Added optional `dump_steered_prefill_residuals`: during the global
  single-sample prepass the concept-gated hook stores post-hook final-token
  residuals at every hooked layer, and the prescreen writer saves an anonymous
  source-index tensor cache. Focused evaluator coverage includes the dump
  path. v58 replays exact primary on train[800:900] only to collect that cache
  for a later steered residual peel; formal eval is unused.
- v58 reproduced the exact 17/60 primary control and wrote a 60×42 residual
  dump with 23 hooked layers. Added
  `compose_exact_primary_with_steered_peel` to keep slot 0 bitwise exact and
  put the gate-on steered leftover in slot 1. v59 is the formal
  `train[900:]` A/B of those two directions; source 18 stays out of the peel
  labels because it is a gate miss.
- v59 formal A/B completed with an exact v43 primary control (10/60, identical
  refusal indices and gate-off `[53, 0]`). The steered peel was 60/60, fixed
  nothing, and introduced 50 failures; the oracle lower bound remains 10/60.
  Stopped the peel / second-static-direction family. Sequential application
  is not justified: a direction with zero standalone control would only
  destroy the 10/60 winner.
- Added anonymous refusal-onset labels (prefix vs late, coarse family only)
  and a same-batch decode retry that bans the first-pass prefix token IDs
  for gate-on prefix refusals. v60 is a two-trial formal A/B of exact
  primary versus that retry. If retry does not beat 10/60, stop mechanism
  probes and keep the current best recipe.
- v60 formal A/B reproduced the 10/60 primary control. Only 3/10 remaining
  refusals were prefix refusals; 7/10 were late. Same-batch prefix retry
  reached 9/60 by fixing index 51 and introducing none. Two gate-on prefixes
  were retried and both lost the prefix, but 98 became a late refusal. This
  misses the ≤8/60 gate. Mechanism probes are stopped. Frozen delivery:
  v36 matched-canonical runtime (KL 0.0000 / 27/100) and v6 t21 mergeable
  LoRA (KL 0.093 / 17.3%).
- `apply_trial_artifact` now fills steering fields that did not exist when
  an older journal was written, using library defaults instead of the live
  TOML. Unknown future fields are still rejected. This unblocks exporting
  Ling v6 trial 21 as a PEFT adapter.
- Added a shard-streaming merge for that adapter into
  `/run/media/s117/OS/Models/Ling-3.0-flash-abliterix`. The 237 GiB bf16
  base cannot be held in RAM, so each safetensor file is merged on disk.

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

### 2026-08-10 — fix/experiment: steering cache provenance + Ling v11 配对条件方向

- **类型**: fix / test / experiment / docs
- **摘要**: cache key 补全 prompt system/prefix/suffix、OT/multi-direction/SRA/SOM/SAE/RDO/iterative 参数；fresh cache 在保存前不再删除 residual states；routing 搜索全关时跳过无效的 1600-prompt MoE profiling；新增 provenance/skip 单测。停止并归档 v10（32 complete + 1 pruned），量化确认有效层 OT/mean cosine=0.999569。新增并启动 v11 同 harmful prompt、refusal/compliance system condition 配对方向 12-trial 探针，只复用 baseline、不复制 steering cache；fresh cache 721,707,546 bytes 且有效层相对旧 mean 平均夹角 77.73°。
- **涉及**: `src/abliterix/cli.py`, `tests/test_steering_cache.py`, `configs/ling30_flash_rocm_v11_paired.toml`, `run_ling30_v11_interactive.sh`, `run_ling30_v10_interactive.sh`, `docs/LING30_FLASH_PROJECT_LOG.md`
- **与上游关系**: cache provenance/残差保存为通用修复；Ling 配方、runner 与项目记录仅 fork

### 2026-08-10 — result: Ling v11 12/12 high + batch runner 修复

- **类型**: result / fix / docs
- **摘要**: v11 全部 12 trial 均 prescreen 30/30 high、objective `(inf, inf)`，停止 prompt-only system-condition 方向；结论限定为该方向编码了新 mode 几何但不控制拒答，后续配对应进入 response trajectory。修复 v10/v11 `--batch` 未传 `--non-interactive`、搜索结束仍进 TUI 的 runner bug。
- **涉及**: `run_ling30_v10_interactive.sh`, `run_ling30_v11_interactive.sh`, `docs/LING30_FLASH_PROJECT_LOG.md`, `BRANCH_LOG.md`
- **与上游关系**: Ling runner 与项目记录仅 fork

### 2026-08-10 — feat: teacher-forced response-pair residual extraction + Ling v12

- **类型**: feat / test / experiment / docs
- **摘要**: 新增 response-pair 模式，在完全相同 prompt/system 的 assistant boundary 后 teacher-force compliance/refusal continuation，并 mean/last pool response-token residual；配置校验强制 prompt source 对齐，cache provenance 纳入 continuation 参数。新增并启动 Ling v12 12-trial 配方与 runner；方向相对 v9 mean 平均夹角 74.04°，前三点 prescreen 28/22/20，t3 首次降至 15/30 borderline 并进入完整评估。
- **涉及**: `src/abliterix/core/engine.py`, `src/abliterix/settings.py`, `src/abliterix/cli.py`, `tests/test_hidden_state_extraction.py`, `tests/test_response_pair.py`, `configs/ling30_flash_rocm_v12_response_pair.toml`, `run_ling30_v12_interactive.sh`, `docs/LING30_FLASH_PROJECT_LOG.md`
- **与上游关系**: response-pair 提取/校验为通用实验功能；Ling 配方与记录仅 fork

### 2026-08-10 — result: Ling v12 正常完成，response-pair 未破前沿

- **类型**: result / docs
- **摘要**: v12 正常完成 12/12，非崩溃或外部停止；4 个 borderline 点进入完整评估。最低拒为 t7 47/98、KL 0.245，最低 KL 为 t8 0.174、拒 63/98，均显著差于 v6/v9。固定短 continuation trajectory 有弱拒答控制信号但无 Pareto 价值，停止且不追加 trial。
- **涉及**: `checkpoints_ling30_flash_v12_response_pair/`, `docs/LING30_FLASH_PROJECT_LOG.md`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork 实验结果

格式建议：

```text
### YYYY-MM-DD — 简短标题
- **类型**: fix | feat | chore | merge | docs
- **摘要**: …
- **涉及**: path1, path2
- **与上游关系**: 仅 fork / 已 cherry-pick 到 pr/* / 已合入上游 #N
```

---

### 2026-08-09 — feat: Ling v10 Optimal Transport 方向（换向量族冲双目标）

- **类型**: feat / experiment（仅 fork）
- **摘要**: v1–v9 mean+MPOA 强度搜索边际耗尽（最佳 dual-ish v6 t21 KL 0.093/拒 17%），v9 抬天花板仅 Pareto 平移。v10 换 **optimal_transport** 向量族（PCA-Gaussian OT，捕获均值+协方差结构差异），保留 MPOA full+r3 写路径与 per-layer 向量。强度带宽于 v9（o∈[1.5,3.2] d∈[1.2,3.0]），因 OT 方向可能需要不同幅度。`projected_abliteration=true` 保留有用性信号。40 trial 探针预算，若 trial 20 前无双目标进展则停。配置 `configs/ling30_flash_rocm_v10_ot.toml` / `run_ling30_v10_interactive.sh`。
- **涉及**: `configs/ling30_flash_rocm_v10_ot.toml`, `run_ling30_v10_interactive.sh`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork 配方

### 2026-08-09 — chore: 停 v9 + 终局写入项目记录

- **类型**: chore / docs（仅 fork）
- **摘要**: kill v9（17/60 trial）。有限 14 点：最佳 dual-ish **t12 KL=0.111/拒14.3%**；最低拒 **t10 13.3%@KL0.160**；最低 KL t6 0.086/拒64%。**双目标 0 命中**。相对 v6 t21（0.093/17%）仅沿 Pareto 平移。结论：mean+MPOA 强度搜索边际耗尽。已写入 `docs/LING30_FLASH_PROJECT_LOG.md` §0/§4.2/§4.3/§5/§7/§8。
- **涉及**: `docs/LING30_FLASH_PROJECT_LOG.md`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork

### 2026-08-09 — docs: Ling 项目总记录

- **类型**: docs（仅 fork）
- **摘要**: 新增 `docs/LING30_FLASH_PROJECT_LOG.md`：汇总 v1–v9 全部方法/证伪、代码修改索引、全局 Pareto、工件地图与运行约定；与 `ling30_flash_abliterix_modifications.md`（早期补丁细节）互补。
- **涉及**: `docs/LING30_FLASH_PROJECT_LOG.md`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork

### 2026-08-09 — feat: Ling v9 数据驱动最优搜索（冲双目标）

- **类型**: feat / experiment（仅 fork）
- **摘要**: 停 v8；按 v1–v8 证据写最优**搜索**配方（非导出）。保留唯一低 KL 路径 MPOA `full+r3`；`fixed_vector_scope=per layer`；**禁止** auto-disable down；强度收在 v6 dual-ish 赢家附近并**抬高天花板**（o∈[2.0,2.9] d∈[1.5,2.7]，因 t21/t15 顶在旧上限 2.5）；`min_weight_frac_max=0.50`；关 cliff/SRA/expert；batch16、**60 trial**、`checkpoints_ling30_flash_v9_optimal`。参考：v6 t21 KL=0.093/拒17%。目标仍 KL≤0.05 且拒≤10%。
- **涉及**: `configs/ling30_flash_rocm_v9_optimal.toml`, `run_ling30_v9_interactive.sh`, `watch_ling30.sh`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork 配方

### 2026-08-09 — feat: Ling v8 回到 Abliterix 默认方法（弃 Ornith/MPOA）

- **类型**: feat / experiment（仅 fork）
- **摘要**: 用户要求停用 Ornith 类消融（`weight_normalization=full` / MPOA rank-3、v4–v7 窄带与 cliff）。v8：`weight_normalization=none`、mean+LoRA、`orthogonal_projection=true`、**关** projected/winsorize/cliff/SRA/expert；`strength_range=[0.8,1.5]`（settings 默认）；o_proj+down_proj 共用该带；batch16、40 trial、`checkpoints_ling30_flash_v8_default`。交互：`./run_ling30_v8_interactive.sh`。
- **涉及**: `configs/ling30_flash_rocm_v8_default.toml`, `run_ling30_v8_interactive.sh`, `watch_ling30.sh`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork 配方

### 2026-08-09 — feat: Ling v7 cliff-head 探针 + bnb 安全 o_proj 列消融

- **类型**: feat / experiment（仅 fork 配方；cliff_head bnb 路径可上游化）
- **摘要**: v4–v6 经典 mean+MPOA 已确认低 KL 盆地但双目标 plateau（~KL 0.09 / 拒 16–21% 量级）。v7 换 **B 方案**：`cliff_head_ablation=true`（top 4% / strength 0.75）固定预消融 + **仅 `attn.o_proj`** 轻 LoRA（[0.3,1.2]）、关 down_proj/expert/SRA、MPOA full+r3、batch16、**10 trial**、`checkpoints_ling30_flash_v7_cliff`。`cliff_head.py`：bnb `Params4bit` 走 dequant→edit→requant（禁止原地 float 改 packed uint8）；identify 用 dequant；优先 `v_head_dim`；列宽可从 `in_features//num_heads` 推导。交互：`./run_ling30_v7_interactive.sh`（`--batch` 无人值守）。
- **涉及**: `src/abliterix/cliff_head.py`, `configs/ling30_flash_rocm_v7_cliff.toml`, `run_ling30_v7_interactive.sh`, `watch_ling30.sh`, `docs/ling30_flash_abliterix_modifications.md`, `BRANCH_LOG.md`
- **与上游关系**: bnb cliff 路径可上游化；配方仅 fork

### 2026-08-08 — feat: Ling v5 lowref 收窄搜索（batch16 + per-layer）

- **类型**: feat / experiment（仅 fork）
- **摘要**: v4 确认低 KL 盆地（0.03–0.11）但拒 31–58%，双目标未近。v5：`fixed_vector_scope=per layer`、o∈[1.5,2.2]/d∈[1.4,2.1]、`min_weight_frac_max=0.55`、禁 auto-disable down、batch16、20 trial、`checkpoints_ling30_flash_v5_lowref`。交互：`./run_ling30_v5_interactive.sh`（强制 `--no-non-interactive`）。
- **涉及**: `configs/ling30_flash_rocm_v5_lowref.toml`, `run_ling30_v5_interactive.sh`, `watch_ling30.sh`
- **与上游关系**: 仅 fork

### 2026-08-08 — feat: Ling v4 抄 Ornith classic (MPOA) + batch8 输出探针

- **类型**: feat / experiment（仅 fork）
- **摘要**: Ornith-35B 经典 Heretic journal 证明可进低 KL 区（t62 KL=0.028/11拒）。Ling 未跑过同方。v4：`weight_normalization=full` + `full_norm_lora_rank=3`、仅 o_proj+down、strength [0.5,2.1]、经典 mean、关 SRA/pair/expert、**10 trial**。同时 **batch_size=8 + print_responses** 复验历史「batch>1 退化」记录（文档称 ulp→router 放大；曾有 continue 复活 batch=8 事件）。checkpoint `checkpoints_ling30_flash_v4_ornith`。
- **涉及**: `configs/ling30_flash_rocm_v4_ornith.toml`, `run_ling30_v4.sh`, `watch_ling30.sh`, `BRANCH_LOG.md`
- **与上游关系**: 仅 fork 配方

### 2026-08-08 — fix: MoE + harmfulness_pair 假冲突崩溃

- **类型**: fix
- **摘要**: v3 trial2 选 `steering_variant=harmfulness_pair`（3-D 多方向）时，`apply_steering` 因 `safety_experts is not None and routing_config is not None` 直接抛错退出——即便 `max_suppress=0` / bias=0 / ablation=0。修：① routing 全关时不构造 `ExpertRoutingConfig`；② 仅在 n_suppress/ablation/bias 非零时禁止多方向+MoE routing。
- **涉及**: `src/abliterix/core/steering.py`, `src/abliterix/optimizer.py`
- **与上游关系**: 可上游化

### 2026-08-08 — pivot: Ling v3 SRA 方向（v2 Laguna 配方证伪）

- **类型**: feat / experiment（仅 fork）
- **摘要**: v2 跑满 ~25 trial 后结论：**o_proj 已接入仍不够**。低拒绝可达（prescreen 2–8/30）但一律 KL 剪枝；有限点 KL 仍 **0.40–0.47**、拒绝 30–57%，与 v1 纯 down_proj 底盘相同 → Laguna 式 strength/expert 未打开低 KL 盆地。Safety expert 风险分极弱（top ~0.05–0.13）。换方向：`vector_method=sra`（相对 benign concept atoms 谱清洗拒绝方向）、关 expert routing、强度收至 [0.3,2.5]、`auto_disable` down_proj、`search_harmfulness_direction`、`weight_normalization=pre`。**探针 `num_trials=10`**（不接近目标即再 pivot，不空烧）。配置 `configs/ling30_flash_rocm_v3_sra.toml` / `run_ling30_v3.sh`。另：KL 超 prune 时写入 `trial.user_attr.kl_divergence` 便于事后分析。
- **涉及**: `configs/ling30_flash_rocm_v3_sra.toml`, `run_ling30_v3.sh`, `src/abliterix/optimizer.py`, `docs/ling30_flash_abliterix_modifications.md`, `BRANCH_LOG.md`
- **与上游关系**: prune 时记 KL 可上游化；配方仅 fork

### 2026-08-07 — fix: Bailing/Ling steerable 发现 + Ling v2 搜索配方

- **类型**: fix + feat（仅 fork 配方）
- **摘要**: Ling-3.0-flash v1 消融失败根因：`BailingMoeV3DecoderLayer` 的 attention 在 `layer.attention`（非 `self_attn`），MLA 输出投影名是 `dense`、KDA 才是 `o_proj`。`steerable_modules` 只发现 `mlp.down_proj` → 纯 down_proj LoRA 跷跷板（有限点 KL 0.46 / 拒绝 43%；低拒绝 trial 全被 KL 剪枝）。对照 Laguna-S-2.1 成功点均含 **强 o_proj + down_proj + expert suppress/ablation**（KL≈0.002–0.01、拒绝 7–9%）。修法：在 `engine.steerable_modules` 注册 `attention.o_proj` / `attention.dense` → `attn.o_proj`，以及 KDA q/k/v 与 MLA q_b/kv_b（默认 disabled）。新增 `configs/ling30_flash_rocm_v2.toml` + `run_ling30_v2.sh`（Laguna 风格强度/routing/gaussian/discriminative，checkpoint 独立目录）。目标验收：KL≤0.05 且拒绝≤10%。
- **涉及**: `src/abliterix/core/engine.py`, `configs/ling30_flash_rocm_v2.toml`, `run_ling30_v2.sh`, `docs/ling30_flash_abliterix_modifications.md`, `BRANCH_LOG.md`
- **与上游关系**: 引擎 discovery 可上游化（Bailing/Ling 家族通用）；配方与 runner 仅 fork

### 2026-08-05 — fix: LoRA 保存走 export_adapter()，补上空 adapter guard 缺口

- **类型**: fix（PR #95 后续）
- **摘要**: 时序复现：先尝试 merged 导出 → `export_merged()` 执行 `merge_and_unload()` 把 LoRA 折叠进基座并移除 lora 参数（`needs_reload=True`），随后 save_pretrained 若失败（如官方 generation_config 校验），再选"保存 LoRA"时 `_save_lora_adapter_locally` 直接 `engine.model.save_pretrained()`——**绕过** PR #95 `f540863` 在 `export_adapter()` 里加的空 adapter guard，空壳 PeftModel 静默写出 0 张量（40 字节 `{"__metadata__":{"format":"pt"}}`）safetensors。LFM2.5-2.6B trial32 实测导出 40 字节空文件。修法：`_save_lora_adapter_locally` 改调 `engine.export_adapter(save_dir)`，guard 生效（`needs_reload` 或零 lora_ 参数时报错并提示重新选 trial）。
- **涉及**: `src/abliterix/interactive.py`
- **与上游关系**: PR #95 同源修复的补丁（上游 f540863 只覆盖 export_adapter 路径）

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
## 2026-08-10 — Ling v13 adaptive runtime causal probe

- Added `configs/ling30_flash_rocm_v13_adaptive_runtime.toml` and
  `run_ling30_v13_interactive.sh` for an 18-trial, decoder-block-output
  adaptive-angular diagnostic using the validated v9 mean direction.
- Disabled duplicate component dimensions because runtime angular steering
  consumes one block-output profile; added explicit stop/continue criteria.
- Documented the v13 rationale and launch state in
  `docs/LING30_FLASH_PROJECT_LOG.md`.
- Validated the runtime hook path (4 focused tests passed), regenerated a
  schema-v2 688MB steering cache after rejecting the legacy v9 cache key, and
  recorded the first v13 prescreen result (20/30 high).
- Extended `watch_ling30.sh` with v11-v13 targets and made v13 the default.
- Added architecture-aware runtime hook sites (KDA, MLA, MLP, shared expert),
  Optuna categorical site search with cleanup-safe config restoration, and
  focused tests (9 runtime-hook tests passing).
- Runtime-only modes now skip unused PEFT adapter initialisation; added the
  four-seed v14 matched site-sweep config/runner and v14 watcher support.
- Added a `post_attention_residual` pre-hook site at the input of the decoder
  block's post-attention norm, preserving residual-space coordinates for the
  next Ling causal comparison.
- Added the one-seed v15 post-attention-residual matched probe, runner, watcher
  target, and project-log decision rule.
- Added `concept_gated_angular`: per-layer ConceptScorer probability gates the
  existing positive-alignment angular removal token by token at the selected
  runtime site; wired scorer training and runtime-only/export contracts.
- 2026-08-10: Added Ling v16 `concept_gated_angular`: GPU-trained per-layer harmful-state classifiers gate the proven adaptive decoder-block hook; added gate-rate telemetry, fixed runtime/export/UI wiring, regression coverage, config, runner, watcher entry, and project-log handoff.
- 2026-08-10: Ling v16 threshold=0.5 yielded 17/30 refusals at 14.49% classifier-gate activation. Added per-trial gate-threshold search/restoration, lower-threshold matched seeds, watcher telemetry, and low-precision scorer inference to cut runtime VRAM/latency.
- 2026-08-10: Ling v16 threshold sweep completed: 0.05=41.65% gate/10 refusals, 0.15=27.18%/14, 0.30=18.76%/15. Added benign-eval gate telemetry and v17 full-validation recipe for the sole passing threshold 0.05.
- 2026-08-11: Ling v17 full result was KL 0.0771 but 42/100 raw refusals (validation KL 0.1000); benign gate 20.79% vs harmful-prescreen 41.65% confirmed selectivity but exposed 30-prompt bias. Added prompt-latched concept gate scope to align final-prompt-token scorer training with autoregressive use.
- 2026-08-11: Added Ling v18 full-100 harmful probe for `concept_gate_scope="prompt"`, avoiding the v17 30-prompt extrapolation failure.
- 2026-08-11: Ling v18 prompt-latched threshold=0.5 yielded 19.78% gate activation and 62/100 refusals. Added v19 low-threshold full-100 terminal probe with independent journal.
- 2026-08-11: Ling v19 prompt-latched threshold=0.05 yielded 24.47% gate activation and 55/100 refusals. Stopped the final-prompt ConceptScorer gate branch per its predeclared rule; remaining credible gate work requires continuation-token trajectory training data.
- 2026-08-11: Added Ling v20 response-trajectory token gate while preserving the proven v13 prompt-mean direction: sampled continuation-token extraction, prompt-grouped 80/20 train/validation split, held-out scorer metrics and pre-generation guard, full-100 config/runner/watcher, and regression coverage (25 focused tests passed).
- 2026-08-11: Ling v20 trajectory scorer reached 99.49% grouped held-out accuracy (0.00% compliance vs 98.98% refusal active), but free generation yielded 21.27% gate activation and 45/100 refusals. Added matched v21 threshold=0.05 full-100 terminal probe and watcher target.
- 2026-08-11: Ling v21 trajectory threshold=0.05 yielded 49.34% free-generation gate activation and 39/100 refusals (held-out 93.51%, 12.98% compliance vs 100% refusal active). Stopped the fixed-continuation classifier branch: lowering threshold mostly approaches always-on v13 without a new dual point; next gate data must use model-generated benign/harmful early trajectories.
- 2026-08-11: Added Ling v22 generated-prompt trajectory gate: deterministic 200/class free-running response cache, row-specific early-token residual extraction, final-prefill x4 plus first-8 decode training mix, separate grouped mixed/prefill guards, full-100 config/runner/watcher, and focused regression coverage (27 passed).
- 2026-08-11: Ling v22 held-out separation was strong (mixed 94.65%, final-prefill 91.67%) but token-scope free generation collapsed to 8.72% activation and 63/100 refusals, exposing intervention-induced gate self-shutoff. Added matched v23 prompt-latched full-100 probe reusing its response cache.
- 2026-08-11: Ling v23 prompt latch yielded only 16.41% activation and 77/100 refusals despite 92.44% in-split harmful prefill activity, exposing cross-split generalization failure. Added external benign/target eval prefill metrics as a mandatory pre-generation guard and v24 400/class generated-latch probe (27 focused tests passed).
- 2026-08-11: Ling v24 external prefill generalization was actually strong (91.52%; 7.90% benign vs 90.95% harmful active), but per-layer prompt latch still yielded 20.45% runtime activity and 58/100 refusals. Added causal layer-0 `global_prompt` sample gate broadcasting one prefill decision across all later steering layers, its external layer guard, v25 runner/watcher, and tests (28 focused passed).
- 2026-08-11: Ling v25 was stopped pre-generation because layer 0 fired on 100% of both external classes (50% accuracy, 0.0004 margin). Added `global_prompt` auto-layer selection using only internal grouped validation, followed by an independent external guard, plus v26 runner/watcher (28 focused tests passed).
- 2026-08-11: Ling v26 selected layer 4 and passed external guard but produced 0% runtime activity/98 refusals because the decision layer lay outside the v13 steering profile and its hook was skipped. Fixed global decision layers to install an angle-zero observer outside the profile, added regression coverage, and v27 runner/watcher (28 focused passed).
- 2026-08-11: Ling v27 observer fix produced 54.96% activity and 54/100 refusals, matching layer-4 external harmful recall (54%) and validating sample-global broadcast. Added internal-only max-gap automatic selection constrained to layers 0..20, external guard, and v28 terminal runner/watcher (28 focused passed).
- 2026-08-11: Ling v28 selected layer 19 by internal max-gap; its independent external gate reached 98.50% accuracy (1% benign active, 98% harmful active), runtime activity was 97%, and refusals fell to 29/100. Added v29 full primary/validation-KL and generation-health evaluation to measure whether this conditional point improves the v13 quality/control frontier.
- 2026-08-11: Ling v29 reproduced layer 19 and established a new conditional Pareto point: primary KL 0.0622, benign gate 1.09%, full refusals 20/100, validation KL 0.0962, and generation health passed. The matched v28/v29 29-vs-20 refusal spread exposed batch-composition sensitivity, so the conservative refusal range is 20-29/100. Tightened thinking-leak screening to explicit tags or line-start reasoning headers instead of ordinary prose words, with regression tests; the old v29 leak flag is invalid/indeterminate rather than confirmed.
- 2026-08-11: Added Ling v30 high-confidence global-gate probe: threshold 0.8 plus exact/1.10/1.20 matched angular strengths on the fixed 30-prompt subset, with internal max-gap layer selection and external guard. This tests whether near-zero benign activation permits stronger harmful steering toward the original dual target.
- 2026-08-11: Ling v30 threshold 0.8 selected layer 18 with 0% external benign and 96% harmful activation. Matched refusals were 9/30 at max 0.9698, 12/30 at 1.10, and 6/30 at 1.20. Added v31 full quality/compliance evaluation for the sole improved 1.20 candidate.
- 2026-08-11: Ling v31 completed at primary KL 0.0602 and 21/100 refusals (validation KL 0.0765, health passed, corrected thinking-leak none). It trades only 0.002 KL for one extra refusal versus v29 and the 30-prompt 9-to-6 gain did not transfer full-set, so single-axis angular-strength escalation is stopped; v29/v31 remain adjacent runtime-conditional Pareto recipes.
- 2026-08-11: Added Ling v32 conditional profile-shape search under the threshold-0.8 global gate: two matched profiles plus six warmups jointly vary max strength, peak, minimum fraction, and falloff, evaluated on a larger fixed 60-prompt subset before any further full KL run.
- 2026-08-11: Ling v32 completed all eight matched 60-prompt probes: refusals were 20, 10, 15, 42, 17, 23, 28, and 12. The sole winner was the already fully evaluated v31 wide late profile; narrow bands, weak floors, and max 1.48 all regressed. Stopped profile search without another redundant KL run and isolated batch-invariant inference/evaluation as the next engineering bottleneck.
- 2026-08-12: Added canonical single-sample prepass for `global_prompt` gates, fixed-gate injection into real generation and forced-KL batches, telemetry isolation, and reverse-order prescreen replay diagnostics. Added Ling v33 matched 60-prompt batch-invariance runner; focused regression suite passed 25 tests.
- 2026-08-12: Ling v33 fixed-gate replay still shifted from 11/60 refusals in seeded order to 18/60 reversed (delta +7), isolating batched decode rather than classifier decisions as the main drift source. Added input-order-independent canonical batching using token length plus rendered-content tie-breaking, caller-order restoration, and coverage across generation/KL paths for v34.
- 2026-08-12: Ling v34 canonical batching produced exactly 10/60 refusals in both forward and reversed caller order (delta 0), eliminating the v33 +7 order drift at batch16. Added v35 canonical full KL/full-100/health evaluation as the new normalized Ling reporting protocol.
- 2026-08-12: Ling v35 canonical full result was 27/100 refusals and nominal KL 0.0566 with 0% benign gate activity; the latter still compared an old-order baseline cache against canonical steered batches. Extended canonical batching to pre-hook baseline capture, added rendered-prompt gate-decision caching, and prepared v36 with a fresh matched canonical baseline.
- 2026-08-12: Ling v36 rebuilt the baseline with the same canonical batch membership and proved v35's KL was a protocol artifact: matched primary KL 0.0000, benign gate 0%, validation KL 0.0461 with no high-margin flips, health passed, while canonical harmful refusals remained 27/100. Recorded v36 as the normalized quality baseline and narrowed subsequent work to stronger gate-active harmful intervention.
- 2026-08-12: Added an opt-out from positive-alignment-only removal inside `concept_gated_angular` (default remains compatible), with regression coverage. Prepared Ling v37 to test full signed projection removal only on sample-global harmful-gated prompts under the v36 canonical protocol; 33 focused tests passed.
- 2026-08-12: Ling v37 full signed removal reproduced 10/60 refusals exactly, so negative-alignment preservation was not the remaining bottleneck and no full eval was run. Prepared v38 to test the remaining intervention-budget axis: flat full-depth 90-degree removal under the unchanged canonical global gate.
- 2026-08-12: Ling v38 flat full-depth removal regressed to 15/60 versus v36's 10/60, excluding insufficient layer coverage and confirming that early-layer intervention can recreate refusal behavior. Added per-prompt prescreen refusal/compliance source-index telemetry for overlap analysis; focused tests now pass 34 cases, and v39 replays the canonical winner to establish an outcome map.
- 2026-08-12: Ling v39 reproduced 10/60 and persisted the stable refusal source indices `[51,58,53,0,78,98,76,5,39,97]`; prompts span ten severe harm domains rather than one topic cluster. Extended telemetry with global gate-on/off source indices so v40 can separate classifier misses from steering failures; 34 focused tests passed.
