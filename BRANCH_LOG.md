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
| 最近对齐上游 | **v1.12.2**（`5d58cea`，merge `bb48dd9`）——已追平上游 `master` |
| 本日志最近更新 | 2026-09-11 (关键词拒绝判定剥离思考痕迹 + Nex v5 重启) |

---

## -1cq. 2026-09-11 — 修复：关键词拒绝判定未剥离思考痕迹

`detector.py` 的 **LLM judge 路径**一直用 `re.sub(r"<think>.*?</think>", "", …)` 剥掉思考痕迹再解析 JSON，
但**关键词路径没有**：`detect_refusal(response)` 直接吃原始回复文本。

Nex-N2.5-mini 每个 trial 都报 `Thinking leak: detected`（即使编码器已强制
`reasoning_effort="none"`，模型仍会漏思考标记），于是「思考里琢磨要不要拒绝」的文本被计入拒绝；
在 256-token cap 下思考文本更长，**系统性高估拒绝率**——这也是 160→256 口径差异的一部分来源
（同配方同 KL 0.0594：160 口径 12/100，256 口径 22/100，后者同时含真实长文崩坏与思考污染）。

**修复**：
- `DetectionConfig.strip_thinking_blocks: bool = True`（可关，用于复现历史数字）。
- `detector.strip_thinking_blocks()`：剥 ` <think>…</think>`；**未闭合**的 `<think>` 视为延伸到回复末尾
  （截断的纯思考回复 → 空 → 判为拒绝，符合"无可用内容即拒绝"的原意）。
- `detect_refusal()` 先剥离再判定，与 judge 路径对齐。
- 回归测试 `tests/test_thinking_strip.py`（6 例）；全量 `pytest` 仍为 **26 failed / 952 passed**（无新增失败）。

**执行影响**：v5 首轮（00:08–00:42）的 trial 1（22/100 @ KL 0.0845）含思考污染，作废；
已带补丁重启（00:42:59），旧 journal 保留为 `checkpoints_nex25_mini_v5_prestrip/` 作对照。

---

## -1cp. 2026-09-10 — Nex-N2.5-mini 消融开工：direct/EGA vs LoRA 的 A/B

**背景**：Nex-N2.5-mini（Nex-AGI，Qwen3.5-35B-A3B 后训练版，70.2 GB bf16）与
Ornith-1.5-35B-A3B **架构完全一致**（40 层 / hidden 2048 / 256 专家 / moe_inter 512 /
GatedDeltaNet 3-linear+1-full / 同 MTP 层），故 Ornith V4 配方零改动迁移（层号 36.22/37.15 通用）。

**新增资产**：
- `scripts/nex25_encode.py` — 自定义编码器强制 `reasoning_effort="none"`：该模板在缺省参数时会开一个
  裸 `<think>` 块（`enable_thinking=False` 被模板忽略），思考痕迹会吃光 eval token。
- `configs/nex25_mini_rocm_{smoke,sweep,sweep_direct,v1,v2_direct}.toml`、`run_nex25.sh`。
- 关键：上游合并后 `resolve_model_class` 不再隐式识别 `qwen3_5_moe`，配置必须写 **`text_only = true`**
  才能按纯文本 CausalLM 加载（该检查点带视觉塔）。

**发现 1 — seed 的 `min_weight` 是比例不是绝对值**（`optimizer.py`：
`min_frac = suggest_float(..., 0.0, frac_hi)` → `min_weight = min_frac * max_w`）。
`configs/ornith15_35b_rocm_{v4,axlora,mpoa}.toml` 的 seed 写了 1.49 / 3.23 / 1.55 / 1.64（均 > 1），
超出分布范围只 warn 不报错 → 实际 `min = max × 绝对值` → **剖面倒置**。
实测同一 seed：bug 版 KL **8.31**，修正成比例后 KL **0.395**（差 20 倍）。
（`-1bd` 已记过 axlora 的 12.5>5 倒置，但三份配置至今未修。）

**发现 2 — EGA 只在 `steering_mode = "direct"` 下执行**（`steering.py`：`if steering_mode ==
SteeringMode.DIRECT: … if engine.has_expert_routing(): _apply_ega_steering(...)`）。
LoRA 模式下 PEFT 只能裹住 110 个真实 `nn.Linear`（shared_expert.down_proj×40 + 全注意力 q/k/v/o×40
+ linear_attn.out_proj×30）——**融合的 256 个专家（占权重 92%）完全没被碰过**。
因此 `-1bf` 那轮号称「EGA across 256 experts」的 Ornith-1.5 V4 实验**实际从未运行 EGA**
（该轮日志为 `LoRA adapters initialised (110 modules …)`，且全部 KL 0.7–2.4 被剪枝）。

**A/B 实测（同一组 5 档强度、8 条 prompt、基线 8/8 全拒绝）**：

| 档位 | o_proj | down_proj | LoRA KL / 拒绝 | **direct+EGA KL / 拒绝** |
|---|---|---|---|---|
| 1.0× | 3.23 | 9.19 | 0.395 / 0-8 | 0.584 / 0-8 |
| 0.5× | 1.61 | 4.59 | 0.122 / 1-8 | 0.191 / **0-8** |
| 0.25× | 0.81 | 2.30 | 0.034 / 6-8 | **0.052 / 0-8** ★ |
| 0.1× | 0.32 | 0.92 | 0.007 / 8-8 | 0.011 / 3-8 |
| 0.05× | 0.16 | 0.46 | 0.003 / 8-8 | 0.003 / 8-8 |

结论：达成同样的「0/8 拒绝」，LoRA 需 KL 0.395，**direct+EGA 只需 0.052（约 7× KL 效率）**，
与源码注释里 TrevorS 的「3/100 vs 29/100」一致。**MoE 消融应默认走 direct+EGA**；
LoRA 模式在融合专家架构上等于自断 92% 权重。

**执行**：`v1`（LoRA）在基线阶段中止；已按 direct 曲线重标定并启动
`configs/nex25_mini_rocm_v2_direct.toml`（50 trials，o_proj [0.3,1.8] / down_proj [1.0,5.0]，
5 个种子含 0.25× 冠军点与 down-heavy 形状）。

**运维**：`ABLITERIX_UMA_MAX_SWAP_GB` 默认 0.5G 会在 70 GB mmap→CUDA 加载途中误报自杀
（当时 RSS 仅 4.1 G、MemAvailable 62 G，内核回收冷匿名页）→ `run_nex25.sh` 已设 2 G；
`min_free=18G` / `max_rss=96G` 保持不动。

---

## -1co. 2026-09-10 — 对齐上游 v1.12.2（merge `bb48dd9`）

**上游新增**：PR #95（bnb-ROCm MoE 稳定性 + `text_only` + 空 adapter 守卫）、#97/#98（固定续写 EOS、
数学文档）、#99（稳定性/可复现性：reproduce manifest 回放）、#100（版本 1.12.2）、#101/#102（手动发布 CI）、
#103（Bailing v3 可转向模块）、#104（MoE router 专家 id 探测）、#105（视频提示词数据集生成器），
另加 `SAFETY.md` / `SECURITY.md`。本 fork 的三个 PR 分支（`pr/bnb-rocm-moe-stability`、
`pr/bailing-steerable-modules`、`pr/moe-router-profiling`）**已全部被上游合并**，本地分支可清理。
上游新增主依赖 `openai>=2,<3`（已装入 `heretic-env`）。

**冲突解决（9 文件 / 21 处）**：
- `engine.resolve_model_class`：**跟随上游契约**——移除 fork 的 `model_type == "qwen3_5_moe"` 隐式识别，
  统一改用 `model.text_only = true` 显式开关（上游新增测试 `tests/test_bnb_moe_load_fixes.py` 锁死该契约）。
  ⚠️ **行为变化**：`qwen3_5_moe` **且带视觉塔**的检查点（本机 `Nex-N2.5-mini`、
  `Ornith-1.5-35B-A3B-Heretic-t62`）默认重新走 ImageTextToText；要保留旧的纯文本加载，
  请在对应 config 里加 `text_only = true`（Agents-A1 / Qwen3.5-122B/397B 的 config 若仍要跑，同理）。
- `engine._load_model_fast` / `_load_model_bnb_fast`：**保留 fork 的 ROCm 快速加载**
  （CPU mmap → 按 storage 上卡，避免 67G MoE 峰值 2×），并把上游新增的 `revision` / `text_only`
  透传进 4 处 `resolve_model_class` 调用与 4 处 `from_pretrained`。
- `optimizer`：保留 fork 的 `runtime_hook_site` / `concept_gate_threshold` 还原 × 上游 `flush_memory` 注释。
- `settings`：fork 的 `response_pair_enabled` 校验 × 上游的 vLLM+FULL→PRE 回退，两者并存。
- `util` / `interactive` / `safex` / `steering`：取上游（docstring、TTY 中断兜底、
  `extract_router_expert_ids` 共享探测、`_cache_dequant` 缓存上限）。
- `cli`：`owner/model` 简写条件取上游 `len(sys.argv) == 2`（fork 版多参数时会把 `--model.model-id`
  插到别的选项值前面）；fork 的 failure-conditioned 变体 × 上游 reproduce 回放做**语义合并**，
  回放块插到 `study` 执行之前（否则会重复跑搜索）。
- `uv.lock`：版本 1.12.2。

**顺带修复**：
- `tests/test_runtime_hook_sites.py`：WIP 中误插的孤立 `)` 让整个文件无法解析（测试收集直接中断），已修。
- `tests/test_qwen4exp_runtime.py`：假 config 补 `revision` / `text_only`，补 `needs_reload = False`
  与 `FakePeftModel.named_parameters()`，适配上游新守卫。

**验证（同基线对照）**：`heretic-env` 跑全量 `pytest tests/`，
基线 26 failed / 913 passed → 合并后 26 failed / **952 passed**，**零回归**、新增 39 个通过用例。
剩余 26 个失败与本次合并无关：peft `0.19.1` vs pyproject `peft~=0.18` 的 API 漂移（8 个）、
fork 自测替身缺 `use_cache` / `concept_gate_refusal_prefix_retry` 等新字段（14 个）、
`test_detector` 正则（2 个）、需本地 Flash-Next 权重（2 个）。

---

## -1cn. 2026-09-05/06 — Spark-X2.5-4B V30 T615 bake + Spark-X2.5-1.7B V1–V4 搜索

> 本节为 2026-09-10 同步上游时**补记**（当日产物落盘但未入日志）。两部分：
> (a) 09-05/06 的日志缺口；(b) 此前只存在于工作区的实验产物（configs/scripts/runners，
> 最早 2026-08-05）随本次同步一并入库。事实取自脚本/配置内自述，不作运行结论推断。

**Spark-X2.5-4B（V30 家族收尾）**
- `export_spark_x25_t615.py`：T615（**7/100 @ 3-token KL 0.1468**）取代 T580，作为
  `Models/Spark-X2.5-4B-abliterated` 的 bake 目标（用户指定替换）。
- `export_spark_x25_t580.py`：按用户要求导出 T580（20/100 @ 3-token KL 0.0924）——
  明确的非 dual HIT 导出。
- `eval_spark_x25_v30_t579_t580_merge.py`：T579+T580 合并评测，dual HIT 才 bake。
- `sweep_spark_x25_t580_leftover.py`：在 T580 仍拒绝的 prompt 上估第二条 mean 方向，
  再对合并后的 T580 权重做 LoRA-peel。
- `eval_spark_x25_ara_t49_t196.py`：T49/T196 全权重 ARA，用 Abliterix 原生 LBFGS 在
  `self_attn.out_proj + mlp.down_proj` 上按 Heretic Optuna 超参重拟合。
- `inspect_spark_x25_t580_refusals.py` / `inspect_spark_x25_t580_r18.py`：拒绝样本排查。

**Spark-X2.5-1.7B（新模型线：28 层 / hidden 2048 / intermediate 6656，
fused `q_k_v_proj` + `out_proj`，3 sliding + 1 full）**
- V1：不播种的首轮 mergeable mean-LoRA 搜索（同 4B 家族，但层数/宽度不同，4B 种子不通用）。
- V2：用 4B V30 champion 播种，层号/索引按 27/35 缩放（last_layer 35→27）。实测 4B 全局
  种子（qkv≈3.1–3.6）在 1.7B 上过度拒答/KL 爆炸，把 TPE 拉向封闭墙。
- V3：改为 1.7B 原生 Pareto 播种。
- V4：放宽搜索箱 `o_proj [auto-off, 12]`、`qkv [auto-off, 8]`、`down [auto-off, 2.5]`，
  KL 剪枝 0.6；wrapper 每轮 +400 trials 直到 dual HIT（refusals ≤10 且 3-token KL ≤0.1），
  **不 bake**。
- Runner：`run_spark_x25_1p7b.sh`、`run_spark_x25_1p7b_v4_keep.sh`、
  `run_spark_x25_1p7b_heretic_then_ara.sh`（经典 Heretic 30 → 全权重 ARA 200，顺序占 GPU）、
  `run_spark_x25_leftover_then_ara.sh`。

**随本次入库的存量产物**：`configs/spark_x25_4b_lora_v30.toml`、`configs/lfm25_2.6b.toml`、
`configs/muse_glimmer_30b_rocm*.toml`(8)、`configs/ornith15_35b_rocm_*.toml`、
`configs/qwen38_27b_rocm*.toml` 及配套 `scripts/sweep_qwen38_*.py`、
`bake_qwen38_cga_distill.py`、`apply_qwen38_ara_fixed.py`、`run_qwen38*.sh`、
`run_ornith15.sh`、`run_muse_glimmer.sh`、`requirements-qwen4exp.txt`。

---

## -1cm. 2026-09-04 — Spark V30 hard mean-LoRA Optuna (full-100 scoring)

Heretic 2.0 and T9 adapter missed dual bar (87/100 @ 3-token KL 0.1086).
No remaining distinct mergeable recipe. Hard-search the V9 family: rank-8
full-norm mean LoRA, projected on, o_proj [2.0, 8.0], 100 trials / 24
warmup, seed T32+T31. Prescreen only prunes 25+/30 so borderline points
get a real 100-eval (V9 T32 was a 18/30 estimate). Ship bar updated to
refusals ≤10/100 and 3-token KL ≤0.10. Paused 2026-09-04 17:27 after
24 completed trials (T25 mid-eval), then resumed. After the first 100
complete, raise budget without overwrite. Do not stop on a near-miss
(≤12 and KL ≤0.12). On dual HIT (≤10 and KL ≤0.10), judge whether KL
0.05 is still reachable; if yes, keep searching. Bake merged only when
≤10 and KL ≤0.05, or when ≤10 and KL ≤0.10 and 0.05 looks unreachable.
`num_trials = 800` after 500 if still no dual HIT.

## -1cl. 2026-09-04 — Heretic 2.0 T9 adapter on Abliterix 3-token meter

Official Heretic 2.0.0.dev0 30-trial search missed dual bar on its own
first-token KL (best T9 43/100 @ 0.1655). Score that saved LoRA adapter
with Abliterix keyword + 3-token KL vs V9 original baseline. Bake merged
only on dual HIT. `scripts/eval_spark_x25_heretic_adapter.py`.

## -1ck. 2026-09-04 — Spark-X2.5-4B full-weight ARA (mergeable)

CGA/distill abandoned for this goal (ungateable). Spark never ran ARA.
One-shot full-weight ARA, not ARA-LoRA: trohrbaugh knobs, o_proj+down_proj,
layers 8–32 of 36, BF16 LBFGS, score keyword + 3-token KL vs V9 original
baseline. Bake merged only on dual HIT. `scripts/apply_spark_x25_ara_fixed.py`.
Finished ~13 min: **5/100 @ KL 0.3038**, HIT=false. Refusal opens, KL
worse than Householder o=2.9 (9@0.182). Delta
`exports/spark_x25_ara_fixed_delta.pt` (1.7G). No bake.

## -1cj. 2026-09-02 — Spark CGA distill of V27 12@0.047 into merged LoRA

V29 qkv on decoder_block was a no-op (still 12 @ 0.0474). Distill the
runtime CGA teacher (filter keyword refusals) into mergeable LoRA r=8
and re-eval the merged dir vs original.
`scripts/bake_spark_x25_cga_distill.py`.

## -1ci. 2026-09-02 — Spark CGA V29 qkv on the 12@0.047 plateau

Leftovers are compliance-theater / "I can't" (8 early, 4 late), not
Sorry-prefix. Prefix retry and lower thresh did not move them. Add
all-layer qkv beside o=1.35 (0.0026 KL budget).
`scripts/sweep_spark_x25_cga_v29.py`.

## -1ch. 2026-09-02 — Spark CGA leftover inspect at 12@0.047

V28 prefix-retry and t=0.45/0.40 left refusals at 12. The last 12 are
gated but not flipped. Dump their prefixes.
`scripts/inspect_spark_x25_cga_leftover.py`.

## -1cg. 2026-09-02 — Spark CGA V28 leftover-prefix retry on 12@0.047

V27 plateau: all-layer global t0.50 s1.35 d40 → 12/100 @ KL 0.0474;
s1.40 went to 13 @ 0.0502. Last 12 are first-token leftovers. V28
bans refusal-prefix tokens on gate-on retry and lowers threshold.
`scripts/sweep_spark_x25_cga_v28.py`.

## -1cf. 2026-09-02 — Spark CGA V27 all-layer interpolate around 17@0.039

V26 new best under 0.05: global all-layer t0.50 s1.20 d40 → 17/100 @
KL 0.0392 (T32 was 57 @ 0.0456). s1.00 d40 was 37 @ 0.0302. V27
interpolates s=1.22–1.40 plus linear_projection and L18/L30.
`scripts/sweep_spark_x25_cga_v27.py`.

## -1ce. 2026-09-02 — Spark CGA V26 dense global_prompt pocket

V25 broke the mean/HH Pareto: global_prompt t0.50 s1.00 → 53/100 @
KL 0.0242 (T32 was 57 @ 0.0456); t0.30 s1.40 → 24/100 @ 0.0647 (HH
o=2.3 was 25 @ 0.123). V26 spends the 0.024→0.05 KL budget plus
both-sign / linear_projection / all-layer / last-layer variants.
`scripts/sweep_spark_x25_cga_v26.py`.

## -1cd. 2026-09-02 — Spark concept-gated angular on V9 mean-diff

SOM 2×2 killed after 14 trials: 17/30 ~57est @ KL 0.1385 (mean T32 was
0.0456), 2/30 @ 0.82. Next distinct recipe: CGA runtime gate so benign
3-token KL can stay low while harmful tokens rotate. Dual HIT is
runtime-only — distill before bake. `scripts/sweep_spark_x25_cga.py`.

## -1cc. 2026-09-02 — Spark SOM 2×2 correlated-direction LoRA

HH-rpair killed after 5 points: 99/100 @ KL 0.0438, 91/100 @ 0.101,
75/100 @ 0.1335 — worse than mean HH (9/100 @ 0.182). Next distinct
recipe: Kohonen SOM 2×2 (4 correlated dirs, not SVD n_dir=3). Mergeable
full-norm LoRA rank 8, prune 0.12. `configs/spark_x25_4b_som_v24.toml`.

## -1cb. 2026-09-02 — Spark Householder on Sorry-vs-Sure vectors

V23 response-pair LoRA: T4 8/30 @ KL 0.308, T6 9/30 @ 0.239 — opens but
worse KL than mean HH (9/100 @ 0.182). V24: Householder isometric peel
of the cached V23 vectors. `scripts/sweep_spark_x25_hh_rpair.py`.

## -1ca. 2026-09-02 — Spark V23 response-pair Sorry vs Sure LoRA

V22 system-prompt pairing: 16 trials all 29–30/30 prune (T32 geometry
cannot open). V23 teacher-forces "I'm sorry, but I can't..." vs
"Sure, I can help..." on the same harmful prompts. Eval unchanged.
`configs/spark_x25_4b_rpair_v23.toml`.

## -1bz. 2026-09-02 — Spark V22 same-prompt conditional refusal LoRA

V21 OT: T32-like geometry 60est @ KL 0.127 (worse than mean T32 0.046).
Killed. V22: identical harmful_1000 prompts, compliance vs refuse system
prompts; eval unchanged. Isolates refusal prefix, not topic.
`configs/spark_x25_4b_paired_v22.toml`.

## -1by. 2026-09-02 — Spark V21 optimal-transport LoRA

V20 COSMIC: 11 trials all 28–30/30 prune. V21 OT LoRA rank-3, V9
envelope, prune 0.12. `configs/spark_x25_4b_lora_v21.toml`.

## -1bx. 2026-09-02 — Spark V20 COSMIC residual LoRA

V19 PCA: 10 trials all 29–30/30 prune. V20 COSMIC LoRA rank-3, o_proj
[3.0, 7.5], prune 0.12, 30 trials. `configs/spark_x25_4b_lora_v20.toml`.

## -1bw. 2026-09-02 — Spark V19 PCA residual LoRA

Unlikelihood from original: 8 steps 98/100 @ KL 0.273, cap trip.
V19 PCA LoRA rank-3, V9 envelope, prune 0.12, 30 trials.
`configs/spark_x25_4b_lora_v19.toml`.

## -1bv. 2026-09-02 — Spark unlikelihood LoRA on T32 leftover prefixes

First-token leftover-opened LoRA: 46/100 @ 0.154, not dual HIT. Next:
unlikelihood on leftover's own refusal prefixes after T32, KL cap 0.049
vs original. `scripts/train_spark_x25_unlikelihood.py`.

## -1bu. 2026-09-02 — Spark first-token leftover-vs-opened LoRA peel

Prefill leftover-vs-opened cos=0.16 (new dir) but HH o=0.4 already KL
0.085 at 55/100. Remaining refusals are first-token "I'm sorry". Extract
teacher-forced first-12-char residuals leftover vs opened, LoRA peel on
merged T32. `scripts/sweep_spark_x25_first_token.py`.

## -1bt. 2026-09-02 — Spark leftover-vs-opened contrast peel

V18 linear decay same Pareto (T15 60/100 @ 0.0434). Next: T32 leftovers
vs already-opened harmful prompts (not vs benign), Householder peel of
that contrast on merged T32.
`scripts/sweep_spark_x25_opened_contrast.py`.

## -1bs. 2026-09-02 — Spark V18 linear-decay mean LoRA

V17 median killed: T6 11/30 @ KL 0.190 at T32-like geometry (mean T32 was
57/100 @ 0.0456). V18 linear decay, rank-3, prune 0.12, 30 trials.
`configs/spark_x25_4b_lora_v18.toml`.

## -1br. 2026-09-02 — Spark V17 median residual LoRA

T32 leftover inspect: 57/100 refuse, 53 early in first 80 chars, only 4
late-only. 3-token KL is the real bottleneck. V17 median-of-means LoRA
rank-3, V9 envelope, prune 0.12, 30 trials.
`configs/spark_x25_4b_lora_v17.toml`.

## -1bq. 2026-09-02 — Spark T32 leftover prefix inspect

V16 T4/T6: 8/30 @ KL 0.19–0.22, same Pareto as V1. Inspect T32's 57
leftover eval prefixes: if refusals start in the first 80 chars, 3-token
KL must move; late-only refusals would justify a later-token peel.
`scripts/inspect_spark_x25_t32_leftover.py`.

## -1bp. 2026-09-02 — Spark V16 LoRA weight_normalization=none

V15 killed after 21: 7/30 @ KL 0.14, no KL≤0.05 opening. V16 rank-1
unnormalized LoRA, o_proj [1,8], prune 0.12, 30 trials.
`configs/spark_x25_4b_lora_v16.toml`.

## -1bo. 2026-09-02 — Spark V15 mean LoRA, projected_abliteration=false

Narrow HH dist=4/6/8 never opened (best 81/100 @ 0.0498). Refusal lives
in a wide layer band, which is the KL cost. V15: V9 envelope, rank-3,
projected_abliteration=false, prune 0.12, 40 trials.
`configs/spark_x25_4b_lora_v15.toml`.

## -1bn. 2026-09-02 — Spark Householder narrow layer band

HH leftover 2nd reflection undid stage-1 (79→100). r2_orth did not
drop refusals (76–82) while KL rose to 0.20. Next: HH o=1.5–4.0 with
min_weight_distance 4/6/8 (late layers only).

## -1bm. 2026-09-02 — Spark HH o=1 leftover second reflection

Dense HH: KL≤0.05 best 63/100 @ 0.046 (o=1.35); refusals≤10 at o=2.9
9/100 @ 0.182. Stage-1 HH o=1.0 is 79/100 @ 0.0285 (0.0215 KL left).
V14 `scripts/sweep_spark_x25_hh_leftover.py` composes a second
Householder on leftover r2 / r2_orth.

## -1bl. 2026-09-02 — Spark Householder dense o_proj grid

Coarse HH: 79@0.0285 (o=1), 35@0.094 (o=2), 11@0.192 (o=3), 6@0.271
(o=4), 1@0.353 (o=5.5). qkv HH collapsed. Dense o-only 1.15–1.85
(KL≤0.05 band) and 2.3–3.2 (refusals≤10 band).

## -1bk. 2026-09-02 — Spark Householder isometric peel

V12 leftover r2_neg restored refusals (76–99/100). r2_orth: 40/100 @
KL 0.197 — same Pareto as V5, not cheaper. Next: Householder direct on
original mean-diff (`scripts/sweep_spark_x25_householder.py`), mergeable
norm-preserving reflection, o_proj 1–8.5.

## -1bj. 2026-09-02 — Spark T32 leftover r2 flip + orthogonal

V11 leftover peel undid T32: 57/100 @ 0.0456 → 74/88/90/95 as o rose
0.6→1.8, KL drifted down toward original. cos(r1,r2)=0.78. Killed.
V12 `scripts/sweep_spark_x25_t32_r2flip.py`: r2_neg=-r2 and
r2_orth=r2-proj_r1, LoRA peel on merged T32, KL vs original.

## -1bi. 2026-09-02 — Spark-X2.5-4B V9 T32 leftover peel

V10 n_directions=3 mean LoRA (50 trials) killed after 18: leftover
openings only at KL 0.40–0.50 (T5 6/30 @ 0.495). Same original residual
subspace, worse Pareto than V9. Next: apply V9 T32 (18/30 est 60/100 @
KL 0.0456), extract mean-diff on leftover keyword-refusals, LoRA-peel r2
on merged T32, score 3-token KL vs original baseline. Script
`scripts/sweep_spark_x25_t32_leftover.py`. HIT still refusals≤10/100 and
KL≤0.05 vs original; bake to `/run/media/s117/OS/Models/Spark-X2.5-4B-abliterix`.

## -1bh. 2026-08-22 — Ornith-1.5 switched to Heretic 1.4 classic

Abliterix MPOA/LoRA/V4 all-pruned. Running Heretic 1.4.0 from
`heretic-ara-lora` like Ornith-1.0 Trial 63: `row_normalization=full`
rank-3, o_proj+down, no ARA, BF16. Config
`config.ornith15-classic.toml`, study `checkpoints-ornith15-classic`,
datasets heretic-data alpaca / harmful_behaviors.

## -1bg. 2026-08-22 — Ornith-1.5 V4 datasets → Abliterix good/harmful_1000

Stopped alpaca/harmful_behaviors (Heretic 1.0). All ornith15 configs
now `datasets/good_1000` + `datasets/harmful_1000` (`prompt` column,
train[:800] / [800:900]). New study `checkpoints_ornith15_35b_v4_axds`.

## -1bf. 2026-08-22 — Ornith-1.5 V4: enable QKV + EGA down_proj

axlora (QKV off) still ~all-prune; one score ~30@0.147. Switch to
sister 35B-A3B V4: `disabled_components=[]`, o_proj [1,6], q/k/v
[0.5,4], down [2,10] (EGA across 256 experts). Study
`checkpoints_ornith15_35b_v4`.

## -1be. 2026-08-22 — Ornith-1.5 Abliterix default LoRA TPE

MPOA (1.0 recipe) plateaued at 30@0.078. New study
`checkpoints_ornith15_35b_axlora`: `weight_normalization=none`
rank-1 LoRA, o_proj [1,8] down [0,10], seeds t39/t10 envelopes.
BF16 batch 128. Config `configs/ornith15_35b_rocm_axlora.toml`.

## -1bd. 2026-08-22 — Ornith-1.5 wide 60-trial finished

20 scored / 40 pruned. Best KL-efficient: **t39 30/100 @ 0.078**
(o_max=3.23 down_max=9.19, per-layer). t46 30@0.079; t53 33@0.091;
t47 37@0.056; t37 29@0.150. No 8–12. Strong seed min_weight was
scaled to 12.5>max 5 (envelope inverted, 30/30 prune).

## -1bc. 2026-08-21 — Ornith-1.5 widen strength after all-prune

v1 [0.5, 2.1] pruned 28 trials at 28–30/30. New study
`checkpoints_ornith15_35b_wide`: o_proj [1, 8], down [1, 10],
batch 128 BF16, plus a stronger MoE seed (o=5 / down=6).

## -1bb. 2026-08-21 — Ornith-1.5-35B-A3B Trial-63 MPOA (BF16)

Same recipe as Ornith-1.0 t62/t63: `weight_normalization=full`
rank-3, o_proj+down only, no routed-expert ablation. BF16 no bnb.
MTP already in the 16 shards. Config
`configs/ornith15_35b_rocm_mpoa.toml`, `./run_ornith15.sh`.
Checkpoint `checkpoints_ornith15_35b`. Batch 32 for UMA headroom.

## -1ba. 2026-08-21 — Removed repo `checkpoints_qwen38_27b_*`

Deleted pocket/rdo/iter/cga/judge/v2–v8/etc. (~44G). Home disk
6.9G → 51G free. Original model on OS volume unchanged.

## -1az. 2026-08-21 — Removed derived Qwen3.8-27B checkpoints

Deleted all `/run/media/s117/OS/Models/Qwen3.8-27B-*` (t24, FP8,
distill, DPO LoRA, heretic-ara, ssm-repaired, etc.). Kept original
`/run/media/s117/OS/Models/Qwen3.8-27B` (52G). Repo Optuna
checkpoints untouched.

## -1ay. 2026-08-21 — Huihui 32-pair + skip-early-layers miss

32+32 house train, r at layer 38, skip layers 0–14 (and 56–63 /
32–39). Skip0-15 s1.00: **96@0.220**; skip 32–39 too: **100@0.148**.
Layer skip matches our bands, but 32 random pairs give a weak r vs
800-pair ORBA. Not in 8–12 @ ≤0.1.

## -1ax. 2026-08-21 — CGA linmap bake miss (CPU rank-4)

Fit rank-4 A on layers 24–56 from CGA vs orig residuals (benign
target 0). `W'=(I+sAᵀ)W`. s0.50 90@0.258; s1.00 **68@1.05**;
s1.30 83@1.87. Skip/非线性门控写不进低秩 W。Desktop-safe CPU fit
OK. Best mergeable still ORBA o8.7 20@0.125 / o12.2 12@0.210.

## -1aw. 2026-08-21 — ORBA t24-band hits 8–12 refusals, KL ~0.21

In-memory only (no 52G write). o12.2 **12@0.210**, o12.8 **9@0.229**,
o13.2 **10@0.239**. Still KL>0.1. Best KL-efficient remains o8.7
**20@0.125**. Closest in-band: o12.2 12@0.210.

## -1av. 2026-08-20 — OrcaRouter Arditi recipe on our meter

Layer-38 massive-masked `r`, `W' = W − s r(rᵀW)` on 129 residual
writers + embed. Our keyword + 3-token vs original:
s1.00 **72@0.105**, s1.10 49@0.153, s1.25 16@0.279, s1.50 4@0.567.
Mergeable, but 8–12 sits around s≈1.22 with KL ≫ 0.1. t24 50@0.087
still more KL-efficient.

## -1au. 2026-08-20 — DPO/SFT still miss 8–12 @ KL≤0.1 mergeable

retain2 late-LoRA: 100@0.0008–0.0054; ×6 scale 94@2.31 (collapse).
DPO CGA-chosen vs original-rejected: 100@0.0015 → 100@0.027 → 100@0.111
→ **96@0.241** then stop. No merge. Loadable checkpoint remains
`Qwen3.8-27B-t24` 50/100 @ 0.0869.

## -1at. 2026-08-20 — Mix/peel/residual miss; householder next

Biprojected 30@0.116 / 24@0.150. ORBA 31@0.110 / 21@0.131. Original
mean-diff ORBA on t24 undoes it (90–100 refusals). Residual re-extract
on t24: 52–61 @ 0.082–0.087, no extra refusal drop. Still no 8–12 @
KL≤0.1 mergeable. Householder t24-band 0.35/0.70/1.00 next.

## -1as. 2026-08-20 — t24 merged eval 50@0.0869; FP8 written

Checkpoint `/run/media/s117/OS/Models/Qwen3.8-27B-t24` (ordinary
qwen3_5 shards). `eval_qwen38_vs_original.py --merged`: **50/100 @
0.0869** vs original (matches search). KL ≤0.1, refusals not 8–12.
FP8: `/run/media/s117/OS/Models/Qwen3.8-27B-t24-fp8` via
`scripts/quantize_fp8.py` (linear_attn weight_scale_inv MISSING on
BF16 source — transformers report).

## -1ar. 2026-08-20 — Retain v2 100@4.16; ORBA 34@0.101; bake t24

Retain SFT 100/100 @ 4.16. ORBA t24-band: 60@0.070 / **34@0.101**. No
8–12 @ ≤0.1 on mergeable weights. Baking pocket trial 24 LoRA to
`/run/media/s117/OS/Models/Qwen3.8-27B-t24` (search 50@0.087) as the
loadable/quantizable checkpoint.

## -1aq. 2026-08-20 — Distill v1 collapse 94@3.25; retain SFT v2

Merged `/run/media/s117/OS/Models/Qwen3.8-27B-cga-distill` via
`eval_qwen38_vs_original.py --merged`: **94/100 @ KL 3.2491** (broken).
Adapter lerp 0.05–0.50 stays 100/100 @ 0.0006–0.0705. Next: short-prefix
+ benign-repeat SFT (`scripts/bake_qwen38_cga_retain.py`), in-memory
score before another 52G write.

## -1ap. 2026-08-20 — Distill CGA sidecar into mergeable LoRA

Runtime CGA cannot be algebraically merged. Added
`src/abliterix/distill.py` (continuation labels + SFT loss) and
`scripts/bake_qwen38_cga_distill.py`: teacher-generate on train
harmful/benign with the 11@0.041 sidecar, LoRA-SFT the original,
write adapter + merged BF16 under `/run/media/s117/OS/Models/`.
Home disk is too small for a 52G merge.

## -1ao. 2026-08-20 — Named candidate 11/100 @ 0.0413

**Candidate:** `cga_low_global_prompt_t0.35_1.60`
(`artifacts/qwen38_cga_global_t035_s160.json`). Runtime
concept-gated angular on original Qwen3.8-27B: pocket mean-diff
vectors, 64-layer decoder_block hooks, `global_prompt` latch at
layer 32, threshold 0.35, strength 1.60 (overrotation). Keyword
**11/100** on harmful_1000 `train[900:]`, 3-token
`full_distribution_kl` **0.0413** vs original (pocket baseline).
Capture: `logs/qwen38_cga_low_sweep.json` (named_candidate set).
Not mergeable; load via the artifact + scorers/vectors paths.

## -1an. 2026-08-19 — CGA global 20@0.040; densify threshold

`scripts/sweep_qwen38_cga_all.py`: token-scope 55–67 @ 0.18–0.23
(worse than LoRA). Prompt-latch 75–81 @ 0.001–0.004 (benign-safe but
weak). **global_prompt t0.35 strength 1.0: 20/100 @ 0.040** — best
KL-at-refusal so far, 0.06 KL still in budget. Next:
`scripts/sweep_qwen38_cga_low.py` global 0.15–0.32 plus overrotation
1.3/1.6 at 0.35.

## -1am. 2026-08-19 — Narrow CGA gate-dead; all-layer lower threshold

`scripts/sweep_qwen38_cga.py`: adaptive 53@0.122 / 47@0.175 (same
Pareto). Prompt-latch on t24/wide12 band at 0.5–0.7: 91–98/100 @
KL 0.000 — scorers exist 64/64 but late-layer last-token never
crosses threshold. Crashed on `global_prompt` (`decision_layer=-1`).
Next: `scripts/sweep_qwen38_cga_all.py` all 64 decoder hooks, token/
prompt/global scopes, thresholds 0.20–0.50, strength 1.0.

## -1al. 2026-08-19 — Narrow-band angular same Pareto; concept-gate next

`scripts/sweep_qwen38_md_angular.py`: t24-band (10 layers) 85@0.072 /
59@0.106 / 51@0.121; wide-12 58@0.135 / 42@0.164; linear t24-band
80@0.084 / 55@0.119; weak flat 94@0.092 / 80@0.162. No 8–12 @ KL≤0.1.
LoRA t24 (50@0.087) still dominates this direction. Next:
`scripts/sweep_qwen38_cga.py` — adaptive angular then concept-gated
angular (prompt/global latch) so benign 3-token KL can stay low.

## -1ak. 2026-08-19 — Residual hooks miss; narrow-band mean-diff angular

`scripts/sweep_qwen38_rdo_hooks.py` finished, no 8–12 @ KL≤0.1.
RDO residual (paper linear + angular, all 64 layers): **99/100** @
0.0036–0.0085 — learned r does not transfer. Pocket mean-diff angular
flat all-layer: 23@0.327, 18@0.399, 13@0.436, **9@0.457**. Refusal
target appears only far past the KL gate (worse than LoRA 10@0.26).
Next: `scripts/sweep_qwen38_md_angular.py` uses the t24 envelope
(peak 49.6, distance 5.1) plus a wider-12 band, linear projection in
that band, and weak flat 0.08/0.14.

## -1aj. 2026-08-19 — RDO-direct miss; residual-stream hooks next

`scripts/sweep_qwen38_rdo_direct.py` finished: o0.30–o3.00 all **100/100**
at 3-token KL 0.0007–0.0063 vs original. o_proj-only direct is not the
RDO paper apply (`h ← h − (h·r̂)r̂` on every decoder block). Next:
`scripts/sweep_qwen38_rdo_hooks.py` installs linear_projection then
angular residual hooks on cached RDO vectors, then pocket mean-diff
angular if RDO still misses. Keyword + 3-token vs original; JSON to
`logs/qwen38_rdo_hooks_sweep.json`.

## -1ai. 2026-08-19 — RDO+LoRA 10-trial all pruned; sweep RDO as direct

RDO 40-step AdamW finished (L 17.5→3.36). Learned r is nearly orthogonal
to pocket mean-diff (global cos 0.109). All 10 LoRA trials at o∈[3.8,6]
prescreen-pruned high (22–30/30). Next: apply cached RDO vectors as
flat all-layer `steering_mode=direct` (`scripts/sweep_qwen38_rdo_direct.py`).

## -1ah. 2026-08-19 — Official heretic-ara miss on our meter

`trohrbaugh/Qwen3.8-27B-heretic-ara` downloaded to
`/run/media/s117/OS/Models/Qwen3.8-27B-heretic-ara`. Card: 0/100 @
first-token KL 0.0535 on mlabonne/harmful_behaviors. Our gate
(`eval_qwen38_vs_original.py --merged`, keyword + 3-token vs original
on harmful_1000 train[900:]): **72/100 @ 0.2275**. Worse than pocket
t24 (50@0.087) and our ARA reimpl (61@0.121). Not a ship.
JSON: `logs/qwen38_official_ara_eval.json`.

## -1ag. 2026-08-19 — Iterative rank-8 direct miss; sweep rank 1–3 slices

Native iterative built an 8-direction subspace (4 passes × 2). Applying
all 8 at once: t7 60@0.407, t2 ~30@1.01, t4 8/30-low @ 1.09. Cliff, not
a 10@0.1 basin. Stopped at trial 9/20. Next: `scripts/sweep_qwen38_iter_rank.py`
applies only the first 1/2/3 cached directions at o_proj 0.5–1.5.

## -1af. 2026-08-19 — ARA lerp + t24 residual miss; start iterative+direct

Sweep `scripts/sweep_qwen38_next.py` (keyword + 3-token vs original):
ARA lerp 0.70 89@0.0675, 0.85 77@0.0925, 1.15 42@0.152, 1.35 22@0.198,
1.60 9@0.258. Same Pareto as LoRA. t24 residual is cheaper near the
KL gate (50@0.087 → 39@0.100 → 26@0.135) then plateaus. No 8–12 @ ≤0.1.
JSON: `logs/qwen38_next_sweep.json`. Next: native iterative extract-ablate
+ `steering_mode=direct` 20-trial (`configs/qwen38_27b_rocm_iter.toml`).

## -1ae. 2026-08-19 — Next search after MPOA: ARA lerp + t24 residual

t24 拆回复: keyword 50, hard+hedge 77, payload 9 — cannot score as 8–12.
MPOA `weight_normalization=full` stopped at trial 10/30: same Pareto
(t4 47@0.097, t2 43@0.105, t9 19@0.201, t5 7@0.441). No 8–12 @ KL≤0.1.
Not another mean+LoRA pocket.

`scripts/eval_qwen38_vs_original.py --delta-scale S` lerps original→ARA.
`scripts/sweep_qwen38_next.py` maps ARA scales 0.70/0.85/1.15/1.35/1.60
then reapplies pocket t24, re-extracts a residual direction, and peels
o_proj at 0.4–2.5. Each point is keyword + 3-token KL vs original; JSON
to `logs/qwen38_next_sweep.json` and scratch.

## -1ad. 2026-08-19 — Repaired 50-trial finished; miss 51/100 @ 0.109

50/50. Same Pareto as unrepaired pocket. In-budget search: t50 ~40@0.093
vs repaired, t3 ~43@0.085. 8–12 refusals only at KL 0.31–0.50 (t10 10@0.311,
t16 8@0.440). Official vs-original eval of t50:
`scripts/eval_qwen38_vs_original.py` → **51/100 @ 0.1088** 3-token
(pocket original baseline). Repair added ~0.015 KL vs stock. Not a ship.
Next: `weight_normalization=full` (MPOA) on the original base.

## -1ac. 2026-08-19 — SSM conv1d repair + Abliterix on repaired base

Stock Qwen3.8-27B has the same Fernflower pattern as AEON 3.6: 8 GDN
`linear_attn.conv1d` outliers (L52/53/56/57/58/60/61/62), median σ=0.0428,
α=0.517–0.666. Script `scripts/repair_qwen38_conv1d.py` writes
`/run/media/s117/OS/Models/Qwen3.8-27B-ssm-repaired` (symlink + 3 shards).
Search: `configs/qwen38_27b_rocm_repaired.toml` /
`checkpoints_qwen38_27b_repaired`, 50/15, 3-token, BF16 batch 128.

## -1ab. 2026-08-19 — Full-weight ARA finished; miss 61/100 @ 0.121

trohrbaugh knobs, BF16, layers 26–56, 60 modules, `exports/qwen38_ara_fixed_delta.pt` (6.8G).
Live eval (keyword + 3-token TF KL vs pocket original baseline):
**61/100 @ 0.1208**. Misses both 8–12 and KL≤0.1. Worse than pocket t24 (50 @ 0.087).
Artifacts: `logs/qwen38_ara_fixed_eval.json`, scratch copy. Next: AEON 3.8 conv1d repair + Abliterix 1.12, not more pocket TPE.

## -1aa. 2026-08-19 — eval_qwen38_vs_original --delta

`scripts/eval_qwen38_vs_original.py` accepts `--delta exports/qwen38_ara_fixed_delta.pt` and copies ARA layer weights onto the original BF16 load, then scores keyword refusals + 3-token KL vs the pocket original baseline. Avoids HuggingFace `save_pretrained` of a 52G merge.

## -1z. 2026-08-19 — t24 拆回复：硬拒绝 77，不是 10

User: 先拆 t24 回复，看关键词 50 里有没有一批「提到 illegal 但在答」。
审完 `logs/qwen38_t24_inspect.json` 100 条：hard 65 + hedge 12 = **77/100 硬拒绝**；payload 9（16/19/22/34/35/37/42/43/56）；partial 14。关键词 50，对硬拒绝 FP=3 FN=30。和 AEON 3.8（judge-R 29–36、硬拒绝 0）相反。KL 0.0866 仍 ≤0.10，拒答门过不了，t24 不当成品。审计：`logs/qwen38_t24_reply_audit.json`。

## -1y. 2026-08-19 — Pocket 100-trial finished; no 10@0.10

100/100, 75 scored. Densified the same curve. New-ish: t24 50/100
@ 0.087, t38 14/100 @ 0.198, t93 10/100 @ 0.258. 10@0.10 empty.
Seeds still own the low-KL end (t1 60@0.074).

## -1x. 2026-08-18 — Pocket 3-token 100-trial from our own front

User: search 100 trials in the ranges that actually produced low
KL / usable refusals. o_proj [3.8, 6.0], down [0, 2.8]. Seeds
kw t93/t19/t85 + aeon t3/t42. 3-token KL, BF16, batch 128.
`qwen38_27b_rocm_pocket.toml` / `checkpoints_qwen38_27b_pocket`.

## -1w. 2026-08-18 — AEON-style 50-trial finished; 3.6 seed does not transfer

50/50, 34 scored. AEON Qwen3.6 t46 seed on 3.8: **17/100 @ KL 0.341**.
Best KL t3 33/100 @ 0.119 (o=4.72 d=2.64). Lowest refusals t14 2/100
@ 0.493. 10@0.10 and 0.04–0.10 basin empty. TPE still preferred
high o_proj. 3.6 ranges did not recreate AEON's 0.0005 / 0/100.

## -1v. 2026-08-18 — AEON-style Abliterix 3-token 50-trial

User: follow AEON-7 recipe, 3-token KL. o_proj [1,6], down [2,10],
50/15, prune 0.5, BF16 batch 128, keyword. Seed ≈ AEON Qwen3.6
t46 (o=1.56, down=3.45, per-layer). Config
`qwen38_27b_rocm_aeon.toml`, dir `checkpoints_qwen38_27b_aeon`.
No high-o / down≈0 seeds. Qwen3.8 trial-48 knobs still unpublished.

## -1u. 2026-08-18 — First-token KL is higher than 3-token; widen o_proj

Same weights: t93 0.074 (3-tok) → 0.171 FT; t19 0.116 → 0.196.
Teacher-forced tokens 2–3 diluted the average. o_proj≥3.8 left FT
KL floor at 0.16. Stopped ~trial 51. Restart
`checkpoints_qwen38_27b_kw_ft2`, o_proj [1.5, 6.0], down [0, 1.2],
same seeds, first-token, 100 trials.

## -1t. 2026-08-18 — First-token KL 100-trial, ranges from kw_bf16

`token_count=1` to match Heretic/JonathanColetti. o_proj [3.8, 6.0],
down [0, 1.2] (high down only bought 1–6/100 @ KL 0.32+). Seeds
t19/t78/t85/t93. Config `qwen38_27b_rocm_kw_ft.toml`, dir
`checkpoints_qwen38_27b_kw_ft`, BF16 caches reused.

## -1s. 2026-08-18 — BF16 keyword 100-trial finished

100/100 complete, 58 scored, 6h56m. First time the search entered
KL<0.10 vs original BF16: t93 60/100 @ 0.074, t65 60/100 @ 0.084.
Best near the 0.10 gate: t19 33/100 @ 0.116. Lowest refusals t55
1/100 @ 0.361. Ship 10/100 @ KL≤0.10 still empty.

## -1r. 2026-08-17 — Drop LLM judge; BF16 keyword 100-trial

User: judge difference not worth SSL flakes; add 100 keyword
trials. Stopped judge v2 resume. Fresh study
`configs/qwen38_27b_rocm_kw_bf16.toml` / `checkpoints_qwen38_27b_kw_bf16`
(keyword only, 100/30, prune 0.5, batch 128). Copied BF16
baseline/steering caches. Did not mix judge scores into TPE.

## -1q. 2026-08-17 — BF16 judge v2: prune 0.5, drop 4-bit seeds

v1 50/50 done: only t18 60/100 @ 0.150 and t46 57/100 @ 0.150
scored; 4-bit seeds failed prescreen; prune 0.16 skipped almost
all unlocks (KL 0.27–0.97). Fresh study
`checkpoints_qwen38_27b_judge_bf16_v2`, copy BF16
baseline/steering/prefix, no seeds, prune 0.5, 50/15, batch 128.

## -1p. 2026-08-17 — Judge search switched to BF16 / 110G / auto-batch

User: 4-bit vs BF16 matters; use BF16, VRAM cap 110G, auto-detect
batch. Stopped 4-bit judge run. New
`configs/qwen38_27b_rocm_judge_bf16.toml`: `quant_method=none`,
`dtype=bfloat16`, `max_memory 110GB`, `batch_size=0`. Fresh
`checkpoints_qwen38_27b_judge_bf16` (no 4-bit residual reuse).
UMA guard `MAX_RSS=110` `MIN_FREE=12`.

## -1o. 2026-08-17 — MiMo-V2.5 LLM-judge 50-trial (AEON-style)

User provided OpenCode Go for `mimo-v2.5` as the judge. New study
`configs/qwen38_27b_rocm_judge.toml`: 4-bit, same v1 knobs, house
prescreen 30/8/19, **prune 0.16**, 50/15, seed t69+t127. Judge via
`llm_judge_base_url=https://opencode.ai/zen/go/v1` (key in env
`LLM_JUDGE_API_KEY`, not in git). json_schema works; urllib timeout
raised to 120s + User-Agent. Fresh dir
`checkpoints_qwen38_27b_judge`, v1 caches copied.

## -1n. 2026-08-17 — Stop 200; HF Qwen3.8 uncensored cards

User: stop and look at HF uncensored releases near KL 0.1.
Stopped default200 at ~trial 133; journal kept. Cards that matter:
JonathanColetti 12/100 @ first-token KL 0.1191 (Heretic 200,
**bf16**, o_proj+down only). trohrbaugh ARA 0/100 @ 0.0535
(full-weight, layers 26–56). 0bserverx RVN is stacked ARA on
that. Our 4-bit TPE never entered KL≤0.12.

## -1m. 2026-08-17 — Resume default-100 and raise budget to 200

User: continue and add 100 trials. Journal at stop: ~89 complete /
55 scored, `finished=false`, frozen `num_trials=100`. Continue
without overwrite; incoming config can raise `num_trials` (cli
patch). `qwen38_27b_rocm_default100.toml` now 200.

## -1l. 2026-08-17 — Resume Abliterix default-100 at 75/100

Wrapper hit max_runtime and killed the search mid trial 75.
Journal `finished=false`, 44 scored. Lowest refusals t13 2/100 @
KL 0.46; t50 4/100 @ 0.288; t69 16/100 @ 0.152. Still empty below
KL 0.12. Resume **without** `--overwrite-checkpoint`.

## -1k. 2026-08-16 — Classic Heretic 13-point front is worse; back to Abliterix

Classic Heretic (strength [0.8, 1.5]) after 13 scored trials:
lowest refusals **t8 29/100 @ KL 0.30**, t5 30/100 @ 0.36; KL≤0.05
is 97–100/100. Same tradeoff as Abliterix v1 but the 1.5 cap cannot
reach v1's o_proj≈5.3 pocket (17/100 @ 0.18). User: go back to
Abliterix. Restart `qwen38_27b_rocm_default100.toml` (100/30,
prescreen 30/8/19).

## -1j. 2026-08-16 — Classic Heretic (no ARA) 100-trial

User: try classic Heretic. Stopped the Abliterix default-100
restart. Launch `heretic-ara-lora` with `use_ara=false`, rank-1
LoRA, 4-bit, projected directions, strength [0.8, 1.5] (Heretic
hardcoded), 100/30, thinking off, skip prefix.
`config.qwen38-classic.toml` / `./run-qwen38-classic.sh`.

## -1i. 2026-08-16 — ARA-LoRA t1 is a dead end; resume Abliterix default-100

ARA-LoRA trial 1: **0/100 refusals, KL 11.69**, ~1h LBFGS, ETA ~100h
remaining. That is smashed-model, not a path to 10/100 @ 0.05. Stopped
heretic-ara-lora. Restarted Abliterix default recipe 100/30,
prescreen 30/8/19, `configs/qwen38_27b_rocm_default100.toml`.

## -1h. 2026-08-16 — Stop default-100; launch Qwen3.8 ARA-LoRA

User stopped the default 100-trial Abliterix search (killed
`run_qwen38.sh` / trial 1). Checked heretic remotes: upstream
`ara` is still `25979ad` (ARA-LoRA #332); no new ARA algorithm
commits. `origin/ara` is behind. `heretic-ara-lora` /
`sc117-ling-ara` already contains that ARA tip plus the
transformers 5.x `is_torch_fx_available` shim.

Run from `/home/s117/heretic-ara-lora` with muse-glimmer-env:
`config.qwen38-ara-lora.toml`, `./run-qwen38-ara.sh`. 4-bit
ARA-LoRA rank 128, thinking off, **100/30 trials** (first
launch was 30/10; user called that out, bumped and
relaunched). Local 800/100 splits. Restored missing
Evaluator fields on the master-merged Settings.

## -1g. 2026-08-16 — Qwen3.8 default 100-trial search

User: restart on the default recipe, 100 trials. Same v1/Qwen3.6
knobs (mean+LoRA, projected+winsorize, o_proj[1,6], down[1,5],
QKV off, KL prune 0.5). Fresh study in
`checkpoints_qwen38_27b_default100` — v1 journal left intact.
Copied v1 baseline/steering/prefix caches. Prescreen 30/8/19,
warmup 30. Config `configs/qwen38_27b_rocm_default100.toml`.
Launch: `./run_qwen38.sh configs/qwen38_27b_rocm_default100.toml`.

## -1f. 2026-08-16 — Restore Qwen3.8 prescreen to house 30/8/19

User house screen is 30 prompts, pass ≤8, prune ≥19 (`settings.py`
defaults; Ling / LFM / Muse v1). Qwen3.8 v1–v8 TOMLs were written
with Muse v3's 16/6/13 (faster 16-prompt screen from the batch-1
era). That was an agent copy, not a user change. Restored all
`configs/qwen38_27b_rocm*.toml` to 30/8/19. Completed v1–v8 journals
still used 16/6/13; next search uses 30.

---

## -1e. 2026-08-15 — Wire FLA Triton GDN + causal_conv1d on gfx1151

Qwen3.8 generate was 4 tok/s because transformers fell through to the
Python `for t in range(seq)` GDN path. `fla-core` 0.5.2 + ROCm Triton
3.7.1 already work on gfx1151 (`fused_recurrent` 0.04 ms vs Python
0.13 ms, max-abs 2.4e-4), but:

- transformers looks up `fla.ops.gated_delta_rule.recurrent_gated_delta_rule`
  (missing; only `fused_recurrent_gated_delta_rule`)
- `causal_conv1d` NVIDIA package is absent on ROCm

Alias added in heretic-env `fla`; muse-env got a `causal_conv1d` shim
that transposes `[B,D,T]↔[B,T,D]` onto FLA's Triton conv. Decode is
still mostly 4-bit weight-bandwidth bound; prefill/extract should
improve more.

Post-FLA batch retune (`scripts/bench_qwen38_batch.py`, fixed 32 tok):
1=4.2, 2=1.8, 4=3.5, 8=6.8, 16=13.0, 32=23.0, 64=39.2, 128=58.5,
256=77.3 tok/s. Locked `batch_size=256`.

v1 search (30/30): baseline 100/100. Best scored **trial 23 = 17/100,
KL 0.181 / val 0.172**. v2 is a manual peel: bake t23 LoRA → merged
`Qwen3.8-27B-t23`, fresh extract, weaker ranges.

v2 (30/30, ~6h): t23-merge baseline **17/100**. Incremental ≤10/100
looked good (t15 9/100 @ 0.015) but cumulative KL is still ~0.20.
User rejected peel-on-0.18. Deleted `Qwen3.8-27B-t23` merge + t23
LoRA. Stacked-search KL is vs the *loaded* base, not the original:
KL(C||A) ≠ KL(C||B)+KL(B||A). Ship gate is
`scripts/eval_qwen38_vs_original.py`: teacher-forced 3-token
KL(candidate || original) on the original's continuations, plus 100
eval refusals. v3 is a new first pass on the original weights:
o_proj ≤ 3.5, KL prune 0.08, target 0.03, 40 trials, reuse v1
residual/baseline cache.

v3 (40/40): **no scored trial**. 20 prescreen-pruned, 18 KL-pruned
(>0.08). Only t2/t32 stayed under the KL cap (0.075 / 0.068) and
both estimated **75/100**. The 10/100 @ 0.05 box is empty on this
o_proj+down_proj LoRA envelope vs the original.

v4: same original weights + KL prune 0.08, but unlock full-attn
Q/K/V ([0, 2.5]) so TPE can buy refusals without pushing every-layer
o_proj past 3.5. Reuse v1 residual/baseline cache.

v4 (40/40): **worse**. 30 prescreen-pruned, 10 KL-pruned, **zero**
trials under KL 0.08. QKV spends KL faster than it buys refusals.
v5: angular decoder-block, o_proj 0.35–1.0 (≤90°), same KL cap and
original residual cache. Runtime-only probe of the direction.

v5 (30/30): **empty**. 28 KL-pruned, 2 prescreen-pruned, lowest KL
0.111 (~69/100). Angular is louder than LoRA on this residual.
v6: re-extract with `response_pair` (same prompt, forced
compliance vs refusal continuation) then LoRA o_proj≤3.5, KL
prune 0.08. New steering cache.

v6 (30/30): **null**. Every trial 15–16/16 prescreen; the pair
direction does not move keyword refusals. v7: COSMIC extract +
adaptive angular (Qwen3.5-4B quality recipe), KL prune 0.08.

v7 (30/30): empty. 25 prescreen, 5 KL-prune. Best KL 0.096 at
~69/100. v8 killed mid-run (same empty pattern). Further recipe
churn stopped: v1 already was the Heretic-like default Pareto.

---

## -1d. 2026-08-15 — Qwen3.8-27B takes priority; Muse v6 is last Muse run

User: finish the current Muse v6 search, then stop. Do not start v7.
Qwen/Qwen3.8-27B is the next model (higher priority).

Weights go to `/run/media/s117/OS/Models/Qwen3.8-27B` (32 files, 51.77 GiB,
`Qwen3_5ForConditionalGeneration`, 64-layer GDN/full-attn hybrid — same
family as `configs/qwen3.6_27b.toml`). Isolated env is still
`/home/s117/muse-glimmer-env` (transformers 5.15, has `Qwen3_5*`).

Muse v6 was killed once Qwen weights finished (32/32 files, 51.77 GiB).
BF16 load suicided uma_guard at ~60% (`avail=22.4G`, `cuda_reserved=51G`,
`rss=44G` — UMA double-copy). 4-bit loads at ~17G VRAM. Auto-batch
measured `bs1=4 tok/s`, `bs2=2 tok/s` (GDN on ROCm does not scale);
first search is locked at `batch_size=1` + uma_guard. Thinking is
forced off via `scripts/qwen38_encode.py`.
Launch: `./run_qwen38.sh` (smoke / batch probe) or
`./run_qwen38.sh configs/qwen38_27b_rocm.toml`. Target ≤10/100 @ KL 0.05.

---

## -1c. 2026-08-14 — Muse v6: unclamp angular past 90°

t12's 65/100 is the 90° ceiling (strength 1.20 ≡ 5.16 after clamp).
Wired `steering.angular_overrotation` into the plain decoder-block hook
(`fraction` up to 2.0). v6 searches `attn.o_proj` in [1.0, 2.0] on the
v3 cache toward 10/100 @ KL 0.05.

---

## -1b. 2026-08-14 — Muse v5 BF16 direct search

Dedicated direct study (`configs/muse_glimmer_30b_rocm_v5_direct.toml`):
per-layer, `to=user`, down_proj kept, strength 0.25–1.60 in *direct*
units, TPE also picks standard / orba / biprojected. Reuses the v3
residual + 95/100 baseline caches. Launch with
`ABLITERIX_UMA_MIN_FREE_GB=4`.

**Result (18/18):** every trial pruned at 15–16/16. Five trials hit
15/16 (higher-strength standard/orba/biprojected ~1.2–1.5). No KL
numbers. Angular works because it rotates the **decoder-block output
(including the skip)**; that is not a rank-1 edit of `o_proj` /
`down_proj`. v3 trial 12's `down_proj` envelope is unused by angular
(hook strength comes from the first profile, `attn.o_proj`).

**Locked eval:** trial 12 angular on bnb, `to=user`, 32 decoder-block
hooks: **65/100** refusals (baseline 95/100). Search KL 0.0274
nats/token (16-prompt). Sidecar:
`artifacts/muse_glimmer_t12_angular.pt` via
`scripts/export_muse_glimmer_t12_artifact.py` /
`scripts/load_muse_glimmer_t12.py`.

---

## -1. 2026-08-14 — Muse Glimmer-30B v3 lock; BF16 direct bake did not transfer

v3 bnb + per-layer angular is still the only search that moved refusals
(trial 12: 10/16 ~62%, KL 0.0274). LoRA is eaten by 4×RMSNorm; v2 BF16
direct at Heretic 0.8–1.5 + skipped `to=user` prefix + auto-disabled
`down_proj` was 12/12 prune.

Tried to ship trial 12 by replaying the v3 residual cache as
`steering_mode=direct` on BF16 (`scripts/bake_muse_glimmer_t12.py`,
`configs/muse_glimmer_30b_rocm_v4_direct.toml`). Two mappings both
scored **15/16** on the same 16-prompt eval (raw angular 5.16, and
clamped-to-1.0 + row-norm restore to approximate 90° rotation). No
checkpoint written. Angular is a 90° activation rotation; it is not
a 1:1 weight edit on this gated-`o_proj` 4-norm decoder.

Runtime recipe remains `configs/muse_glimmer_30b_rocm_v3_best.toml`.
A shippable HF/GGUF needs a dedicated BF16 direct search, not a bake
of the angular numbers.

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
branch, not #95):**
- #103 `pr/bailing-steerable-modules` — Bailing `layer.attention` discovery
- #104 `pr/moe-router-profiling` — router dtype probe + bincount

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
