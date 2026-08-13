#!/bin/bash
# Ling-3.0-flash v47: train-holdout failure-conditioned primary variants

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
SOURCE_CKPT="checkpoints_ling30_flash_v46_train_counterfactual_directions"
CKPT="checkpoints_ling30_flash_v47_failure_conditioned_variants"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt _concept_scorers.pt; do
  for f in "$SOURCE_CKPT"/*"$suffix"; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$CKPT/$(basename "$f")"
  done
done

REFUSALS='[26,51,20,15,67,72,42,8,94,53,18,56,7,37,1,2,82]'
COMPLIANCES='[30,23,21,71,52,89,44,58,83,4,16,47,55,80,49,54,0,78,75,98,61,76,29,22,43,57,5,39,85,13,97,70,96,41,88,19,24,35,74,90,9,92,73]'
ALPHAS='[0.25,0.5,1.0,2.0]'

COMMON='"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544'
V47_OPTIMIZATION_JSON="{\"num_trials\":5,\"num_warmup_trials\":5,\"checkpoint_dir\":\"$CKPT\",\"refusal_prescreen_enabled\":true,\"refusal_prescreen_size\":60,\"refusal_prescreen_pass_max\":0,\"refusal_prescreen_prune_min\":0,\"refusal_prescreen_seed\":117,\"validation_kl_enabled\":false,\"generation_health_enabled\":false,\"thinking_leak_detection_enabled\":false,\"seed_trials\":[{\"steering_variant\":\"primary\",$COMMON},{\"steering_variant\":\"failure_alpha_0p25\",$COMMON},{\"steering_variant\":\"failure_alpha_0p5\",$COMMON},{\"steering_variant\":\"failure_alpha_1\",$COMMON},{\"steering_variant\":\"failure_alpha_2\",$COMMON}]}"

echo "=== Ling-3.0-flash v47 FAILURE-CONDITIONED VARIANTS ==="
echo "  calibration: target train[800:900], same canonical 60 labels as v46"
echo "  variants: exact primary + alpha*(failure-success orthogonal residual)"
echo "  guard: eval[900:] is not used; no prompt text is persisted"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V47_OPTIMIZATION_JSON" \
  --target-eval-prompts.split 'train[800:900]' \
  --steering.n-directions 2 \
  --steering.concept-gate-global-decision-layer 18 \
  --no-steering.concept-gate-positive-alignment-only \
  --steering.concept-gate-global-single-sample-prepass \
  --steering.concept-gate-global-canonical-batching \
  --steering.calibration-failure-refusal-indices "$REFUSALS" \
  --steering.calibration-failure-compliance-indices "$COMPLIANCES" \
  --steering.calibration-failure-alphas "$ALPHAS" \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
