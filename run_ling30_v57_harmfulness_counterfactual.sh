#!/bin/bash
# Ling-3.0-flash v57: train-holdout exact-primary vs Zhao harmfulness

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
LEGACY_CKPT="checkpoints_ling30_flash_v43_legacy_rank1_frozen_gate"
TEMPLATE_CKPT="checkpoints_ling30_flash_v46_train_counterfactual_directions"
CKPT="checkpoints_ling30_flash_v57_harmfulness_counterfactual"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _generated_gate_responses.pt _concept_scorers.pt; do
  for f in "$LEGACY_CKPT"/*"$suffix"; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$CKPT/$(basename "$f")"
  done
done

legacy=$(find "$LEGACY_CKPT" -maxdepth 1 -type f -name '*_steering.pt' -print -quit)
template=$(find "$TEMPLATE_CKPT" -maxdepth 1 -type f -name '*_steering.pt' -print -quit)
output="$CKPT/$(basename "$template")"
python scripts/build_ling30_v57_steering_cache.py \
  --legacy "$legacy" --rank2-template "$template" --output "$output"

V57_OPTIMIZATION_JSON="{\"num_trials\":2,\"num_warmup_trials\":2,\"checkpoint_dir\":\"$CKPT\",\"refusal_prescreen_enabled\":true,\"refusal_prescreen_size\":60,\"refusal_prescreen_pass_max\":0,\"refusal_prescreen_prune_min\":0,\"refusal_prescreen_seed\":117,\"validation_kl_enabled\":false,\"generation_health_enabled\":false,\"thinking_leak_detection_enabled\":false,\"seed_trials\":[{\"direction_index\":0,\"vector_scope\":\"per layer\",\"vector_index\":30.25491805410134,\"attn.o_proj.max_weight\":1.20,\"attn.o_proj.max_weight_position\":38.01626146913563,\"attn.o_proj.min_weight\":0.3145186120734486,\"attn.o_proj.min_weight_distance\":18.562455387145544},{\"direction_index\":1,\"vector_scope\":\"per layer\",\"vector_index\":30.25491805410134,\"attn.o_proj.max_weight\":1.20,\"attn.o_proj.max_weight_position\":38.01626146913563,\"attn.o_proj.min_weight\":0.3145186120734486,\"attn.o_proj.min_weight_distance\":18.562455387145544}]}"

echo "=== Ling-3.0-flash v57 HARMFULNESS COUNTERFACTUAL ==="
echo "  calibration: target train[800:900], fixed canonical 60 subset"
echo "  trials: exact primary index 0 vs Zhao harmfulness index 1"
echo "  guard: eval[900:] is not used; no sequential joint removal yet"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V57_OPTIMIZATION_JSON" \
  --target-eval-prompts.split 'train[800:900]' \
  --steering.n-directions 2 \
  --steering.concept-gate-global-decision-layer 18 \
  --steering.search-concept-gate-fixed-direction \
  --no-steering.concept-gate-positive-alignment-only \
  --steering.concept-gate-global-single-sample-prepass \
  --steering.concept-gate-global-canonical-batching \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
