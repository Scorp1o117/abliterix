#!/bin/bash
# Ling-3.0-flash v59: formal A/B of exact primary vs steered residual peel

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
DUMP_CKPT="checkpoints_ling30_flash_v58_steered_residual_dump"
FORMAL_BASELINE_CKPT="checkpoints_ling30_flash_v36_matched_canonical_full"
CKPT="checkpoints_ling30_flash_v59_steered_peel_formal"
mkdir -p "$CKPT" logs

for suffix in _generated_gate_responses.pt _concept_scorers.pt; do
  for f in "$LEGACY_CKPT"/*"$suffix"; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$CKPT/$(basename "$f")"
  done
done
for f in "$FORMAL_BASELINE_CKPT"/*_baseline.pt; do
  [[ -f "$f" ]] || continue
  ln -f "$f" "$CKPT/$(basename "$f")"
done

legacy=$(find "$LEGACY_CKPT" -maxdepth 1 -type f -name '*_steering.pt' -print -quit)
template=$(find "$DUMP_CKPT" -maxdepth 1 -type f -name '*_steering.pt' -print -quit)
dump="$DUMP_CKPT/steered_prefill_residuals.pt"
output="$CKPT/$(basename "$template")"
python scripts/build_ling30_v59_steering_cache.py \
  --legacy "$legacy" \
  --dump "$dump" \
  --rank2-template "$template" \
  --output "$output" \
  --exclude-source-indices 18

V59_OPTIMIZATION_JSON="{\"num_trials\":2,\"num_warmup_trials\":2,\"checkpoint_dir\":\"$CKPT\",\"refusal_prescreen_enabled\":true,\"refusal_prescreen_size\":60,\"refusal_prescreen_pass_max\":0,\"refusal_prescreen_prune_min\":0,\"refusal_prescreen_seed\":117,\"validation_kl_enabled\":false,\"generation_health_enabled\":false,\"thinking_leak_detection_enabled\":false,\"seed_trials\":[{\"direction_index\":0,\"vector_scope\":\"per layer\",\"vector_index\":30.25491805410134,\"attn.o_proj.max_weight\":1.20,\"attn.o_proj.max_weight_position\":38.01626146913563,\"attn.o_proj.min_weight\":0.3145186120734486,\"attn.o_proj.min_weight_distance\":18.562455387145544},{\"direction_index\":1,\"vector_scope\":\"per layer\",\"vector_index\":30.25491805410134,\"attn.o_proj.max_weight\":1.20,\"attn.o_proj.max_weight_position\":38.01626146913563,\"attn.o_proj.min_weight\":0.3145186120734486,\"attn.o_proj.min_weight_distance\":18.562455387145544}]}"

echo "=== Ling-3.0-flash v59 STEERED PEEL FORMAL A/B ==="
echo "  formal: target train[900:], fixed canonical 60 subset"
echo "  trials: exact primary vs steered residual peel from v58"
echo "  peel labels: v58 gate-on failures only; source 18 excluded"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V59_OPTIMIZATION_JSON" \
  --target-eval-prompts.split 'train[900:]' \
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
