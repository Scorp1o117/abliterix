#!/bin/bash
# Ling-3.0-flash v55: anonymous signed-cosine and gate-score telemetry

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
SOURCE_CKPT="checkpoints_ling30_flash_v54_strength_feature_probe"
CKPT="checkpoints_ling30_flash_v55_signed_gate_feature_probe"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt _concept_scorers.pt; do
  for f in "$SOURCE_CKPT"/*"$suffix"; do
    [[ -f "$f" ]] || continue
    ln -f "$f" "$CKPT/$(basename "$f")"
  done
done

V55_OPTIMIZATION_JSON="{\"num_trials\":1,\"num_warmup_trials\":1,\"checkpoint_dir\":\"$CKPT\",\"refusal_prescreen_enabled\":true,\"refusal_prescreen_size\":60,\"refusal_prescreen_pass_max\":0,\"refusal_prescreen_prune_min\":0,\"refusal_prescreen_seed\":117,\"validation_kl_enabled\":false,\"generation_health_enabled\":false,\"thinking_leak_detection_enabled\":false,\"seed_trials\":[{\"vector_scope\":\"per layer\",\"vector_index\":30.25491805410134,\"attn.o_proj.max_weight\":1.20,\"attn.o_proj.max_weight_position\":38.01626146913563,\"attn.o_proj.min_weight\":0.3145186120734486,\"attn.o_proj.min_weight_distance\":18.562455387145544}]}"

echo "=== Ling-3.0-flash v55 ANONYMOUS SIGNED/GATE FEATURE PROBE ==="
echo "  calibration: target train[800:900], fixed canonical 60 subset"
echo "  intervention: linear projection 1.20 reproduction"
echo "  telemetry: anonymous layer-18 abs/signed cosine + gate score only"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V55_OPTIMIZATION_JSON" \
  --target-eval-prompts.split 'train[800:900]' \
  --steering.n-directions 2 \
  --steering.concept-gate-fixed-direction-index 0 \
  --steering.concept-gate-global-decision-layer 18 \
  --no-steering.concept-gate-positive-alignment-only \
  --steering.concept-gate-angular-overrotation \
  --steering.concept-gate-intervention-geometry linear_projection \
  --steering.component-strength-ranges '{"attn.o_proj":[0.95,1.40]}' \
  --steering.concept-gate-global-single-sample-prepass \
  --steering.concept-gate-global-canonical-batching \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
