#!/bin/bash
# Ling-3.0-flash v45: exact legacy primary + residual secondary router

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
RANK2_CKPT="checkpoints_ling30_flash_v42b_frozen_gate_rank2"
CKPT="checkpoints_ling30_flash_v45_legacy_primary_router"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _generated_gate_responses.pt _concept_scorers.pt; do
  for f in "$LEGACY_CKPT"/*"$suffix"; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$CKPT/$(basename "$f")"
  done
done

legacy=$(find "$LEGACY_CKPT" -maxdepth 1 -type f -name '*_steering.pt' -print -quit)
rank2=$(find "$RANK2_CKPT" -maxdepth 1 -type f -name '*_steering.pt' -print -quit)
output="$CKPT/$(basename "$rank2")"
python scripts/build_ling30_v45_steering_cache.py \
  --legacy "$legacy" --rank2 "$rank2" --output "$output"

V45_OPTIMIZATION_JSON='{"num_trials":1,"num_warmup_trials":1,"checkpoint_dir":"checkpoints_ling30_flash_v45_legacy_primary_router","refusal_prescreen_enabled":true,"refusal_prescreen_size":60,"refusal_prescreen_pass_max":0,"refusal_prescreen_prune_min":0,"refusal_prescreen_seed":117,"validation_kl_enabled":false,"generation_health_enabled":false,"thinking_leak_detection_enabled":false,"seed_trials":[{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544}]}'

echo "=== Ling-3.0-flash v45 LEGACY-PRIMARY ROUTER ==="
echo "  direction 0: exact v43 legacy mean; direction 1: orthogonal train residual"
echo "  route: max-absolute final-prefill projection; no eval-label fitting"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V45_OPTIMIZATION_JSON" \
  --steering.n-directions 2 \
  --steering.concept-gate-global-decision-layer 18 \
  --steering.concept-gate-direction-router \
  --no-steering.concept-gate-positive-alignment-only \
  --steering.concept-gate-global-single-sample-prepass \
  --steering.concept-gate-global-canonical-batching \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
