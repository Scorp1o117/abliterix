#!/bin/bash
# Ling-3.0-flash v32: robust conditional profile-shape search

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
CKPT="checkpoints_ling30_flash_v32_global_profile_search"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v31_global_strong_full/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

V32_OPTIMIZATION_JSON='{"num_trials":8,"num_warmup_trials":8,"checkpoint_dir":"checkpoints_ling30_flash_v32_global_profile_search","refusal_prescreen_enabled":true,"refusal_prescreen_size":60,"refusal_prescreen_pass_max":0,"refusal_prescreen_prune_min":0,"refusal_prescreen_seed":117,"validation_kl_enabled":false,"generation_health_enabled":false,"thinking_leak_detection_enabled":false,"seed_trials":[{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":0.9698154112336964,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544},{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544}]}'

echo "=== Ling-3.0-flash v32 CONDITIONAL PROFILE SEARCH ==="
echo "  gate: threshold 0.8, internal max-gap layer <=20, external guard"
echo "  search: max 0.95..1.50 plus peak/min-fraction/falloff"
echo "  evaluation: fixed 60-prompt prescreen; 2 matched seeds + 6 warmups"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V32_OPTIMIZATION_JSON" \
  --steering.component-strength-ranges '{"attn.o_proj":[0.95,1.50]}' \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
