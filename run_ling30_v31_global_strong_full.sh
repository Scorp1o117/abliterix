#!/bin/bash
# Ling-3.0-flash v31: full evaluation of v30's max=1.20 winner

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
CKPT="checkpoints_ling30_flash_v31_global_strong_full"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v30_global_strength/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

# Passing the complete nested object avoids ambiguous list[dict] CLI parsing
# for seed_trials while leaving the steering/data sections in the TOML intact.
V31_OPTIMIZATION_JSON='{"num_trials":1,"num_warmup_trials":1,"checkpoint_dir":"checkpoints_ling30_flash_v31_global_strong_full","refusal_prescreen_enabled":true,"refusal_prescreen_size":30,"refusal_prescreen_pass_max":30,"refusal_prescreen_prune_min":31,"refusal_prescreen_seed":117,"validation_kl_enabled":true,"validation_kl_size":30,"generation_health_enabled":true,"thinking_leak_detection_enabled":true,"seed_trials":[{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544}]}'

echo "=== Ling-3.0-flash v31 STRONG GLOBAL GATE FULL EVAL ==="
echo "  gate: threshold 0.8, internal max-gap layer <=20, external guard"
echo "  steering: v30 winner, max=1.20"
echo "  evaluation: primary KL + full harmful + validation KL + health"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V31_OPTIMIZATION_JSON" \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
