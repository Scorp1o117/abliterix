#!/bin/bash
# Ling-3.0-flash v39: canonical per-prompt refusal outcome map

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
CKPT="checkpoints_ling30_flash_v39_outcome_map"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v36_matched_canonical_full/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

V39_OPTIMIZATION_JSON='{"num_trials":1,"num_warmup_trials":1,"checkpoint_dir":"checkpoints_ling30_flash_v39_outcome_map","refusal_prescreen_enabled":true,"refusal_prescreen_size":60,"refusal_prescreen_pass_max":0,"refusal_prescreen_prune_min":0,"refusal_prescreen_seed":117,"validation_kl_enabled":false,"generation_health_enabled":false,"thinking_leak_detection_enabled":false,"seed_trials":[{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544}]}'

echo "=== Ling-3.0-flash v39 CANONICAL OUTCOME MAP ==="
echo "  profile: exact v36 canonical winner"
echo "  telemetry: persistent refusal/compliance source indices"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V39_OPTIMIZATION_JSON" \
  --steering.concept-gate-global-single-sample-prepass \
  --steering.concept-gate-global-canonical-batching \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
