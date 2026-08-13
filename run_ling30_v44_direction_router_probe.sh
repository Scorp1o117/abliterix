#!/bin/bash
# Ling-3.0-flash v44: frozen-gate rank-2 single-direction router probe

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
GATE_CKPT="checkpoints_ling30_flash_v42a_frozen_gate_rank1"
RANK2_CKPT="checkpoints_ling30_flash_v42b_frozen_gate_rank2"
CKPT="checkpoints_ling30_flash_v44_direction_router_probe"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _generated_gate_responses.pt _concept_scorers.pt; do
  for f in "$GATE_CKPT"/*"$suffix"; do
    [[ -f "$f" ]] || continue
    cp -f "$f" "$CKPT/$(basename "$f")"
  done
done
for f in "$RANK2_CKPT"/*_steering.pt; do
  [[ -f "$f" ]] || continue
  cp -f "$f" "$CKPT/$(basename "$f")"
done

V44_OPTIMIZATION_JSON='{"num_trials":1,"num_warmup_trials":1,"checkpoint_dir":"checkpoints_ling30_flash_v44_direction_router_probe","refusal_prescreen_enabled":true,"refusal_prescreen_size":60,"refusal_prescreen_pass_max":0,"refusal_prescreen_prune_min":0,"refusal_prescreen_seed":117,"validation_kl_enabled":false,"generation_health_enabled":false,"thinking_leak_detection_enabled":false,"seed_trials":[{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544}]}'

echo "=== Ling-3.0-flash v44 SINGLE-DIRECTION ROUTER PROBE ==="
echo "  gate: exact frozen scorer; fixed decision layer 18"
echo "  route: max-absolute final-prefill projection; one of two train directions"
echo "  guard: no eval outcome labels used for routing"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V44_OPTIMIZATION_JSON" \
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
