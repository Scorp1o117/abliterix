#!/bin/bash
# Ling-3.0-flash v42b: frozen-gate paired treatment (rank 2)

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
CONTROL_CKPT="checkpoints_ling30_flash_v42a_frozen_gate_rank1"
CKPT="checkpoints_ling30_flash_v42b_frozen_gate_rank2"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v36_matched_canonical_full/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done
for f in checkpoints_ling30_flash_v41_rank2_subspace_probe/*_steering.pt; do
  [[ -f "$f" ]] || continue
  bn=$(basename "$f")
  [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
done

scorer_cache=$(find "$CONTROL_CKPT" -maxdepth 1 -type f -name '*_concept_scorers.pt' -print -quit)
if [[ -z "$scorer_cache" || ! -f "$scorer_cache" ]]; then
  echo "v42a scorer cache is required before v42b" >&2
  exit 2
fi
cp -f "$scorer_cache" "$CKPT/$(basename "$scorer_cache")"

V42_OPTIMIZATION_JSON='{"num_trials":1,"num_warmup_trials":1,"checkpoint_dir":"checkpoints_ling30_flash_v42b_frozen_gate_rank2","refusal_prescreen_enabled":true,"refusal_prescreen_size":60,"refusal_prescreen_pass_max":0,"refusal_prescreen_prune_min":0,"refusal_prescreen_seed":117,"validation_kl_enabled":false,"generation_health_enabled":false,"thinking_leak_detection_enabled":false,"seed_trials":[{"vector_scope":"per layer","vector_index":30.25491805410134,"attn.o_proj.max_weight":1.20,"attn.o_proj.max_weight_position":38.01626146913563,"attn.o_proj.min_weight":0.3145186120734486,"attn.o_proj.min_weight_distance":18.562455387145544}]}'

echo "=== Ling-3.0-flash v42b FROZEN-GATE RANK-2 TREATMENT ==="
echo "  gate: exact scorer cache from v42a; fixed decision layer 18"
echo "  intervention: rank 2, sign-agnostic, exact canonical winner profile"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --optimization "$V42_OPTIMIZATION_JSON" \
  --steering.n-directions 2 \
  --steering.concept-gate-global-decision-layer 18 \
  --no-steering.concept-gate-positive-alignment-only \
  --steering.concept-gate-global-single-sample-prepass \
  --steering.concept-gate-global-canonical-batching \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
