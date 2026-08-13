#!/bin/bash
# Ling-3.0-flash v29: full quality evaluation of the v28 max-gap global gate

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v22_generated_gate.toml"
CKPT="checkpoints_ling30_flash_v29_global_prompt_full"
mkdir -p "$CKPT" logs

# Reuse immutable baseline/direction/generated-response caches.  Scorers and
# layer selection are deliberately rebuilt so the v28 choice is reproduced.
for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v28_global_prompt_maxgap/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v29 MAX-GAP GLOBAL GATE FULL EVAL ==="
echo "  selection: reproduce v28 using internal max-gap over layers 0..20"
echo "  prescreen: 30 prompts, always pass (telemetry only)"
echo "  evaluation: primary KL + full harmful + validation KL + health"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --steering.concept-gate-scope global_prompt \
  --steering.concept-gate-global-decision-layer -2 \
  --steering.concept-gate-global-candidate-max-layer 20 \
  --steering.concept-gate-generated-prompts-per-class 400 \
  --optimization.checkpoint-dir "$CKPT" \
  --optimization.refusal-prescreen-size 30 \
  --optimization.refusal-prescreen-pass-max 30 \
  --optimization.refusal-prescreen-prune-min 31 \
  --optimization.validation-kl-enabled \
  --optimization.validation-kl-size 30 \
  --optimization.generation-health-enabled \
  --optimization.thinking-leak-detection-enabled \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
