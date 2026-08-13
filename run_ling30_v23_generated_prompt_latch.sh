#!/bin/bash
# Ling-3.0-flash v23: generated-trajectory classifier with prompt latch

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v22_generated_gate.toml"
CKPT="checkpoints_ling30_flash_v23_generated_prompt_latch"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v22_generated_gate/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v23 GENERATED-TRAJECTORY PROMPT LATCH ==="
echo "  classifier/data/profile: identical to v22"
echo "  only runtime change: token gate -> final-prefill prompt latch"
echo "  evaluation: all 100 harmful prompts"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --steering.concept-gate-scope prompt \
  --optimization.checkpoint-dir "$CKPT" \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
