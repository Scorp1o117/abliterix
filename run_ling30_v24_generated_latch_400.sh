#!/bin/bash
# Ling-3.0-flash v24: 400/class generated trajectory prompt latch + external guard

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v22_generated_gate.toml"
CKPT="checkpoints_ling30_flash_v24_generated_latch_400"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v22_generated_gate/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v24 GENERATED LATCH 400/CLASS ==="
echo "  coverage: 400 prompts/class, grouped 320/80"
echo "  guard: mixed + internal prefill + external eval prefill"
echo "  generation: full 100 only if all guards pass"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --steering.concept-gate-scope prompt \
  --steering.concept-gate-generated-prompts-per-class 400 \
  --optimization.checkpoint-dir "$CKPT" \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
