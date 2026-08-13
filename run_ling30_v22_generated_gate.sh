#!/bin/bash
# Ling-3.0-flash v22: generated benign/harmful early-trajectory gate

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v22_generated_gate.toml}"
CKPT="checkpoints_ling30_flash_v22_generated_gate"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v20_trajectory_gate/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v22 GENERATED BENIGN/HARMFUL TRAJECTORY GATE ==="
echo "  classifier data: 200 prompts/class, 12 generated tokens"
echo "  states: final-prefill x4 + first 8 decode tokens"
echo "  direction/site/profile: exact v13 prompt mean t2"
echo "  evaluation if both guards pass: all 100 harmful prompts"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
