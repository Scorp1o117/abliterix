#!/bin/bash
# Ling-3.0-flash v19: low-threshold prompt latch, full-100 harmful probe

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

# v19 deliberately reuses the v18 recipe and overrides only the threshold and
# checkpoint path on the CLI. The complete resolved recipe is journaled.
CONFIG="configs/ling30_flash_rocm_v18_prompt_gate.toml"
CKPT="checkpoints_ling30_flash_v19_prompt_gate_low"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v13_adaptive_runtime/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v19 LOW-THRESHOLD PROMPT GATE ==="
echo "  base recipe: v18 prompt latch, exact v13 t2"
echo "  override: threshold 0.05, full 100 harmful"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --steering.concept-gate-threshold 0.05 \
  --optimization.checkpoint-dir "$CKPT" \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
