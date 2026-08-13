#!/bin/bash
# Ling-3.0-flash v30: threshold-0.8 global gate, three matched strengths

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="configs/ling30_flash_rocm_v30_global_strength.toml"
CKPT="checkpoints_ling30_flash_v30_global_strength"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt _generated_gate_responses.pt; do
  for f in checkpoints_ling30_flash_v29_global_prompt_full/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v30 HIGH-CONFIDENCE GLOBAL STRENGTH PROBE ==="
echo "  gate: internal max-gap layer <=20, threshold 0.8, external guard"
echo "  strengths: exact v29, max=1.10, max=1.20"
echo "  evaluation: matched 30-prompt prescreen only"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
