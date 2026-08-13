#!/bin/bash
# Ling-3.0-flash v16: learned concept gate, matched v13 t2 prescreen

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v16_concept_gate.toml}"
CKPT="checkpoints_ling30_flash_v16_concept_gate"
mkdir -p "$CKPT" logs

# The baseline and residual tensors have the same model, datasets, extraction
# settings, and mean-vector provenance as v13. Abliterix validates cache keys.
for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v13_adaptive_runtime/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v16 CONCEPT-GATED BLOCK PROBE ==="
echo "  site/profile: decoder-block output, exact v13 t2"
echo "  gate sweep: harmful-state thresholds 0.05 / 0.15 / 0.30 + positive alignment"
echo "  evaluation: fixed 30-prompt prescreen + gate activation telemetry"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
