#!/bin/bash
# Ling-3.0-flash v14: matched KDA/MLA/MLP/shared-expert prescreen sweep

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v14_site_sweep.toml}"
CKPT="checkpoints_ling30_flash_v14_site_sweep"
mkdir -p "$CKPT" logs

for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v13_adaptive_runtime/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v14 ARCHITECTURE SITE SWEEP (4 matched trials) ==="
echo "  sites: KDA / MLA / MLP / shared expert output"
echo "  profile: exact v13 t2 strong/wide intervention"
echo "  evaluation: fixed 30-prompt prescreen only"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
