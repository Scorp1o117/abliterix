#!/bin/bash
# Run abliterix search on LiquidAI LFM2.5-2.6B (ROCm gfx1151).
# Usage: ./run_lfm2.5.sh [configs/lfm2.5_2.6b_rocm.toml|configs/lfm2.5_2.6b_rocm_smoke.toml]
set -euo pipefail
cd "$(dirname "$0")"

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# CRITICAL: strip Hermes PYTHONPATH leakage (venv python3.11 site-packages)
unset PYTHONPATH

source /home/s117/heretic-env/bin/activate

CONFIG="${1:-configs/lfm2.5_2.6b_rocm.toml}"
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_lfm25_$(date +%Y%m%d_%H%M%S).log}"

echo "=== Abliterix LFM2.5-2.6B Search ==="
echo "  config: $CONFIG"
echo "  log: $LOGFILE"
echo

PYTHONUNBUFFERED=1 abliterix \
  --config "$CONFIG" \
  --non-interactive \
  --overwrite-checkpoint \
  2>&1 | ( trap '' INT; tee "$LOGFILE" )
