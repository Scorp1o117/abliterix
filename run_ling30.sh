#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH   # CRITICAL: strip Hermes venv leak
source /home/s117/heretic-env/bin/activate
CONFIG="${1:-configs/ling30_flash_rocm.toml}"
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_ling30_$(date +%Y%m%d_%H%M%S).log}"
PYTHONUNBUFFERED=1 abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --overwrite-checkpoint \
  2>&1 | ( trap '' INT; tee "$LOGFILE" )
