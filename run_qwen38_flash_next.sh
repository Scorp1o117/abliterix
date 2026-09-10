#!/bin/bash
# Qwen3.8-Flash-Next mergeable LoRA search. UMA guard stays on.
set -euo pipefail
cd "$(dirname "$0")"
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ABLITERIX_UMA_GUARD=1
export ABLITERIX_UMA_MIN_FREE_GB=20
export ABLITERIX_UMA_MAX_RSS_GB=88
export ABLITERIX_MAX_SEQ=2048
unset PYTHONPATH
source /home/s117/heretic-env/bin/activate
CONFIG="${1:-configs/qwen38_flash_next_rocm.toml}"
LOGFILE="${ABLITERIX_LOG:-logs/abliterix_qwen38_flash_next_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOGFILE")" logs
PYTHONUNBUFFERED=1 abliterix \
  --config "$CONFIG" \
  --seed 117 \
  2>&1 | ( trap '' INT; tee "$LOGFILE" )
