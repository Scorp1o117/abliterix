#!/bin/bash
# Run full abliterix search on Agents-A1.
set -euo pipefail
cd "$(dirname "$0")"

export QWEN35_MOE_UNPACK_EXPERTS=1
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

source /home/s117/heretic-env/bin/activate

CONFIG="${1:-configs/agents_a1_full.toml}"
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_agents_a1_$(date +%Y%m%d_%H%M%S).log}"

echo "=== Abliterix Agents-A1 Search ==="
echo "  config: $CONFIG"
echo "  log: $LOGFILE"
echo "  env: QWEN35_MOE_UNPACK_EXPERTS=$QWEN35_MOE_UNPACK_EXPERTS"
echo

PYTHONUNBUFFERED=1 abliterix \
  --config "$CONFIG" \
  2>&1 | ( trap '' INT; tee "$LOGFILE" )
