#!/bin/bash
# Ornith-1.5-35B-A3B on Linux ROCm gfx1151 (BF16, no bitsandbytes).
# Isolated transformers 5.15 env (muse-glimmer-env).
#
#   ./run_ornith15.sh
#
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-18}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-92}"
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH
unset AX_NON_INTERACTIVE

PYTHON="/home/s117/muse-glimmer-env/bin/python"
export PYTHONPATH="/home/s117/heretic-abliterix/src"

CONFIG="${1:-configs/ornith15_35b_rocm_mpoa.toml}"
if [[ $# -gt 0 ]]; then
  shift
fi
mkdir -p logs
LOGFILE="${ABLITERIX_LOG:-logs/ornith15_$(date +%Y%m%d_%H%M%S).log}"

echo "=== Abliterix Ornith-1.5-35B-A3B ==="
echo "  python: $PYTHON"
echo "  config: $CONFIG"
echo "  log:    $LOGFILE"
echo

PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_abliterix.py \
  --config "$CONFIG" \
  --seed 117 \
  --non-interactive \
  "$@" \
  2>&1 | tee "$LOGFILE"
