#!/bin/bash
# Nex-N2.5-mini on Linux ROCm gfx1151 (BF16, no bitsandbytes).
# Isolated transformers 5.15 env (muse-glimmer-env).
#
#   ./run_nex25.sh                          # 正式配方 v1
#   ./run_nex25.sh configs/nex25_mini_rocm_smoke.toml
#
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# UMA guard: the 70 GB checkpoint plus extraction activations must never push
# the desktop session into the OOM reaper (redline). 96 GB RSS cap leaves
# ~26 GB of the 122 GB pool for the desktop.
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-18}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-96}"
# The 0.5 GB default swap trigger fired while the kernel reclaimed cold anon
# pages during the 70 GB mmap->CUDA load (2026-09-10 21:31: process RSS 4.1 GB,
# MemAvailable 62 GB — a false positive, not desktop pressure). 2 GB still
# trips long before the 7 GB swap is exhausted, and min_free/max_rss keep doing
# the real desktop protection.
export ABLITERIX_UMA_MAX_SWAP_GB="${ABLITERIX_UMA_MAX_SWAP_GB:-2}"
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH
unset AX_NON_INTERACTIVE

PYTHON="/home/s117/muse-glimmer-env/bin/python"
export PYTHONPATH="/home/s117/heretic-abliterix/src"

CONFIG="${1:-configs/nex25_mini_rocm_v1.toml}"
if [[ $# -gt 0 ]]; then
  shift
fi
mkdir -p logs
LOGFILE="${ABLITERIX_LOG:-logs/nex25_$(date +%Y%m%d_%H%M%S).log}"

echo "=== Abliterix Nex-N2.5-mini ==="
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
