#!/bin/bash
# Nex-N2.5-mini leftover peel (stage 2) — see scripts/nex25_peel.py
#
#   ./run_nex25_peel.sh --trial 0              # all grid points
#   ./run_nex25_peel.sh --trial 0 --grid 3     # first three grid points
#
# Requires the stage-1 bake to exist first (scripts/nex25_bake.py + run_nex25.sh).
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-18}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-96}"
export ABLITERIX_UMA_MAX_SWAP_GB="${ABLITERIX_UMA_MAX_SWAP_GB:-2}"
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH
unset AX_NON_INTERACTIVE

PYTHON="/home/s117/muse-glimmer-env/bin/python"
mkdir -p logs
LOGFILE="${ABLITERIX_LOG:-logs/nex25_peel_$(date +%Y%m%d_%H%M%S).log}"

echo "=== Abliterix Nex-N2.5-mini leftover peel (stage 2) ==="
echo "  python: $PYTHON"
echo "  log:    $LOGFILE"
echo

PYTHONUNBUFFERED=1 "$PYTHON" scripts/nex25_peel.py "$@" 2>&1 | tee "$LOGFILE"
