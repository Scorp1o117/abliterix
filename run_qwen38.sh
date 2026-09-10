#!/bin/bash
# Qwen3.8-27B on Linux ROCm gfx1151.
# Uses the isolated transformers 5.15 env (do not activate heretic-env).
#
#   ./run_qwen38.sh                                    # smoke
#   ./run_qwen38.sh configs/qwen38_27b_rocm.toml       # first search
#
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-24}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-88}"
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH
unset AX_NON_INTERACTIVE

PYTHON="/home/s117/muse-glimmer-env/bin/python"
export PYTHONPATH="/home/s117/heretic-abliterix/src"

CONFIG="${1:-configs/qwen38_27b_rocm_smoke.toml}"
if [[ $# -gt 0 ]]; then
  shift
fi
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_qwen38_$(date +%Y%m%d_%H%M%S).log}"

EXTRA=()
if [[ "$CONFIG" == *smoke* || "$CONFIG" == *qwen38_27b_rocm*.toml ]]; then
  EXTRA+=(--non-interactive)
fi

echo "=== Abliterix Qwen3.8-27B ==="
echo "  python: $PYTHON"
echo "  config: $CONFIG"
echo "  extra:  ${EXTRA[*]:-(none)}"
echo "  log:    $LOGFILE"
echo

if [ -t 1 ]; then
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_abliterix.py \
    --config "$CONFIG" \
    --seed 117 \
    --overwrite-checkpoint \
    "${EXTRA[@]}" \
    "$@"
else
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_abliterix.py \
    --config "$CONFIG" \
    --seed 117 \
    --overwrite-checkpoint \
    "${EXTRA[@]}" \
    "$@" \
    2>&1 | ( trap '' INT; tee "$LOGFILE" )
fi
