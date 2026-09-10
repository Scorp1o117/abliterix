#!/bin/bash
# Muse Glimmer-30B on Linux ROCm gfx1151.
# Uses the isolated transformers 5.15 env (do not activate heretic-env).
#
#   ./run_muse_glimmer.sh                         # smoke
#   ./run_muse_glimmer.sh configs/muse_glimmer_30b_rocm.toml
#
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Suicide this job before the kernel OOM-kills the desktop.
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-24}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-88}"
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH
unset AX_NON_INTERACTIVE

PYTHON="/home/s117/muse-glimmer-env/bin/python"
export PYTHONPATH="/home/s117/heretic-abliterix/src"

CONFIG="${1:-configs/muse_glimmer_30b_rocm_smoke.toml}"
if [[ $# -gt 0 ]]; then
  shift
fi
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_muse_glimmer_$(date +%Y%m%d_%H%M%S).log}"

# pydantic CliSettingsSource sits above TOML: a missing --non-interactive
# flag becomes False and overrides the file. Smoke/v2 must pass it.
EXTRA=()
if [[ "$CONFIG" == *smoke* || "$CONFIG" == *_v2.toml || "$CONFIG" == *_v3.toml || "$CONFIG" == *_v3_best.toml || "$CONFIG" == *_v4_direct.toml || "$CONFIG" == *_v5_direct.toml || "$CONFIG" == *_v6_overrot.toml ]]; then
  EXTRA+=(--non-interactive)
fi

echo "=== Abliterix Muse Glimmer-30B ==="
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
