#!/bin/bash
# Abliterix search on XHToken Spark-X2.5-1.7B (ROCm gfx1151).
# Usage: ./scripts/run_spark_x25_1p7b.sh [configs/spark_x25_1p7b_lora_v1.toml]
# Resume: ABLITERIX_OVERWRITE=0 ./scripts/run_spark_x25_1p7b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-24}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-56}"
export ABLITERIX_UMA_POLL_SEC="${ABLITERIX_UMA_POLL_SEC:-0.1}"
export ABLITERIX_UMA_PSI_SOME_AVG10="${ABLITERIX_UMA_PSI_SOME_AVG10:-15}"
export ABLITERIX_UMA_MAX_SWAP_GB="${ABLITERIX_UMA_MAX_SWAP_GB:-0.5}"
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH

source /home/s117/heretic-env/bin/activate

CONFIG="${1:-configs/spark_x25_1p7b_lora_v1.toml}"
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_spark_x25_1p7b_$(date +%Y%m%d_%H%M%S).log}"
MODEL="${SPARK_MODEL:-/run/media/s117/OS/Models/Spark-X2.5-1.7B}"

echo "=== Abliterix Spark-X2.5-1.7B Search ==="
echo "  model:  $MODEL"
echo "  config: $CONFIG"
echo "  log:    $LOGFILE"
echo

python scripts/patch_spark_x25_hidden_states.py --model "$MODEL"
rm -rf "$HOME/.cache/huggingface/modules/transformers_modules/Spark_hyphen_X2_dot_5_hyphen_1_dot_7B"

SCOPE_UNIT="abliterix-spark-1p7b-$(date +%s)"
SCOPE=(systemd-run --user --scope --collect --unit "$SCOPE_UNIT"
  -p "MemoryMax=${ABLITERIX_MEMORY_MAX:-64G}"
  -p "MemoryHigh=${ABLITERIX_MEMORY_HIGH:-48G}"
  -p "MemorySwapMax=${ABLITERIX_MEMORY_SWAP_MAX:-512M}")
echo "  cgroup: $SCOPE_UNIT  MemoryMax=${ABLITERIX_MEMORY_MAX:-64G}"
echo

OVERWRITE_FLAG=()
if [ "${ABLITERIX_OVERWRITE:-1}" != "0" ]; then
  OVERWRITE_FLAG=(--overwrite-checkpoint)
  echo "  checkpoint: overwrite"
else
  echo "  checkpoint: resume (ABLITERIX_OVERWRITE=0)"
fi

if [ -t 1 ]; then
  "${SCOPE[@]}" -- env PYTHONUNBUFFERED=1 SPARK_MODEL="$MODEL" abliterix \
    --config "$CONFIG" \
    --seed 117 \
    "${OVERWRITE_FLAG[@]}"
else
  "${SCOPE[@]}" -- env PYTHONUNBUFFERED=1 SPARK_MODEL="$MODEL" abliterix \
    --config "$CONFIG" \
    --seed 117 \
    "${OVERWRITE_FLAG[@]}" \
    --non-interactive \
    2>&1 | ( trap '' INT; tee -a "$LOGFILE" )
fi
