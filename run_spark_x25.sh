#!/bin/bash
# Run abliterix search on XHToken Spark-X2.5-4B (ROCm gfx1151).
# Usage: ./run_spark_x25.sh [configs/spark_x25_4b_rocm.toml|configs/spark_x25_4b_rocm_smoke.toml]
# Resume unfinished V30: ABLITERIX_OVERWRITE=0 ./run_spark_x25.sh configs/spark_x25_4b_lora_v30.toml
set -euo pipefail
cd "$(dirname "$0")"

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ABLITERIX_UMA_GUARD="${ABLITERIX_UMA_GUARD:-1}"
export ABLITERIX_UMA_MIN_FREE_GB="${ABLITERIX_UMA_MIN_FREE_GB:-24}"
export ABLITERIX_UMA_MAX_RSS_GB="${ABLITERIX_UMA_MAX_RSS_GB:-56}"
export ABLITERIX_UMA_POLL_SEC="${ABLITERIX_UMA_POLL_SEC:-0.1}"
export ABLITERIX_UMA_PSI_SOME_AVG10="${ABLITERIX_UMA_PSI_SOME_AVG10:-15}"
export ABLITERIX_UMA_MAX_SWAP_GB="${ABLITERIX_UMA_MAX_SWAP_GB:-0.5}"
# Spark advertises 1M context / 131k tokenizer max; cap KV prealloc on UMA.
export ABLITERIX_MAX_SEQ="${ABLITERIX_MAX_SEQ:-4096}"
unset PYTHONPATH

source /home/s117/heretic-env/bin/activate

CONFIG="${1:-configs/spark_x25_4b_rocm.toml}"
LOGFILE="${ABLITERIX_LOG:-/tmp/abliterix_spark_x25_$(date +%Y%m%d_%H%M%S).log}"
MODEL="${SPARK_MODEL:-/run/media/s117/OS/Models/Spark-X2.5-4B}"

echo "=== Abliterix Spark-X2.5-4B Search ==="
echo "  model:  $MODEL"
echo "  config: $CONFIG"
echo "  log:    $LOGFILE"
echo

python scripts/patch_spark_x25_hidden_states.py --model "$MODEL"
# trust_remote_code copies modeling_spark.py into a hash-named cache;
# drop it so the patched file is the one that actually loads.
rm -rf "$HOME/.cache/huggingface/modules/transformers_modules/Spark_hyphen_X2_dot_5_hyphen_4B"

# Hard cap THIS job so systemd-oomd cannot harvest Edge/Grok instead.
# MemoryMax is enforced by the kernel on this scope only.
SCOPE_UNIT="abliterix-spark-$(date +%s)"
SCOPE=(systemd-run --user --scope --collect --unit "$SCOPE_UNIT"
  -p "MemoryMax=${ABLITERIX_MEMORY_MAX:-64G}"
  -p "MemoryHigh=${ABLITERIX_MEMORY_HIGH:-48G}"
  -p "MemorySwapMax=${ABLITERIX_MEMORY_SWAP_MAX:-512M}")
echo "  cgroup: $SCOPE_UNIT  MemoryMax=${ABLITERIX_MEMORY_MAX:-64G}"
echo

# Default overwrites. Resume an unfinished study with ABLITERIX_OVERWRITE=0.
OVERWRITE_FLAG=()
if [ "${ABLITERIX_OVERWRITE:-1}" != "0" ]; then
  OVERWRITE_FLAG=(--overwrite-checkpoint)
  echo "  checkpoint: overwrite"
else
  echo "  checkpoint: resume (ABLITERIX_OVERWRITE=0)"
fi

if [ -t 1 ]; then
  "${SCOPE[@]}" -- env PYTHONUNBUFFERED=1 abliterix \
    --config "$CONFIG" \
    --seed 117 \
    "${OVERWRITE_FLAG[@]}"
else
  "${SCOPE[@]}" -- env PYTHONUNBUFFERED=1 abliterix \
    --config "$CONFIG" \
    --seed 117 \
    "${OVERWRITE_FLAG[@]}" \
    --non-interactive \
    2>&1 | ( trap '' INT; tee "$LOGFILE" )
fi
