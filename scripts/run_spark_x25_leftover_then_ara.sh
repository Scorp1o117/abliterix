#!/bin/bash
# T580 leftover peel, then ARA T49/T196 3-token retest if leftover does not bake.
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

MODEL="${SPARK_MODEL:-/run/media/s117/OS/Models/Spark-X2.5-4B}"
python scripts/patch_spark_x25_hidden_states.py --model "$MODEL"
rm -rf "$HOME/.cache/huggingface/modules/transformers_modules/Spark_hyphen_X2_dot_5_hyphen_4B"

echo "=== leftover T580 ==="
python scripts/sweep_spark_x25_t580_leftover.py

LEFTOVER_JSON=logs/spark_x25_t580_leftover.json
if python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('baked') else 1)" "$LEFTOVER_JSON"; then
  echo "LEFTOVER_BAKED"
  exit 0
fi

echo "=== leftover did not bake, ARA T49/T196 ==="
python scripts/eval_spark_x25_ara_t49_t196.py
echo "BOTH_PHASES_DONE"
