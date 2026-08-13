#!/bin/bash
# run_ling30_v2.sh — Ling-3.0-flash v2 search (o_proj + down_proj + experts)
# Goal: KL ≤ 0.05 AND refusal ≤ 10%.
#
# PYTORCH_ALLOC_CONF=expandable_segments is MANDATORY on this APU.
set -euo pipefail
cd "$(dirname "$0")"

# Ling has ~63k weight shards; default nofile 1024 → Errno 24 mid-load.
ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

# Fresh study under checkpoints_ling30_flash_v2 (does not reuse broken v1 journal).
# Steering/baseline caches can still be copied if keys match — optional.
exec abliterix \
  --config configs/ling30_flash_rocm_v2.toml \
  --seed 117 \
  --overwrite-checkpoint \
  --non-interactive \
  "$@"
