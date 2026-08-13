#!/bin/bash
# run_ling30_v4.sh — Ling-3.0-flash v4: Ornith classic (MPOA) recipe
# Goal: KL ≤ 0.05 AND refusal ≤ 10%. print_responses on for batch quality checks.
# NOTE: continuing a checkpoint restores batch_size from the study settings JSON,
# not this script. To change batch you must start a fresh journal (overwrite or
# new checkpoint_dir).
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

exec abliterix \
  --config configs/ling30_flash_rocm_v4_ornith.toml \
  --seed 117 \
  --overwrite-checkpoint \
  --non-interactive \
  "$@"
