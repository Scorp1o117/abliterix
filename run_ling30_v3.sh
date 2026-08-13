#!/bin/bash
# run_ling30_v3.sh — Ling-3.0-flash v3: SRA direction + mild LoRA, no experts
# Goal: KL ≤ 0.05 AND refusal ≤ 10%.
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

exec abliterix \
  --config configs/ling30_flash_rocm_v3_sra.toml \
  --seed 117 \
  --overwrite-checkpoint \
  --non-interactive \
  "$@"
