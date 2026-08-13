#!/bin/bash
# run_ling30_debug.sh — Ling-3.0-flash debug run (prints model responses).
# PYTORCH_ALLOC_CONF=expandable_segments is MANDATORY: without it the
# bnb 4-bit load dies at ~3% with "CUDA out of memory (20 MiB)" even
# though the GPU reports 116 GB free (allocator segment fragmentation
# on the unified-memory pool). Do not remove it.
set -e
cd "$(dirname "$0")"

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
source /home/s117/heretic-env/bin/activate

exec abliterix \
  --config configs/ling30_flash_rocm_debug.toml \
  --seed 117 \
  --overwrite-checkpoint \
  --non-interactive \
  "$@"
