#!/bin/bash
# Ling-3.0-flash v13: adaptive runtime causal probe

set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
unset AX_NON_INTERACTIVE

# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v13_adaptive_runtime.toml}"
CKPT="checkpoints_ling30_flash_v13_adaptive_runtime"
mkdir -p "$CKPT" logs

CONTINUE=0
BATCH=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --continue|-c) CONTINUE=1 ;;
    --batch|-b) BATCH=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

# v13 uses the same mean-direction provenance and evaluation set as v9.
# Reusing both caches avoids a redundant 1600-prompt extraction and preserves
# exact baseline comparability. Abliterix validates the embedded cache key.
for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v9_optimal/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    if [[ ! -f "$CKPT/$bn" ]]; then
      cp -f "$f" "$CKPT/$bn"
      echo "copied v9 cache: $bn"
    fi
  done
done

J=""
for g in "$CKPT"/*Ling*flash.jsonl; do
  [[ -f "$g" ]] || continue
  J=$g
  break
done

echo "=== Ling-3.0-flash v13 ADAPTIVE RUNTIME PROBE (18 trials) ==="
echo "  site: decoder-block output (post-attention + post-MoE residual)"
echo "  intervention: positive-alignment-gated angular removal"
echo "  checkpoint_dir: $CKPT"
echo "  mode: $([[ $CONTINUE -eq 1 ]] && echo CONTINUE || echo FRESH)"

COMMON=(
  --config "$CONFIG"
  --seed 117
  --inference.batch-size 16
  --inference.min-batch-size 16
  --inference.max-batch-size 16
)
if [[ $BATCH -eq 1 ]]; then
  COMMON+=(--non-interactive)
else
  COMMON+=(--no-non-interactive)
fi

if [[ $CONTINUE -eq 1 ]]; then
  echo "  journal: ${J:-none}"
  exec abliterix "${COMMON[@]}" --no-overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
fi

if [[ -n "${J:-}" && -s "$J" ]]; then
  bak="${J}.bak_$(date +%Y%m%d_%H%M%S)"
  mv -f "$J" "$bak"
  echo "  archived journal -> $bak"
fi

exec abliterix "${COMMON[@]}" --overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
