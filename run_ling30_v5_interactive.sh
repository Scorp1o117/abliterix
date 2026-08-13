#!/bin/bash
# Interactive Ling-3.0-flash v5 low-refusal continuation (batch=16).
#
# Fresh study by default (new checkpoint_dir). Forces TUI after search.
#
#   ./run_ling30_v5_interactive.sh              # fresh
#   ./run_ling30_v5_interactive.sh --continue   # resume v5 journal + TUI
#
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
unset AX_NON_INTERACTIVE

# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${HERETIC_CONFIG:-configs/ling30_flash_rocm_v5_lowref.toml}"
CKPT="checkpoints_ling30_flash_v5_lowref"
mkdir -p "$CKPT"

CONTINUE=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --continue|-c) CONTINUE=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

# Reuse mean + baseline caches from v4 (same model / datasets / vector method)
for src_dir in checkpoints_ling30_flash_v4_b16 checkpoints_ling30_flash_v4_ornith; do
  for f in "$src_dir"/*_baseline.pt "$src_dir"/*_steering.pt; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    if [[ ! -f "$CKPT/$bn" ]]; then
      cp -f "$f" "$CKPT/$bn"
      echo "copied cache: $bn ← $src_dir"
    fi
  done
done

J=""
for g in "$CKPT"/*Ling*flash.jsonl; do
  [[ -f "$g" ]] || continue
  J=$g
  break
done

echo "=== Ling-3.0-flash v5 lowref (batch=16, per-layer, narrow strength) ==="
echo "  config: $CONFIG"
echo "  checkpoint_dir: $CKPT"
echo "  mode: $([[ $CONTINUE -eq 1 ]] && echo CONTINUE || echo FRESH)"
echo "  forced: --no-non-interactive"
echo

COMMON=(
  --config "$CONFIG"
  --seed 117
  --no-non-interactive
  --inference.batch-size 16
  --inference.min-batch-size 16
  --inference.max-batch-size 16
)

if [[ $CONTINUE -eq 1 ]]; then
  echo "  journal: ${J:-none}"
  exec abliterix "${COMMON[@]}" --no-overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
fi

if [[ -n "${J:-}" && -s "$J" ]]; then
  bak="${J}.bak_$(date +%Y%m%d_%H%M%S)"
  mv -f "$J" "$bak"
  echo "  archived journal → $bak"
fi

exec abliterix "${COMMON[@]}" --overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
