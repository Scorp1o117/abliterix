#!/bin/bash
# Ling-3.0-flash v8: Abliterix default method (weight_norm=none, no Ornith/MPOA).
#
#   ./run_ling30_v8_interactive.sh              # fresh + TUI
#   ./run_ling30_v8_interactive.sh --continue   # resume + TUI
#   ./run_ling30_v8_interactive.sh --batch      # non-interactive
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

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v8_default.toml}"
CKPT="checkpoints_ling30_flash_v8_default"
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

# Mean-vector caches from any prior Ling run (compatible with weight_norm=none)
for src_dir in \
  checkpoints_ling30_flash_v6_stack \
  checkpoints_ling30_flash_v4_b16 \
  checkpoints_ling30_flash_v2 \
  checkpoints_ling30_flash_debug
do
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

echo "=== Ling-3.0-flash v8 Abliterix DEFAULT (weight_norm=none, no Ornith) ==="
echo "  config: $CONFIG"
echo "  checkpoint_dir: $CKPT"
echo "  mode: $([[ $CONTINUE -eq 1 ]] && echo CONTINUE || echo FRESH)"
if [[ $BATCH -eq 1 ]]; then
  echo "  interactive: OFF (--batch)"
else
  echo "  forced: --no-non-interactive"
fi
echo "  method: mean + LoRA + orthog; weight_normalization=none; strength [0.8,1.5]"
echo

COMMON=(
  --config "$CONFIG"
  --seed 117
  --inference.batch-size 16
  --inference.min-batch-size 16
  --inference.max-batch-size 16
)

if [[ $BATCH -eq 0 ]]; then
  COMMON+=(--no-non-interactive)
fi

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
