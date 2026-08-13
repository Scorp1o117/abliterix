#!/bin/bash
# Ling-3.0-flash v6: classic abliteration, batch=16, large trial stack + TUI.
#
#   ./run_ling30_v6_interactive.sh              # fresh study
#   ./run_ling30_v6_interactive.sh --continue   # resume + TUI (no reload needed mid-session)
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

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v6_stack.toml}"
CKPT="checkpoints_ling30_flash_v6_stack"
mkdir -p "$CKPT"

CONTINUE=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --continue|-c) CONTINUE=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

# Reuse baseline/steering from prior Ling runs (mean vectors)
for src_dir in \
  checkpoints_ling30_flash_v5_lowref \
  checkpoints_ling30_flash_v4_b16 \
  checkpoints_ling30_flash_v4_ornith \
  checkpoints_ling30_flash_v2
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

echo "=== Ling-3.0-flash v6 stack (classic abliteration, batch=16, 80 trials) ==="
echo "  config: $CONFIG"
echo "  checkpoint_dir: $CKPT"
echo "  mode: $([[ $CONTINUE -eq 1 ]] && echo CONTINUE || echo FRESH)"
echo "  forced: --no-non-interactive (TUI after search for more trials / export)"
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
