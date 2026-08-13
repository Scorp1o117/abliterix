#!/bin/bash
# Interactive Ling-3.0-flash v4 (Ornith MPOA), batch=16, fresh-friendly.
#
# Why --no-non-interactive is REQUIRED:
#   Past non-interactive runs bake non_interactive=true into the study
#   settings JSON. On resume, that flag can stick and skip the TUI with:
#     "Non-interactive mode: optimization finished with N completed trials."
#   --no-non-interactive forces the interactive results menu after search.
#
# Usage:
#   ./run_ling30_v4_interactive.sh              # fresh batch16 (default)
#   ./run_ling30_v4_interactive.sh --continue   # resume same journal + TUI
#   ./run_ling30_v4_interactive.sh --continue --optimization.num-trials 50
#
set -euo pipefail
cd "$(dirname "$0")"

ulimit -n 65536 || ulimit -n 8192 || true

export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH
# Kill any env leak that forces batch mode
unset AX_NON_INTERACTIVE

# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${HERETIC_CONFIG:-configs/ling30_flash_rocm_v4_ornith.toml}"
CKPT="checkpoints_ling30_flash_v4_b16"
mkdir -p "$CKPT"

CONTINUE=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --continue|-c) CONTINUE=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

slug_globs=("$CKPT"/*Ling*flash.jsonl)
J=""
for g in "${slug_globs[@]}"; do
  if [[ -f "$g" ]]; then J=$g; break; fi
done

# Seed caches from batch8 study if present (same model/datasets/mean vectors)
B8=checkpoints_ling30_flash_v4_ornith
for f in baseline.pt steering.pt; do
  # match actual slug filenames
  src=$(ls "$B8"/*"_$f" 2>/dev/null | head -1 || true)
  dst_name=""
  if [[ -n "${src:-}" ]]; then
    base=$(basename "$src")
    if [[ ! -f "$CKPT/$base" ]]; then
      cp -f "$src" "$CKPT/$base"
      echo "copied cache: $base"
    fi
  fi
done

echo "=== Ling-3.0-flash v4 interactive (batch=16, Ornith MPOA) ==="
echo "  config: $CONFIG"
echo "  checkpoint_dir: $CKPT"
echo "  mode: $([[ $CONTINUE -eq 1 ]] && echo 'CONTINUE journal' || echo 'FRESH study (overwrite journal if present)')"
echo "  forced: --no-non-interactive  (always enter TUI after search)"
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
  if [[ -z "${J:-}" ]]; then
    echo "No journal in $CKPT — starting fresh instead."
    exec abliterix "${COMMON[@]}" --overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
  fi
  echo "  journal: $J"
  echo "  tip: pick 继续/continue in any menu; do not wipe."
  # Explicitly disable overwrite so finished studies can still open TUI path
  exec abliterix "${COMMON[@]}" --no-overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
else
  if [[ -n "${J:-}" ]]; then
    bak="${J}.bak_$(date +%Y%m%d_%H%M%S)"
    mv -f "$J" "$bak"
    echo "  archived old journal → $bak"
  fi
  # Fresh Optuna study with batch=16 baked into settings
  exec abliterix "${COMMON[@]}" --overwrite-checkpoint "${ARGS[@]+"${ARGS[@]}"}"
fi
