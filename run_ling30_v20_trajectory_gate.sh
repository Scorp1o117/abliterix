#!/bin/bash
# Ling-3.0-flash v20: response-trajectory-trained token gate, full-100 probe

set -euo pipefail
cd "$(dirname "$0")"
ulimit -n 65536 || ulimit -n 8192 || true
export TRANSFORMERS_SKIP_ALLOCATOR_WARMUP=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset PYTHONPATH AX_NON_INTERACTIVE
# shellcheck source=/dev/null
source /home/s117/heretic-env/bin/activate

CONFIG="${ABLITERIX_CONFIG:-configs/ling30_flash_rocm_v20_trajectory_gate.toml}"
CKPT="checkpoints_ling30_flash_v20_trajectory_gate"
mkdir -p "$CKPT" logs

# Direction provenance is unchanged from v13. The trajectory tensors and gate
# scorers are deliberately recomputed and are not part of this cache.
for suffix in _baseline.pt _steering.pt; do
  for f in checkpoints_ling30_flash_v13_adaptive_runtime/*"$suffix"; do
    [[ -f "$f" ]] || continue
    bn=$(basename "$f")
    [[ -f "$CKPT/$bn" ]] || cp -f "$f" "$CKPT/$bn"
  done
done

echo "=== Ling-3.0-flash v20 RESPONSE-TRAJECTORY TOKEN GATE ==="
echo "  direction/site/profile: v13 prompt mean, decoder block, exact t2"
echo "  gate training: 4 sampled tokens/prompt, compliance vs refusal"
echo "  guard: grouped 20% hold-out, accuracy >=70%, active gap >=25pp"
echo "  evaluation if guard passes: all 100 harmful prompts"

exec abliterix \
  --config "$CONFIG" \
  --seed 117 \
  --inference.batch-size 16 \
  --inference.min-batch-size 16 \
  --inference.max-batch-size 16 \
  --non-interactive \
  --overwrite-checkpoint \
  "$@"
