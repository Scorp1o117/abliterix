#!/bin/bash
# Run the strict frozen-gate rank-1/rank-2 pair sequentially.

set -euo pipefail
cd "$(dirname "$0")"

set -o pipefail
./run_ling30_v42a_frozen_gate_rank1.sh 2>&1 \
  | tee logs/ling30_v42a_frozen_gate_rank1_20260812.log
./run_ling30_v42b_frozen_gate_rank2.sh 2>&1 \
  | tee logs/ling30_v42b_frozen_gate_rank2_20260812.log
