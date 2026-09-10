#!/bin/bash
# 1.7B Abliterix V4: wide box, raise trial budget by 400 until dual HIT.
# Does not bake. Dual HIT = refusals≤10 and 3-token KL≤0.1.
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=configs/spark_x25_1p7b_lora_v4.toml
JOURNAL=checkpoints_spark_x25_1p7b_lora_v4/--run--media--s117--OS--Models--Spark-X2--5-1--7B.jsonl
LOG="${ABLITERIX_LOG:-/home/s117/logs/spark-x25-1p7b-lora-v4-$(date +%Y%m%d-%H%M%S).log}"
echo "$LOG" > /home/s117/logs/spark-x25-1p7b-lora-v4-logpath.txt
mkdir -p checkpoints_spark_x25_1p7b_lora_v4
echo "=== V4 keep-search until dual HIT ==="
echo "  config: $CFG"
echo "  log:    $LOG"

hit() {
  /home/s117/heretic-env/bin/python - "$JOURNAL" <<'PY'
import json, math, sys
p = sys.argv[1]
trials = {}
try:
    with open(p) as fh:
        for line in fh:
            o = json.loads(line)
            tid = o.get("trial_id")
            if tid is None:
                continue
            d = trials.setdefault(tid, {"values": None})
            if o.get("op_code") == 6:
                d["values"] = o.get("values")
except FileNotFoundError:
    raise SystemExit(1)
for d in trials.values():
    v = d.get("values")
    if not v or len(v) < 2:
        continue
    kl, rf = v[0], v[1]
    if kl != kl or (isinstance(kl, float) and math.isinf(kl)):
        continue
    if int(round(rf * 100)) <= 10 and kl <= 0.1:
        print(f"{int(round(rf*100))}@{kl:.4f}")
        raise SystemExit(0)
raise SystemExit(1)
PY
}

set_n() {
  local n=$1
  /home/s117/heretic-env/bin/python - "$CFG" "$n" <<'PY'
from pathlib import Path
import re, sys
p, n = Path(sys.argv[1]), sys.argv[2]
t = p.read_text()
t2, c = re.subn(r"(?m)^num_trials = \d+", f"num_trials = {n}", t, count=1)
if c != 1:
    raise SystemExit(f"failed to set num_trials ({c})")
p.write_text(t2)
print(f"num_trials -> {n}")
PY
}

n=800
while true; do
  set_n "$n"
  if [[ -s "$JOURNAL" ]]; then
    export ABLITERIX_OVERWRITE=0
    echo "=== resume / extend to $n ==="
  else
    export ABLITERIX_OVERWRITE=1
    echo "=== fresh start $n ==="
  fi
  ABLITERIX_LOG="$LOG" ./scripts/run_spark_x25_1p7b.sh "$CFG"
  rc=$?
  if hit; then
    echo "DUAL_HIT: $(hit 2>/dev/null || true)"
    echo "KEEP_SEARCH_DONE"
    exit 0
  fi
  if [[ $rc -ne 0 ]]; then
    echo "FAILED: abliterix exit $rc"
    exit $rc
  fi
  echo "no dual HIT at $n trials; raising budget"
  n=$((n + 400))
done
