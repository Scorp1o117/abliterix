#!/bin/bash
# Sequential GPU: classic Heretic 30, then full-weight ARA 200, Spark-X2.5-1.7B.
# Does not bake to Models/Spark-X2.5-1.7B-abliterix.
set -euo pipefail

HERETIC_TS="$(date +%Y%m%d-%H%M%S)"
HLOG="/home/s117/logs/spark-x25-1p7b-heretic-${HERETIC_TS}.log"
ALOG="/home/s117/logs/spark-x25-1p7b-ara-${HERETIC_TS}.log"
echo "$HLOG" > /home/s117/logs/spark-x25-1p7b-heretic-logpath.txt
echo "$ALOG" > /home/s117/logs/spark-x25-1p7b-ara-logpath.txt

echo "=== PHASE 1: classic Heretic 30 ==="
HERETIC_LOG="$HLOG" /home/s117/heretic-latest/run-spark-x25-1p7b.sh
echo "HERETIC_PHASE_DONE"

echo "=== PHASE 2: full-weight ARA 200 ==="
HERETIC_LOG="$ALOG" /home/s117/heretic-ara-lora/run-spark-x25-1p7b-ara.sh
echo "BOTH_PHASES_DONE"
