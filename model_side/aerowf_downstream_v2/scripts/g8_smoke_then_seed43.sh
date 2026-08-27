#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/aerowf_downstream_v2
PIPE="$ROOT/src/aerowf_full_pipeline_v2.py"
SMOKE_OUT="$ROOT/results/harness/g8_smoke_seed1001"
CONS_OUT="$ROOT/results/full_pipeline/seed43_v2_g8_consistency"
CHAIN_LOG="$ROOT/results/analysis/g8_smoke_then_seed43.log"
SMOKE_LOG="$ROOT/results/harness/g8_smoke_seed1001_launch.log"
CONS_LOG="$ROOT/results/full_pipeline/seed43_v2_g8_consistency_launch.log"

mkdir -p "$ROOT/results/analysis" "$ROOT/results/harness" "$ROOT/results/full_pipeline"
: > "$SMOKE_LOG"
: > "$CONS_LOG"

cd /root/autodl-tmp/aerowf_baseline/AeroWF

echo "[$(date -Is)] G-8 smoke start -> $SMOKE_OUT" | tee -a "$CHAIN_LOG"
python -u "$PIPE" \
  --seed 1001 \
  --batch-size 128 \
  --pretrain-epochs 1 \
  --downstream-epochs 1 \
  --patience 10 \
  --min-delta 1e-4 \
  --num-workers 0 \
  --output-root "$SMOKE_OUT" \
  >> "$SMOKE_LOG" 2>&1

if [[ ! -f "$SMOKE_OUT/pipeline_summary.json" ]]; then
  echo "[$(date -Is)] SMOKE FAILED: missing pipeline_summary.json" | tee -a "$CHAIN_LOG"
  exit 1
fi
python3 - << PY
import json
from pathlib import Path
p = Path("$SMOKE_OUT") / "pipeline_summary.json"
s = json.loads(p.read_text())
assert s.get("status") == "success", s.get("status")
print("smoke status", s["status"])
PY

echo "[$(date -Is)] G-8 smoke OK; starting seed43 consistency -> $CONS_OUT" | tee -a "$CHAIN_LOG"
python -u "$PIPE" \
  --seed 43 \
  --batch-size 128 \
  --pretrain-epochs 100 \
  --downstream-epochs 30 \
  --patience 10 \
  --min-delta 1e-4 \
  --num-workers 0 \
  --output-root "$CONS_OUT" \
  >> "$CONS_LOG" 2>&1

echo "[$(date -Is)] seed43 g8 python finished" | tee -a "$CHAIN_LOG"
python3 "$ROOT/scripts/g8_compare_seed43.py" | tee -a "$CHAIN_LOG"
echo "[$(date -Is)] G-8 chain done" | tee -a "$CHAIN_LOG"
