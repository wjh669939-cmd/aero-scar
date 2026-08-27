#!/usr/bin/env bash
set -euo pipefail
TARGET="/root/autodl-tmp/aerowf_downstream_v2/results/full_pipeline/seed2027_v2/pipeline_summary.json"
SCRIPT="/root/autodl-tmp/aerowf_downstream_v2/scripts/three_seed_effect_noise_report.py"
LOG="/root/autodl-tmp/aerowf_downstream_v2/results/analysis/wait_seed2027.log"

mkdir -p "$(dirname "$LOG")"
echo "[$(date -Is)] waiting for $TARGET" | tee -a "$LOG"

while [[ ! -f "$TARGET" ]]; do
  if ! pgrep -f "aerowf_full_pipeline_v2.py.*2027" >/dev/null 2>&1; then
    if [[ ! -f "$TARGET" ]]; then
      echo "[$(date -Is)] pipeline process gone but summary missing — check logs" | tee -a "$LOG"
      exit 1
    fi
  fi
  sleep 120
done

echo "[$(date -Is)] seed2027 complete, generating report" | tee -a "$LOG"
python3 "$SCRIPT" | tee -a "$LOG"
echo "[$(date -Is)] done" | tee -a "$LOG"
