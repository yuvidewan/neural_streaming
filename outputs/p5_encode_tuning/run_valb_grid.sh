#!/usr/bin/env bash
# Encode-time tuning, VAL-B settings search (docs/PLAN_encode_tuning.md). Reports land in
# outputs/p4_wide_latent/tune_valb_*.json; the log is outputs/p5_encode_tuning/valb_grid.log.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
L=outputs/p5_encode_tuning/valb_grid.log
"$PY" -u scripts/p4_gate3.py --val-b >> $L 2>&1
for steps in 20 50; do
  for lr in 0.01 0.03 0.1; do
    "$PY" -u scripts/p4_gate3.py --val-b --tune-steps $steps --tune-lr $lr >> $L 2>&1
  done
done
echo "[$(date +%m-%d\ %H:%M)] grid done" >> $L
