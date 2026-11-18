#!/usr/bin/env bash
# Encode-time tuning, VAL-B settings search (docs/PLAN_encode_tuning.md). Reports land in
# outputs/p4_wide_latent/tune_valb_*.json; the log is outputs/p5_encode_tuning/valb_grid.log.
# Only one copy may run (two copies on the 8 GB GPU ran out of memory on 2026-11-17), and a
# setting whose report already exists is skipped, so the script can simply be rerun.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python
L=outputs/p5_encode_tuning/valb_grid.log
LOCK=outputs/p5_encode_tuning/.grid.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "another grid is running ($LOCK)"; exit 1; fi
trap 'rmdir "$LOCK"' EXIT
run() {  # steps lr
  local out=outputs/p4_wide_latent/tune_valb_s$1_lr$2_wide_gop30.json
  [ -f "$out" ] && { echo "[skip] $out exists" >> $L; return; }
  echo "[$(date +%m-%d\ %H:%M)] start s$1 lr$2" >> $L
  "$PY" -u scripts/p4_gate3.py --val-b --tune-steps $1 --tune-lr $2 >> $L 2>&1 || echo "[FAIL] s$1 lr$2" >> $L
}
run 0 0.03
for steps in 20 50; do for lr in 0.01 0.03 0.1; do run $steps $lr; done; done
echo "[$(date +%m-%d\ %H:%M)] grid done" >> $L
