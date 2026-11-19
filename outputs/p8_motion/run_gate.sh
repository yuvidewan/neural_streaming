#!/usr/bin/env bash
# docs/PLAN_motion_rate.md, the whole chain, one job at a time:
# 1) VAL-B grid over lambda_m (all 5 codec lambdas); 2) pick lambda_m per lambda on VAL-B;
# 3) DAVIS TEST and 4) UVG 1080p with the chosen values; 5) the report.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p8_motion
PY=.venv/Scripts/python.exe
COMMON="--lambdas 1e-2 3e-3 1e-3 3e-4 1e-4 --low-rate-dir outputs/p6_low_rate/vimeo --output-dir $O"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
log "VAL-B grid"
for M in 0 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8 16; do
  $PY -u scripts/p4_gate3.py --val-b --motion-lambda $M $COMMON --output-name valb_m$M.json >> $O/valb.log 2>&1
  log "VAL-B lambda_m=$M exit $?"
done
$PY scripts/p8_select.py $O/valb_m*.json --output $O/selection.json >> $O/gate.log 2>&1 || { log "select FAIL"; exit 1; }
ARG=$(grep '^ARG ' $O/gate.log | tail -1 | cut -d' ' -f2)
log "chosen: $ARG; DAVIS TEST"
$PY -u scripts/p4_gate3.py --motion-lambda "$ARG" $COMMON --output-name test_motion.json >> $O/test.log 2>&1
log "TEST exit $?; UVG"
$PY -u scripts/p4_gate3.py --motion-lambda "$ARG" $COMMON --eval-manifest C:/uvg_scratch/uvg/manifest.json \
    --output-name uvg_motion.json >> $O/uvg.log 2>&1
log "UVG exit $?; report"
$PY outputs/p8_motion/report.py >> $O/gate.log 2>&1
log "report exit $?; done"
