#!/usr/bin/env bash
# docs/PLAN_audit2.md, C2 amendment 2: re-pick GOP 10/20/30 on VAL-B, after the I-frame re-pick.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p12_audit2
PY=.venv/Scripts/python.exe
ML="1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
until grep -q "C2 re-pick done" $O/gate.log; do sleep 120; done
for G in 30 20 10; do
  $PY -u scripts/p4_gate3.py --val-b --i-frames wide --gop $G --lambdas 1e-2 3e-3 1e-3 3e-4 1e-4 \
      --motion-lambda $ML --low-rate-dir outputs/p6_low_rate/vimeo \
      --output-dir $O --output-name c2_valb_gop$G.json --no-resume >> $O/c2_gop.log 2>&1
  log "C2 VAL-B GOP $G exit $?"
done
log "C2 GOP re-pick done; ALL AUDIT RUNS DONE"
