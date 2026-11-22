#!/usr/bin/env bash
# docs/PLAN_audit2.md, C2 amendment: re-pick old vs wide I-frames on VAL-B, after the heavy checks.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p12_audit2
PY=.venv/Scripts/python.exe
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
until grep -q "heavy checks done" $O/gate.log; do sleep 120; done
for ARM in old wide; do
  $PY -u scripts/p4_gate3.py --val-b --i-frames $ARM --gop 30 --lambdas 1e-3 3e-4 1e-4 \
      --output-dir $O --output-name c2_valb_$ARM.json --no-resume >> $O/c2.log 2>&1
  log "C2 VAL-B $ARM exit $?"
done
log "C2 re-pick done; all audit runs done"
