#!/usr/bin/env bash
# docs/PLAN_audit2.md: the heavy checks, one at a time, after C1 (the Vimeo hashing) finishes.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p12_audit2
PY=.venv/Scripts/python.exe
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
until grep -q "C1 exit" $O/gate.log; do sleep 120; done
log "A1 encode"
$PY -u $O/bitstream.py encode >> $O/a.log 2>&1;            log "A1 encode exit $?"
$PY -u $O/bitstream.py decode >> $O/a.log 2>&1;            log "A1 decode (fresh process) exit $?"
$PY -u $O/bitstream.py wrong >> $O/a.log 2>&1;             log "A2 wrong models exit $?"
$PY -u $O/bitstream.py damage >> $O/a.log 2>&1;            log "A5 damage exit $?"
$PY -u $O/bitstream.py decode --device cpu --only-davis >> $O/a.log 2>&1; log "A6 cpu decode exit $?"
$PY -u $O/fairness.py b1 >> $O/b.log 2>&1;                 log "B1 lossless exit $?"
$PY -u $O/fairness.py b3 >> $O/b.log 2>&1;                 log "B3 anchor settings exit $?"
log "heavy checks done"
