#!/usr/bin/env bash
# Resume gate 1 after the 3e-4 model has trained: the remaining steps of run_pilot.sh.
set -euo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p1_pframe
log() { echo "[$(date +%H:%M)] $*"; }
log "cache 1e-3";  "$PY" -u scripts/p1_pframe.py cache --lambda 1e-3 > $O/cache_1e-3.log 2>&1
log "cache 1e-4";  "$PY" -u scripts/p1_pframe.py cache --lambda 1e-4 > $O/cache_1e-4.log 2>&1
log "early gate";  "$PY" -u scripts/p1_pframe.py gate > $O/gate_early.log 2>&1
log "train 1e-3";  "$PY" -u scripts/p1_pframe.py train --lambda 1e-3 > $O/train_1e-3.log 2>&1
log "train 1e-4";  "$PY" -u scripts/p1_pframe.py train --lambda 1e-4 > $O/train_1e-4.log 2>&1
log "gate";        "$PY" -u scripts/p1_pframe.py gate > $O/gate.log 2>&1
log "done"
