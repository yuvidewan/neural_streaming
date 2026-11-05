#!/usr/bin/env bash
# Gate 1 of docs/PLAN_learned_pframe.md, end to end. The 3e-4 model trains first and gets an
# early gate read once the other two intra points exist; then the other two models train.
set -euo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p1_pframe
log() { echo "[$(date +%H:%M)] $*"; }
log "train 3e-4";  "$PY" -u scripts/p1_pframe.py train --lambda 3e-4 > $O/train_3e-4.log 2>&1
log "cache 1e-3";  "$PY" -u scripts/p1_pframe.py cache --lambda 1e-3 > $O/cache_1e-3.log 2>&1
log "cache 1e-4";  "$PY" -u scripts/p1_pframe.py cache --lambda 1e-4 > $O/cache_1e-4.log 2>&1
log "early gate";  "$PY" -u scripts/p1_pframe.py gate > $O/gate_early.log 2>&1
log "train 1e-3";  "$PY" -u scripts/p1_pframe.py train --lambda 1e-3 > $O/train_1e-3.log 2>&1
log "train 1e-4";  "$PY" -u scripts/p1_pframe.py train --lambda 1e-4 > $O/train_1e-4.log 2>&1
log "gate";        "$PY" -u scripts/p1_pframe.py gate > $O/gate.log 2>&1
log "done"
