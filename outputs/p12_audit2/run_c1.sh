#!/usr/bin/env bash
# docs/PLAN_audit2.md check C1 (Vimeo re-download, one chunk at a time, deleted after hashing).
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p12_audit2
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
log "C1 start"
.venv/Scripts/python.exe -u $O/c1_leakage.py >> $O/c1.log 2>&1
log "C1 exit $?"
