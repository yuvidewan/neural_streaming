#!/usr/bin/env bash
# Continue the gate 3 P-coders on the 7 Vimeo chunks they have not seen (4-10), on the laptop.
# Unroll 3 (chained P-frames, what the real codec does), starting from gate 2's stage 2 models.
# Every chunk downloads from Kaggle into C:/vimeo_scratch and is deleted once all three
# lambdas have trained on it. Then gate 3 re-runs on the new models.
# Resumable: rerun this script after a stop and finished (lambda, chunk) pairs are skipped.
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PATH="$PWD/.venv/Scripts:$PATH"   # kaggle CLI lives in the venv
PY=".venv/Scripts/python.exe"; O=outputs/p1_pframe/gate2_local
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
log "chunks 4-10"
"$PY" -u scripts/p1_pframe.py train-vimeo --scratch-dir C:/vimeo_scratch \
    --lambdas 1e-3 3e-4 1e-4 --output-dir $O/stage3 --init-from $O/stage2 \
    --chunks 4 5 6 7 8 9 10 --steps-per-chunk 3500 --unroll 3 \
    --batch 8 --crop 256 --lr 1e-4 --lr-decay-chunks 2 --save-every 500 --seed 43 \
    --num-workers 2 >> $O/stage3.log 2>&1
log "gate 3 on the new models"
"$PY" -u scripts/p1_gate3.py --models-dir $O/stage3 --output-name gate3_stage3.json \
    > outputs/p1_pframe/gate3_stage3.log 2>&1
log "done"
