#!/usr/bin/env bash
# Plan A gate 2: retrain the A2 P-coder and train the wide I-frame coder on Vimeo chunks 4-10,
# on the laptop. Every chunk downloads from Kaggle into C:/vimeo_scratch and is deleted once
# all six models (2 roles x 3 lambdas) have trained on it.
# Resumable: rerun after a stop and finished (role, lambda, chunk) triples are skipped.
set -euo pipefail
export PYTHONIOENCODING=utf-8
export PATH="$PWD/.venv/Scripts:$PATH"   # kaggle CLI lives in the venv
PY=".venv/Scripts/python.exe"; O=outputs/p4_wide_latent
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
log "gate 2 training, chunks 4-10" >> $O/gate2.log
"$PY" -u scripts/p4_wide_latent.py train-vimeo --scratch-dir C:/vimeo_scratch \
    --output-dir $O/vimeo --chunks 4 5 6 7 8 9 10 --p-steps 2500 --i-steps 2000 \
    --unroll 3 --batch 8 --crop 256 --lr 1e-4 --lr-decay-chunks 2 --seed 43 \
    --num-workers 2 >> $O/gate2.log 2>&1
log "done" >> $O/gate2.log
