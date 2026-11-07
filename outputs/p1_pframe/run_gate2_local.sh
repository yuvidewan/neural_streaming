#!/usr/bin/env bash
# Gate 2 on the laptop instead of Colab: the notebook's two stages, same settings.
# Chunk 1 is already extracted in C:/vimeo_scratch; chunks 2 and 3 download from Kaggle.
# Every chunk, chunk 1 included, is deleted once all three lambdas have trained on it.
set -euo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p1_pframe/gate2_local
mkdir -p $O
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
COMMON=(scripts/p1_pframe.py train-vimeo --scratch-dir C:/vimeo_scratch --lambdas 1e-3 3e-4 1e-4
        --batch 8 --crop 256 --lr 1e-4 --lr-decay-chunks 1 --save-every 500 --seed 42 --num-workers 2)
log "stage 1"
"$PY" -u "${COMMON[@]}" --output-dir $O/stage1 --chunks 1 2 --steps-per-chunk 10000 --unroll 1 \
    --local-vimeo C:/vimeo_scratch/chunk_1/vimeo_settuplet_1/sequences --local-chunk 1 --delete-local-after C:/vimeo_scratch/chunk_1 >> $O/stage1.log 2>&1
log "stage 2"
"$PY" -u "${COMMON[@]}" --output-dir $O/stage2 --chunks 3 --steps-per-chunk 4000 --unroll 3 \
    --init-from $O/stage1 >> $O/stage2.log 2>&1
log "gate stage 1"; "$PY" -u scripts/p1_pframe.py gate --models-dir $O/stage1 --output-name gate2_stage1.json > $O/gate_stage1.log 2>&1
log "gate stage 2"; "$PY" -u scripts/p1_pframe.py gate --models-dir $O/stage2 --output-name gate2_stage2.json > $O/gate_stage2.log 2>&1
log "done"
