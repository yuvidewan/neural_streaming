#!/usr/bin/env bash
# docs/PLAN_low_rate.md: train the wide I-coder and A2 P-coder at lambda 3e-3 and 1e-2 on Vimeo
# chunks 4-6, warm-started from the lambda 1e-3 models. Each chunk downloads into
# C:/vimeo_scratch and is deleted once all four models trained on it. Resumable: rerun after a
# stop or crash and finished (role, lambda, chunk) triples are skipped.
cd "$(dirname "$0")/../.."
set -uo pipefail
export PYTHONIOENCODING=utf-8
export PATH="$PWD/.venv/Scripts:$PATH"   # kaggle CLI lives in the venv
O=outputs/p6_low_rate
echo "[$(date '+%m-%d %H:%M')] low-rate training, chunks 4-6" >> $O/train.log
.venv/Scripts/python.exe -u scripts/p4_wide_latent.py train-vimeo --scratch-dir C:/vimeo_scratch \
    --output-dir $O/vimeo --lambdas 3e-3 1e-2 --chunks 4 5 6 --p-steps 2500 --i-steps 2000 \
    --unroll 3 --batch 8 --crop 256 --lr 1e-4 --lr-decay-chunks 1 --seed 43 --num-workers 2 \
    --init-dir outputs/p4_wide_latent/vimeo --init-lambda 1e-3 --wide-intra-refs "$@" >> $O/train.log 2>&1 \
  && echo "[$(date '+%m-%d %H:%M')] done" >> $O/train.log || echo "[$(date '+%m-%d %H:%M')] FAIL" >> $O/train.log
