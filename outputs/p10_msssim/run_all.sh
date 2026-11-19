#!/usr/bin/env bash
# docs/PLAN_msssim.md, the whole chain, one job at a time: 1) train the MS-SSIM set on Vimeo
# chunk 7 (deleted after use; resumable); 2) DAVIS TEST; 3) UVG 1080p; 4) the report.
cd "$(dirname "$0")/../.."
set -uo pipefail
export PYTHONIOENCODING=utf-8
export PATH="$PWD/.venv/Scripts:$PATH"   # kaggle CLI lives in the venv
O=outputs/p10_msssim
PY=.venv/Scripts/python.exe
ML="1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625"
LAMS="1e-2 3e-3 1e-3 3e-4 1e-4"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
log "training"
$PY -u scripts/p4_wide_latent.py train-vimeo --scratch-dir C:/vimeo_scratch \
    --output-dir $O/vimeo --lambdas $LAMS --chunks 7 --p-steps 2500 --i-steps 2000 \
    --unroll 3 --batch 8 --crop 256 --lr 1e-4 --lr-decay-chunks 0 --seed 44 --num-workers 2 \
    --init-dir outputs/p4_wide_latent/vimeo outputs/p6_low_rate/vimeo --init-lambda same \
    --wide-intra-refs --distortion msssim >> $O/train.log 2>&1
code=$?
log "training exit $code"
[ $code -eq 0 ] || exit 1
COMMON="--models-dir $O/vimeo --low-rate-dir $O/vimeo --lambdas $LAMS --motion-lambda $ML --output-dir $O"
$PY -u scripts/p4_gate3.py $COMMON --output-name test_msssim.json >> $O/test.log 2>&1
log "TEST exit $?; UVG"
$PY -u scripts/p4_gate3.py $COMMON --eval-manifest C:/uvg_scratch/uvg/manifest.json \
    --output-name uvg_msssim.json >> $O/uvg.log 2>&1
log "UVG exit $?; report"
$PY outputs/p10_msssim/report.py >> $O/gate.log 2>&1
log "report exit $?; done"
