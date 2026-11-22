#!/usr/bin/env bash
# docs/PLAN_mcljcv.md, the whole chain, one job at a time:
# 1) fetch and convert the 30 sources; 2) x264/x265 anchors, 4:2:0 then 4:4:4;
# 3) the MSE set; 4) the MS-SSIM set; 5) the report. Every step resumes on a rerun.
cd "$(dirname "$0")/../.."
set -uo pipefail
export PYTHONIOENCODING=utf-8
O=outputs/p11_mcljcv
B=outputs/benchmarks
PY=.venv/Scripts/python.exe
M=C:/mcljcv_scratch/mcljcv/manifest.json
LAMS="1e-2 3e-3 1e-3 3e-4 1e-4"
ML="1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625"
CRF="8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
log "prepare"
$PY -u scripts/prepare_mcljcv.py --output C:/mcljcv_scratch/mcljcv >> $O/prepare.log 2>&1
code=$?; log "prepare exit $code"; [ $code -eq 0 ] || exit 1
for FMT in 420 444; do
  D=$B/mcljcv_classical_${FMT}_gop30; mkdir -p $D
  PIX=yuv${FMT}p
  $PY -u scripts/benchmark_parity.py --skip-nvc --manifest $M --gop 30 --classical-pix-fmt $PIX \
      --crf $CRF --device cuda --sequence-major --output-dir $D --output-name parity.json >> $D/run.log 2>&1
  log "anchors $FMT exit $?"
done
COMMON="--lambdas $LAMS --motion-lambda $ML --eval-manifest $M --output-dir $O"
$PY -u scripts/p4_gate3.py $COMMON --low-rate-dir outputs/p6_low_rate/vimeo \
    --output-name mcljcv_mse.json >> $O/mse.log 2>&1
log "MSE set exit $?"
$PY -u scripts/p4_gate3.py $COMMON --models-dir outputs/p10_msssim/vimeo --low-rate-dir outputs/p10_msssim/vimeo \
    --output-name mcljcv_msssim.json >> $O/msssim.log 2>&1
log "MS-SSIM set exit $?; report"
$PY $O/report.py >> $O/gate.log 2>&1
log "report exit $?; done"
