#!/usr/bin/env bash
# docs/PLAN_audit2_fixes.md, the whole chain, one heavy job at a time. Every step resumes.
cd "$(dirname "$0")/../.."
set -uo pipefail
export PYTHONIOENCODING=utf-8
O=outputs/p13_audit_fixes
B=outputs/benchmarks
PY=.venv/Scripts/python.exe
UVG=C:/uvg_scratch/uvg/manifest.json
MCL=C:/mcljcv_scratch/mcljcv/manifest.json
LAMS="1e-2 3e-3 1e-3 3e-4 1e-4"
ML="1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625"
CRF="8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44"
CRF_UVG120="0 2 4 6 $CRF"   # amendment 1: at 120 fps CRF 8 no longer reaches our top quality
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }

if ! grep -q "PROOF 2: PASS" $O/gate.log; then   # a rerun after a stop does not redo it
  log "fix 1 proof 2: encode"
  $PY -u $O/proof.py encode >> $O/proof.log 2>&1; log "proof encode exit $?"
  $PY -u $O/proof.py decode >> $O/proof.log 2>&1; log "proof decode exit $?"
  $PY $O/proof.py check >> $O/gate.log 2>&1 || { log "PROOF 2 FAILED - chain stopped"; exit 1; }
fi

MSE="--lambdas $LAMS --motion-lambda $ML --low-rate-dir outputs/p6_low_rate/vimeo --output-dir $O"
MSS="--lambdas $LAMS --motion-lambda $ML --models-dir outputs/p10_msssim/vimeo --low-rate-dir outputs/p10_msssim/vimeo --output-dir $O"
$PY -u scripts/p4_gate3.py $MSE --output-name test_motion.json >> $O/davis.log 2>&1
log "DAVIS MSE set exit $?"
$PY -u scripts/p4_gate3.py $MSS --output-name test_msssim.json >> $O/davis.log 2>&1
log "DAVIS MS-SSIM set exit $?"

for FMT in 420 444; do
  D=$B/uvg_classical_${FMT}_gop30_realfps; mkdir -p $D
  $PY -u scripts/benchmark_parity.py --skip-nvc --manifest $UVG --gop 30 --classical-pix-fmt yuv${FMT}p \
      --crf $CRF_UVG120 --device cuda --sequence-major --real-fps --output-dir $D --output-name parity.json >> $D/run.log 2>&1
  log "UVG anchors $FMT at 120 fps exit $?"
done

$PY -u scripts/p4_gate3.py $MSE --eval-manifest $UVG --output-name uvg_motion.json >> $O/uvg.log 2>&1
log "UVG MSE set exit $?"

for FMT in 420 444; do
  D=$B/mcljcv_classical_${FMT}_gop30_realfps; mkdir -p $D
  $PY -u scripts/benchmark_parity.py --skip-nvc --manifest $MCL --gop 30 --classical-pix-fmt yuv${FMT}p \
      --crf $CRF --device cuda --sequence-major --real-fps --output-dir $D --output-name parity.json >> $D/run.log 2>&1
  log "MCL-JCV anchors $FMT at real fps exit $?"
done

$PY $O/report.py >> $O/gate.log 2>&1
log "report exit $?; done"
