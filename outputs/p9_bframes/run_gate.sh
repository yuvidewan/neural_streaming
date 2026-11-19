#!/usr/bin/env bash
# docs/PLAN_bframes.md, the whole chain, one job at a time:
# 1) VAL-B grid M in {2, 4} x B model in {same, coarser} (M = 1 is outputs/p8_motion);
# 2) pick per lambda on VAL-B; 3) DAVIS TEST and 4) UVG 1080p for every lambda that chose
# B-frames (a lambda that chose M = 1 reuses outputs/p8_motion); 5) the report.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p9_bframes
PY=.venv/Scripts/python.exe
ML="1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625"
COMMON="--motion-lambda $ML --low-rate-dir outputs/p6_low_rate/vimeo --output-dir $O"
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/gate.log; }
log "VAL-B grid"
for M in 2 4; do
  for BM in same coarser; do
    $PY -u scripts/p4_gate3.py --val-b --lambdas 1e-2 3e-3 1e-3 3e-4 1e-4 --bframes $M --b-model $BM \
        $COMMON --output-name valb_M${M}_$BM.json >> $O/valb.log 2>&1
    log "VAL-B M=$M $BM exit $?"
  done
done
$PY scripts/p9_bselect.py $O/valb_M*.json --output $O/selection.json >> $O/gate.log 2>&1 || { log "select FAIL"; exit 1; }
grep '^CFG ' $O/gate.log | tail -5 > $O/cfg.txt
log "chosen: $(tr '\n' ' ' < $O/cfg.txt); DAVIS TEST"
while read -r _ LAM M BM; do
  [ "$M" = "1" ] && continue
  $PY -u scripts/p4_gate3.py --lambdas $LAM --bframes $M --b-model $BM $COMMON \
      --output-name test_$LAM.json >> $O/test.log 2>&1
  log "TEST $LAM exit $?"
done < $O/cfg.txt
log "UVG"
while read -r _ LAM M BM; do
  [ "$M" = "1" ] && continue
  $PY -u scripts/p4_gate3.py --lambdas $LAM --bframes $M --b-model $BM $COMMON \
      --eval-manifest C:/uvg_scratch/uvg/manifest.json --output-name uvg_$LAM.json >> $O/uvg.log 2>&1
  log "UVG $LAM exit $?"
done < $O/cfg.txt
$PY outputs/p9_bframes/report.py >> $O/gate.log 2>&1
log "report exit $?; done"
