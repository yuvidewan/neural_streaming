#!/usr/bin/env bash
# docs/PLAN_budget_pilot.md, the whole chain, one heavy job at a time. Every step resumes.
# Downloads (approved by the user before this runs): the 4 UVG 4K archives (10.88 GiB, deleted
# once their frames are out) and Vimeo-90K chunks 4 and 5 (8.70 + 8.86 GiB, each deleted after
# its last training phase).
cd "$(dirname "$0")/../.."
set -uo pipefail
export PYTHONIOENCODING=utf-8
export PATH="$PWD/.venv/Scripts:$PATH"   # kaggle CLI lives in the venv
O=outputs/p16_budget_pilot
F=outputs/p15_fair_eval
PY=.venv/Scripts/python.exe
S=C:/pilot_scratch
DEV=$S/dev
M=$DEV/manifest.json
T=$O/train
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $O/chain.log; }
stop() { log "STOPPED: $*"; exit 1; }
mkdir -p $S/archives $DEV

# 1. Development set: download, downscale to 1080p YUV, delete the archives.
if [ ! -f $M ]; then
  for N in CityAlley FlowerKids RaceNight SunBath; do
    FILE=${N}_3840x2160_50fps_420_8bit_YUV_RAW.7z
    if [ ! -f $DEV/${N}_1920x1080_50fps_96f.yuv ] && [ ! -f $S/archives/$FILE ]; then
      log "download $FILE"
      curl -sS -L -C - --retry 5 -o $S/archives/$FILE.part https://ultravideo.fi/video/$FILE \
        || stop "download $FILE failed (rerun resumes it)"
      mv $S/archives/$FILE.part $S/archives/$FILE
    fi
  done
  $PY -u scripts/fair_eval.py prepare-uvg4k --archives $S/archives --output $DEV >> $O/prepare.log 2>&1 \
    || stop "prepare-uvg4k failed"
  log "development set ready"
fi

# 2. Proof 3 on a real video, the conversion ceiling (proof 4), the anchors (CPU).
grep -q "PROOF 3: PASS" $O/proof3.log 2>/dev/null || {
  $PY -u scripts/fair_eval.py proof3 --manifest $M --video CityAlley --scratch $S \
      --out $O/proof3.json > $O/proof3.log 2>&1 || stop "PROOF 3 FAILED"; log "proof 3 pass"; }
[ -f $O/ceiling.json ] || $PY -u scripts/fair_eval.py ceiling --manifest $M --out $O/ceiling.json >> $O/ceiling.log 2>&1
$PY -u scripts/fair_eval.py anchors --manifest $M --store $O/anchors.json --streams $S/anchor_streams \
    >> $O/anchors.log 2>&1 || stop "anchors failed"
log "anchors done"

# 3. The budget test must pass before any training counts.
$PY -m pytest -q tests/test_pilot_budget.py >> $O/budget_test.log 2>&1 || stop "BUDGET TEST FAILED"
log "budget test passed"

# 4. Training: the fixed schedule, 72 GPU hours at most (configs/pilot_schedule.json).
if [ ! -f $T/model.pt ]; then
  $PY -u scripts/pilot_budget.py train --out $T --chunks 4 5 --scratch-dir $S/vimeo >> $O/train.log 2>&1 \
    || stop "training failed (it resumes from $T/latest.pt on a rerun)"
fi
log "training done"

# 5. The pilot on the development set, the rate check, speed.
$PY -u scripts/pilot_budget.py eval --checkpoint $T/model.pt --manifest $M --out $O/pilot.json >> $O/eval.log 2>&1 \
  || stop "pilot eval failed"
[ -f $O/ratecheck.json ] || $PY -u scripts/pilot_budget.py ratecheck --checkpoint $T/model.pt --manifest $M \
    --out $O/ratecheck.json >> $O/eval.log 2>&1 || stop "rate check failed"
[ -f $O/speed.json ] || $PY -u scripts/pilot_budget.py speed --checkpoint $T/model.pt --out $O/speed.json >> $O/eval.log 2>&1
log "pilot evaluated"

# 6. The current codec on the same videos (reference point; also proof 2's real 1080p clip).
$PY -u scripts/fair_eval.py ours --manifest $M --out $O/ours.json --streams $S/our_streams >> $O/ours.log 2>&1 \
  || stop "current codec run failed"
log "current codec done"

# 7. Coverage rule, then the verdict and the reports.
for R in pilot ours; do
  $PY -u scripts/fair_eval.py anchors --manifest $M --store $O/anchors.json --streams $S/anchor_streams \
      --cover $O/$R.json >> $O/anchors.log 2>&1 || stop "anchor extension failed"
done
$PY -u scripts/pilot_budget.py verdict --manifest $M --anchors $O/anchors.json --results $O/pilot.json \
    --ratecheck $O/ratecheck.json --out $O/verdict.json >> $O/chain.log 2>&1
$PY -u scripts/fair_eval.py report --manifest $M --store $O/anchors.json --test $O/ours.json \
    --out $O/report_ours.json > $O/report_ours.txt 2>&1
$PY -u scripts/fair_eval.py report --manifest $M --store $O/anchors.json --test $O/pilot.json \
    --out $O/report_pilot.json > $O/report_pilot.txt 2>&1
log "done"
