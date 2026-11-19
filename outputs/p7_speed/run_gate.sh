#!/usr/bin/env bash
# docs/PLAN_speed.md gate, one job at a time: checks 1+2 on DAVIS TEST (lambda 1e-3, full),
# check 1 on the first 31 frames of every UVG video, then check 3 (the profile again).
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p7_speed
echo "[$(date '+%m-%d %H:%M')] check DAVIS" >> $O/gate.log
.venv/Scripts/python.exe -u scripts/p7_check_search.py --lambdas 1e-3 --no-resume \
    --output-dir $O --output-name check_davis.json >> $O/check_davis.log 2>&1
echo "[$(date '+%m-%d %H:%M')] DAVIS exit $?; check UVG" >> $O/gate.log
.venv/Scripts/python.exe -u scripts/p7_check_search.py --lambdas 1e-3 --no-resume \
    --eval-manifest C:/uvg_scratch/uvg/manifest.json --max-frames 31 \
    --output-dir $O --output-name check_uvg31.json >> $O/check_uvg.log 2>&1
echo "[$(date '+%m-%d %H:%M')] UVG exit $?; profile" >> $O/gate.log
.venv/Scripts/python.exe -u scripts/p7_speed_profile.py --output $O/profile_after.json >> $O/profile_after.log 2>&1
echo "[$(date '+%m-%d %H:%M')] profile exit $?; done" >> $O/gate.log
