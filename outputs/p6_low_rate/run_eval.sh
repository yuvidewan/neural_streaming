#!/usr/bin/env bash
# docs/PLAN_low_rate.md: code the two new lambdas, one job at a time. 1) DAVIS TEST gate
# (real bytes, GOP 30, untuned); 2) UVG 1080p (not gated). Both resume per video.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
O=outputs/p6_low_rate
echo "[$(date '+%m-%d %H:%M')] DAVIS TEST gate" >> $O/eval.log
.venv/Scripts/python.exe -u scripts/p4_gate3.py --models-dir $O/vimeo --lambdas 3e-3 1e-2 \
    --output-dir $O --output-name gate_test.json >> $O/gate_run.log 2>&1 \
  && echo "[$(date '+%m-%d %H:%M')] TEST done" >> $O/eval.log || echo "[$(date '+%m-%d %H:%M')] TEST FAIL" >> $O/eval.log
.venv/Scripts/python.exe -u scripts/p4_gate3.py --models-dir $O/vimeo --lambdas 3e-3 1e-2 \
    --eval-manifest C:/uvg_scratch/uvg/manifest.json --output-dir $O --output-name uvg_low_rate.json >> $O/uvg_run.log 2>&1 \
  && echo "[$(date '+%m-%d %H:%M')] UVG done" >> $O/eval.log || echo "[$(date '+%m-%d %H:%M')] UVG FAIL" >> $O/eval.log
