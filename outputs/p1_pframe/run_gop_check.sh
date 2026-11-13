#!/usr/bin/env bash
# I-frame spacing check: rerun gate 3 (real bytes, full DAVIS TEST) on the stage 3 P-coders
# with an I-frame every 20 and every 30 frames instead of every 10. No training.
# Rule, fixed before running: adopt the longer spacing only if it beats +77.8% vs x264
# (default GOP, B-frames: the arm --gop does not touch) by at least 3 points.
set -uo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p1_pframe
for G in 20 30; do
    echo "[$(date '+%m-%d %H:%M')] GOP $G"
    "$PY" -u scripts/p1_gate3.py --models-dir $O/gate2_local/stage3 --gop $G \
        --output-name gate3_stage3_gop$G.json > $O/gate3_stage3_gop$G.log 2>&1
    echo "[$(date '+%m-%d %H:%M')] GOP $G exit $?"
done
