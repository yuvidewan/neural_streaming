#!/usr/bin/env bash
# One heavy job at a time (two crashes on 2026-11-17 under long GPU load):
# 1) x264/x265 4:4:4 UVG anchors (CPU, resumes from parity.json); 2) finish the tuning TEST run
# (GPU, resumes finished lambdas/videos); 3) our codec on UVG at 1080p (GPU, resumes per video).
cd "$(dirname "$0")/../.."
Q=outputs/benchmarks/queue.log
echo "[$(date +%m-%d\ %H:%M)] queue start; 4:4:4 anchors" >> $Q
bash outputs/benchmarks/run_uvg_444.sh
echo "[$(date +%m-%d\ %H:%M)] 4:4:4 anchors finished ($(tail -1 outputs/benchmarks/uvg_classical_444_gop30/run.log)); tuning TEST run" >> $Q
.venv/Scripts/python -u scripts/p4_gate3.py --tune-steps 20 --tune-lr 0.01 >> outputs/p5_encode_tuning/test_run.log 2>&1 \
  && echo "[$(date +%m-%d\ %H:%M)] TEST done" >> $Q || echo "[$(date +%m-%d\ %H:%M)] TEST FAIL" >> $Q
.venv/Scripts/python -u scripts/p4_gate3.py --eval-manifest C:/uvg_scratch/uvg/manifest.json \
  --output-name uvg_a2_wide_gop30.json >> outputs/p4_wide_latent/uvg_run.log 2>&1 \
  && echo "[$(date +%m-%d\ %H:%M)] UVG ours done" >> $Q || echo "[$(date +%m-%d\ %H:%M)] UVG ours FAIL" >> $Q
echo "[$(date +%m-%d\ %H:%M)] queue done" >> $Q
