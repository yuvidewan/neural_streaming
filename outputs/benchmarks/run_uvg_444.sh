#!/usr/bin/env bash
# x264/x265 anchors on UVG in full colour (yuv444p), GOP 30. Resumes from parity.json.
cd "$(dirname "$0")/../.."
D=outputs/benchmarks/uvg_classical_444_gop30; mkdir -p $D
echo "=== resumed $(date +%m-%d\ %H:%M) ===" >> $D/run.log
.venv/Scripts/python -u scripts/benchmark_parity.py --skip-nvc --manifest C:/uvg_scratch/uvg/manifest.json \
  --gop 30 --classical-pix-fmt yuv444p --crf 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 \
  --device cpu --output-dir $D --output-name parity.json >> $D/run.log 2>&1 && echo "[done $(date +%H:%M)]" >> $D/run.log || echo "[FAIL $(date +%H:%M)]" >> $D/run.log
