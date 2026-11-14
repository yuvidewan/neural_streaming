#!/usr/bin/env bash
# Plan A gate 1: fine-tune A1 and A2 like the plain coder, then score all three on VAL-B.
# The plain arm is reused from outputs/p3_hyperprior/finetune_plain (same recipe).
# Resumable: finished models are skipped.
set -uo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p4_wide_latent
{
for LAM in 1e-3 3e-4 1e-4; do
  for KIND in a1 a2; do
    if [ -f $O/finetune_$KIND/lambda$LAM/model.pt ]; then continue; fi
    echo "[$(date '+%m-%d %H:%M')] train $KIND $LAM"
    "$PY" -u scripts/p4_wide_latent.py train --lambda $LAM --kind $KIND || exit 1
  done
done
echo "[$(date '+%m-%d %H:%M')] evaluate"
"$PY" -u scripts/p4_wide_latent.py evaluate
echo "[$(date '+%m-%d %H:%M')] done"
} >> $O/gate1.log 2>&1
