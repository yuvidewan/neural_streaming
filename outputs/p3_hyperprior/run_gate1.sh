#!/usr/bin/env bash
# Hyperprior gate 1: fine-tune plain and hyperprior copies of the stage 3 P-coder equally,
# then score both on VAL-B. Resumable: finished models are skipped.
set -uo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p3_hyperprior
mkdir -p $O
{
for LAM in 1e-3 3e-4 1e-4; do
  for KIND in plain hyper; do
    if [ -f $O/finetune_$KIND/lambda$LAM/model.pt ]; then continue; fi
    echo "[$(date '+%m-%d %H:%M')] train $KIND $LAM"
    "$PY" -u scripts/p3_hyperprior.py train --lambda $LAM --kind $KIND || exit 1
  done
done
echo "[$(date '+%m-%d %H:%M')] evaluate"
"$PY" -u scripts/p3_hyperprior.py evaluate
echo "[$(date '+%m-%d %H:%M')] done"
} >> $O/gate1.log 2>&1
