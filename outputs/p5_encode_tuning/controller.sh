#!/usr/bin/env bash
# 1) wait until s50 lr0.01 has finished, then stop the grid (before s50 lr0.03/0.1 run);
# 2) rerun s20 lr0.01 (the old report was partial); 3) if s50 lr0.01 == s20 lr0.01, skip the
# other 50-step runs, else run them; 4) pick the setting by the plan's rule and run TEST.
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python; L=outputs/p5_encode_tuning/valb_grid.log; D=outputs/p4_wide_latent
export CUDA_LAUNCH_BLOCKING=1
run() { echo "[$(date +%m-%d\ %H:%M)] start s$1 lr$2" >> $L
        "$PY" -u scripts/p4_gate3.py --val-b --tune-steps $1 --tune-lr $2 >> $L 2>&1 || echo "[FAIL] s$1 lr$2" >> $L; }
until grep -q '"valid"' $D/tune_valb_s50_lr0.01_wide_gop30.json 2>/dev/null; do sleep 20; done
powershell -NoProfile -c "Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'run_valb_grid.sh|p4_gate3.py --val-b --tune-steps 50' -and \$_.Name -ne 'powershell.exe' } | % { taskkill /PID \$_.ProcessId /T /F | Out-Null }"
rmdir outputs/p5_encode_tuning/.grid.lock 2>/dev/null
echo "[$(date +%m-%d\ %H:%M)] s50 lr0.01 done; grid stopped by controller" >> $L
run 20 0.01
if "$PY" outputs/p5_encode_tuning/compare_and_pick.py identical s50_lr0.01 s20_lr0.01 2>> $L; then
  echo "[controller] s50 == s20 at lr0.01: other 50-step runs dropped (agreed rule)" >> $L
else
  echo "[controller] s50 != s20 at lr0.01: running the other 50-step settings" >> $L
  run 50 0.03; run 50 0.1
fi
echo "[$(date +%m-%d\ %H:%M)] grid done" >> $L
read STEPS LR < <("$PY" outputs/p5_encode_tuning/compare_and_pick.py pick 2>> $L)
T=outputs/p5_encode_tuning/test_run.log
echo "[$(date +%m-%d\ %H:%M)] TEST run: s$STEPS lr$LR" | tee -a $L >> $T
"$PY" -u scripts/p4_gate3.py --tune-steps $STEPS --tune-lr $LR >> $T 2>&1 && echo "[$(date +%m-%d\ %H:%M)] TEST done" >> $T || echo "[FAIL] TEST" >> $T
