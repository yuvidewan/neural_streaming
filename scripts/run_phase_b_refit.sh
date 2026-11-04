#!/usr/bin/env bash
# The Phase B video refit for one lambda, in run4's order (docs/HANDOFF_Aditya_video_refit.md):
# alphabet width, M10K, M10L, M11 offline gate, M11 training, M13, M14, then the full-video
# benchmark. Usage: scripts/run_phase_b_refit.sh 1e-4 [first step]
# where the first step (default m10k) is one of m10k m10l m11_offline m11_train m13 m14 parity,
# to resume a chain whose earlier steps already passed.
set -euo pipefail
LAMBDA="$1"
FROM="${2:-m10k}"
STEPS=(m10k m10l m11_offline m11_train m13 m14 parity)
TAG="phase_b_lambda${LAMBDA}"
CK="outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda${LAMBDA}/checkpoints/best.pt"
PY=".venv/Scripts/python.exe"
R="outputs/stage1_refit"
export PYTHONIOENCODING=utf-8
log() { echo "[$(date +%H:%M)] ${TAG}: $*"; }
FROM_INDEX=-1
for i in "${!STEPS[@]}"; do [[ "${STEPS[$i]}" == "$FROM" ]] && FROM_INDEX=$i; done
(( FROM_INDEX >= 0 )) || { echo "unknown first step: $FROM" >&2; exit 2; }
# true when step $1 is at or after the first step
run() { for i in "${!STEPS[@]}"; do [[ "${STEPS[$i]}" == "$1" ]] && (( i >= FROM_INDEX )) && return 0; done; return 1; }

# The unit step's stream width is a fact about the checkpoint, read off the wrapper's refusal.
BITS=$("$PY" - "$CK" <<'EOF'
import importlib.util, re, sys
from pathlib import Path
import torch
def load(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
from nvc.evaluation.sequences import discover_sequences
from nvc.training.checkpoint import load_model_from_checkpoint
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
checkpoint = Path(sys.argv[1])
model, _ = load_model_from_checkpoint(checkpoint, device=get_device())
train = discover_sequences(load_default_config().processed_data_dir / "manifest.json", split="train")
try:
    load("unit_step_grids").calibrate_grids(
        load("m10h_motion_compensation"), model, train, checkpoint=checkpoint, bits=10,
        mode="per_channel", gop_size=10, block_size=16, search_range=16,
        reference_mode="mc", max_frames=400)
    print(10)
except ValueError as error:
    print(re.search(r"stream is (\d+) bits wide", str(error)).group(1))
EOF
)
BITS=$(echo "$BITS" | tail -1)
log "stream width ${BITS} bits"

if run m10k; then
log "M10K"
"$PY" -u scripts/m10k_learned_entropy.py --checkpoint "$CK" --output-dir "$R/m10k_$TAG" \
    --rate-points "$BITS" > "$R/m10k_${TAG}_gate.log" 2>&1
fi
if run m10l; then
log "M10L"
"$PY" -u scripts/m10l_offline_gate.py --checkpoint "$CK" --model-dir "$R/m10k_$TAG" \
    --output-dir "$R/m10l_$TAG" --rate-points "$BITS" --metric-comparison-bits 0 \
    > "$R/m10l_${TAG}_gate.log" 2>&1
fi
if run m11_offline; then
log "M11 offline gate"
"$PY" -u scripts/m11_offline_gate.py --checkpoint "$CK" --m10k-dir "$R/m10k_$TAG" \
    --m10l-dir "$R/m10l_$TAG/codebooks" --output-dir "$R/m11_offline_$TAG" \
    --rate-points "$BITS" > "$R/m11_offline_${TAG}.log" 2>&1
fi
if run m11_train; then
log "M11 training"
"$PY" -u scripts/m11_train.py --checkpoint "$CK" --m10k-dir "$R/m10k_$TAG" \
    --m10l-dir "$R/m10l_$TAG/codebooks" --output-dir "$R/m11_$TAG" \
    --rate-points "$BITS" > "$R/m11_train_${TAG}.log" 2>&1
fi
if run m13; then
log "M13"
"$PY" -u scripts/m13_offline_gate.py --checkpoint "$CK" --m11-dir "$R/m11_$TAG" \
    --output-dir "$R/m13_$TAG" --rate-points "$BITS" > "$R/m13_${TAG}_gate.log" 2>&1
fi
if run m14; then
log "M14"
"$PY" -u scripts/m14_offline_gate.py --checkpoint "$CK" --output-dir "$R/m14_$TAG" \
    --rate-points "$BITS" > "$R/m14_${TAG}_gate.log" 2>&1
fi
if run parity; then
log "full-video benchmark"
mkdir -p "outputs/benchmarks/parity_$TAG"
"$PY" -u scripts/benchmark_parity.py --checkpoint "$CK" --m11-dir "$R/m11_$TAG" \
    --m10k-dir "$R/m10k_$TAG" --arms nvc_deployed \
    --motion-cache "$R/m14_$TAG/motion_table.json" --output-dir "outputs/benchmarks/parity_$TAG" \
    --output-name "parity_$TAG.json" --rate-points "$BITS" \
    > "outputs/benchmarks/parity_$TAG/run.log" 2>&1
fi
log "done"
