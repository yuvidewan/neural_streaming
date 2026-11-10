#!/usr/bin/env bash
# Fairness check for the motion oracle: fine-tune the P-coder equally on block and dense
# warps, then score each warp with the coder tuned for it. Shares the GPU with the
# chunk 4-10 training run.
set -euo pipefail
export PYTHONIOENCODING=utf-8
PY=".venv/Scripts/python.exe"; O=outputs/p2_motion
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
for lam in 1e-3 3e-4 1e-4; do
  [ -f "$TEMP/nvc_p1_cache/train_dense_$lam.pt" ] || { log "cache $lam"; "$PY" -u scripts/p2_flow_finetune.py cache --lambda $lam; }
done
for lam in 1e-3 3e-4 1e-4; do for kind in block dense; do
  [ -f "$O/finetune_$kind/lambda$lam/model.pt" ] || { log "train $kind $lam"; "$PY" -u scripts/p2_flow_finetune.py train --lambda $lam --kind $kind; }
done; done
log "oracle, block-tuned"
"$PY" -u scripts/p2_motion_oracle.py --models-dir $O/finetune_block --arms block16 dense_oracle --output $O/fairness_block.json
log "oracle, dense-tuned"
"$PY" -u scripts/p2_motion_oracle.py --models-dir $O/finetune_dense --arms block16 dense_oracle --output $O/fairness_dense.json
"$PY" - <<'PY'
import json, sys
sys.path.insert(0, "scripts")
import benchmark_parity as bp
b = json.load(open("outputs/p2_motion/fairness_block.json"))["points"]
d = json.load(open("outputs/p2_motion/fairness_dense.json"))["points"]
curve = lambda pts, arm: sorted((pts[l][arm]["bpp"], pts[l][arm]["p_psnr"]) for l in pts)
result = {"dense_tuned_dense_vs_block_tuned_block16": bp.bd_rate(curve(b, "block16"), curve(d, "dense_oracle")),
          "block_tuned_dense_vs_block16": bp.bd_rate(curve(b, "block16"), curve(b, "dense_oracle")),
          "dense_tuned_block16_vs_block_tuned_block16": bp.bd_rate(curve(b, "block16"), curve(d, "block16"))}
result["passed"] = result["dense_tuned_dense_vs_block_tuned_block16"] <= -25.0
json.dump(result, open("outputs/p2_motion/fairness.json", "w"), indent=2)
print(json.dumps(result, indent=2))
PY
log "done"
