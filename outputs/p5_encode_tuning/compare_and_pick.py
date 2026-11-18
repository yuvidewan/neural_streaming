"""VAL-B tuning: the identity check and the setting pick (docs/PLAN_encode_tuning.md).

`identical A B`: exit 0 when two reports have byte-identical totals and identical PSNR at
every lambda (used to drop the other 50-step runs, agreed with the user 2026-11-17: if
s50 lr0.01 equals s20 lr0.01, more steps cannot change the pick, since a tie goes to fewer steps).
`pick`: applies the plan's rule (lowest BD-rate vs untuned; within 0.5 points -> fewer
steps) and prints "STEPS LR"."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import benchmark_parity as bp
D = Path("outputs/p4_wide_latent")
def load(name):
    return json.loads((D / f"tune_valb_{name}_wide_gop30.json").read_text())
def curve(r):
    return [(p["bpp"], p["quality"][bp.PRIMARY_CONVENTION]["psnr"]) for p in r["points"].values()]
if sys.argv[1] == "identical":
    a, b = load(sys.argv[2]), load(sys.argv[3])
    same = a["points"].keys() == b["points"].keys() and all(
        a["points"][k]["total_bytes"] == b["points"][k]["total_bytes"]
        and a["points"][k]["quality"][bp.PRIMARY_CONVENTION]["psnr"]
        == b["points"][k]["quality"][bp.PRIMARY_CONVENTION]["psnr"] for k in a["points"])
    print(f"{sys.argv[2]} vs {sys.argv[3]}: identical={same}", file=sys.stderr)
    sys.exit(0 if same else 1)
base = curve(load("s0_lr0.03"))
rows = []
for path in sorted(D.glob("tune_valb_s*_wide_gop30.json")):
    r = json.loads(path.read_text())
    if r.get("tune_steps", 0) > 0 and "valid" in r and r["valid"]:
        rows.append((bp.bd_rate(base, curve(r)), r["tune_steps"], r["tune_lr"]))
for bd, s, lr in sorted(rows):
    print(f"  s{s} lr{lr:g}: BD-rate vs untuned {bd:+.2f}%", file=sys.stderr)
best = min(rows)[0]
close = [row for row in rows if row[0] <= best + 0.5]
bd, steps, lr = min(close, key=lambda row: (row[1], row[0]))
print(f"pick: s{steps} lr{lr:g} ({bd:+.2f}%)", file=sys.stderr)
print(steps, f"{lr:g}")
