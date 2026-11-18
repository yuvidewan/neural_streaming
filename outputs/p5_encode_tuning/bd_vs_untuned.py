"""BD-rate (PSNR, sequence_mean) of every finished VAL-B tuning report vs the untuned one."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
import benchmark_parity as bp
D = Path("outputs/p4_wide_latent")
def curve(path):
    r = json.loads(path.read_text())
    return [(p["bpp"], p["quality"][bp.PRIMARY_CONVENTION]["psnr"]) for p in r["points"].values()], r
base, _ = curve(D / "tune_valb_s0_lr0.03_wide_gop30.json")
for path in sorted(D.glob("tune_valb_s*_wide_gop30.json")):
    pts, r = curve(path)
    if r.get("tune_steps", 0) == 0:
        continue
    done = "valid" in r
    bd = bp.bd_rate(base, pts) if len(pts) >= 3 else None
    print(f"{path.name:40s} points={len(pts)} finished={done} BD-rate vs untuned: "
          + (f"{bd:+.2f}%" if bd is not None else "n/a"))
