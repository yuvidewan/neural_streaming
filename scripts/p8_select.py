"""docs/PLAN_motion_rate.md: pick lambda_m per codec lambda from the VAL-B grid runs.

For each lambda, the grid value with the lowest VAL-B global-pooled MSE + lambda * bpp (real
bytes) wins. Writes the choice as JSON and prints the --motion-lambda argument.

    p8_select.py outputs/p8_motion/valb_m*.json --output outputs/p8_motion/selection.json
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
table: dict[str, dict[str, dict]] = {}
for path in args.reports:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["valid"] and report["split"] == "valb", path
    for lam, point in report["points"].items():
        mse = 10 ** (-point["quality"]["global_pooled"]["psnr"] / 10)
        table.setdefault(lam, {})[str(point["motion_lambda"])] = {
            "cost": mse + float(lam) * point["bpp"], "mse": mse, "bpp": point["bpp"],
            "psnr": point["quality"]["sequence_mean"]["psnr"],
            "motion_bytes": point["split"]["motion_bytes"], "report": str(path)}
choice = {lam: min(rows, key=lambda k: rows[k]["cost"]) for lam, rows in table.items()}
args.output.write_text(json.dumps({"choice": choice, "table": table}, indent=2), encoding="utf-8")
for lam, rows in table.items():
    print(lam, " ".join(f"{k}:{v['cost']:.6f}" for k, v in sorted(rows.items(), key=lambda kv: float(kv[0]))),
          "->", choice[lam])
print("ARG " + ",".join(f"{lam}:{v}" for lam, v in choice.items()))
