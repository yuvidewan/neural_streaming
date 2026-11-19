"""docs/PLAN_bframes.md: pick (M, B model) per codec lambda from the VAL-B grid.

Candidates per lambda: M = 1 (today's codec: the outputs/p8_motion VAL-B report of the chosen
lambda_m), and every B-frame grid report. The lowest VAL-B global-pooled MSE + lambda * bpp
(real bytes) wins; ties keep the earlier candidate (M = 1 first). Prints one `CFG lam M model`
line per lambda.

    p9_bselect.py outputs/p9_bframes/valb_M*.json --output outputs/p9_bframes/selection.json
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+", type=Path)
parser.add_argument("--baseline", type=Path, default=Path("outputs/p8_motion/selection.json"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()


def row(point, path):
    mse = 10 ** (-point["quality"]["global_pooled"]["psnr"] / 10)
    lam = point["lambda"]
    return {"cost": mse + float(lam) * point["bpp"], "mse": mse, "bpp": point["bpp"],
            "psnr": point["quality"]["sequence_mean"]["psnr"],
            "b_share": point["split"].get("b_bytes", 0) / point["total_bytes"],
            "report": str(path)}


table: dict[str, dict[str, dict]] = {}
baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
for lam, value in baseline["choice"].items():
    path = Path(baseline["table"][lam][value]["report"])
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["valid"] and report["split"] == "valb", path
    table.setdefault(lam, {})["M1_same"] = row(report["points"][lam], path)
for path in args.reports:
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["valid"] and report["split"] == "valb" and report["bframes"] > 1, path
    for lam, point in report["points"].items():
        table.setdefault(lam, {})[f"M{report['bframes']}_{report['b_model']}"] = row(point, path)
choice = {lam: min(rows, key=lambda k: rows[k]["cost"]) for lam, rows in table.items()}
args.output.write_text(json.dumps({"choice": choice, "table": table}, indent=2), encoding="utf-8")
for lam, rows in table.items():
    print(lam, " ".join(f"{k}:{v['cost']:.6f}" for k, v in rows.items()), "->", choice[lam])
for lam, key in choice.items():
    m, model = key[1:].split("_")
    print(f"CFG {lam} {m} {model}")
