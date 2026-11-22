"""docs/PLAN_mcljcv.md: the reported numbers (no pass/fail) and the UVG consistency check.

Both model sets against x264/x265 on MCL-JCV (30 videos, 96 frames, 1920x1072). Writes
outputs/p11_mcljcv/results.json. Run from the repo root:
    .venv/Scripts/python outputs/p11_mcljcv/report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
O, B = Path("outputs/p11_mcljcv"), Path("outputs/benchmarks")
LOW = ("1e-2", "3e-3", "1e-3")
UVG_X264_420_PSNR_MSE = -18.1   # outputs/p8_motion/README.md, fixed in the plan
KEYS = {"psnr": "mean_psnr", "msssim": "mean_msssim"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def curve(pts, key, arm=None, lams=None):
    return sorted((p["bpp"], p["quality"][C][key]) for lam, p in pts.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams))


def seq_curve(pts, seq, key, arm=None):
    return sorted((s["bpp"], s[KEYS[key]]) for p in pts.values()
                  if arm is None or p.get("arm") == arm
                  for s in p["per_sequence"] if s["sequence"] == seq)


def bd(base, test):
    v = bp.bd_rate(base, test)
    return None if v is None else round(v, 2)


anc = {}
for fmt in ("420", "444"):
    anc.update({k: v for k, v in load(B / f"mcljcv_classical_{fmt}_gop30/parity.json")["points"].items()
                if v.get("arm", "").startswith("h26")})
arms = sorted({p["arm"] for p in anc.values()})
results = {}
for name, key_metric in (("mse", "psnr"), ("msssim", "msssim")):
    report = load(O / f"mcljcv_{name}.json")
    pts = report["points"]
    seqs = sorted({s["sequence"] for p in pts.values() for s in p["per_sequence"]})
    r = {"decode_exact": report["valid"], "headline_metric": key_metric,
         "points": {lam: {"bpp": round(p["bpp"], 4), "psnr": round(p["quality"][C]["psnr"], 3),
                          "msssim": round(p["quality"][C]["msssim"], 4),
                          "motion_share_of_stream": round(p["split"]["motion_bytes"] / p["total_bytes"], 4),
                          "encode_s_per_frame": round(p["encode_seconds"] / p["frames"], 3),
                          "decode_s_per_frame": round(p["decode_seconds"] / p["frames"], 3)}
                    for lam, p in sorted(pts.items(), key=lambda kv: float(kv[0]), reverse=True)},
         "vs_anchors": {f"{arm}/{key}": {"5pt": bd(curve(anc, key, arm), curve(pts, key)),
                                          "low": bd(curve(anc, key, arm), curve(pts, key, lams=LOW))}
                        for arm in arms for key in KEYS},
         "per_video_vs_x264_444": {seq: {key: bd(seq_curve(anc, seq, key, "h264_444"),
                                                 seq_curve(pts, seq, key)) for key in KEYS}
                                   for seq in seqs}}
    per = [v[key_metric] for v in r["per_video_vs_x264_444"].values() if v[key_metric] is not None]
    r["wins_vs_x264_444"] = {"metric": key_metric, "wins": sum(v < 0 for v in per),
                             "losses": sum(v >= 0 for v in per), "no_overlap": len(seqs) - len(per)}
    results[name] = r
mcl = results["mse"]["vs_anchors"]["h264/psnr"]["5pt"]
results["uvg_consistency"] = {
    "mcljcv_vs_x264_420_psnr_mse_5pt": mcl, "uvg": UVG_X264_420_PSNR_MSE,
    "uvg_specific": None if mcl is None else mcl > UVG_X264_420_PSNR_MSE + 20,
    "rule": "UVG result called UVG-specific if MCL-JCV is more than 20 points worse"}
(O / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for name, r in results.items():
    if name == "uvg_consistency":
        continue
    print(f"== {name} set ({r['headline_metric']}): decode-exact {r['decode_exact']}")
    for lam, p in r["points"].items():
        print("  ", lam, p)
    for k, v in r["vs_anchors"].items():
        print(f"   {k:26s} {v}")
    print("   wins vs x264 4:4:4:", r["wins_vs_x264_444"])
    print("   per video vs x264 4:4:4:", r["per_video_vs_x264_444"])
print("UVG consistency:", results["uvg_consistency"])
