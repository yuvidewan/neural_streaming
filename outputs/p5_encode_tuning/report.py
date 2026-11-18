"""Encode-time tuning verdict (DAVIS TEST) and the first UVG 1080p run, from the saved reports.

Writes outputs/p5_encode_tuning/results.json. Run from the repo root:
    .venv/Scripts/python outputs/p5_encode_tuning/report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
P4 = Path("outputs/p4_wide_latent")
B = Path("outputs/benchmarks")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def curve(points, key, arm=None):
    return sorted((p["bpp"], p["quality"][C][key]) for p in points.values()
                  if arm is None or p.get("arm") == arm)


def merged(*paths):
    out = {}
    for path in paths:
        out.update(load(path)["points"])
    return out


def bd(base, test):
    value = bp.bd_rate(base, test)
    return None if value is None else round(value, 2)


def vs_anchors(ours, anchors):
    rows = {}
    for arm in sorted({p["arm"] for p in anchors.values()}):
        for key in ("psnr", "msssim"):
            rows[f"{arm}/{key}"] = {"bd_rate": bd(curve(anchors, key, arm), curve(ours, key)),
                                    "overlap": bp.overlap(curve(anchors, key, arm), curve(ours, key))}
    return rows


def per_video(report):
    videos = {}
    for lam, point in report["points"].items():
        for s in point["per_sequence"]:
            videos.setdefault(s["sequence"], {})[lam] = (s["bpp"], s["mean_psnr"], s["mean_msssim"])
    return videos


untuned = load(P4 / "gate3_a2_wide_gop30.json")
tuned = load(P4 / "tune_test_s20_lr0.01_wide_gop30.json")
assert untuned["valid"] and tuned["valid"]

# DAVIS TEST anchors over the codec's full range (as in gate3_vs_full_range_anchors.json)
davis_420 = merged(B / "parity_classical_gop30/parity_classical_gop30.json",
                   B / "parity_classical_420_hq_gop30/parity.json")
davis_444 = merged(B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                   B / "parity_classical_444_hq_gop30/parity.json")
davis_anchors = {k: v for k, v in {**davis_420, **davis_444}.items() if v.get("arm", "").startswith("h26")}

pv_u, pv_t = per_video(untuned), per_video(tuned)
videos = {}
for seq in pv_u:
    u = [(b, q) for b, q, _ in pv_u[seq].values()]
    t = [(b, q) for b, q, _ in pv_t[seq].values()]
    videos[seq] = bd(sorted(u), sorted(t))

results = {
    "davis_test_tuning": {
        "setting": {"tune_steps": tuned["tune_steps"], "tune_lr": tuned["tune_lr"]},
        "gate": "PASS <= -5%, SMALL WIN -5%..-2%, NOT WORTH IT > -2% (docs/PLAN_encode_tuning.md)",
        "bd_vs_untuned/psnr": bd(curve(untuned["points"], "psnr"), curve(tuned["points"], "psnr")),
        "bd_vs_untuned/msssim": bd(curve(untuned["points"], "msssim"), curve(tuned["points"], "msssim")),
        "per_video_bd_vs_untuned/psnr": videos,
        "points": {lam: {"untuned": {"bpp": untuned["points"][lam]["bpp"],
                                     "psnr": untuned["points"][lam]["quality"][C]["psnr"],
                                     "encode_s_per_frame": untuned["points"][lam]["encode_seconds"] / untuned["points"][lam]["frames"]},
                         "tuned": {"bpp": p["bpp"], "psnr": p["quality"][C]["psnr"],
                                   "encode_s_per_frame": p["encode_seconds"] / p["frames"]}}
                   for lam, p in tuned["points"].items()},
        "untuned_vs_anchors": vs_anchors(untuned["points"], davis_anchors),
        "tuned_vs_anchors": vs_anchors(tuned["points"], davis_anchors),
    },
}

uvg = load(P4 / "uvg_a2_wide_gop30.json")
assert uvg["valid"] and uvg["tune_steps"] == 0
uvg_anchors = merged(B / "uvg_classical_420_gop30/parity.json", B / "uvg_classical_444_gop30/parity.json")
uvg_anchors = {k: v for k, v in uvg_anchors.items() if v.get("arm", "").startswith("h26")}
results["uvg_1080p_untuned"] = {
    "points": {lam: {"bpp": p["bpp"], "psnr": p["quality"][C]["psnr"], "msssim": p["quality"][C]["msssim"],
                     "encode_s_per_frame": p["encode_seconds"] / p["frames"],
                     "decode_s_per_frame": p["decode_seconds"] / p["frames"]}
               for lam, p in uvg["points"].items()},
    "vs_anchors": vs_anchors(uvg["points"], uvg_anchors),
    "per_video": per_video(uvg),
}
for arm in ("h264_444", "h265_444"):
    base = curve(uvg_anchors, "psnr", arm)
    results["uvg_1080p_untuned"][f"bits_ratio_vs_{arm}"] = {
        lam: bp.rate_ratio_at_quality(base, p["bpp"], p["quality"][C]["psnr"]) for lam, p in uvg["points"].items()}

Path("outputs/p5_encode_tuning/results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
d = results["davis_test_tuning"]
print("DAVIS TEST tuned vs untuned: PSNR %+.2f%%  MS-SSIM %+.2f%%" % (d["bd_vs_untuned/psnr"], d["bd_vs_untuned/msssim"]))
print("per video:", d["per_video_bd_vs_untuned/psnr"])
for k in d["tuned_vs_anchors"]:
    print(f"  {k:28s} untuned {d['untuned_vs_anchors'][k]['bd_rate']}  tuned {d['tuned_vs_anchors'][k]['bd_rate']}")
for lam, p in d["points"].items():
    print(lam, "encode s/frame %.2f -> %.2f" % (p["untuned"]["encode_s_per_frame"], p["tuned"]["encode_s_per_frame"]))
print("UVG:")
for k, v in results["uvg_1080p_untuned"]["vs_anchors"].items():
    print(f"  {k:28s} {v['bd_rate']}  overlap {v['overlap']}")
for k in ("bits_ratio_vs_h264_444", "bits_ratio_vs_h265_444"):
    print(" ", k, results["uvg_1080p_untuned"][k])
for lam, p in results["uvg_1080p_untuned"]["points"].items():
    print(" ", lam, {k: round(v, 3) for k, v in p.items()})
for seq, pts in results["uvg_1080p_untuned"]["per_video"].items():
    print(" ", seq, {lam: f"{b:.3f}bpp/{q:.2f}dB" for lam, (b, q, _) in pts.items()})
