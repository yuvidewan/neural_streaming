"""docs/PLAN_motion_rate.md: gate verdict and the always-reported numbers.

Writes outputs/p8_motion/results.json. Run from the repo root:
    .venv/Scripts/python outputs/p8_motion/report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
O = Path("outputs/p8_motion")
P4, P6, B = Path("outputs/p4_wide_latent"), Path("outputs/p6_low_rate"), Path("outputs/benchmarks")
LOW = ("1e-2", "3e-3", "1e-3")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def points(*paths):
    out = {}
    for path in paths:
        out.update(load(path)["points"])
    return out


def anchors(*paths):
    return {k: v for k, v in points(*paths).items() if v.get("arm", "").startswith("h26")}


def curve(pts, key, arm=None, lams=None):
    return sorted((p["bpp"], p["quality"][C][key]) for lam, p in pts.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams))


def bd(base, test):
    v = bp.bd_rate(base, test)
    return None if v is None else round(v, 2)


def seq_curve(pts, seq, arm=None, lams=None):
    return sorted((s["bpp"], s["mean_psnr"]) for lam, p in pts.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams)
                  for s in p["per_sequence"] if s["sequence"] == seq)


def section(new_path, old_paths, anchor_paths):
    new_report = load(new_path)
    new, old, anc = new_report["points"], points(*old_paths), anchors(*anchor_paths)
    seqs = sorted({s["sequence"] for p in new.values() for s in p["per_sequence"]})
    out = {
        "decode_exact": new_report["valid"],
        "bd_vs_current": {key: bd(curve(old, key), curve(new, key)) for key in ("psnr", "msssim")},
        "bd_vs_current_low_band_psnr": bd(curve(old, "psnr", lams=LOW), curve(new, "psnr", lams=LOW)),
        "points": {lam: {"motion_lambda": p["motion_lambda"], "bpp_new": round(p["bpp"], 4),
                         "bpp_old": round(old[lam]["bpp"], 4),
                         "psnr_new": round(p["quality"][C]["psnr"], 3),
                         "psnr_old": round(old[lam]["quality"][C]["psnr"], 3),
                         "motion_share_of_stream": round(p["split"]["motion_bytes"] / p["total_bytes"], 4),
                         "encode_s_per_frame": round(p["encode_seconds"] / p["frames"], 3),
                         "decode_s_per_frame": round(p["decode_seconds"] / p["frames"], 3)}
                   for lam, p in sorted(new.items(), key=lambda kv: float(kv[0]), reverse=True)},
        "vs_anchors": {},
        "per_video_bd_vs_current_psnr": {seq: bd(seq_curve(old, seq), seq_curve(new, seq)) for seq in seqs},
        "per_video_bd_vs_x264_444_psnr": {seq: {"new": bd(seq_curve(anc, seq, "h264_444"), seq_curve(new, seq)),
                                                "old": bd(seq_curve(anc, seq, "h264_444"), seq_curve(old, seq))}
                                          for seq in seqs},
    }
    for arm in sorted({p["arm"] for p in anc.values()}):
        for key in ("psnr", "msssim"):
            base = curve(anc, key, arm)
            out["vs_anchors"][f"{arm}/{key}"] = {
                "new_5pt": bd(base, curve(new, key)), "old_5pt": bd(base, curve(old, key)),
                "new_low": bd(base, curve(new, key, lams=LOW)), "old_low": bd(base, curve(old, key, lams=LOW))}
    return out


davis_anchor_files = (B / "parity_classical_gop30/parity_classical_gop30.json",
                      B / "parity_classical_420_hq_gop30/parity.json",
                      B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                      B / "parity_classical_444_hq_gop30/parity.json")
uvg_anchor_files = (B / "uvg_classical_420_gop30/parity.json", B / "uvg_classical_444_gop30/parity.json")
results = {
    "selection": load(O / "selection.json"),
    "davis_test": section(O / "test_motion.json", (P4 / "gate3_a2_wide_gop30.json", P6 / "gate_test.json"),
                          davis_anchor_files),
}
if (O / "uvg_motion.json").exists():
    results["uvg_1080p"] = section(O / "uvg_motion.json",
                                   (P4 / "uvg_a2_wide_gop30.json", P6 / "uvg_low_rate.json"), uvg_anchor_files)
    uvg, davis = results["uvg_1080p"]["bd_vs_current"]["psnr"], results["davis_test"]["bd_vs_current"]["psnr"]
    exact = results["uvg_1080p"]["decode_exact"] and results["davis_test"]["decode_exact"]
    if not exact or uvg is None or davis is None:
        verdict = "NOT PASSED"
    elif uvg <= -10 and davis <= 0:
        verdict = "PASS"
    elif uvg <= -3 and davis <= 0:
        verdict = "SMALL WIN"
    else:
        verdict = "NOT PASSED"
    results["gate"] = {"uvg_bd_vs_current_psnr": uvg, "davis_bd_vs_current_psnr": davis,
                       "decode_exact": exact, "verdict": verdict,
                       "rule": "PASS: UVG <= -10% and DAVIS <= 0%; SMALL WIN: UVG -10..-3%, DAVIS <= 0%"}

(O / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("choice:", results["selection"]["choice"])
for name in ("davis_test", "uvg_1080p"):
    if name not in results:
        continue
    r = results[name]
    print(f"== {name}: decode-exact {r['decode_exact']}  BD vs current PSNR {r['bd_vs_current']['psnr']}  "
          f"MS-SSIM {r['bd_vs_current']['msssim']}  low band PSNR {r['bd_vs_current_low_band_psnr']}")
    for lam, p in r["points"].items():
        print("  ", lam, p)
    for k, v in r["vs_anchors"].items():
        print(f"   {k:26s} {v}")
    print("   per video vs current:", r["per_video_bd_vs_current_psnr"])
    print("   per video vs x264 4:4:4:", r["per_video_bd_vs_x264_444_psnr"])
if "gate" in results:
    print("GATE:", results["gate"])
