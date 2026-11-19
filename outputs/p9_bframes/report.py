"""docs/PLAN_bframes.md: gate verdict and the always-reported numbers.

The oracle curve takes, per lambda, the test_<lam>.json / uvg_<lam>.json point when VAL-B chose
B-frames, else the current codec's point (outputs/p8_motion, the same encoder by construction).
Writes outputs/p9_bframes/results.json. Run from the repo root:
    .venv/Scripts/python outputs/p9_bframes/report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
O, P8, B = Path("outputs/p9_bframes"), Path("outputs/p8_motion"), Path("outputs/benchmarks")
LOW = ("1e-2", "3e-3", "1e-3")
LOW_DELAY_NOTE = "B-frames add delay: the low-delay anchors are no longer like-for-like"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def anchors(*paths):
    out = {}
    for path in paths:
        out.update({k: v for k, v in load(path)["points"].items() if v.get("arm", "").startswith("h26")})
    return out


def curve(pts, key, arm=None, lams=None):
    return sorted((p["bpp"], p["quality"][C][key]) for lam, p in pts.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams))


def seq_curve(pts, seq, arm=None):
    return sorted((s["bpp"], s["mean_psnr"]) for p in pts.values()
                  if arm is None or p.get("arm") == arm
                  for s in p["per_sequence"] if s["sequence"] == seq)


def bd(base, test):
    v = bp.bd_rate(base, test)
    return None if v is None else round(v, 2)


def section(prefix, current_path, anchor_paths, choice):
    current_report = load(current_path)
    old = current_report["points"]
    new, exact = {}, current_report["valid"]
    for lam in old:
        path = O / f"{prefix}_{lam}.json"
        if choice[lam] != "M1_same":
            report = load(path)
            exact &= report["valid"]
            new[lam] = report["points"][lam]
        else:
            new[lam] = old[lam]
    anc = anchors(*anchor_paths)
    seqs = sorted({s["sequence"] for p in new.values() for s in p["per_sequence"]})
    out = {
        "decode_exact": exact,
        "bd_vs_current": {key: bd(curve(old, key), curve(new, key)) for key in ("psnr", "msssim")},
        "bd_vs_current_low_band_psnr": bd(curve(old, "psnr", lams=LOW), curve(new, "psnr", lams=LOW)),
        "points": {},
        "vs_anchors": {"note": LOW_DELAY_NOTE},
        "per_video_bd_vs_current_psnr": {s: bd(seq_curve(old, s), seq_curve(new, s)) for s in seqs},
        "per_video_bd_vs_x264_444_psnr": {s: {"new": bd(seq_curve(anc, s, "h264_444"), seq_curve(new, s)),
                                              "old": bd(seq_curve(anc, s, "h264_444"), seq_curve(old, s))}
                                          for s in seqs},
    }
    for lam, p in sorted(new.items(), key=lambda kv: float(kv[0]), reverse=True):
        split = p["split"]
        out["points"][lam] = {
            "config": choice[lam], "bpp_new": round(p["bpp"], 4), "bpp_old": round(old[lam]["bpp"], 4),
            "psnr_new": round(p["quality"][C]["psnr"], 3), "psnr_old": round(old[lam]["quality"][C]["psnr"], 3),
            "b_share_of_stream": round(split.get("b_bytes", 0) / p["total_bytes"], 4),
            "bytes_per_p_frame": round(split["bytes_per_p_frame"], 1),
            "bytes_per_b_frame": round(split.get("bytes_per_b_frame", 0), 1),
            "pooled_psnr_p": split["pooled_psnr_p"], "pooled_psnr_b": split.get("pooled_psnr_b"),
            "b_modes_fwd_bwd_avg": split.get("b_modes_fwd_bwd_avg"),
            "motion_share_of_stream": round(split["motion_bytes"] / p["total_bytes"], 4),
            "encode_s_per_frame": round(p["encode_seconds"] / p["frames"], 3),
            "decode_s_per_frame": round(p["decode_seconds"] / p["frames"], 3)}
    for arm in sorted({p["arm"] for p in anc.values()}):
        for key in ("psnr", "msssim"):
            base = curve(anc, key, arm)
            out["vs_anchors"][f"{arm}/{key}"] = {
                "new_5pt": bd(base, curve(new, key)), "old_5pt": bd(base, curve(old, key)),
                "new_low": bd(base, curve(new, key, lams=LOW)), "old_low": bd(base, curve(old, key, lams=LOW))}
    return out


selection = load(O / "selection.json")
choice = selection["choice"]
davis_anchor_files = (B / "parity_classical_gop30/parity_classical_gop30.json",
                      B / "parity_classical_420_hq_gop30/parity.json",
                      B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                      B / "parity_classical_444_hq_gop30/parity.json")
uvg_anchor_files = (B / "uvg_classical_420_gop30/parity.json", B / "uvg_classical_444_gop30/parity.json")
results = {"selection": selection,
           "davis_test": section("test", P8 / "test_motion.json", davis_anchor_files, choice),
           "uvg_1080p": section("uvg", P8 / "uvg_motion.json", uvg_anchor_files, choice)}
uvg, davis = results["uvg_1080p"]["bd_vs_current"]["psnr"], results["davis_test"]["bd_vs_current"]["psnr"]
exact = results["uvg_1080p"]["decode_exact"] and results["davis_test"]["decode_exact"]
if not exact or uvg is None or davis is None:
    verdict = "NOT PASSED"
elif uvg <= -5 and davis <= -5:
    verdict = "PASS"
elif uvg <= 0 and davis <= 0 and min(uvg, davis) <= -2:
    verdict = "SMALL WIN"
else:
    verdict = "NOT PASSED"
results["gate"] = {"uvg_bd_vs_current_psnr": uvg, "davis_bd_vs_current_psnr": davis,
                   "decode_exact": exact, "verdict": verdict,
                   "rule": "PASS: UVG <= -5% and DAVIS <= -5%; SMALL WIN: both <= 0%, one <= -2%"}
(O / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("choice:", choice)
for name in ("davis_test", "uvg_1080p"):
    r = results[name]
    print(f"== {name}: decode-exact {r['decode_exact']}  BD vs current PSNR {r['bd_vs_current']['psnr']}  "
          f"MS-SSIM {r['bd_vs_current']['msssim']}  low band PSNR {r['bd_vs_current_low_band_psnr']}")
    for lam, p in r["points"].items():
        print("  ", lam, p)
    for k, v in r["vs_anchors"].items():
        print(f"   {k:26s} {v}")
    print("   per video vs current:", r["per_video_bd_vs_current_psnr"])
    print("   per video vs x264 4:4:4:", r["per_video_bd_vs_x264_444_psnr"])
print("GATE:", results["gate"])
