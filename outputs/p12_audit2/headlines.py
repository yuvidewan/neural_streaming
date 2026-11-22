"""docs/PLAN_audit2.md, checks D1 and B2: every headline in the README.md "Latest" block,
recomputed from the raw point files, and with two other BD-rate methods.

D1: each number is rebuilt here from the raw coded-point JSONs. Merges and lambda bands are
written again from the milestone plans, not imported from the milestone report.py, so a bug
in a report.py shows up as a mismatch. A mismatch over 0.1 point is a finding.

B2: the same curves through
  (a) the project's piecewise-linear BD-rate (benchmark_parity.bd_rate, the quoted method);
  (b) the classic Bjontegaard cubic: log10(rate) as a cubic in quality, integrated over the
      overlap only. Each curve is fitted on its points inside the overlap plus one point
      beyond each end; with fewer than 4 points the degree is n-1;
  (c) PCHIP (monotone piecewise cubic), as in the JVET common test conditions, for context.
Rule (fixed in the plan): a finding only if (b) and (a) disagree on the sign of a headline, or
by 1 point or more on a number quoted as a win.

    .venv/Scripts/python outputs/p12_audit2/headlines.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
B = Path("outputs/benchmarks")
O = Path("outputs/p12_audit2")
LOW = ("1e-2", "3e-3", "1e-3")


def pts(*files):
    out = {}
    for f in files:
        out.update(json.loads(Path(f).read_text(encoding="utf-8"))["points"])
    return out


def anchors(*files):
    return {k: v for k, v in pts(*files).items() if v.get("arm", "").startswith("h26")}


def curve(points, key, arm=None, lams=None):
    return sorted((p["bpp"], p["quality"][C][key]) for lam, p in points.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams))


DAVIS = anchors(B / "parity_classical_gop30/parity_classical_gop30.json",
                B / "parity_classical_420_hq_gop30/parity.json",
                B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                B / "parity_classical_444_hq_gop30/parity.json")
UVG = anchors(B / "uvg_classical_420_gop30/parity.json", B / "uvg_classical_444_gop30/parity.json")
MCL = anchors(B / "mcljcv_classical_420_gop30/parity.json", B / "mcljcv_classical_444_gop30/parity.json")
P4, P6, P8 = Path("outputs/p4_wide_latent"), Path("outputs/p6_low_rate"), Path("outputs/p8_motion")
TUNED = pts(P4 / "tune_test_s20_lr0.01_wide_gop30.json")
UVG3 = pts(P4 / "uvg_a2_wide_gop30.json")
DAVIS_P6 = pts(P4 / "gate3_a2_wide_gop30.json", P6 / "gate_test.json")
UVG_P6 = pts(P4 / "uvg_a2_wide_gop30.json", P6 / "uvg_low_rate.json")
DAVIS_P8, UVG_P8 = pts(P8 / "test_motion.json"), pts(P8 / "uvg_motion.json")
DAVIS_MS, UVG_MS = pts("outputs/p10_msssim/test_msssim.json"), pts("outputs/p10_msssim/uvg_msssim.json")
MCL_MSE, MCL_MS = pts("outputs/p11_mcljcv/mcljcv_mse.json"), pts("outputs/p11_mcljcv/mcljcv_msssim.json")

# (label, quoted value, base curve, test curve). Base first: + = the test needs more bits.
HEADLINES = [
    ("DAVIS tuned vs x264 4:2:0, PSNR", 20.6, curve(DAVIS, "psnr", "h264"), curve(TUNED, "psnr")),
    ("DAVIS tuned vs x264 4:4:4, PSNR", 69.4, curve(DAVIS, "psnr", "h264_444"), curve(TUNED, "psnr")),
    ("UVG untuned vs x264 4:4:4, PSNR", -18.9, curve(UVG, "psnr", "h264_444"), curve(UVG3, "psnr")),
    ("UVG untuned vs x265 4:4:4, PSNR", 17.2, curve(UVG, "psnr", "h265_444"), curve(UVG3, "psnr")),
    ("DAVIS low band vs x264 4:4:4, PSNR", 47.1, curve(DAVIS, "psnr", "h264_444"),
     curve(DAVIS_P6, "psnr", lams=LOW)),
    ("DAVIS low band vs x264 4:2:0, PSNR", 35.1, curve(DAVIS, "psnr", "h264"),
     curve(DAVIS_P6, "psnr", lams=LOW)),
    ("UVG low band vs x264 4:4:4, PSNR (p6)", 95.6, curve(UVG, "psnr", "h264_444"),
     curve(UVG_P6, "psnr", lams=LOW)),
    ("UVG 5-point vs x264 4:4:4, PSNR (p6)", 27.9, curve(UVG, "psnr", "h264_444"), curve(UVG_P6, "psnr")),
    ("Motion: UVG new vs current, PSNR", -10.7, curve(UVG_P6, "psnr"), curve(UVG_P8, "psnr")),
    ("Motion: DAVIS new vs current, PSNR", -4.75, curve(DAVIS_P6, "psnr"), curve(DAVIS_P8, "psnr")),
    ("DAVIS vs x264 4:2:0, PSNR (p8)", 27.6, curve(DAVIS, "psnr", "h264"), curve(DAVIS_P8, "psnr")),
    ("DAVIS vs x264 4:4:4, PSNR (p8)", 55.8, curve(DAVIS, "psnr", "h264_444"), curve(DAVIS_P8, "psnr")),
    ("UVG vs x264 4:2:0, PSNR (p8)", -18.1, curve(UVG, "psnr", "h264"), curve(UVG_P8, "psnr")),
    ("UVG vs x264 4:4:4, PSNR (p8)", 17.9, curve(UVG, "psnr", "h264_444"), curve(UVG_P8, "psnr")),
    ("UVG low band vs x264 4:4:4, PSNR (p8)", 65.8, curve(UVG, "psnr", "h264_444"),
     curve(UVG_P8, "psnr", lams=LOW)),
    ("MS-SSIM set vs MSE set, DAVIS, MS-SSIM", -22.4, curve(DAVIS_P8, "msssim"), curve(DAVIS_MS, "msssim")),
    ("MS-SSIM set vs MSE set, UVG, MS-SSIM", -9.7, curve(UVG_P8, "msssim"), curve(UVG_MS, "msssim")),
    ("MS-SSIM set, UVG vs x264 4:2:0, MS-SSIM", 0.4, curve(UVG, "msssim", "h264"), curve(UVG_MS, "msssim")),
    ("MS-SSIM set, DAVIS vs x264 4:2:0, MS-SSIM", 16.3, curve(DAVIS, "msssim", "h264"),
     curve(DAVIS_MS, "msssim")),
    ("MCL-JCV MSE set vs x264 4:2:0, PSNR", -1.4, curve(MCL, "psnr", "h264"), curve(MCL_MSE, "psnr")),
    ("MCL-JCV MSE set vs x264 4:2:0 low-delay, PSNR", -12.3, curve(MCL, "psnr", "h264_lowdelay"),
     curve(MCL_MSE, "psnr")),
    ("MCL-JCV MSE set vs x264 4:4:4, PSNR", 40.1, curve(MCL, "psnr", "h264_444"), curve(MCL_MSE, "psnr")),
    ("MCL-JCV MS-SSIM set vs x264 4:2:0, MS-SSIM", -0.4, curve(MCL, "msssim", "h264"),
     curve(MCL_MS, "msssim")),
]


def _local(points, low, high):
    """The points inside [low, high] plus one beyond each end (sorted by quality)."""
    p = sorted(points, key=lambda t: t[1])
    q = [t[1] for t in p]
    i0 = max(0, max([i for i, v in enumerate(q) if v <= low], default=0))
    i1 = min(len(p) - 1, min([i for i, v in enumerate(q) if v >= high], default=len(p) - 1))
    return p[i0:i1 + 1]


def bd_cubic(base, test):
    low = max(min(q for _, q in base), min(q for _, q in test))
    high = min(max(q for _, q in base), max(q for _, q in test))
    if high <= low:
        return None
    ints = []
    for c in (base, test):
        c = _local(c, low, high)
        q, r = np.array([t[1] for t in c]), np.log10([t[0] for t in c])
        poly = np.polyint(np.polyfit(q, r, min(3, len(c) - 1)))
        ints.append(np.polyval(poly, high) - np.polyval(poly, low))
    return float((10 ** ((ints[1] - ints[0]) / (high - low)) - 1) * 100)


def _pchip(x, y, grid):
    x, y = np.asarray(x, float), np.asarray(y, float)
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.zeros_like(y)
    for k in range(1, len(x) - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0], m[-1] = d[0], d[-1]
    idx = np.clip(np.searchsorted(x, grid) - 1, 0, len(x) - 2)
    t = (grid - x[idx]) / h[idx]
    return ((2 * t**3 - 3 * t**2 + 1) * y[idx] + (t**3 - 2 * t**2 + t) * h[idx] * m[idx]
            + (-2 * t**3 + 3 * t**2) * y[idx + 1] + (t**3 - t**2) * h[idx] * m[idx + 1])


def bd_pchip(base, test):
    low = max(min(q for _, q in base), min(q for _, q in test))
    high = min(max(q for _, q in base), max(q for _, q in test))
    if high <= low:
        return None
    grid = np.linspace(low, high, 2000)
    vals = []
    for c in (base, test):
        c = sorted(c, key=lambda t: t[1])
        vals.append(_pchip([t[1] for t in c], np.log10([t[0] for t in c]), grid))
    return float((10 ** np.mean(vals[1] - vals[0]) - 1) * 100)


def main() -> int:
    rows, d1_findings, b2_findings = [], [], []
    for label, quoted, base, test in HEADLINES:
        linear, cubic, pchip = bp.bd_rate(base, test), bd_cubic(base, test), bd_pchip(base, test)
        mismatch = abs(linear - quoted)
        decimals = len(str(quoted).split(".")[1]) if "." in str(quoted) else 0
        d1 = round(linear, decimals) != quoted and mismatch > 0.1
        sign = (cubic is not None) and (np.sign(cubic) != np.sign(linear))
        win_gap = quoted < 0 and cubic is not None and abs(cubic - linear) >= 1.0
        row = {"headline": label, "quoted": quoted, "recomputed_linear": round(linear, 2),
               "d1_mismatch": d1, "cubic": None if cubic is None else round(cubic, 2),
               "pchip": None if pchip is None else round(pchip, 2),
               "b2_finding": bool(sign or win_gap), "test_points": len(test),
               "overlap_db": [round(x, 2) for x in bp.overlap(base, test)]}
        rows.append(row)
        (d1_findings if d1 else []).append(label)
        (b2_findings if row["b2_finding"] else []).append(label)
        print(f"{label:48s} quoted {quoted:+7.2f}  linear {linear:+7.2f}  cubic "
              f"{row['cubic']!s:>7}  pchip {row['pchip']!s:>7}  {'D1!' if d1 else ''}"
              f"{'B2!' if row['b2_finding'] else ''}")
    # Ranges quoted for UVG untuned (36.8-40.9 dB) and the DAVIS low band (28.6-33.0 dB).
    extra = {"UVG untuned PSNR range": [round(min(q for _, q in curve(UVG3, "psnr")), 1),
                                         round(max(q for _, q in curve(UVG3, "psnr")), 1)],
             "DAVIS low-band PSNR range": [round(min(q for _, q in curve(DAVIS_P6, "psnr", lams=LOW)), 1),
                                            round(max(q for _, q in curve(DAVIS_P6, "psnr", lams=LOW)), 1)]}
    sel = json.loads(Path("outputs/p9_bframes/results.json").read_text(encoding="utf-8"))["selection"]["table"]
    ratios = [100 * (t["M2_same"]["bpp"] / t["M1_same"]["bpp"] - 1) for t in sel.values()]
    extra["B-frames M=2 bits vs all-P on VAL-B (quoted +0.1..+10.1%)"] = [round(min(ratios), 1),
                                                                         round(max(ratios), 1)]
    mcl = json.loads((Path("outputs/p11_mcljcv/results.json")).read_text(encoding="utf-8"))
    extra["MCL-JCV wins vs x264 4:4:4 (quoted 6/30 PSNR, 11/30 MS-SSIM)"] = [
        mcl["mse"]["wins_vs_x264_444"]["wins"], mcl["msssim"]["wins_vs_x264_444"]["wins"]]
    print(json.dumps(extra, indent=1))
    out = {"headlines": rows, "other_quoted": extra, "d1_mismatches": d1_findings,
           "b2_findings": b2_findings}
    (O / "d1_b2_headlines.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
