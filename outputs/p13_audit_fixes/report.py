"""docs/PLAN_audit2_fixes.md: the checks and the re-quote table (old, after fix 1, after both).

    .venv/Scripts/python outputs/p13_audit_fixes/report.py

Works on whatever has finished: a row whose new files are missing is reported as pending.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
O = Path("outputs/p13_audit_fixes")
B = Path("outputs/benchmarks")
BLOCKS = 1024
LOW = ("1e-2", "3e-3", "1e-3")


def load(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def points(*files):
    out = {}
    for f in files:
        d = load(f)
        if d is None:
            return None
        out.update(d["points"])
    return out


def anchors(*files):
    p = points(*files)
    return None if p is None else {k: v for k, v in p.items() if v.get("arm", "").startswith("h26")}


def derived(pts):
    """Fix 1 without re-coding: every stream is exactly BLOCKS bytes smaller (proof 2, and the
    real re-runs check it on three whole sets)."""
    if pts is None:
        return None
    out = copy.deepcopy(pts)
    for p in out.values():
        p["total_bytes"] -= BLOCKS * len(p["per_sequence"])
        p["bpp"] = p["total_bytes"] * 8 / p["total_pixels"]
    return out


def curve(pts, key, arm=None, lams=None):
    if pts is None:
        return None
    return sorted((p["bpp"], p["quality"][C][key]) for lam, p in pts.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams))


def bd(base, test):
    return None if base is None or test is None else round(bp.bd_rate(base, test), 2)


def same_but_blocks(old, new):
    """Real re-run check: every video exactly BLOCKS bytes smaller, same scores, decode-exact."""
    if old is None or new is None:
        return None
    rows = []
    for lam, p in old.items():
        q = new.get(lam)
        if q is None:
            return None
        olds = {r["sequence"]: r for r in p["per_sequence"]}
        for r in q["per_sequence"]:
            o = olds[r["sequence"]]
            rows.append(o["bytes"] - r["bytes"] == BLOCKS and o["mean_psnr"] == r["mean_psnr"]
                        and o["mean_msssim"] == r["mean_msssim"])
        rows.append(bool(q.get("decode_reconstruction_exact")))
    return all(rows)


VIDEOS_UVG = ("Beauty", "Bosphorus", "HoneyBee", "Jockey", "ReadySetGo", "ShakeNDry", "YachtRide")


def per_video(pts, video, key, arm=None, lams=None):
    if pts is None:
        return None
    out = []
    for lam, p in pts.items():
        if (arm is not None and p.get("arm") != arm) or (lams is not None and lam not in lams):
            continue
        r = next(s for s in p["per_sequence"] if s["sequence"] == video)
        out.append((r["bpp"], r[f"mean_{key}"]))
    return sorted(out)


def mean_per_video_bd(anchor, test, key, arm, videos, lams=None):
    """Amendment 1, report-only: BD-rate per video, then the mean (not the quoted method)."""
    if anchor is None or test is None:
        return None
    values = [bp.bd_rate(per_video(anchor, v, key, arm), per_video(test, v, key, lams=lams))
              for v in videos]
    return round(sum(values) / len(values), 2)


def coverage(anchor_old, anchor_new, test, arms=("h264", "h264_444")):
    """Amendment 1: top anchor PSNR vs our top PSNR, per video."""
    if anchor_new is None or test is None:
        return None
    out, short = {}, []
    for v in VIDEOS_UVG:
        ours = max(q for _, q in per_video(test, v, "psnr"))
        row = {"ours_top": round(ours, 2)}
        for arm in arms:
            for tag, a in (("30fps", anchor_old), ("120fps", anchor_new)):
                top = max(q for _, q in per_video(a, v, "psnr", arm))
                row[f"{arm}_{tag}_top"] = round(top, 2)
                if tag == "120fps" and top < ours:
                    short.append(f"{v} {arm}")
        out[v] = row
    return {"per_video": out, "anchor_ends_below_ours_at_120fps": short}


def unchanged_30fps(old_anchor, new_anchor, rates):
    """MCL-JCV: the videos that really are 30 fps must give byte-identical anchors."""
    if old_anchor is None or new_anchor is None:
        return None
    thirty = {n for n, r in rates.items() if r == 30}
    same = []
    for label, p in old_anchor.items():
        q = {r["sequence"]: r for r in new_anchor[label]["per_sequence"]}
        same += [r["bytes"] == q[r["sequence"]]["bytes"] for r in p["per_sequence"]
                 if r["sequence"] in thirty]
    return {"videos": len(thirty), "rows_checked": len(same), "all_identical": all(same)}


def main() -> int:
    davis = anchors(B / "parity_classical_gop30/parity_classical_gop30.json",
                    B / "parity_classical_420_hq_gop30/parity.json",
                    B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                    B / "parity_classical_444_hq_gop30/parity.json")
    uvg_old = anchors(B / "uvg_classical_420_gop30/parity.json", B / "uvg_classical_444_gop30/parity.json")
    uvg_new = anchors(B / "uvg_classical_420_gop30_realfps/parity.json",
                      B / "uvg_classical_444_gop30_realfps/parity.json")
    mcl_old = anchors(B / "mcljcv_classical_420_gop30/parity.json", B / "mcljcv_classical_444_gop30/parity.json")
    mcl_new = anchors(B / "mcljcv_classical_420_gop30_realfps/parity.json",
                      B / "mcljcv_classical_444_gop30_realfps/parity.json")

    d_mse_old, d_mse_new = points("outputs/p8_motion/test_motion.json"), points(O / "test_motion.json")
    d_ms_old, d_ms_new = points("outputs/p10_msssim/test_msssim.json"), points(O / "test_msssim.json")
    u_mse_old, u_mse_new = points("outputs/p8_motion/uvg_motion.json"), points(O / "uvg_motion.json")
    u_ms_old = points("outputs/p10_msssim/uvg_msssim.json")
    m_mse_old = points("outputs/p11_mcljcv/mcljcv_mse.json")
    m_ms_old = points("outputs/p11_mcljcv/mcljcv_msssim.json")
    u_ms_new, m_mse_new, m_ms_new = derived(u_ms_old), derived(m_mse_old), derived(m_ms_old)

    def row(label, how, metric, arm, anchor_old, anchor_new, test_old, test_new, lams=None):
        base_old, base_new = curve(anchor_old, metric, arm), curve(anchor_new, metric, arm)
        return {"headline": label, "fix1": how,
                "old": bd(base_old, curve(test_old, metric, lams=lams)),
                "after_fix1": bd(base_old, curve(test_new, metric, lams=lams)),
                "after_both": bd(base_new, curve(test_new, metric, lams=lams))}

    def relative(label, how, metric, base_old, base_new, test_old, test_new):
        old = bd(curve(base_old, metric), curve(test_old, metric))
        new = bd(curve(base_new, metric), curve(test_new, metric))
        return {"headline": label, "fix1": how, "old": old, "after_fix1": new, "after_both": new}

    table = [
        row("DAVIS MSE set vs x264 4:2:0, PSNR", "real", "psnr", "h264", davis, davis, d_mse_old, d_mse_new),
        row("DAVIS MSE set vs x264 4:4:4, PSNR", "real", "psnr", "h264_444", davis, davis, d_mse_old, d_mse_new),
        row("UVG MSE set vs x264 4:2:0, PSNR", "real", "psnr", "h264", uvg_old, uvg_new, u_mse_old, u_mse_new),
        row("UVG MSE set vs x264 4:4:4, PSNR", "real", "psnr", "h264_444", uvg_old, uvg_new, u_mse_old, u_mse_new),
        row("UVG MSE set vs x264 4:2:0 low-delay, PSNR", "real", "psnr", "h264_lowdelay", uvg_old, uvg_new,
            u_mse_old, u_mse_new),
        row("UVG low band vs x264 4:4:4, PSNR", "real", "psnr", "h264_444", uvg_old, uvg_new, u_mse_old,
            u_mse_new, lams=LOW),
        relative("MS-SSIM set vs MSE set, DAVIS, MS-SSIM", "real", "msssim", d_mse_old, d_mse_new,
                 d_ms_old, d_ms_new),
        relative("MS-SSIM set vs MSE set, UVG, MS-SSIM", "real/derived", "msssim", u_mse_old, u_mse_new,
                 u_ms_old, u_ms_new),
        row("MS-SSIM set, DAVIS vs x264 4:2:0, MS-SSIM", "real", "msssim", "h264", davis, davis, d_ms_old, d_ms_new),
        row("MS-SSIM set, UVG vs x264 4:2:0, MS-SSIM", "derived", "msssim", "h264", uvg_old, uvg_new,
            u_ms_old, u_ms_new),
        row("MCL-JCV MSE set vs x264 4:2:0, PSNR", "derived", "psnr", "h264", mcl_old, mcl_new, m_mse_old, m_mse_new),
        row("MCL-JCV MSE set vs x264 4:2:0 low-delay, PSNR", "derived", "psnr", "h264_lowdelay", mcl_old,
            mcl_new, m_mse_old, m_mse_new),
        row("MCL-JCV MSE set vs x264 4:4:4, PSNR", "derived", "psnr", "h264_444", mcl_old, mcl_new,
            m_mse_old, m_mse_new),
        row("MCL-JCV MS-SSIM set vs x264 4:2:0, MS-SSIM", "derived", "msssim", "h264", mcl_old, mcl_new,
            m_ms_old, m_ms_new),
    ]
    rates = {i["source_name"]: i["fps"] for i in json.loads(
        Path("C:/mcljcv_scratch/mcljcv/manifest.json").read_text(encoding="utf-8"))["items"]}
    proof = load(O / "proof_encode.json"), load(O / "proof_decode.json")
    checks = {
        "proof2_streams": None if proof[0] is None else len(proof[0]),
        "proof2_pass": (None if None in proof else
                        all(e["equals_old_minus_blocks"] and e["saved"] == BLOCKS
                            and e["same_reconstruction_as_audit"] for e in proof[0].values())
                        and all(d["same"] for d in proof[1].values())),
        "rerun_davis_mse_exactly_1024_smaller": same_but_blocks(d_mse_old, d_mse_new),
        "rerun_davis_msssim_exactly_1024_smaller": same_but_blocks(d_ms_old, d_ms_new),
        "rerun_uvg_mse_exactly_1024_smaller": same_but_blocks(u_mse_old, u_mse_new),
        "mcljcv_30fps_anchors_unchanged": unchanged_30fps(mcl_old, mcl_new, rates),
    }
    mcl_videos = sorted(rates)
    spec = {  # headline -> (anchor old, anchor new, test old, test new, metric, arm, videos, lams)
        "UVG MSE set vs x264 4:2:0, PSNR": (uvg_old, uvg_new, u_mse_old, u_mse_new, "psnr", "h264", VIDEOS_UVG, None),
        "UVG MSE set vs x264 4:4:4, PSNR": (uvg_old, uvg_new, u_mse_old, u_mse_new, "psnr", "h264_444", VIDEOS_UVG, None),
        "UVG MSE set vs x264 4:2:0 low-delay, PSNR": (uvg_old, uvg_new, u_mse_old, u_mse_new, "psnr", "h264_lowdelay", VIDEOS_UVG, None),
        "UVG low band vs x264 4:4:4, PSNR": (uvg_old, uvg_new, u_mse_old, u_mse_new, "psnr", "h264_444", VIDEOS_UVG, LOW),
        "MS-SSIM set, UVG vs x264 4:2:0, MS-SSIM": (uvg_old, uvg_new, u_ms_old, u_ms_new, "msssim", "h264", VIDEOS_UVG, None),
        "MCL-JCV MSE set vs x264 4:2:0, PSNR": (mcl_old, mcl_new, m_mse_old, m_mse_new, "psnr", "h264", mcl_videos, None),
        "MCL-JCV MSE set vs x264 4:2:0 low-delay, PSNR": (mcl_old, mcl_new, m_mse_old, m_mse_new, "psnr", "h264_lowdelay", mcl_videos, None),
        "MCL-JCV MSE set vs x264 4:4:4, PSNR": (mcl_old, mcl_new, m_mse_old, m_mse_new, "psnr", "h264_444", mcl_videos, None),
        "MCL-JCV MS-SSIM set vs x264 4:2:0, MS-SSIM": (mcl_old, mcl_new, m_ms_old, m_ms_new, "msssim", "h264", mcl_videos, None),
    }
    for r in table:
        s = spec.get(r["headline"])
        if s:
            a_old, a_new, t_old, t_new, key, arm, videos, lams = s
            r["per_video_mean_old"] = mean_per_video_bd(a_old, t_old, key, arm, videos, lams)
            r["per_video_mean_after_both"] = mean_per_video_bd(a_new, t_new, key, arm, videos, lams)
    checks["uvg_coverage"] = coverage(uvg_old, uvg_new, u_mse_new)
    out = {"checks": checks, "table": table,
           "note": "DAVIS has no recorded frame rate, so its anchors stay at 30 fps; "
                   "'derived' = bpp from (bytes - 1,024 per video), not re-coded"}
    (O / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    for r in table:
        print(f"{r['headline']:48s} {r['fix1']:13s} old {r['old']!s:>7}  fix1 "
              f"{r['after_fix1']!s:>7}  both {r['after_both']!s:>7}"
              + (f"   per-video mean {r['per_video_mean_old']} -> {r['per_video_mean_after_both']}"
                 if "per_video_mean_old" in r else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
