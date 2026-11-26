"""docs/PLAN_dcvcrt_check.md: score the published DCVC-RT against x264 and our current codec,
video by video and pooled, from its real stream sizes. Coverage is listed before any BD-rate.

    python outputs/p14_dcvcrt_check/report.py   ->  results.json + a printed summary
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "outputs/p13_audit_fixes")
sys.path.insert(0, "scripts")
import report as r  # noqa: E402  (p13 helpers: anchors, curve, per_video, bd)

O = Path("outputs/p14_dcvcrt_check")
STREAMS = Path("C:/dcvc_scratch/streams")
C = r.C
B = r.B
ARMS = ("h264", "h264_lowdelay", "h264_444")
QPS = [0, 13, 25, 38, 50, 63]          # --rate_num 6: DCVC-RT's own uniform sampling


def dcvc_points(log: dict, ds: str, config: dict) -> dict:
    seqs = config["test_classes"][ds]["sequences"]
    points = {}
    for idx, qp in enumerate(QPS):
        rows, total_bytes, total_pixels, enc, dec = [], 0, 0, [], []
        for seq, s in seqs.items():
            res = log[ds][seq][f"{idx:03d}"]
            size = (STREAMS / ds / f"{seq}_q{qp}.bin").stat().st_size
            pixels = s["width"] * s["height"] * s["frames"]
            assert res["frame_pixel_num"] * (res["i_frame_num"] + res["p_frame_num"]) == pixels
            rows.append({"sequence": seq, "bytes": size, "bpp": size * 8 / pixels,
                         "mean_psnr": res["ave_all_frame_psnr"],
                         "bpp_reported": res["ave_all_frame_bpp"]})
            total_bytes += size
            total_pixels += pixels
            enc.append(res["avg_frame_encoding_time"])
            dec.append(res["avg_frame_decoding_time"])
        points[f"q{qp}"] = {
            "arm": "dcvcrt", "qp": qp, "total_bytes": total_bytes, "total_pixels": total_pixels,
            "bpp": total_bytes * 8 / total_pixels,
            "quality": {C: {"psnr": statistics.fmean(x["mean_psnr"] for x in rows)}},
            "per_sequence": rows, "encode_s_per_frame": statistics.fmean(enc),
            "decode_s_per_frame": statistics.fmean(dec)}
    return points


def ours_speed(points: dict) -> dict:
    enc = sum(p["encode_seconds"] for p in points.values())
    dec = sum(p["decode_seconds"] for p in points.values())
    frames = sum(p["frames"] for p in points.values())
    return {"encode_s_per_frame": round(enc / frames, 4), "decode_s_per_frame": round(dec / frames, 4)}


def score_set(name: str, anchors: dict, dcvc: dict, ours: dict) -> dict:
    videos = [x["sequence"] for x in next(iter(dcvc.values()))["per_sequence"]]
    out = {"videos": {}, "pooled": {}, "wins": {}}
    rivals = {arm: (anchors, arm) for arm in ARMS} | {"ours": (ours, None)}
    for v in videos:
        d = r.per_video(dcvc, v, "psnr")
        row = {"dcvcrt_psnr_range": [round(d[0][1], 2), round(d[-1][1], 2)],
               "dcvcrt_bpp_range": [round(d[0][0], 4), round(d[-1][0], 4)]}
        for rival, (pts, arm) in rivals.items():
            base = r.per_video(pts, v, "psnr", arm)
            window = r.bp.overlap(base, d)
            row[f"{rival}_overlap_db"] = None if window is None else [round(w, 2) for w in window]
            row[f"vs_{rival}"] = None if window is None else round(r.bp.bd_rate(base, d), 2)
        out["videos"][v] = row
    for rival, (pts, arm) in rivals.items():
        scored = [row[f"vs_{rival}"] for row in out["videos"].values()]
        out["wins"][rival] = {"wins": sum(x is not None and x < 0 for x in scored),
                              "of": len(scored), "not_scored": sum(x is None for x in scored)}
        out["pooled"][rival] = r.bd(r.curve(pts, "psnr", arm), r.curve(dcvc, "psnr"))
    return out


def main() -> int:
    config = json.loads((O / "dcvc_config.json").read_text(encoding="utf-8"))
    log = json.loads((O / "dcvcrt.json").read_text(encoding="utf-8"))
    davis_anchor = r.anchors(B / "parity_classical_gop30/parity_classical_gop30.json",
                             B / "parity_classical_420_hq_gop30/parity.json",
                             B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                             B / "parity_classical_444_hq_gop30/parity.json")
    uvg_anchor = r.anchors(B / "uvg_classical_420_gop30_realfps/parity.json",
                           B / "uvg_classical_444_gop30_realfps/parity.json")
    ours = {"DAVIS": r.points(r.O / "test_motion.json"), "UVG": r.points(r.O / "uvg_motion.json")}
    anchors = {"DAVIS": davis_anchor, "UVG": uvg_anchor}
    result = {"plan": "docs/PLAN_dcvcrt_check.md", "qps": QPS, "sets": {}, "speed": {}}
    for ds in ("DAVIS", "UVG"):
        dcvc = dcvc_points(log, ds, config)
        result["sets"][ds] = score_set(ds, anchors[ds], dcvc, ours[ds])
        result["sets"][ds]["dcvcrt_points"] = {k: {"bpp": round(p["bpp"], 5),
                                                    "psnr": round(p["quality"][C]["psnr"], 3)}
                                               for k, p in dcvc.items()}
        enc = statistics.fmean(p["encode_s_per_frame"] for p in dcvc.values())
        dec = statistics.fmean(p["decode_s_per_frame"] for p in dcvc.values())
        result["speed"][ds] = {"dcvcrt_pytorch_fallback": {"encode_s_per_frame": round(enc, 4),
                                                           "decode_s_per_frame": round(dec, 4),
                                                           "encode_fps": round(1 / enc, 1),
                                                           "decode_fps": round(1 / dec, 1)},
                               "ours": ours_speed(ours[ds])}

    wins = {arm: sum(result["sets"][ds]["wins"][arm]["wins"] for ds in ("DAVIS", "UVG"))
            for arm in (*ARMS, "ours")}
    rule1 = wins["h264_lowdelay"] >= 15 and wins["h264"] >= 14
    vs_ours = {ds: result["sets"][ds]["pooled"]["ours"] for ds in ("DAVIS", "UVG")}
    rule2 = all(x is not None and x <= -25.0 for x in vs_ours.values())
    hard = {v: {arm: result["sets"]["DAVIS"]["videos"][v][f"vs_{arm}"] for arm in ARMS}
            for v in ("cows", "schoolgirls")}
    result["decision"] = {"wins_of_16": wins, "rule1_wins": rule1, "pooled_vs_ours": vs_ours,
                          "rule2_vs_ours": rule2, "verdict": "GO" if rule1 and rule2 else "NO-GO",
                          "cows_schoolgirls_vs_x264": hard}
    (O / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    for ds in ("DAVIS", "UVG"):
        s = result["sets"][ds]
        print(f"== {ds}: pooled DCVC-RT vs " + ", ".join(f"{k} {v:+.1f}%" if v is not None else f"{k} n/a"
                                                     for k, v in s["pooled"].items()))
        print("   wins: " + ", ".join(f"{k} {v['wins']}/{v['of']}" + (f" ({v['not_scored']} unscored)" if v["not_scored"] else "")
                                     for k, v in s["wins"].items()))
        for v, row in s["videos"].items():
            print(f"   {v:14s} {row['dcvcrt_psnr_range'][0]:5.1f}-{row['dcvcrt_psnr_range'][1]:5.1f} dB  "
                  + "  ".join(f"{k} {row['vs_' + k]:+7.1f}" if row["vs_" + k] is not None else f"{k}    n/a"
                              for k in (*ARMS, "ours")))
        print(f"   speed: {result['speed'][ds]}")
    print(json.dumps(result["decision"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
