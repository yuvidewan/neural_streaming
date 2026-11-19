"""Low-rate models: gate verdict (docs/PLAN_low_rate.md) on DAVIS TEST, plus UVG 1080p (not gated).

Writes outputs/p6_low_rate/results.json. Run from the repo root:
    .venv/Scripts/python outputs/p6_low_rate/report.py
UVG is skipped if outputs/p6_low_rate/uvg_low_rate.json does not exist yet.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402

C = bp.PRIMARY_CONVENTION
O = Path("outputs/p6_low_rate")
P4 = Path("outputs/p4_wide_latent")
B = Path("outputs/benchmarks")
BAR = 80.9                      # untuned 3-point codec vs x264 4:4:4 over its own range
LOW = ("1e-2", "3e-3", "1e-3")  # the gated low band
ORDER = ("1e-2", "3e-3", "1e-3", "3e-4", "1e-4")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merged(*paths):
    out = {}
    for path in paths:
        out.update(load(path)["points"])
    return {k: v for k, v in out.items() if v.get("arm", "").startswith("h26")}


def curve(points, key, arm=None, lams=None):
    return sorted((p["bpp"], p["quality"][C][key]) for lam, p in points.items()
                  if (arm is None or p.get("arm") == arm) and (lams is None or lam in lams))


def bd(base, test):
    value = bp.bd_rate(base, test)
    return None if value is None else round(value, 2)


def vs_anchors(ours, anchors, lams=None):
    rows = {}
    for arm in sorted({p["arm"] for p in anchors.values()}):
        for key in ("psnr", "msssim"):
            base, test = curve(anchors, key, arm), curve(ours, key, lams=lams)
            rows[f"{arm}/{key}"] = {"bd_rate": bd(base, test), "overlap": bp.overlap(base, test)}
    return rows


def seq_curve(points, seq, arm=None, lams=None):
    out = []
    for lam, p in points.items():
        if (arm is not None and p.get("arm") != arm) or (lams is not None and lam not in lams):
            continue
        for s in p["per_sequence"]:
            if s["sequence"] == seq:
                out.append((s["bpp"], s["mean_psnr"]))
    return sorted(out)


def per_video(ours, anchors, arm):
    seqs = sorted({s["sequence"] for p in ours.values() for s in p["per_sequence"]})
    return {seq: {"low_band": bd(seq_curve(anchors, seq, arm), seq_curve(ours, seq, lams=LOW)),
                  "five_point": bd(seq_curve(anchors, seq, arm), seq_curve(ours, seq)),
                  "points": {lam: [round(b, 4), round(q, 2)] for lam, p in ours.items()
                             for s in p["per_sequence"] if s["sequence"] == seq
                             for b, q in [(s["bpp"], s["mean_psnr"])]}}
            for seq in seqs}


def section(new_path, old_path, anchors, gated):
    new, old = load(new_path), load(old_path)
    ours = {**old["points"], **new["points"]}
    bpp = {lam: ours[lam]["bpp"] for lam in ORDER}
    ordered = all(bpp[a] < bpp[b] for a, b in zip(ORDER, ORDER[1:3]))
    out = {
        "decode_exact": {"new": new["valid"], "old": old["valid"]},
        "points": {lam: {"bpp": round(ours[lam]["bpp"], 4), "psnr": round(ours[lam]["quality"][C]["psnr"], 3),
                         "msssim": round(ours[lam]["quality"][C]["msssim"], 5)} for lam in ORDER},
        "bpp_ordered_1e-2<3e-3<1e-3": ordered,
        "low_band_vs_anchors": vs_anchors(ours, anchors, LOW),
        "five_point_vs_anchors": vs_anchors(ours, anchors),
        "old_three_point_vs_anchors": vs_anchors(old["points"], anchors),
        "per_video_psnr_vs_h264_444": per_video(ours, anchors, "h264_444"),
    }
    if gated:
        score = out["low_band_vs_anchors"]["h264_444/psnr"]["bd_rate"]
        valid = ordered and new["valid"] and old["valid"]
        out["gate"] = {"score_low_band_vs_x264_444_psnr": score, "bar": BAR,
                       "verdict": "INVALID" if not valid else ("PASS" if score is not None and score <= BAR
                                                                else "NOT PASSED")}
    return out


results = {"davis_test": section(O / "gate_test.json", P4 / "gate3_a2_wide_gop30.json",
                                 merged(B / "parity_classical_gop30/parity_classical_gop30.json",
                                        B / "parity_classical_420_hq_gop30/parity.json",
                                        B / "parity_classical_444_gop30/parity_classical_444_gop30.json",
                                        B / "parity_classical_444_hq_gop30/parity.json"), gated=True)}
if (O / "uvg_low_rate.json").exists():
    results["uvg_1080p"] = section(O / "uvg_low_rate.json", P4 / "uvg_a2_wide_gop30.json",
                                   merged(B / "uvg_classical_420_gop30/parity.json",
                                          B / "uvg_classical_444_gop30/parity.json"), gated=False)

(O / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for name, r in results.items():
    print(f"== {name}  decode-exact {r['decode_exact']}  bpp ordered {r['bpp_ordered_1e-2<3e-3<1e-3']}")
    if "gate" in r:
        print("   GATE:", r["gate"])
    for lam, p in r["points"].items():
        print(f"   {lam:5s} {p['bpp']:.4f} bpp  {p['psnr']:.2f} dB  {p['msssim']:.4f}")
    print(f"   {'anchor':26s} {'low band':>9s} {'5-point':>9s} {'old 3-pt':>9s}")
    for k in r["low_band_vs_anchors"]:
        print(f"   {k:26s} {r['low_band_vs_anchors'][k]['bd_rate']!s:>9s} "
              f"{r['five_point_vs_anchors'][k]['bd_rate']!s:>9s} {r['old_three_point_vs_anchors'][k]['bd_rate']!s:>9s}")
    for seq, v in r["per_video_psnr_vs_h264_444"].items():
        print(f"   {seq:22s} low {v['low_band']!s:>8s}  5pt {v['five_point']!s:>8s}  {v['points']}")
