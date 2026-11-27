"""docs/PLAN_fair_eval.md proofs 3 and 5 on synthetic data: the metric code is right, and the
anchor path adds nothing (lossless x264 and x265 decode to the source, byte for byte)."""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from nvc.evaluation import fair

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def frames(n, w, h, value=0):
    return np.full(n * w * h * 3 // 2, value, dtype=np.uint8)


def synthetic_video(tmp_path, n=8, w=64, h=48, seed=0):
    """Smooth moving gradients plus a little noise, so encoders have real work."""
    rng = np.random.default_rng(seed)
    out = []
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(n):
        y = (16 + 200 * (0.5 + 0.5 * np.sin((xx + 2 * t) / 9.0) * np.cos(yy / 7.0)))
        y = np.clip(y + rng.normal(0, 3, y.shape), 16, 235).astype(np.uint8)
        u = np.clip(128 + 40 * np.sin((xx[::2, ::2] - t) / 5.0), 16, 240).astype(np.uint8)
        v = np.clip(128 + 40 * np.cos(yy[::2, ::2] / 4.0 + t), 16, 240).astype(np.uint8)
        out += [y.ravel(), u.ravel(), v.ravel()]
    path = tmp_path / "synthetic.yuv"
    np.concatenate(out).tofile(path)
    return fair.YuvVideo("synthetic", path, w, h, 30.0, n)


# --------------------------------------------------------------------------------- metrics


def test_plane_psnr_known_values():
    ref = np.zeros((2, 4, 4), np.uint8)
    dist = np.ones((2, 4, 4), np.uint8)
    assert fair.plane_psnr(ref, dist) == pytest.approx([20 * math.log10(255)] * 2)
    dist[1] = 2   # second frame: MSE 4
    assert fair.plane_psnr(ref, dist)[1] == pytest.approx(10 * math.log10(255 ** 2 / 4))


def test_identical_planes_hit_the_cap():
    ref = np.full((1, 4, 4), 7, np.uint8)
    assert fair.plane_psnr(ref, ref)[0] == fair.PSNR_CAP


def test_score_yuv_weights_planes_6_1_1():
    w, h = 8, 4
    ref = frames(1, w, h, 100)
    dist = ref.copy()
    y, u, v = fair.split_planes(dist, w, h)  # views into a read-only buffer: copy then edit
    dist = dist.copy()
    luma, chroma = w * h, (w // 2) * (h // 2)
    dist[:luma] += 1                       # Y MSE 1
    dist[luma:luma + chroma] += 2          # U MSE 4
    dist[luma + chroma:] += 4              # V MSE 16
    s = fair.score_yuv(ref, dist, w, h)
    py, pu, pv = (10 * math.log10(255 ** 2 / m) for m in (1, 4, 16))
    assert s["psnr_y"] == pytest.approx(py)
    assert s["psnr_u"] == pytest.approx(pu)
    assert s["psnr_v"] == pytest.approx(pv)
    assert s["psnr_yuv"] == pytest.approx((6 * py + pu + pv) / 8)


def test_score_yuv_averages_psnr_over_frames_not_mse():
    w, h = 4, 4
    ref = frames(2, w, h, 50)
    dist = ref.copy()
    dist[: w * h] += 1                     # frame 0 Y MSE 1, frame 1 identical (capped)
    s = fair.score_yuv(ref, dist, w, h)
    assert s["psnr_y"] == pytest.approx((20 * math.log10(255) + fair.PSNR_CAP) / 2)


def test_split_planes_layout():
    w, h = 4, 2
    raw = np.arange(12, dtype=np.uint8)
    y, u, v = fair.split_planes(raw, w, h)
    assert y.tolist() == [[[0, 1, 2, 3], [4, 5, 6, 7]]]
    assert u.tolist() == [[[8, 9]]] and v.tolist() == [[[10, 11]]]
    with pytest.raises(ValueError):
        fair.split_planes(np.zeros(13, np.uint8), w, h)


# --------------------------------------------------------------------------------- BD-rate


ANCHOR = [(0.02, 32.0), (0.05, 35.0), (0.12, 38.0), (0.3, 41.0), (0.7, 44.0)]


def scaled(curve, factor):
    return [(r * factor, q) for r, q in curve]


def test_identical_curves_give_zero():
    assert fair.bd_rate_pchip(ANCHOR, ANCHOR)["bd_rate"] == pytest.approx(0.0, abs=1e-9)
    assert fair.bd_rate_linear(ANCHOR, ANCHOR) == pytest.approx(0.0, abs=1e-9)


def test_twice_the_rate_gives_plus_100_and_half_gives_minus_50():
    assert fair.bd_rate_pchip(ANCHOR, scaled(ANCHOR, 2))["bd_rate"] == pytest.approx(100.0)
    assert fair.bd_rate_pchip(ANCHOR, scaled(ANCHOR, 0.5))["bd_rate"] == pytest.approx(-50.0)
    assert fair.bd_rate_linear(ANCHOR, scaled(ANCHOR, 2)) == pytest.approx(100.0)


def test_order_of_points_does_not_matter():
    shuffled = [ANCHOR[i] for i in (3, 0, 4, 2, 1)]
    assert fair.bd_rate_pchip(shuffled, scaled(ANCHOR, 2))["bd_rate"] == pytest.approx(100.0)


def test_overlap_only_and_band():
    test = [(r * 2, q) for r, q in ANCHOR if q >= 35.0]
    out = fair.bd_rate_pchip(ANCHOR, test)
    assert (out["low"], out["high"]) == (35.0, 44.0)
    assert out["bd_rate"] == pytest.approx(100.0, abs=1.0)   # 3-point PCHIP: nearly the shape
    band = fair.bd_rate_pchip(ANCHOR, scaled(ANCHOR, 2), band=(36.0, 40.0))
    assert (band["low"], band["high"]) == (36.0, 40.0)
    assert band["bd_rate"] == pytest.approx(100.0)
    assert fair.bd_rate_pchip(ANCHOR, [(1.0, 50.0), (2.0, 52.0)])["bd_rate"] is None


def test_known_nonuniform_case_matches_numeric_integral():
    # Test curve 2x the rate below 38 dB and the same rate above: the mean log-rate gap over
    # [32, 44] is what a fine numeric integral of the two PCHIP curves gives.
    from scipy.interpolate import PchipInterpolator
    test = [(r * (2 if q < 38 else 1), q) for r, q in ANCHOR]
    out = fair.bd_rate_pchip(ANCHOR, test)["bd_rate"]
    grid = np.linspace(32, 44, 200001)
    fa = PchipInterpolator([q for _, q in ANCHOR], np.log10([r for r, _ in ANCHOR]))
    ft = PchipInterpolator([q for _, q in test], np.log10([r for r, _ in test]))
    expected = (10 ** np.mean(ft(grid) - fa(grid)) - 1) * 100
    assert out == pytest.approx(expected, rel=1e-4)
    assert 0 < out < 100


def test_streaming_band_and_quality_at_rate():
    assert fair.quality_at_rate(ANCHOR, 0.05) == pytest.approx(35.0)
    assert fair.quality_at_rate(ANCHOR, 0.001) == pytest.approx(32.0)   # clipped to the range
    low, high = fair.streaming_band(ANCHOR)
    assert low == pytest.approx(32.0)          # 0.016 bpp is below our lowest point: clipped
    assert 38.0 < high < 38.5                  # 0.129 bpp, just above the 0.12 point
    assert fair.streaming_band([(0.5, 40.0), (0.7, 40.5)]) is None      # narrower than 1 dB


def test_coverage_and_extension_rule():
    ours = [(0.05, 33.0), (0.5, 43.0)]
    points = [{"crf": c, "bpp": r, "psnr_yuv": q}
              for c, (r, q) in zip((30, 26, 22, 18, 14), ANCHOR)]
    assert fair.coverage(fair.curve_of(points), ours) == {"top": True, "bottom": True}
    ours_wide = [(0.01, 30.0), (1.0, 46.0)]
    assert fair.extra_crfs(points, ours_wide) == [12, 32]
    assert fair.extra_crfs(points, ours) == []


def test_summary_counts_wins():
    s = fair.summarize({"a": -5.0, "b": -1.0, "c": 10.0, "d": None})
    assert s == {"videos": 4, "scored": 3, "mean": pytest.approx(4 / 3), "median": -1.0,
                 "wins": 2, "robust_wins": 1}


# --------------------------------------------------------------------------------- anchors


def test_arm_arguments():
    ld = fair.ARMS["x264_lowdelay"].arguments(30)
    assert ld[:6] == ["-c:v", "libx264", "-preset", "medium", "-tune", "psnr"]
    assert ["-crf", "30"] == ld[6:8] and "-bf" in ld and "-g" in ld
    x5 = fair.ARMS["x265_lowdelay"].arguments(30)
    assert x5[-1] == "log-level=error:bframes=0:keyint=30"
    assert "-tune" not in fair.ARMS["x264_default_visual"].arguments(30)
    assert fair.ARMS["x264_default"].arguments(lossless=True)[-2:] == ["-qp", "0"]
    assert fair.ARMS["x265_default"].arguments(lossless=True)[-1].endswith("lossless=1")
    assert fair.HEADLINE_ARMS == ("x264_default", "x264_lowdelay", "x265_default",
                                  "x265_lowdelay")


@needs_ffmpeg
@pytest.mark.parametrize("arm", ["x264_default", "x265_default"])
def test_proof3_lossless_anchor_is_byte_identical(tmp_path, arm):
    video = synthetic_video(tmp_path)
    point = fair.encode_anchor(video, fair.ARMS[arm], None, tmp_path / f"l{fair.ARMS[arm].suffix}",
                               lossless=True)
    assert point["identical_to_source"]
    assert point["psnr_yuv"] == fair.PSNR_CAP


@needs_ffmpeg
def test_lossy_anchor_point(tmp_path):
    video = synthetic_video(tmp_path)
    point = fair.encode_anchor(video, fair.ARMS["x264_lowdelay"], 30, tmp_path / "a.264")
    assert point["bytes"] == (tmp_path / "a.264").stat().st_size
    assert point["bpp"] == pytest.approx(point["bytes"] * 8 / (64 * 48 * 8))
    assert 25 < point["psnr_yuv"] < 60 and not point["identical_to_source"]
    # a raw elementary stream, not MP4: an H.264 Annex B start code
    assert (tmp_path / "a.264").read_bytes()[:4] in (b"\x00\x00\x00\x01", b"\x00\x00\x01\x09")


@needs_ffmpeg
def test_rgb_round_trip_is_close(tmp_path):
    # Smooth, in-gamut colour, like real video. (Per-pixel random chroma does not survive
    # 4:2:0 -> 4:4:4 -> 4:2:0, and out-of-gamut YUV clips in RGB; proof 4 measures real video.)
    w, h, n = 64, 48, 8
    yy, xx = np.mgrid[0:h, 0:w]
    parts = []
    for t in range(n):
        parts += [(60 + 120 * (0.5 + 0.5 * np.sin((xx + 2 * t) / 9.0))).astype(np.uint8).ravel(),
                  (128 + 12 * np.sin(xx[::2, ::2] / 8.0)).astype(np.uint8).ravel(),
                  (128 + 12 * np.cos(yy[::2, ::2] / 6.0)).astype(np.uint8).ravel()]
    np.concatenate(parts).tofile(tmp_path / "smooth.yuv")
    video = fair.YuvVideo("smooth", tmp_path / "smooth.yuv", w, h, 30.0, n)
    raw = fair.read_yuv(video)
    rgb = fair.yuv_to_rgb(raw, video.width, video.height)
    assert rgb.shape == (8, 48, 64, 3)
    back = fair.rgb_to_yuv420(rgb)
    s = fair.score_yuv(raw, back, video.width, video.height)
    assert s["psnr_y"] > 45 and s["psnr_yuv"] > 40


@needs_ffmpeg
def test_cli_anchors_cover_and_report(tmp_path):
    """anchors -> coverage extension -> report, end to end on a tiny clip. The test codec is
    the anchor itself at 1.5x the rate, so every BD-rate must come out at +50%."""
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import fair_eval

    video = synthetic_video(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"videos": [video.to_json()]}), encoding="utf-8")
    store = tmp_path / "anchors.json"
    assert fair_eval.main(["anchors", "--manifest", str(manifest), "--store", str(store),
                           "--arms", "x264_lowdelay", "--crf", "24", "28", "32"]) == 0
    points = fair.AnchorStore.open(store).points("synthetic", "x264_lowdelay")
    assert [p["crf"] for p in points] == [24, 28, 32]
    # A test codec spanning a wider quality range than CRF 24-32 forces the extension rule.
    wide = [{"label": str(i), "bpp": p["bpp"] * 1.5, "psnr_yuv": p["psnr_yuv"] + d}
            for i, (p, d) in enumerate(zip(points, (1.0, 0.0, -1.0)))]
    test = tmp_path / "test.json"
    test.write_text(json.dumps({"videos": {"synthetic": wide}}), encoding="utf-8")
    assert fair_eval.main(["anchors", "--manifest", str(manifest), "--store", str(store),
                           "--arms", "x264_lowdelay", "--crf", "24", "28", "32",
                           "--cover", str(test)]) == 0
    points = fair.AnchorStore.open(store).points("synthetic", "x264_lowdelay")
    assert min(p["crf"] for p in points) < 24 and max(p["crf"] for p in points) > 32
    exact = [{"label": str(p["crf"]), "bpp": p["bpp"] * 1.5, "psnr_yuv": p["psnr_yuv"]}
             for p in points]
    test.write_text(json.dumps({"videos": {"synthetic": exact}}), encoding="utf-8")
    out = tmp_path / "report.json"
    assert fair_eval.main(["report", "--manifest", str(manifest), "--store", str(store),
                           "--test", str(test), "--arms", "x264_lowdelay",
                           "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))["arms"]["x264_lowdelay"]
    assert report["full_range"]["mean"] == pytest.approx(50.0, abs=0.01)
    assert report["coverage_ok"]
