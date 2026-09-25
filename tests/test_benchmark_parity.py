"""Stage 0 scoreboard - tests for `scripts/benchmark_parity.py`.

The load-bearing tests here are:

  * `test_conventions_differ_on_the_same_frames` - the reason the script
    exists. The old "4.8x" figure compared a mean-of-per-frame-dB number
    against a pooled-MSE one; if these conventions did not diverge on uneven
    per-frame error, that comparison would have been harmless;
  * `test_every_arm_is_scored_by_the_same_function` - NVC and FFmpeg must not
    grow separate metric paths;
  * `test_ffmpeg_round_trip_preserves_frame_count_and_shape` - a real FFmpeg
    encode/decode over the raw pipe, skipped when FFmpeg is absent.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.evaluation.ffmpeg import has_encoders

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def parity():
    spec = importlib.util.spec_from_file_location(
        "benchmark_parity", ROOT / "scripts" / "benchmark_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(mse_values, msssim_values=None):
    return {
        "mse": list(mse_values),
        "psnr": [10 * math.log10(1 / v) for v in mse_values],
        "msssim": list(msssim_values or [0.9] * len(mse_values)),
    }


def test_conventions_differ_on_the_same_frames(parity):
    # One sequence with very uneven per-frame error.
    quality = parity.aggregate_quality([_row([1e-4, 1e-2])])
    mean_of_db = (40.0 + 20.0) / 2
    pooled_db = 10 * math.log10(1 / ((1e-4 + 1e-2) / 2))
    assert quality["sequence_mean"]["psnr"] == pytest.approx(mean_of_db)
    assert quality["sequence_pooled"]["psnr"] == pytest.approx(pooled_db)
    # Jensen: mean of dB is never below pooled dB, and here by a lot.
    assert quality["sequence_mean"]["psnr"] - quality["sequence_pooled"]["psnr"] > 7.0


def test_sequence_mean_weights_sequences_equally_and_frame_mean_weights_frames(parity):
    short = _row([1e-2])                       # 20 dB, 1 frame
    long = _row([1e-4] * 3)                    # 40 dB, 3 frames
    quality = parity.aggregate_quality([short, long])
    assert quality["sequence_mean"]["psnr"] == pytest.approx(30.0)
    assert quality["frame_mean"]["psnr"] == pytest.approx(35.0)
    assert quality["global_pooled"]["psnr"] == pytest.approx(
        10 * math.log10(1 / ((1e-2 + 3e-4) / 4)))


def test_primary_convention_is_declared_and_known(parity):
    assert parity.PRIMARY_CONVENTION in parity.CONVENTIONS
    assert parity.PRIMARY_CONVENTION == "sequence_mean"


def test_summarize_point_uses_pooled_bpp(parity):
    rows = [{"sequence": "a", "bytes": 100, "pixels": 800, **_row([1e-3])},
            {"sequence": "b", "bytes": 300, "pixels": 800, **_row([1e-3])}]
    point = parity.summarize_point("x/1", "x", "1", rows)
    assert point["bpp"] == pytest.approx(400 * 8 / 1600)
    assert point["frames"] == 2
    assert "mse" not in point["per_sequence"][0]      # per-frame lists are not stored


def test_rate_ratio_at_quality_interpolates_in_log_rate(parity):
    base = [(0.1, 30.0), (1.0, 40.0)]
    # 35 dB sits halfway in log-rate: sqrt(0.1 * 1.0)
    assert parity.rate_ratio_at_quality(base, 0.632455532, 35.0) == pytest.approx(2.0, rel=1e-6)
    assert parity.rate_ratio_at_quality(base, 1.0, 45.0) is None


def test_bd_rate_sign_means_nvc_needs_more_bits(parity):
    base = [(0.1, 30.0), (0.2, 33.0), (0.4, 36.0)]
    double = [(0.2, 30.0), (0.4, 33.0), (0.8, 36.0)]
    assert parity.bd_rate(base, double) == pytest.approx(100.0, rel=1e-6)
    assert parity.bd_rate(double, base) == pytest.approx(-50.0, rel=1e-6)


def test_uint8_rounding_matches_a_real_decoder(parity):
    frames = torch.tensor([[[[0.6 / 255, 1.49 / 255, 1.2, -0.1]]]])
    rounded = parity.to_uint8_frames(frames) * 255
    assert rounded.flatten().tolist() == pytest.approx([1.0, 1.0, 255.0, 0.0])


def test_source_frames_are_never_requantized(parity):
    exact = torch.randint(0, 256, (2, 3, 4, 4)).float() / 255
    array = parity.frames_to_uint8(exact)
    assert array.shape == (2, 4, 4, 3) and array.dtype == np.uint8
    assert torch.equal(parity.uint8_to_frames(array), exact)
    # channels-last layout gave wrong CUDA MS-SSIM; decoded frames must be NCHW
    assert parity.uint8_to_frames(array).is_contiguous()
    with pytest.raises(ValueError, match="not exact 8-bit"):
        parity.frames_to_uint8(exact + 0.3 / 255)


def test_every_arm_is_scored_by_the_same_function(parity):
    for runner in (parity.run_nvc, parity.run_classical):
        source = inspect.getsource(runner)
        assert "score_frames(" in source
        assert "psnr_from_mse" not in source and "log10" not in source


def test_lowdelay_arms_force_nvc_structure(parity):
    arms = {arm.name: arm for arm in parity.classical_arms(10)}
    assert set(arms) == {"h264", "h265", "h264_lowdelay", "h265_lowdelay"}
    x264 = arms["h264_lowdelay"].encoder_arguments(30)
    for flag, value in (("-g", "10"), ("-keyint_min", "10"), ("-bf", "0"),
                        ("-sc_threshold", "0")):
        assert x264[x264.index(flag) + 1] == value
    x265 = arms["h265_lowdelay"].encoder_arguments(30)
    params = x265[x265.index("-x265-params") + 1]
    assert {"keyint=10", "min-keyint=10", "bframes=0", "scenecut=0"} <= set(params.split(":"))
    # the default arms leave the encoder's own structure alone
    assert "-g" not in arms["h264"].encoder_arguments(30)
    assert "keyint" not in arms["h265"].encoder_arguments(30)[-1]


@pytest.mark.skipif(not has_encoders(["libx264", "libx265"]),
                    reason="FFmpeg with libx264/libx265 not available")
@pytest.mark.parametrize("arm_name", ["h264", "h265_lowdelay"])
def test_ffmpeg_round_trip_preserves_frame_count_and_shape(parity, tmp_path, arm_name):
    arm = {a.name: a for a in parity.classical_arms(4)}[arm_name]
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(7, 64, 64, 3), dtype=np.uint8)
    result = parity.ffmpeg_round_trip(frames, arm, 30, tmp_path)
    assert result["decoded"].shape == frames.shape
    assert result["bytes"] > 0
    assert not any(tmp_path.iterdir())              # the .mp4 was measured, then removed


# --- Intra-only mode (--gop 1), the Stage 1 gate's denominator --------------------


def test_gop_one_renames_the_forced_arms_to_what_they_actually_are(parity):
    """At GOP 1 the forced arms are all-intra, not low-delay. The name is what
    a reader six months from now sees first, so it has to be the truth."""
    arms = {arm.name: arm for arm in parity.classical_arms(1)}

    assert set(arms) == {"h264", "h265", "h264_intra", "h265_intra"}
    assert "all-intra" in arms["h264_intra"].describe()["structure"]
    assert arms["h264"].describe()["structure"] == "encoder default GOP and B-frames"


def test_gop_one_forced_arms_are_genuinely_all_intra(parity):
    arms = {arm.name: arm for arm in parity.classical_arms(1)}

    x264 = arms["h264_intra"].encoder_arguments(30)
    for flag, value in (("-g", "1"), ("-keyint_min", "1"), ("-bf", "0")):
        assert x264[x264.index(flag) + 1] == value
    x265 = arms["h265_intra"].encoder_arguments(30)
    params = set(x265[x265.index("-x265-params") + 1].split(":"))
    assert {"keyint=1", "min-keyint=1", "bframes=0"} <= params


def test_the_default_arms_keep_their_b_frames_at_gop_one(parity):
    """--gop only reaches the forced arms. `h264`/`h265` stay at the encoder's
    own GOP whatever --gop says, so they are NOT the intra reference - reading
    the gate off `nvc_*_vs_h264` instead of `..._vs_h264_intra` would compare
    all-intra NVC against H.264 with B-frames."""
    arms = {arm.name: arm for arm in parity.classical_arms(1)}

    assert "-g" not in arms["h264"].encoder_arguments(30)
    assert "keyint" not in arms["h265"].encoder_arguments(30)[-1]


def test_comparisons_find_the_classical_arms_by_name_not_by_a_fixed_list(parity):
    """compare() used to iterate a hardcoded arm tuple, which would silently
    drop the renamed intra arms and produce a report with no gate number in it."""
    report = {"points": {}}
    for arm, bpps in (("nvc_deployed", (0.3, 0.7)), ("h264_intra", (0.2, 0.6))):
        for index, bpp in enumerate(bpps):
            report["points"][f"{arm}/{index}"] = {
                "arm": arm, "bpp": bpp, "bits": 4 + index, "total_bytes": 1,
                "quality": {c: {"psnr": 28.0 + index, "msssim": 0.95 + 0.01 * index}
                            for c in parity.CONVENTIONS},
            }

    results = parity.compare(report)

    assert "nvc_deployed_vs_h264_intra" in results


def test_the_reproduction_check_is_skipped_away_from_the_deployed_gop(parity):
    """M22 recorded its DAVIS totals at the deployed GOP. At any other GOP the
    codec is configured differently, so a byte mismatch is expected and must not
    be reported as a failed reproduction - that would mark the run invalid."""
    source = inspect.getsource(parity.run_nvc)

    assert "args.gop == parity.DEPLOYED_GOP".replace("parity.", "") in source
    assert parity.DEPLOYED_GOP == 10


def test_intra_only_runs_calibrate_at_the_deployed_gop(parity):
    """The residual grid, context model, codebooks and motion table are all
    fitted on P-frame data, and GOP 1 has none - calibrating there raises. The
    intra tables this measurement uses do not depend on the GOP, so calibration
    stays at the deployed GOP and only the coding GOP changes."""
    source = inspect.getsource(parity.run_nvc)

    assert "calibration_gop = DEPLOYED_GOP if args.gop == 1 else args.gop" in source
    assert "gop_size=calibration_gop" in source
    # the encode/decode calls must still use the requested GOP, not the
    # calibration one, or --gop 1 would silently measure the GOP-10 codec
    assert "gop_size=args.gop" in source
