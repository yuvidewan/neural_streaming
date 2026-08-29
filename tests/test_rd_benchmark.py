"""Tests for the Milestone 7 rate-distortion benchmark harness.

Covers sequence discovery/validation, the bitrate and compression-ratio
arithmetic, weighted aggregation, the result schema, plotting, and the
codec configuration objects.

Nothing here needs DAVIS, Vimeo-90K, a checkpoint, or CUDA. The one real
FFmpeg round-trip is skipped gracefully when libx264 is unavailable, and
uses a tiny synthetic sequence rather than any real dataset.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from nvc.data.validation import DatasetValidationError
from nvc.evaluation.codecs import (
    DEFAULT_CRF_VALUES,
    DEFAULT_NVC_BIT_DEPTHS,
    CodecResult,
    FFmpegCodecConfig,
    FFmpegVideoCodec,
    FrameMetrics,
    H264_CODEC,
    H265_CODEC,
)
from nvc.evaluation.ffmpeg import has_encoders
from nvc.evaluation.rd_benchmark import (
    CALIBRATION_CLIP_WARNING_PERCENT,
    METHODOLOGY_NOTE,
    BenchmarkRun,
    CalibrationMismatchError,
    aggregate_results,
    check_calibration_fit,
    create_run_directory,
    file_sha256,
    require_calibration_fit,
    write_results,
)
from nvc.evaluation.sequences import (
    FFMPEG_FRAME_PATTERN,
    BenchmarkSequence,
    discover_sequences,
    materialize_sequence_for_ffmpeg,
    validate_sequence_frames,
)

from helpers import make_tiny_checkpoint, make_tiny_manifest

# MS-SSIM needs >160px; sequence tests that only exercise ordering/IO use
# smaller frames, and the FFmpeg round-trip uses an even multiple of 2.
_FFMPEG_SIZE = 64


def _make_sequence(tmp_path, *, frames=5, width=32, height=32, name="seq"):
    import cv2
    import numpy as np

    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(1, frames + 1):
        image = np.full((height, width, 3), (index * 20) % 256, dtype=np.uint8)
        path = directory / f"{name}_{index:06d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return BenchmarkSequence(
        dataset="synthetic", sequence_id=name, split="test",
        frame_paths=tuple(paths), width=width, height=height,
    )


def _result(**overrides) -> CodecResult:
    defaults = dict(
        codec="nvc", configuration="8bit-per_channel", sequence_id="seq",
        dataset="synthetic", frame_count=10, width=256, height=256,
        total_bytes=160_000, mean_psnr=27.0, mean_msssim=0.95,
        pooled_mse=0.002, encode_seconds=1.0, decode_seconds=1.0, details={},
    )
    defaults.update(overrides)
    return CodecResult(**defaults)


# --- Sequence discovery ---


def test_discover_sequences_groups_frames_by_manifest_item(tmp_path):
    manifest_path = make_tiny_manifest(
        tmp_path, num_sequences=4, frames_per_sequence=8, train_ratio=0.5,
        val_ratio=0.25, test_ratio=0.25,
    )

    sequences = discover_sequences(manifest_path, split="train")

    assert sequences
    for sequence in sequences:
        assert sequence.frame_count == 8
        assert all(path.is_file() for path in sequence.frame_paths)


def test_discover_sequences_preserves_frame_ordering(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=12)

    sequence = discover_sequences(manifest_path, split="train")[0]

    # Numeric order, not lexicographic: frame 2 must precede frame 10.
    indices = [int(p.stem.rsplit("_", 1)[1]) for p in sequence.frame_paths]
    assert indices == sorted(indices)
    assert indices == list(range(1, len(indices) + 1))


def test_max_sequences_truncates_deterministically(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, num_sequences=4, frames_per_sequence=6)

    first = discover_sequences(manifest_path, split="train", max_sequences=1)
    second = discover_sequences(manifest_path, split="train", max_sequences=1)

    assert len(first) == 1
    assert [s.sequence_id for s in first] == [s.sequence_id for s in second]


def test_max_frames_per_sequence_truncates_from_the_front(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=10)

    sequences = discover_sequences(manifest_path, split="train", max_frames_per_sequence=3)

    assert all(s.frame_count == 3 for s in sequences)
    assert sequences[0].frame_paths[0].name.endswith("000001.png")


def test_discover_sequences_missing_manifest_raises(tmp_path):
    with pytest.raises(DatasetValidationError):
        discover_sequences(tmp_path / "missing.json")


def test_discover_sequences_missing_frame_raises(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=6)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_item = next(i for i in manifest["items"] if i["split"] == "train")
    frame_dir = manifest_path.parent / train_item["frame_directory"]
    (frame_dir / train_item["frame_filename_pattern"].format(index=2)).unlink()

    with pytest.raises(DatasetValidationError, match="missing"):
        discover_sequences(manifest_path, split="train")


def test_discover_sequences_unknown_split_raises(tmp_path):
    manifest_path = make_tiny_manifest(tmp_path)

    with pytest.raises(DatasetValidationError, match="No sequences"):
        discover_sequences(manifest_path, split="nonexistent")


# --- Sequence validation ---


def test_validate_sequence_frames_accepts_a_consistent_sequence(tmp_path):
    validate_sequence_frames(_make_sequence(tmp_path))


def test_validate_sequence_frames_rejects_inconsistent_dimensions(tmp_path):
    import cv2
    import numpy as np

    sequence = _make_sequence(tmp_path, frames=3, width=32, height=32)
    # Rewrite one frame at a different resolution: FFmpeg would silently
    # rescale it, misaligning every subsequent per-frame comparison.
    cv2.imwrite(str(sequence.frame_paths[1]), np.zeros((64, 64, 3), dtype=np.uint8))

    with pytest.raises(DatasetValidationError, match="must share one resolution"):
        validate_sequence_frames(sequence)


# --- FFmpeg staging ---


def test_materialize_renumbers_frames_contiguously(tmp_path):
    sequence = _make_sequence(tmp_path, frames=4)
    destination = tmp_path / "staged"

    directory, pattern = materialize_sequence_for_ffmpeg(sequence, destination)

    assert pattern == FFMPEG_FRAME_PATTERN
    staged = sorted(directory.glob("frame_*.png"))
    assert [p.name for p in staged] == [
        "frame_000001.png", "frame_000002.png", "frame_000003.png", "frame_000004.png"
    ]


def test_materialize_preserves_frame_content_and_order(tmp_path):
    from nvc.data.image_io import read_image_as_tensor

    sequence = _make_sequence(tmp_path, frames=4)
    directory, _ = materialize_sequence_for_ffmpeg(sequence, tmp_path / "staged")

    for index, original in enumerate(sequence.frame_paths, start=1):
        staged = directory / f"frame_{index:06d}.png"
        assert torch.equal(read_image_as_tensor(staged), read_image_as_tensor(original))


def test_materialize_is_idempotent(tmp_path):
    sequence = _make_sequence(tmp_path, frames=3)
    destination = tmp_path / "staged"

    materialize_sequence_for_ffmpeg(sequence, destination)
    directory, _ = materialize_sequence_for_ffmpeg(sequence, destination)

    assert len(list(directory.glob("frame_*.png"))) == 3


# --- Bitrate / ratio arithmetic ---


def test_bpp_is_bytes_times_eight_over_total_pixels():
    result = _result(frame_count=10, width=256, height=256, total_bytes=160_000)

    # 160000 * 8 / (10 * 256 * 256) = 1280000 / 655360
    assert result.bpp == pytest.approx(1_280_000 / 655_360)


def test_total_pixels_spans_every_frame():
    assert _result(frame_count=7, width=100, height=50).total_pixels == 35_000


def test_compression_ratio_is_against_raw_uint8_rgb():
    result = _result(frame_count=10, width=256, height=256, total_bytes=196_608)

    # Raw = 10 * 256 * 256 * 3 = 1,966,080 bytes -> exactly 10x
    assert result.compression_ratio == pytest.approx(10.0)


def test_raw_rgb_bytes_uses_three_channels(tmp_path):
    sequence = _make_sequence(tmp_path, frames=4, width=10, height=20)

    assert sequence.raw_rgb_bytes() == 4 * 10 * 20 * 3


def test_pooled_psnr_derives_from_pooled_mse():
    from nvc.evaluation.basic_metrics import psnr_from_mse

    result = _result(pooled_mse=0.001)

    assert result.pooled_psnr == pytest.approx(float(psnr_from_mse(0.001)))


def test_to_row_exposes_the_required_schema_fields():
    row = _result().to_row()

    for field in (
        "dataset", "sequence_id", "codec", "codec_configuration", "frame_count",
        "width", "height", "total_bytes", "bpp", "compression_ratio",
        "mean_psnr", "mean_msssim", "encode_time_seconds", "decode_time_seconds",
    ):
        assert field in row


def test_to_row_includes_codec_specific_details():
    row = _result(details={"quantization_bits": 8, "entropy_model_id": "abc123"}).to_row()

    assert row["quantization_bits"] == 8
    assert row["entropy_model_id"] == "abc123"


# --- FrameMetrics ---


def test_frame_metrics_scores_identical_frames_perfectly():
    metrics = FrameMetrics()
    frame = torch.rand(1, 3, 200, 200)

    metrics.add(frame, frame)

    assert metrics.mean_psnr == float("inf")
    assert metrics.mean_msssim == pytest.approx(1.0, abs=1e-5)
    assert metrics.pooled_mse == 0.0


def test_frame_metrics_skips_msssim_for_frames_below_its_size_floor(caplog):
    # Small frames must still yield PSNR, with MS-SSIM reported as absent
    # rather than as a fabricated number - and the drop must be logged, not
    # silent, so a caller can tell a clean result apart from a partial one.
    metrics = FrameMetrics()
    frame = torch.rand(1, 3, 32, 32)

    with caplog.at_level("WARNING"):
        metrics.add(frame, frame, frame_label="seq/000001")

    assert metrics.mean_psnr == float("inf")
    assert metrics.mean_msssim is None
    assert metrics.msssim_dropped == 1
    assert metrics.msssim_frame_count == 0
    assert "seq/000001" in caplog.text


def test_frame_metrics_mean_psnr_is_not_poisoned_by_one_perfect_frame():
    # A single bit-exact frame alongside imperfect ones must not make the
    # whole sequence's mean_psnr report +inf - that would silently hide the
    # real error in every other frame. Regression test for the bug where
    # mean_psnr was a plain sum()/len() over per-frame PSNR values.
    metrics = FrameMetrics()
    perfect = torch.rand(1, 3, 32, 32)
    metrics.add(perfect, perfect)  # MSE == 0 -> PSNR == +inf
    reference = torch.zeros(1, 3, 32, 32)
    noisy = torch.full((1, 3, 32, 32), 0.1)
    metrics.add(noisy, reference)  # MSE == 0.01 -> finite PSNR

    from nvc.evaluation.basic_metrics import psnr_from_mse

    assert math.isfinite(metrics.mean_psnr)
    assert metrics.mean_psnr == pytest.approx(float(psnr_from_mse(0.01)))


def test_frame_metrics_mean_psnr_is_inf_only_when_every_frame_is_perfect():
    metrics = FrameMetrics()
    frame = torch.rand(1, 3, 32, 32)
    metrics.add(frame, frame)
    metrics.add(frame, frame)

    assert metrics.mean_psnr == float("inf")


def test_frame_metrics_pools_error_across_frames():
    metrics = FrameMetrics()
    reference = torch.zeros(1, 3, 32, 32)
    metrics.add(torch.zeros(1, 3, 32, 32), reference)
    metrics.add(torch.full((1, 3, 32, 32), 0.5), reference)

    # Half the elements have error 0.25, half have 0 -> pooled MSE 0.125
    assert metrics.pooled_mse == pytest.approx(0.125)


# --- Weighted aggregation ---


def test_aggregate_bpp_is_pixel_weighted_not_a_mean_of_means():
    # Two sequences of very different lengths: a naive mean of per-sequence
    # BPP would weight them equally, which is the bug this guards against.
    long_sequence = _result(sequence_id="long", frame_count=100, total_bytes=100 * 1000)
    short_sequence = _result(sequence_id="short", frame_count=10, total_bytes=10 * 5000)

    aggregate = aggregate_results([long_sequence, short_sequence])[0]

    total_bytes = 100 * 1000 + 10 * 5000
    total_pixels = 110 * 256 * 256
    assert aggregate["aggregate_bpp"] == pytest.approx(total_bytes * 8 / total_pixels)
    # The naive mean-of-means would be the average of 0.122 and 0.610.
    naive = (long_sequence.bpp + short_sequence.bpp) / 2
    assert aggregate["aggregate_bpp"] != pytest.approx(naive)


def test_aggregate_psnr_is_frame_weighted():
    long_sequence = _result(sequence_id="long", frame_count=90, mean_psnr=30.0)
    short_sequence = _result(sequence_id="short", frame_count=10, mean_psnr=20.0)

    aggregate = aggregate_results([long_sequence, short_sequence])[0]

    assert aggregate["mean_psnr"] == pytest.approx((30.0 * 90 + 20.0 * 10) / 100)


def test_aggregate_msssim_is_frame_weighted():
    long_sequence = _result(sequence_id="long", frame_count=90, mean_msssim=0.98)
    short_sequence = _result(sequence_id="short", frame_count=10, mean_msssim=0.88)

    aggregate = aggregate_results([long_sequence, short_sequence])[0]

    assert aggregate["mean_msssim"] == pytest.approx((0.98 * 90 + 0.88 * 10) / 100)


def test_aggregate_handles_missing_msssim():
    aggregate = aggregate_results([_result(mean_msssim=None)])[0]

    assert aggregate["mean_msssim"] is None


def test_aggregate_msssim_weighted_by_actual_scored_frames_not_nominal_count():
    # "long" nominally has 90 frames but only 20 actually produced an
    # MS-SSIM score (the rest were, e.g., below the minimum spatial size).
    # Weighting by frame_count=90 would over-count its influence; weighting
    # by msssim_frame_count=20 is what the docstring's "frame-weighted"
    # claim actually means for a partially-scored result.
    long_sequence = _result(
        sequence_id="long", frame_count=90, mean_msssim=0.98, msssim_frame_count=20,
    )
    short_sequence = _result(
        sequence_id="short", frame_count=10, mean_msssim=0.88, msssim_frame_count=10,
    )

    aggregate = aggregate_results([long_sequence, short_sequence])[0]

    assert aggregate["mean_msssim"] == pytest.approx((0.98 * 20 + 0.88 * 10) / 30)


def test_aggregate_msssim_falls_back_to_frame_count_when_unset():
    # Results built without msssim_frame_count (e.g. older code, or tests
    # that don't care about this distinction) must aggregate exactly as
    # before: weighted by the nominal frame_count.
    result = _result(frame_count=90, mean_msssim=0.98)

    assert result.msssim_frame_count is None
    aggregate = aggregate_results([result])[0]
    assert aggregate["mean_msssim"] == pytest.approx(0.98)


def test_aggregate_groups_by_codec_and_configuration():
    results = [
        _result(codec="nvc", configuration="8bit-per_channel"),
        _result(codec="nvc", configuration="6bit-per_channel"),
        _result(codec="h264", configuration="crf23"),
    ]

    aggregates = aggregate_results(results)

    assert len(aggregates) == 3
    assert {(a["codec"], a["codec_configuration"]) for a in aggregates} == {
        ("nvc", "8bit-per_channel"), ("nvc", "6bit-per_channel"), ("h264", "crf23"),
    }


def test_aggregate_compression_ratio_is_pixel_weighted():
    results = [
        _result(sequence_id="a", frame_count=10, total_bytes=196_608),
        _result(sequence_id="b", frame_count=10, total_bytes=196_608),
    ]

    aggregate = aggregate_results(results)[0]

    assert aggregate["compression_ratio"] == pytest.approx(10.0)


# --- Result schema / serialization ---


def test_write_results_produces_every_expected_artifact(tmp_path):
    run = BenchmarkRun(
        output_dir=tmp_path,
        metadata={"dataset": "synthetic", "methodology_note": METHODOLOGY_NOTE},
        results=[_result(), _result(codec="h264", configuration="crf23")],
    )

    paths = write_results(tmp_path, run)

    for key in ("results_json", "results_csv", "aggregate_csv", "metadata_json"):
        assert paths[key].is_file()


def test_results_json_is_valid_and_carries_the_methodology_note(tmp_path):
    run = BenchmarkRun(
        output_dir=tmp_path,
        metadata={"methodology_note": METHODOLOGY_NOTE},
        results=[_result()],
    )
    paths = write_results(tmp_path, run)

    document = json.loads(paths["results_json"].read_text(encoding="utf-8"))

    assert "per_sequence" in document
    assert "aggregate" in document
    assert "aggregation_methodology" in document
    # The temporal-fairness caveat must travel with the numbers.
    note = document["metadata"]["methodology_note"].lower()
    assert "intra-only" in note and "temporal redundancy" in note


def test_results_json_sanitizes_infinite_metrics_to_valid_json(tmp_path):
    # A pixel-perfect frame legitimately produces mean_psnr == +inf (see
    # FrameMetrics.mean_psnr). json.dumps emits the non-standard `Infinity`
    # token for that by default, which is not valid JSON (RFC 8259) and
    # breaks strict parsers (browsers, jq, non-Python JSON libraries).
    # Regression test: the written file must be strict-JSON-readable, and
    # the non-finite value must come back as null rather than crash.
    run = BenchmarkRun(
        output_dir=tmp_path,
        metadata={"methodology_note": METHODOLOGY_NOTE},
        results=[_result(mean_psnr=float("inf"))],
    )
    paths = write_results(tmp_path, run)

    raw_text = paths["results_json"].read_text(encoding="utf-8")
    # Python's own json.loads accepts the non-standard Infinity/NaN tokens
    # (a CPython extension), so the real proof of RFC 8259 compliance is
    # that the token never appears in the file at all - a strict external
    # parser (JS JSON.parse, jq) would reject it if it did.
    assert "Infinity" not in raw_text

    document = json.loads(raw_text)
    assert document["per_sequence"][0]["mean_psnr"] is None


def test_results_csv_has_a_header_and_one_row_per_measurement(tmp_path):
    import csv as csv_module

    run = BenchmarkRun(
        output_dir=tmp_path, metadata={},
        results=[_result(sequence_id="a"), _result(sequence_id="b")],
    )
    paths = write_results(tmp_path, run)

    with paths["results_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))

    assert len(rows) == 2
    assert {row["sequence_id"] for row in rows} == {"a", "b"}
    assert "bpp" in rows[0]


def test_csv_columns_unify_across_codecs_with_different_details(tmp_path):
    import csv as csv_module

    run = BenchmarkRun(
        output_dir=tmp_path, metadata={},
        results=[
            _result(codec="nvc", details={"quantization_bits": 8}),
            _result(codec="h264", configuration="crf23", details={"crf": 23}),
        ],
    )
    paths = write_results(tmp_path, run)

    with paths["results_csv"].open(newline="", encoding="utf-8") as handle:
        reader = csv_module.DictReader(handle)
        fieldnames = reader.fieldnames

    assert "quantization_bits" in fieldnames
    assert "crf" in fieldnames


def test_create_run_directory_does_not_collide(tmp_path):
    first = create_run_directory(tmp_path, "run_a")
    second = create_run_directory(tmp_path, "run_b")

    assert first != second
    assert (first / "plots").is_dir()


def test_file_sha256_is_stable_and_none_for_missing_files(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"neural video compression")

    assert file_sha256(path) == file_sha256(path)
    assert len(file_sha256(path)) == 64
    assert file_sha256(tmp_path / "absent.bin") is None


# --- Calibration guard ---


def test_check_calibration_fit_reports_low_high_clip_split(tmp_path):
    # M8: check_calibration_fit() used to collapse count_clipped()'s
    # low/high breakdown into a single clipped_percent. A calibration whose
    # range is deliberately off-center (not just too narrow) can clip almost
    # entirely on one side - that asymmetry is exactly what a shifted latent
    # distribution (e.g. after QAT) would look like, and the aggregate
    # percentage alone can't distinguish it from symmetric clipping.
    from nvc.compression.calibration import calibrate_quantization_params
    from nvc.evaluation.sequences import discover_sequences
    from nvc.training import load_model_from_checkpoint

    checkpoint_path = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    model, _ = load_model_from_checkpoint(checkpoint_path, device=torch.device("cpu"))

    manifest_path = make_tiny_manifest(tmp_path, num_sequences=4, frames_per_sequence=8)
    sequences = discover_sequences(manifest_path, split="train")

    with torch.no_grad():
        from nvc.data.image_io import read_image_as_tensor
        probe_batch = torch.stack([
            read_image_as_tensor(sequences[0].frame_paths[0]),
            read_image_as_tensor(sequences[0].frame_paths[1]),
        ])
        probe_latents = model.encode(probe_batch)

    # A narrow, deliberately shifted range (well above the latents' own
    # spread) forces clipping that is almost entirely on the low side -
    # the asymmetric case the aggregate-only metric couldn't show.
    shifted = probe_latents + 50.0
    narrow_params = calibrate_quantization_params(
        shifted, bits=8, mode="per_channel", lower_percentile=45.0, upper_percentile=55.0,
    )

    fit = check_calibration_fit(model, sequences, narrow_params, device=torch.device("cpu"))

    assert fit["fits"] is False  # mismatch is expected and intentional here
    assert fit["clipped_low_percent"] > fit["clipped_high_percent"]
    assert fit["clipped_low_percent"] + fit["clipped_high_percent"] == pytest.approx(
        fit["clipped_percent"], abs=1e-6
    )
    assert fit["clipped_total"] > 0
    assert fit["total_values"] > 0


def test_calibration_guard_passes_for_a_fitting_calibration():
    require_calibration_fit(
        {"fits": True, "clipped_percent": 0.09, "threshold_percent": 2.0},
        checkpoint="a.pt", calibration="c.json",
    )


def test_calibration_guard_raises_with_actionable_guidance():
    # A mismatched calibration would still encode successfully and silently
    # produce invalid numbers, so this must be a hard stop.
    with pytest.raises(CalibrationMismatchError) as info:
        require_calibration_fit(
            {"fits": False, "clipped_percent": 17.88, "threshold_percent": 2.0},
            checkpoint="vimeo_epoch17_best.pt",
            calibration="latent_quantization.json",
        )

    message = str(info.value)
    assert "17.88%" in message
    assert "calibrate_quantizer.py" in message
    assert "vimeo_epoch17_best.pt" in message


def test_calibration_threshold_sits_between_known_good_and_known_bad():
    # Measured on this project: 0.09% matched, 17.88% mismatched.
    assert 0.09 < CALIBRATION_CLIP_WARNING_PERCENT < 17.88


# --- Codec configuration ---


def test_h264_and_h265_share_one_implementation_differing_only_in_config():
    assert H264_CODEC.encoder == "libx264"
    assert H265_CODEC.encoder == "libx265"
    assert type(FFmpegVideoCodec(H264_CODEC)) is type(FFmpegVideoCodec(H265_CODEC))


def test_codec_configuration_label_encodes_crf():
    codec = FFmpegVideoCodec(FFmpegCodecConfig(name="h264", encoder="libx264", crf=28))

    assert codec.configuration == "crf28"
    assert codec.describe() == {"codec": "h264", "codec_configuration": "crf28"}


def test_intra_only_is_marked_in_the_configuration_label():
    codec = FFmpegVideoCodec(
        FFmpegCodecConfig(name="h265", encoder="libx265", crf=23, intra_only=True)
    )

    assert codec.configuration == "crf23-intra"


def test_configuration_label_disambiguates_non_default_preset_and_pix_fmt():
    # Two configs sharing crf but differing in preset/pix_fmt must not get
    # the same configuration label, or aggregate_results (which groups by
    # (codec, configuration)) would silently average two different
    # operating points together. Regression test for that bug.
    default_preset = FFmpegCodecConfig(name="h264", encoder="libx264", crf=23)
    other_preset = FFmpegCodecConfig(
        name="h264", encoder="libx264", crf=23, preset="veryslow",
    )
    other_pix_fmt = FFmpegCodecConfig(
        name="h264", encoder="libx264", crf=23, pix_fmt="yuv444p",
    )

    assert default_preset.configuration == "crf23"
    assert other_preset.configuration != default_preset.configuration
    assert other_pix_fmt.configuration != default_preset.configuration
    assert other_preset.configuration != other_pix_fmt.configuration


@pytest.mark.parametrize(
    "encoder, expected_flag",
    [("libx264", "-g"), ("libx265", "-x265-params")],
)
def test_intra_only_emits_encoder_specific_keyframe_flags(encoder, expected_flag):
    config = FFmpegCodecConfig(name="x", encoder=encoder, intra_only=True)

    assert expected_flag in config.intra_arguments()


def test_normal_mode_adds_no_keyframe_flags():
    assert FFmpegCodecConfig(name="h264", encoder="libx264").intra_arguments() == []


def test_default_operating_points_are_sensible():
    assert DEFAULT_CRF_VALUES == (18, 23, 28, 33)
    assert sorted(DEFAULT_CRF_VALUES) == list(DEFAULT_CRF_VALUES)  # ascending = worsening
    assert DEFAULT_NVC_BIT_DEPTHS == (8, 6, 4)


# --- Plotting ---


def test_plot_rate_distortion_generates_both_plots(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path as _Path

    run_dir = tmp_path / "run"
    (run_dir / "plots").mkdir(parents=True)
    document = {
        "metadata": {
            "dataset": "synthetic", "sequence_ids": ["a"], "total_frames": 10,
            "methodology_note": METHODOLOGY_NOTE,
        },
        "per_sequence": [],
        "aggregate": [
            {"codec": "nvc", "codec_configuration": "8bit", "aggregate_bpp": 1.9,
             "mean_psnr": 27.2, "mean_msssim": 0.95},
            {"codec": "nvc", "codec_configuration": "4bit", "aggregate_bpp": 0.9,
             "mean_psnr": 26.2, "mean_msssim": 0.92},
            {"codec": "h264", "codec_configuration": "crf23", "aggregate_bpp": 0.4,
             "mean_psnr": 32.0, "mean_msssim": 0.97},
        ],
    }
    (run_dir / "results.json").write_text(json.dumps(document), encoding="utf-8")

    script = _Path(__file__).resolve().parents[1] / "scripts" / "plot_rate_distortion.py"
    spec = importlib.util.spec_from_file_location("plot_rate_distortion", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["plot_rate_distortion"] = module
    spec.loader.exec_module(module)

    exit_code = module.main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    assert (run_dir / "plots" / "rd_psnr_vs_bpp.png").is_file()
    assert (run_dir / "plots" / "rd_msssim_vs_bpp.png").is_file()


# --- Real FFmpeg round-trip (skipped when unavailable) ---


@pytest.mark.skipif(
    not has_encoders(["libx264"]), reason="FFmpeg build lacks libx264"
)
def test_h264_round_trip_on_a_tiny_synthetic_sequence(tmp_path):
    sequence = _make_sequence(
        tmp_path, frames=6, width=_FFMPEG_SIZE, height=_FFMPEG_SIZE, name="tiny"
    )
    codec = FFmpegVideoCodec(FFmpegCodecConfig(name="h264", encoder="libx264", crf=23))

    result = codec.run(sequence, tmp_path / "work")

    assert result.frame_count == 6
    assert result.total_bytes > 0
    assert result.bpp > 0
    assert result.mean_psnr > 0
    assert result.details["encoder"] == "libx264"
    assert result.details["temporal_prediction"] is True
    # MS-SSIM is unavailable at 64px, and must be reported as absent.
    assert result.mean_msssim is None
