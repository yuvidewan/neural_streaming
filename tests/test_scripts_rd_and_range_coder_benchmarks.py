"""Script-level tests: benchmark_rd.py, plot_rate_distortion.py,
benchmark_range_coder.py.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import make_tiny_calibration, make_tiny_checkpoint, make_tiny_manifest  # noqa: E402

from nvc.evaluation.ffmpeg import has_encoders  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- benchmark_rd.py: NVC-only path (no FFmpeg dependency) ----------------


def test_benchmark_rd_nvc_only_end_to_end(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib_8bit.json", checkpoint_path=checkpoint, bits=8)
    mod = _load_script("benchmark_rd")

    exit_code = mod.main([
        "--manifest", str(manifest), "--split", "test",
        "--codecs", "nvc",
        "--checkpoint", str(checkpoint), "--calibration", str(calibration),
        "--nvc-bits", "8",
        "--max-sequences", "2", "--max-frames-per-sequence", "2",
        "--output-dir", str(tmp_path / "benchmarks"),
        "--run-name", "test_run",
        "--device", "cpu",
        "--allow-calibration-mismatch",  # tiny synthetic latents won't fit a percentile-calibrated grid well
    ])

    assert exit_code == 0
    run_dir = tmp_path / "benchmarks" / "test_run"
    results_path = run_dir / "results.json"
    assert results_path.is_file()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert any(row["codec"] == "nvc" for row in results["aggregate"])


def test_benchmark_rd_nvc_without_checkpoint_fails_cleanly(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=1, frames_per_sequence=4)
    mod = _load_script("benchmark_rd")

    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--codecs", "nvc",
            "--output-dir", str(tmp_path / "benchmarks"),
        ])


def test_benchmark_rd_missing_manifest_fails_cleanly(tmp_path):
    mod = _load_script("benchmark_rd")
    exit_code = mod.main([
        "--manifest", str(tmp_path / "nope.json"), "--codecs", "nvc",
        "--checkpoint", str(tmp_path / "irrelevant.pt"),
        "--calibration", str(tmp_path / "irrelevant.json"),
    ])
    assert exit_code != 0


@pytest.mark.skipif(not has_encoders(["libx264"]), reason="FFmpeg build lacks libx264")
def test_benchmark_rd_h264_intra_only_end_to_end(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=1, frames_per_sequence=4, width=64, height=64)
    mod = _load_script("benchmark_rd")

    exit_code = mod.main([
        "--manifest", str(manifest), "--codecs", "h264",
        "--crf", "30",
        "--intra-only",
        "--max-sequences", "1", "--max-frames-per-sequence", "4",
        "--output-dir", str(tmp_path / "benchmarks"), "--run-name", "h264_run",
    ])

    assert exit_code == 0
    metadata = json.loads((tmp_path / "benchmarks" / "h264_run" / "metadata.json").read_text())
    assert metadata["methodology_note"]  # the intra-only asymmetry disclosure is always present


# --- plot_rate_distortion.py -----------------------------------------


def test_plot_rate_distortion_end_to_end(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib_8bit.json", checkpoint_path=checkpoint, bits=8)
    rd_mod = _load_script("benchmark_rd")
    run_dir = tmp_path / "benchmarks" / "run1"

    assert rd_mod.main([
        "--manifest", str(manifest), "--codecs", "nvc",
        "--checkpoint", str(checkpoint), "--calibration", str(calibration),
        "--nvc-bits", "8", "--max-sequences", "2", "--max-frames-per-sequence", "2",
        "--output-dir", str(tmp_path / "benchmarks"), "--run-name", "run1",
        "--device", "cpu", "--allow-calibration-mismatch",
    ]) == 0

    plot_mod = _load_script("plot_rate_distortion")
    exit_code = plot_mod.main(["--run-dir", str(run_dir)])

    assert exit_code == 0
    assert (run_dir / "plots" / "rd_psnr_vs_bpp.png").is_file()


def test_plot_rate_distortion_missing_run_dir_fails_cleanly(tmp_path):
    mod = _load_script("plot_rate_distortion")
    exit_code = mod.main(["--run-dir", str(tmp_path / "no_such_run")])
    assert exit_code != 0


# --- benchmark_range_coder.py ------------------------------------------


def test_benchmark_range_coder_end_to_end(tmp_path):
    mod = _load_script("benchmark_range_coder")
    output_path = tmp_path / "range_coder_bench.json"

    exit_code = mod.main([
        "--reps", "2", "--seed", "0", "--label", "pytest_run", "--out", str(output_path),
    ])

    assert exit_code == 0
    assert output_path.is_file()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["label"] == "pytest_run"
    assert result["encode"]["mean_ms"] > 0
    assert result["decode"]["mean_ms"] > 0
    assert result["bits_per_symbol"] > 0


def test_benchmark_range_coder_round_trip_is_actually_lossless(tmp_path, capsys):
    mod = _load_script("benchmark_range_coder")
    exit_code = mod.main(["--reps", "2", "--out", str(tmp_path / "out.json")])
    assert exit_code == 0
    assert "round-trip verified lossless" in capsys.readouterr().out
