"""Script-level tests for compare_m8_models.py (Milestone 8).

Builds two small but real benchmark_rd.py runs (real checkpoints, real
calibrations, real NVC encode/decode over synthetic frames - the same
pattern as tests/test_pipelines.py) and feeds both run directories into
compare_m8_models.py, exactly like a real baseline-vs-QAT comparison would.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_manifest  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_run(tmp_path, run_name: str, manifest: Path, seed: int) -> Path:
    """Train, calibrate (8/6/4-bit), and benchmark one tiny model - a
    smaller version of the real M8 workflow, against synthetic data."""
    train = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / f"checkpoints_{run_name}"
    assert train.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "2",
        "--batch-size", "2", "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu", "--seed", str(seed),
    ]) == 0
    checkpoint = checkpoint_dir / "best.pt"

    calibrate = _load_script("calibrate_quantizer")
    calibration = tmp_path / f"calib_{run_name}.json"
    for bits, output in [
        (8, calibration),
        (6, tmp_path / f"calib_{run_name}_6bit.json"),
        (4, tmp_path / f"calib_{run_name}_4bit.json"),
    ]:
        assert calibrate.main([
            "--checkpoint", str(checkpoint), "--manifest", str(manifest),
            "--bits", str(bits), "--batch-size", "2", "--max-batches", "2",
            "--output", str(output),
            "--metrics-dir", str(tmp_path / "metrics" / run_name),
            "--visualizations-dir", str(tmp_path / "viz" / run_name),
            "--device", "cpu",
        ]) == 0

    benchmark_rd = _load_script("benchmark_rd")
    run_dir = tmp_path / "benchmarks" / run_name
    assert benchmark_rd.main([
        "--manifest", str(manifest), "--split", "test", "--codecs", "nvc",
        "--checkpoint", str(checkpoint), "--calibration", str(calibration),
        "--nvc-bits", "8", "6", "4",
        "--max-sequences", "2", "--max-frames-per-sequence", "2",
        "--output-dir", str(tmp_path / "benchmarks"), "--run-name", run_name,
        "--device", "cpu", "--allow-calibration-mismatch",  # tiny synthetic latents, per test_pipelines.py
    ]) == 0
    return run_dir


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("m8_compare")
    manifest = make_tiny_manifest(tmp_path, num_sequences=6, frames_per_sequence=4)
    run_a = _build_run(tmp_path, "model_a", manifest, seed=1)
    run_b = _build_run(tmp_path, "model_b", manifest, seed=2)
    return run_a, run_b, tmp_path


def test_compare_m8_models_end_to_end(two_runs):
    run_a, run_b, tmp_path = two_runs
    compare = _load_script("compare_m8_models")
    output_dir = tmp_path / "comparison"

    exit_code = compare.main([
        "--run", f"model_a={run_a}", "--run", f"model_b={run_b}",
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 0
    assert (output_dir / "comparison.json").is_file()
    assert (output_dir / "comparison.csv").is_file()
    assert (output_dir / "rd_psnr_vs_bpp.png").is_file()
    # No MS-SSIM plot at this synthetic frame size: MS-SSIM needs both
    # spatial dimensions >= 161 (see nvc.evaluation.codecs), same reason
    # plot_rate_distortion.py's own tests don't assert on it either.

    rows = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    models = {row["model"] for row in rows}
    bits = {row["bits"] for row in rows}
    assert models == {"model_a", "model_b"}
    assert bits == {8, 6, 4}
    # 2 models x 3 bit depths.
    assert len(rows) == 6


def test_compare_m8_models_derives_payload_bpp_below_total_bpp(two_runs):
    # Payload excludes the fixed .nvc header, so it must never exceed the
    # total-file bpp it was derived from.
    run_a, run_b, tmp_path = two_runs
    compare = _load_script("compare_m8_models")
    output_dir = tmp_path / "comparison_payload_check"

    assert compare.main([
        "--run", f"model_a={run_a}", "--run", f"model_b={run_b}",
        "--output-dir", str(output_dir),
    ]) == 0

    rows = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    for row in rows:
        assert row["payload_bpp"] is not None
        assert 0 < row["payload_bpp"] < row["aggregate_bpp"]


def test_compare_m8_models_requires_at_least_two_runs(tmp_path):
    compare = _load_script("compare_m8_models")
    with pytest.raises(SystemExit):
        compare.main(["--run", f"solo={tmp_path}", "--output-dir", str(tmp_path / "out")])


def test_compare_m8_models_missing_run_dir_fails_cleanly(tmp_path):
    compare = _load_script("compare_m8_models")
    exit_code = compare.main([
        "--run", f"a={tmp_path / 'nope_a'}", "--run", f"b={tmp_path / 'nope_b'}",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert exit_code != 0
