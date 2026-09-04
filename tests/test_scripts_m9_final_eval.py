"""Tests for scripts/m9_final_calibrate_benchmark.py.

Per TESTING.md's "Added a new script?" rule: `build_arg_parser` defaults, a
missing-required-file error path, and a happy-path `main(argv)` run against
synthetic data checking the exit code and the files written.

The benchmark stage is exercised on a tiny synthetic dataset - the real
719-frame DAVIS runs are recorded in MILESTONE_9_PLAN.md, not repeated here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_checkpoint, make_tiny_manifest  # noqa: E402

from nvc.utils.config import load_default_config  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arg_parser_defaults_follow_the_established_methodology():
    mod = _load_script("m9_final_calibrate_benchmark")
    args = mod.build_arg_parser(load_default_config()).parse_args([])

    # The M7/M8 calibration protocol, unchanged.
    assert args.calibration_batches == 50
    assert args.mode == "per_channel"
    assert args.split == "test", "benchmarking uses the held-out split"
    assert args.seed == 42
    assert args.stage == "both"
    assert args.allow_clipping is False, "clipping over guard must stop by default"
    assert mod.CLIP_GUARD_PERCENT == 2.0
    assert mod.BIT_DEPTHS == (8, 6, 4)


def test_the_comparison_set_covers_m7_m8_the_control_and_three_lambdas():
    mod = _load_script("m9_final_calibrate_benchmark")
    models = mod._model_set(Path("outputs/m9_final"))

    assert [m["key"] for m in models] == ["M7", "M8-QAT", "M9-CTRL", "M9-L", "M9-M", "M9-H"]
    # Baselines come from their own historical locations and are never rewritten.
    assert models[0]["checkpoint"] == Path("outputs/checkpoints/vimeo_epoch17_best.pt")
    assert models[1]["checkpoint"] == Path("outputs/qat_combined/checkpoints_qat_noise/best.pt")
    # M9 models come from the M9 output tree.
    for model in models[2:]:
        assert model["checkpoint"].parts[:2] == ("outputs", "m9_final")
        assert model["checkpoint"].name == "best.pt"


def test_calibration_filenames_match_the_benchmark_lookup_convention(tmp_path):
    """benchmark_rd.py finds non-primary depths by a _<n>bit suffix on the base
    filename. If these two disagree the benchmark silently reuses the 8-bit
    grid for every depth, so the convention is pinned here."""
    mod = _load_script("m9_final_calibrate_benchmark")
    benchmark = _load_script("benchmark_rd")

    base = mod._calibration_path(tmp_path, "M9-L", 8)
    assert base.name == "m9_l.json"
    for bits in (6, 4):
        expected = mod._calibration_path(tmp_path, "M9-L", bits)
        assert expected.name == f"m9_l_{bits}bit.json"
        assert benchmark._calibration_path_for_bits(base, bits, 8) == expected
    assert benchmark._calibration_path_for_bits(base, 8, 8) == base


def test_reports_a_missing_manifest(tmp_path, capsys):
    mod = _load_script("m9_final_calibrate_benchmark")

    exit_code = mod.main(["--manifest", str(tmp_path / "absent.json"), "--device", "cpu"])

    assert exit_code == 1
    assert "--manifest not found" in capsys.readouterr().err


def test_reports_a_missing_checkpoint(tmp_path, capsys):
    mod = _load_script("m9_final_calibrate_benchmark")
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)

    exit_code = mod.main([
        "--manifest", str(manifest), "--output-dir", str(tmp_path / "out"),
        "--stage", "calibrate", "--only", "M9-L", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "checkpoint missing for M9-L" in capsys.readouterr().err


def test_unknown_only_key_is_rejected(tmp_path):
    mod = _load_script("m9_final_calibrate_benchmark")
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)

    with pytest.raises(SystemExit):
        mod.main([
            "--manifest", str(manifest), "--only", "NOPE", "--device", "cpu",
        ])


def test_benchmark_stage_refuses_to_run_without_calibration(tmp_path, capsys):
    """Benchmarking on a stale or absent grid would silently invalidate the
    whole comparison, so it must be an error, not a fallback."""
    mod = _load_script("m9_final_calibrate_benchmark")
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=4, width=32, height=32)
    output_dir = tmp_path / "out"
    (output_dir / "control_lambda0").mkdir(parents=True)
    make_tiny_checkpoint(output_dir / "control_lambda0" / "best.pt")

    exit_code = mod.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--stage", "benchmark", "--only", "M9-CTRL", "--device", "cpu",
    ])

    assert exit_code == 1
    assert "run --stage calibrate first" in capsys.readouterr().err


def test_calibrate_then_benchmark_end_to_end(tmp_path):
    mod = _load_script("m9_final_calibrate_benchmark")
    # Enough sequences and calibration batches that the percentile fit actually
    # represents the latent: a 2-batch sample of a randomly-initialised model
    # clips ~3% and is (correctly) rejected by benchmark_rd's own fit guard.
    manifest = make_tiny_manifest(
        tmp_path, num_sequences=16, frames_per_sequence=8, width=32, height=32,
    )
    output_dir = tmp_path / "out"
    (output_dir / "control_lambda0").mkdir(parents=True)
    make_tiny_checkpoint(output_dir / "control_lambda0" / "best.pt")

    exit_code = mod.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--only", "M9-CTRL", "--batch-size", "4", "--calibration-batches", "16",
        "--max-sequences", "1", "--max-frames-per-sequence", "2", "--device", "cpu",
    ])

    assert exit_code == 0

    report = json.loads((output_dir / "calibration_report.json").read_text())
    assert report["calibration_split"] == "train"
    assert report["lower_percentile"] == 0.1 and report["upper_percentile"] == 99.9
    assert {row["bits"] for row in report["rows"]} == {8, 6, 4}
    for row in report["rows"]:
        assert Path(row["calibration"]).is_file()
        assert 0.0 <= row["clipped_percent"] <= 100.0

    aggregate = json.loads((output_dir / "benchmark_aggregate.json").read_text())
    assert "not the training-time Laplace proxy" in aggregate["note"]
    assert "MEASURED .nvc payload" in aggregate["note"]
    assert aggregate["rows"], "the benchmark must produce measurements"
    for row in aggregate["rows"]:
        assert row["model"] == "M9-CTRL"
        assert row["aggregate_bpp"] > 0
        assert row["bytes_per_frame"] > 0
    assert (output_dir / "benchmark_aggregate.csv").is_file()


def test_a_calibration_over_the_clip_guard_stops_before_benchmarking(tmp_path, capsys, monkeypatch):
    """Pathological clipping is an explicit M9 STOP condition - it must not be
    benchmarked past silently."""
    mod = _load_script("m9_final_calibrate_benchmark")
    manifest = make_tiny_manifest(tmp_path, num_sequences=8, frames_per_sequence=6, width=32, height=32)
    output_dir = tmp_path / "out"
    (output_dir / "control_lambda0").mkdir(parents=True)
    make_tiny_checkpoint(output_dir / "control_lambda0" / "best.pt")

    # Force every calibration to look badly mismatched.
    monkeypatch.setattr(mod, "CLIP_GUARD_PERCENT", -1.0)

    exit_code = mod.main([
        "--manifest", str(manifest), "--output-dir", str(output_dir),
        "--only", "M9-CTRL", "--batch-size", "2", "--calibration-batches", "2",
        "--device", "cpu",
    ])

    assert exit_code == 1
    assert "did not pass the clipping guard" in capsys.readouterr().err
    assert not (output_dir / "benchmark_aggregate.json").exists()
