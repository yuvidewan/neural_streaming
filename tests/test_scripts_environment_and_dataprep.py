"""Script-level tests: scripts/check_environment.py, prepare_dataset.py,
prepare_training_dataset.py, inspect_dataset.py.

These invoke each script's own main(argv) - not just the library functions
underneath - so a regression in argument parsing, file-existence checks, or
CLI wiring is caught here even if the underlying nvc.* functions are
individually correct and separately tested elsewhere.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import make_sequence  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- check_environment.py ---------------------------------------------


def test_check_environment_runs_and_returns_an_int():
    mod = _load_script("check_environment")
    # Real environment (this repo's own venv) - every check should pass in CI/dev.
    assert mod.main() == 0


def test_check_environment_individual_checks_are_booleans():
    mod = _load_script("check_environment")
    assert mod.check_python() is True
    assert isinstance(mod.check_torch(), bool)
    assert isinstance(mod.check_opencv(), bool)
    assert isinstance(mod.check_directories(), bool)


def test_check_environment_fails_when_a_required_directory_is_missing(monkeypatch):
    mod = _load_script("check_environment")
    monkeypatch.setattr(mod, "REQUIRED_DIRS", ["this_directory_does_not_exist_anywhere"])
    assert mod.check_directories() is False


# --- prepare_dataset.py -------------------------------------------------


def test_prepare_dataset_image_sequence_end_to_end(tmp_path):
    mod = _load_script("prepare_dataset")
    input_root = tmp_path / "raw_sequences"
    for i in range(3):
        make_sequence(input_root / f"seq{i}", num_frames=6, width=64, height=64)

    output_dir = tmp_path / "frames"
    manifest_path = tmp_path / "processed" / "manifest.json"

    exit_code = mod.main([
        "--source-type", "image-sequence",
        "--input", str(input_root),
        "--output", str(output_dir),
        "--manifest", str(manifest_path),
        "--width", "64", "--height", "64",
        "--seed", "1",
    ])

    assert exit_code == 0
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["total_items_processed"] == 3
    assert sum(manifest["summary"]["items_per_split"].values()) == 3
    assert output_dir.is_dir()


def test_prepare_dataset_missing_input_directory_fails_cleanly(tmp_path, capsys):
    mod = _load_script("prepare_dataset")
    exit_code = mod.main([
        "--source-type", "image-sequence",
        "--input", str(tmp_path / "does_not_exist"),
        "--output", str(tmp_path / "frames"),
        "--manifest", str(tmp_path / "manifest.json"),
    ])
    assert exit_code != 0


def test_prepare_dataset_build_arg_parser_defaults_are_sane():
    mod = _load_script("prepare_dataset")
    from nvc.utils.config import load_default_config

    parser = mod.build_arg_parser(load_default_config())
    args = parser.parse_args(["--input", "somewhere"])
    assert args.source_type is None  # auto-detect by default
    assert args.width > 0 and args.height > 0


# --- inspect_dataset.py --------------------------------------------------


def test_inspect_dataset_end_to_end_without_visualization(tmp_path):
    from helpers import make_tiny_manifest

    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("inspect_dataset")

    exit_code = mod.main(["--manifest", str(manifest), "--batch-size", "2"])
    assert exit_code == 0


def test_inspect_dataset_with_visualization_writes_a_file(tmp_path):
    from helpers import make_tiny_manifest

    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("inspect_dataset")
    output = tmp_path / "viz" / "grid.png"

    exit_code = mod.main([
        "--manifest", str(manifest), "--batch-size", "2",
        "--visualize", "--grid-size", "4",
    ])
    assert exit_code == 0


def test_inspect_dataset_missing_manifest_fails_cleanly(tmp_path):
    mod = _load_script("inspect_dataset")
    exit_code = mod.main(["--manifest", str(tmp_path / "nope.json")])
    assert exit_code != 0


# --- prepare_training_dataset.py -----------------------------------------


def test_prepare_training_dataset_validate_only(tmp_path):
    from helpers import make_vimeo_dataset

    root = make_vimeo_dataset(
        tmp_path / "vimeo",
        train_sequence_ids=["00001/0001", "00001/0002"],
        test_sequence_ids=["00002/0001"],
        width=64, height=64,
    )
    mod = _load_script("prepare_training_dataset")

    exit_code = mod.main([
        "--dataset", "vimeo90k",
        "--vimeo-root", str(root),
        "--validate-only",
    ])
    assert exit_code == 0


def test_prepare_training_dataset_builds_a_subset_manifest(tmp_path):
    from helpers import make_vimeo_dataset

    root = make_vimeo_dataset(
        tmp_path / "vimeo",
        train_sequence_ids=["00001/0001", "00001/0002", "00001/0003"],
        test_sequence_ids=["00002/0001"],
        width=64, height=64,
    )
    mod = _load_script("prepare_training_dataset")
    manifest_out = tmp_path / "vimeo_train_manifest.json"

    exit_code = mod.main([
        "--dataset", "vimeo90k",
        "--vimeo-root", str(root),
        "--split", "train",
        "--max-sequences", "2",
        "--seed", "42",
        "--output", str(manifest_out),
    ])
    assert exit_code == 0
    assert manifest_out.is_file()
    manifest = json.loads(manifest_out.read_text(encoding="utf-8"))
    assert len(manifest["items"]) == 2


def test_prepare_training_dataset_missing_vimeo_root_fails_cleanly(tmp_path):
    mod = _load_script("prepare_training_dataset")
    exit_code = mod.main([
        "--dataset", "vimeo90k",
        "--vimeo-root", str(tmp_path / "does_not_exist"),
        "--validate-only",
    ])
    assert exit_code != 0
