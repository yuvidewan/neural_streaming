"""Script-level tests: scripts/train_vimeo_qat_combined.py.

The full main() loop needs a real Kaggle download and is out of scope for
a unit test - what's covered here instead, at the same granularity a real
run actually exercises them:

  1. CLI-level: build_arg_parser defaults, and main()'s pre-flight checks
     that don't need network access (missing kaggle CLI, missing
     calibration file).
  2. The zip-extraction collision fix (_extract_reconciling_collisions,
     _ensure_dir) - reproduces the exact crash it fixes with synthetic
     zips, not just "doesn't raise".
  3. The cloud-sync-lock retry fix (_reset_dir_with_retry).
  4. The Defender-exclusion safeguard (_try_add_defender_exclusion) - every
     failure path, with subprocess/shutil.which mocked so this never
     touches real Windows Defender settings.
  5. The combined-training partial pipeline (_bootstrap_run,
     _train_one_chunk_with_early_stopping, _load_or_init_progress,
     _save_progress) end to end against tiny synthetic data - this is the
     actual training logic, and the one place a regression here would be
     expensive to catch by hand (it only shows up hours into a real chunk
     download).
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_calibration, make_tiny_checkpoint, make_tiny_manifest  # noqa: E402

from nvc.data.loaders import create_train_loader, create_val_loader  # noqa: E402
from nvc.models import BaselineAutoencoder  # noqa: E402
from nvc.training import QuantizationNoise  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "train_vimeo_qat_combined", Path("scripts/train_vimeo_qat_combined.py").resolve()
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


# --- CLI: argument defaults and pre-flight checks --------------------


def test_build_arg_parser_defaults(mod):
    from nvc.utils.config import load_default_config

    args = mod.build_arg_parser(load_default_config()).parse_args([])
    assert args.chunks == list(range(1, 11))
    assert args.epochs_per_chunk_max == 10
    assert args.early_stop_patience == 2
    assert args.qat_bits == 4
    assert args.qat_mode == "per_channel"
    assert args.add_defender_exclusion is False
    assert args.output_dir is None  # resolved from defaults inside main(), not here


def test_main_fails_cleanly_when_kaggle_cli_is_missing(mod, monkeypatch):
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    assert mod.main([]) == 1


def test_main_fails_cleanly_when_calibration_file_is_missing(mod, monkeypatch, tmp_path):
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/kaggle" if name == "kaggle" else None)
    exit_code = mod.main([
        "--output-dir", str(tmp_path / "out"),
        "--calibration-path", str(tmp_path / "does_not_exist.json"),
    ])
    assert exit_code == 1


# --- Zip-extraction collision fix ------------------------------------


def test_extract_reconciling_collisions_handles_dir_then_file_ordering(mod, tmp_path):
    """Directory entry listed before a file entry at the same path - plain
    zipfile.extractall() crashes with PermissionError on this ordering."""
    zip_path = tmp_path / "a.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a/b/", "")
        zf.writestr("a/b", b"file-data")
        zf.writestr("a/b/c.txt", b"nested-file-data")

    with pytest.raises((FileExistsError, PermissionError)):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path / "plain")

    dest = tmp_path / "fixed"
    mod._extract_reconciling_collisions(zip_path, dest)
    assert (dest / "a" / "b").is_dir()
    assert (dest / "a" / "b" / "c.txt").read_bytes() == b"nested-file-data"


def test_extract_reconciling_collisions_handles_file_then_dir_ordering(mod, tmp_path):
    """File entry listed before a directory entry at the same path - the
    exact WinError 183 FileExistsError originally reported."""
    zip_path = tmp_path / "b.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("x/y", b"file-data")
        zf.writestr("x/y/", "")
        zf.writestr("x/y/z.txt", b"nested")

    with pytest.raises((FileExistsError, PermissionError)):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path / "plain")

    dest = tmp_path / "fixed"
    mod._extract_reconciling_collisions(zip_path, dest)
    assert (dest / "x" / "y").is_dir()
    assert (dest / "x" / "y" / "z.txt").read_bytes() == b"nested"


def test_extract_reconciling_collisions_normal_zip_extracts_correctly(mod, tmp_path):
    """No collisions at all - the common case must still work exactly like
    extractall()."""
    zip_path = tmp_path / "normal.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("group1/clip1/im1.png", b"frame-bytes-1")
        zf.writestr("group1/clip1/im2.png", b"frame-bytes-2")

    dest = tmp_path / "out"
    mod._extract_reconciling_collisions(zip_path, dest)
    assert (dest / "group1" / "clip1" / "im1.png").read_bytes() == b"frame-bytes-1"
    assert (dest / "group1" / "clip1" / "im2.png").read_bytes() == b"frame-bytes-2"


def test_ensure_dir_replaces_a_file_blocking_the_path(mod, tmp_path):
    blocked = tmp_path / "should_be_a_dir"
    blocked.write_text("i am a file, not a directory")
    mod._ensure_dir(blocked)
    assert blocked.is_dir()


def test_ensure_dir_creates_missing_parents(mod, tmp_path):
    nested = tmp_path / "a" / "b" / "c"
    mod._ensure_dir(nested)
    assert nested.is_dir()


# --- Cloud-sync-lock retry fix -----------------------------------------


def test_reset_dir_with_retry_recreates_a_clean_empty_directory(mod, tmp_path):
    target = tmp_path / "sequences"
    target.mkdir()
    (target / "stale_file.txt").write_text("leftover from a previous chunk")

    mod._reset_dir_with_retry(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_reset_dir_with_retry_succeeds_when_mkdir_fails_once_then_recovers(mod, tmp_path, monkeypatch):
    target = tmp_path / "sequences"
    target.mkdir()
    calls = {"n": 0}
    real_mkdir = Path.mkdir

    def flaky_mkdir(self, *args, **kwargs):
        if self == target and calls["n"] == 0:
            calls["n"] += 1
            raise FileExistsError("simulated sustained lock on first attempt")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)  # don't actually wait in tests

    mod._reset_dir_with_retry(target, attempts=3)
    assert target.is_dir()
    assert calls["n"] == 1


def test_reset_dir_with_retry_raises_a_clear_error_after_exhausting_attempts(mod, tmp_path, monkeypatch):
    target = tmp_path / "sequences"

    def always_fails(self, *args, **kwargs):
        raise FileExistsError("simulated permanent lock")

    monkeypatch.setattr(Path, "mkdir", always_fails)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="synced folder"):
        mod._reset_dir_with_retry(target, attempts=2)


# --- Defender-exclusion safeguard (never touches real Defender) ----------


def test_try_add_defender_exclusion_simulated_success(mod, tmp_path, capsys):
    target = tmp_path / "data"
    with mock.patch.object(mod.shutil, "which", return_value=r"C:\Windows\System32\powershell.exe"), \
         mock.patch.object(mod.subprocess, "run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        mod._try_add_defender_exclusion(target)
    assert "Add-MpPreference" in run_mock.call_args[0][0][-1]
    assert str(target) in run_mock.call_args[0][0][-1]
    assert "Added Windows Defender exclusion" in capsys.readouterr().out


def test_try_add_defender_exclusion_non_elevated_terminal_does_not_raise(mod, tmp_path):
    with mock.patch.object(mod.shutil, "which", return_value="powershell"), \
         mock.patch.object(mod.subprocess, "run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="Access is denied."
        )
        mod._try_add_defender_exclusion(tmp_path)  # must not raise


def test_try_add_defender_exclusion_missing_powershell_does_not_call_subprocess(mod, tmp_path):
    with mock.patch.object(mod.shutil, "which", return_value=None), \
         mock.patch.object(mod.subprocess, "run") as run_mock:
        mod._try_add_defender_exclusion(tmp_path)
        run_mock.assert_not_called()


def test_try_add_defender_exclusion_non_windows_skips_cleanly(mod, tmp_path):
    with mock.patch.object(mod.sys, "platform", "linux"), \
         mock.patch.object(mod.subprocess, "run") as run_mock:
        mod._try_add_defender_exclusion(tmp_path)
        run_mock.assert_not_called()


def test_try_add_defender_exclusion_subprocess_timeout_does_not_raise(mod, tmp_path):
    with mock.patch.object(mod.shutil, "which", return_value="powershell"), \
         mock.patch.object(mod.subprocess, "run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=30)):
        mod._try_add_defender_exclusion(tmp_path)  # must not raise


# --- Progress bookkeeping -------------------------------------------------


def test_load_or_init_progress_returns_empty_state_when_no_file_exists(mod, tmp_path):
    progress = mod._load_or_init_progress(tmp_path / "progress.json")
    assert progress == {"completed_chunks": [], "best_val_loss": None}


def test_save_then_load_progress_round_trips(mod, tmp_path):
    path = tmp_path / "progress.json"
    mod._save_progress(path, {"completed_chunks": [1, 2, 3], "best_val_loss": 0.042})
    assert mod._load_or_init_progress(path) == {"completed_chunks": [1, 2, 3], "best_val_loss": 0.042}


# --- Combined QAT+control partial pipeline (the actual training logic) ----


def _make_source_checkpoint(tmp_path) -> Path:
    path = tmp_path / "source_best.pt"
    return make_tiny_checkpoint(path, epoch=17, history=[{"epoch": e} for e in range(1, 18)])


def test_bootstrap_run_starts_fresh_from_source_checkpoint_when_no_own_checkpoint_exists(mod, tmp_path):
    source = _make_source_checkpoint(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints_qat"
    checkpoint_dir.mkdir()

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch, history = mod._bootstrap_run("qat", model, optimizer, checkpoint_dir, source, torch.device("cpu"))

    assert epoch == 1
    assert history == []


def test_bootstrap_run_resumes_from_its_own_latest_checkpoint_when_present(mod, tmp_path):
    source = _make_source_checkpoint(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints_qat"
    checkpoint_dir.mkdir()
    # Simulate an interrupted-and-restarted run: its own latest.pt already exists.
    make_tiny_checkpoint(checkpoint_dir / "latest.pt", epoch=5, history=[{"epoch": e} for e in range(1, 6)])

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch, history = mod._bootstrap_run("qat", model, optimizer, checkpoint_dir, source, torch.device("cpu"))

    assert epoch == 6
    assert len(history) == 5


def test_bootstrap_run_random_init_when_neither_checkpoint_exists(mod, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints_qat"
    checkpoint_dir.mkdir()

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch, history = mod._bootstrap_run(
        "qat", model, optimizer, checkpoint_dir, tmp_path / "nonexistent.pt", torch.device("cpu"),
    )
    assert epoch == 1
    assert history == []


def test_train_one_chunk_with_early_stopping_stops_after_patience_non_improving_epochs(mod, tmp_path):
    """A model whose weights never move (lr=0) can never improve val_loss -
    it must stop after exactly patience+1 epochs: one to establish the
    baseline against +inf, then `patience` non-improving epochs."""
    manifest = make_tiny_manifest(tmp_path, width=32, height=32, num_sequences=4, frames_per_sequence=6)
    train_loader = create_train_loader(manifest, batch_size=2, seed=1)
    test_loader = create_val_loader(manifest, batch_size=2)

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    patience = 2
    history = []
    next_epoch, best_val_loss = mod._train_one_chunk_with_early_stopping(
        "control", model, optimizer, train_loader, test_loader, torch.device("cpu"),
        start_epoch=1, max_epochs=10, patience=patience, min_delta=1e-5,
        chunk_number=1, history=history, checkpoint_dir=checkpoint_dir,
        model_config=model.config_dict(), best_val_loss=float("inf"),
        run_type="control", qat_enabled=False, qat_bits=4, qat_mode="per_channel",
    )

    steps_used = next_epoch - 1
    assert steps_used == patience + 1
    assert len(history) == steps_used
    assert (checkpoint_dir / "latest.pt").is_file()
    assert (checkpoint_dir / "best.pt").is_file()
    assert (checkpoint_dir / "history.json").is_file()


def test_train_one_chunk_with_early_stopping_respects_max_epochs_ceiling(mod, tmp_path):
    """A very high patience means early stopping never triggers - the
    max_epochs ceiling must still bound the run."""
    manifest = make_tiny_manifest(tmp_path, width=32, height=32, num_sequences=4, frames_per_sequence=6)
    train_loader = create_train_loader(manifest, batch_size=2, seed=1)
    test_loader = create_val_loader(manifest, batch_size=2)

    model = BaselineAutoencoder(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    next_epoch, _ = mod._train_one_chunk_with_early_stopping(
        "qat", model, optimizer, train_loader, test_loader, torch.device("cpu"),
        start_epoch=1, max_epochs=3, patience=1000, min_delta=1e-5,
        chunk_number=1, history=[], checkpoint_dir=checkpoint_dir,
        model_config=model.config_dict(), best_val_loss=float("inf"),
        run_type="qat", qat_enabled=False, qat_bits=4, qat_mode="per_channel",
    )
    assert next_epoch - 1 == 3


def test_train_one_chunk_with_early_stopping_qat_history_records_bits_and_mode(mod, tmp_path):
    manifest = make_tiny_manifest(tmp_path, width=32, height=32, num_sequences=4, frames_per_sequence=6)
    train_loader = create_train_loader(manifest, batch_size=2, seed=1)
    test_loader = create_val_loader(manifest, batch_size=2)

    noise = QuantizationNoise(torch.full((1, TINY_MODEL_KWARGS["latent_channels"], 1, 1), 0.3), bits=4, mode="per_channel")
    model = BaselineAutoencoder(quantization_noise=noise, **TINY_MODEL_KWARGS)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    history = []

    mod._train_one_chunk_with_early_stopping(
        "qat", model, optimizer, train_loader, test_loader, torch.device("cpu"),
        start_epoch=1, max_epochs=1, patience=2, min_delta=1e-5,
        chunk_number=3, history=history, checkpoint_dir=checkpoint_dir,
        model_config=model.config_dict(), best_val_loss=float("inf"),
        run_type="qat", qat_enabled=True, qat_bits=4, qat_mode="per_channel",
    )

    assert history[0]["qat_enabled"] is True
    assert history[0]["qat_bits"] == 4
    assert history[0]["qat_mode"] == "per_channel"
    assert history[0]["chunk"] == 3

    # The saved best.pt must be a plain inference checkpoint - no
    # quantization_noise attached (it's a training-only mechanism).
    from nvc.training import load_model_from_checkpoint
    loaded_model, _ = load_model_from_checkpoint(checkpoint_dir / "best.pt")
    assert loaded_model.quantization_noise is None
    with torch.no_grad():
        out = loaded_model(torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 3, 32, 32)


def test_qat_and_control_runs_pending_logic_skips_completed_chunks(mod, tmp_path):
    """The main loop's own pattern: a chunk already in a run's
    completed_chunks must be excluded from that run's pending list."""
    progress_qat = {"completed_chunks": [1, 2], "best_val_loss": 0.1}
    progress_control = {"completed_chunks": [1], "best_val_loss": 0.2}
    runs = {"qat": {"progress": progress_qat}, "control": {"progress": progress_control}}

    pending_chunk_1 = [name for name, run in runs.items() if 1 not in run["progress"]["completed_chunks"]]
    pending_chunk_2 = [name for name, run in runs.items() if 2 not in run["progress"]["completed_chunks"]]

    assert pending_chunk_1 == []
    assert pending_chunk_2 == ["control"]
