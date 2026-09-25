"""Tests for `scripts/train_vimeo_stage1.py` - the Stage 1 Vimeo-90K trainer.

The Kaggle download and the ten 6-10GB chunks are obviously out of scope here.
What is tested is everything that runs between them: model and optimizer
construction, both objectives (distortion-only Phase A and D+lambda*R Phase B),
checkpointing, early stopping, and that the checkpoint a long Colab run produces
can actually be loaded back as the right architecture.

A 32x32 tiny manifest stands in for Vimeo, and the model is built at toy widths,
so the whole file runs in seconds on CPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from nvc.data.loaders import create_train_loader, create_val_loader
from nvc.models import ResidualGDNAutoencoder
from nvc.training.checkpoint import load_model_from_checkpoint

from helpers import make_tiny_manifest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stage1():
    spec = importlib.util.spec_from_file_location(
        "train_vimeo_stage1", ROOT / "scripts" / "train_vimeo_stage1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(stage1, **overrides):
    from nvc.utils.config import load_default_config
    args = stage1.build_arg_parser(load_default_config()).parse_args([])
    # Toy widths so a test builds in milliseconds instead of 8.4M parameters.
    args.latent_channels, args.base_channels, args.residual_blocks = 4, 8, 1
    args.batch_size, args.crop_size, args.num_workers = 2, 32, 0
    args.epochs_per_chunk_max, args.early_stop_patience = 2, 1
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _loaders(tmp_path, args):
    manifest = make_tiny_manifest(tmp_path, width=64, height=64)
    train = create_train_loader(manifest, batch_size=args.batch_size,
                                num_workers=0, crop_size=args.crop_size)
    val = create_val_loader(manifest, batch_size=args.batch_size,
                            num_workers=0, crop_size=args.crop_size)
    return train, val


def _run_chunk(stage1, args, tmp_path, rate_estimator=None, **overrides):
    train, val = _loaders(tmp_path, args)
    model = stage1.build_model(args)
    optimizer = stage1.build_optimizer(model, rate_estimator, args)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    history: list[dict] = []
    kwargs = dict(
        model=model, optimizer=optimizer, rate_estimator=rate_estimator,
        train_loader=train, val_loader=val, device=torch.device("cpu"), args=args,
        start_epoch=1, chunk_number=3, history=history, checkpoint_dir=checkpoint_dir,
        model_config=model.config_dict(), best_val_loss=float("inf"),
        checkpoint_extra=lambda: None,
    )
    kwargs.update(overrides)
    next_epoch, best = stage1.train_one_chunk(**kwargs)
    return {"next_epoch": next_epoch, "best": best, "history": history,
            "dir": checkpoint_dir, "model": model}


# --- construction -----------------------------------------------------------------


def test_it_builds_the_stage_1_transform_not_the_baseline(stage1):
    """The whole point of the script. A silent fall back to BaselineAutoencoder
    would train the wrong model for hours and look fine doing it."""
    model = stage1.build_model(_args(stage1))

    assert isinstance(model, ResidualGDNAutoencoder)


def test_the_defaults_are_the_stage_1_architecture(stage1):
    from nvc.utils.config import load_default_config
    args = stage1.build_arg_parser(load_default_config()).parse_args([])

    assert (args.latent_channels, args.base_channels, args.residual_blocks) == (192, 192, 1)
    assert stage1.build_model(args).num_parameters() == 8_437_827


def test_phase_a_has_no_rate_estimator(stage1):
    assert stage1.build_rate_estimator(_args(stage1, rate_enabled=False)) is None


def test_phase_b_refuses_to_start_without_a_calibration(stage1):
    """Silently training Phase B with no bin width would make the rate term
    meaningless, and the run would still produce plausible-looking losses."""
    with pytest.raises(SystemExit, match="rate-calibration"):
        stage1.build_rate_estimator(_args(stage1, rate_enabled=True, rate_calibration=None))

    with pytest.raises(SystemExit, match="not found"):
        stage1.build_rate_estimator(_args(stage1, rate_enabled=True,
                                          rate_calibration=Path("no/such/file.json")))


def test_the_rate_estimator_gets_its_own_optimizer_group(stage1):
    """M9C.1: the estimator's 2*C scalars need far larger steps than the model's
    weights. One shared group would leave them effectively frozen."""
    args = _args(stage1, rate_lr=0.5, learning_rate=1e-4)
    model = stage1.build_model(args)
    estimator = torch.nn.Linear(2, 2)  # stands in for RateEstimator's parameters

    optimizer = stage1.build_optimizer(model, estimator, args)

    assert [group["lr"] for group in optimizer.param_groups] == [1e-4, 0.5]


def test_without_a_rate_estimator_the_optimizer_is_a_single_group(stage1):
    args = _args(stage1)

    optimizer = stage1.build_optimizer(stage1.build_model(args), None, args)

    assert len(optimizer.param_groups) == 1


# --- the training loop ------------------------------------------------------------


def test_phase_a_trains_and_writes_resumable_checkpoints(stage1, tmp_path):
    result = _run_chunk(stage1, _args(stage1), tmp_path)

    assert (result["dir"] / "latest.pt").is_file()
    assert (result["dir"] / "best.pt").is_file()
    assert result["next_epoch"] > 1
    assert result["best"] < float("inf")
    history = json.loads((result["dir"] / "history.json").read_text(encoding="utf-8"))
    assert history and history[0]["chunk"] == 3
    assert history[0]["rate_enabled"] is False


def test_the_checkpoint_reloads_as_the_stage_1_architecture(stage1, tmp_path):
    """A Colab run's only durable output is this file. If it cannot be rebuilt
    as the right class, the training time is simply lost."""
    result = _run_chunk(stage1, _args(stage1), tmp_path)

    reloaded, checkpoint = load_model_from_checkpoint(result["dir"] / "best.pt")

    assert checkpoint["architecture"] == "ResidualGDNAutoencoder"
    assert isinstance(reloaded, ResidualGDNAutoencoder)
    original = result["model"].eval()
    frame = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(reloaded(frame), original(frame))


def test_phase_b_trains_under_the_rate_objective(stage1, tmp_path):
    from nvc.training import RateEstimator

    args = _args(stage1, rate_enabled=True, rate_lambda=0.01)
    estimator = RateEstimator(torch.full((1, 4, 1, 1), 0.1), bits=4, mode="per_channel")

    result = _run_chunk(stage1, args, tmp_path, rate_estimator=estimator)

    record = json.loads((result["dir"] / "history.json").read_text(encoding="utf-8"))[0]
    assert record["rate_enabled"] is True and record["rate_lambda"] == 0.01
    # the rate loops report these two; the distortion-only ones do not, and
    # reading the wrong key name would silently drop them from the history
    assert "val_distortion" in record and "val_rate" in record


def test_early_stopping_cuts_a_chunk_short_when_validation_stalls(stage1, tmp_path):
    """Ten chunks times a full epoch ceiling is the difference between a run
    that finishes overnight and one that does not."""
    args = _args(stage1, epochs_per_chunk_max=5, early_stop_patience=1,
                 early_stop_min_delta=1e9)  # nothing can count as an improvement

    result = _run_chunk(stage1, args, tmp_path)

    assert len(result["history"]) < 5


def test_a_crop_size_off_the_stride_is_refused(stage1):
    with pytest.raises(SystemExit, match="divisible by 16"):
        stage1.main(["--crop-size", "40"])


# --- repository conventions -------------------------------------------------------


def test_it_reuses_the_m8b_chunk_machinery_instead_of_copying_it(stage1):
    """Kaggle download, collision-reconciling extraction, symlinking and the
    per-chunk split lists are fiddly and already tested. A second copy would
    drift from the first."""
    source = (ROOT / "scripts" / "train_vimeo_stage1.py").read_text(encoding="utf-8")

    assert '_load_script("train_vimeo_qat_combined")' in source
    for helper in ("download_and_extract_chunk", "relink_sequences_to_chunk",
                   "write_chunk_split_lists", "build_chunk_manifests"):
        assert f"chunks.{helper}" in source
        assert f"def {helper}" not in source


def test_it_does_not_modify_the_milestone_8b_script(stage1):
    """Prior milestone scripts are the record of how their result was produced
    and are imported read-only, never edited by later work."""
    import subprocess

    changed = subprocess.run(
        ["git", "status", "--porcelain", "scripts/train_vimeo_qat_combined.py"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout

    assert changed.strip() == "", f"M8B script modified: {changed}"


# --- the manifest -> loader path ---------------------------------------------------


def test_it_builds_loaders_from_a_real_vimeo_sequence_manifest(stage1, tmp_path):
    """The bug this exists for: Vimeo chunks produce a SEQUENCE manifest
    (`sequence_id` + `frame_filenames`), and the frame loaders want a
    `frame_directory` per item, so `create_train_loader` died with
    `KeyError: 'frame_directory'` - but only after a 9GB download, an
    extraction and a symlink pass, none of which the other tests reach.

    This builds a miniature Vimeo tree, runs it through the same
    `build_chunk_manifests` the script uses, and constructs the loaders from
    the result. It would have caught that in seconds.
    """
    from helpers import make_vimeo_dataset

    chunks = stage1._load_script("train_vimeo_qat_combined")
    root = make_vimeo_dataset(
        tmp_path / "vimeo",
        train_sequence_ids=["00001/0001", "00001/0002", "00002/0001"],
        test_sequence_ids=["00003/0001"],
        width=64, height=64,
    )
    train_manifest, val_manifest = chunks.build_chunk_manifests(
        root, tmp_path / "vimeo_manifest.json", 42)

    args = _args(stage1, crop_size=32)
    train_loader, val_loader = stage1.build_loaders(train_manifest, val_manifest, args)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert train_batch.shape[1:] == (3, 32, 32)
    assert val_batch.shape[1:] == (3, 32, 32)


def test_the_loaders_are_the_sequence_ones_not_the_frame_ones(stage1):
    """A static backstop for the same bug: the frame loaders cannot read a
    Vimeo sequence manifest at all."""
    source = (ROOT / "scripts" / "train_vimeo_stage1.py").read_text(encoding="utf-8")

    assert "create_sequence_train_loader(" in source
    assert "create_sequence_test_loader(" in source
    assert "create_train_loader(" not in source
    assert "create_val_loader(" not in source


# --- chunk reuse ------------------------------------------------------------------


def _fake_chunks(stage1, tmp_path, monkeypatch):
    """The real helper module, with the 9GB download stubbed out."""
    chunks = stage1._load_script("train_vimeo_qat_combined")
    calls = []
    monkeypatch.setattr(chunks, "download_and_extract_chunk",
                        lambda *a, **k: calls.append(a) or (tmp_path / "downloaded"))
    return chunks, calls


def test_reuse_chunk_skips_the_download_when_frames_are_already_extracted(
        stage1, tmp_path, monkeypatch):
    """download_and_extract_chunk wipes its scratch dir first, so without this a
    crash mid-chunk re-pays the whole 6-10GB download."""
    from helpers import make_vimeo_dataset

    chunks, calls = _fake_chunks(stage1, tmp_path, monkeypatch)
    chunk_dir = tmp_path / "chunk_1"
    make_vimeo_dataset(chunk_dir / "vimeo_settuplet_1",
                       train_sequence_ids=["00001/0001"], test_sequence_ids=["00003/0001"],
                       width=64, height=64)

    root = stage1.prepare_chunk(chunks, 1, chunk_dir, _args(stage1, reuse_chunk=True))

    assert calls == [], "downloaded despite the frames already being extracted"
    assert (root / "00001" / "0001" / "im1.png").is_file()


def test_reuse_chunk_is_off_by_default(stage1, tmp_path, monkeypatch):
    """On Colab the VM is wiped between sessions, so reuse would be a lie."""
    from helpers import make_vimeo_dataset

    chunks, calls = _fake_chunks(stage1, tmp_path, monkeypatch)
    chunk_dir = tmp_path / "chunk_1"
    make_vimeo_dataset(chunk_dir / "vimeo_settuplet_1",
                       train_sequence_ids=["00001/0001"], test_sequence_ids=["00003/0001"],
                       width=64, height=64)

    stage1.prepare_chunk(chunks, 1, chunk_dir, _args(stage1))

    assert len(calls) == 1


def test_reuse_chunk_falls_back_to_downloading_a_half_extracted_chunk(
        stage1, tmp_path, monkeypatch):
    """An interrupted extraction leaves a directory with no im1.png in it.
    Training on that would silently use a fraction of the chunk."""
    chunks, calls = _fake_chunks(stage1, tmp_path, monkeypatch)
    chunk_dir = tmp_path / "chunk_1"
    (chunk_dir / "partial").mkdir(parents=True)

    stage1.prepare_chunk(chunks, 1, chunk_dir, _args(stage1, reuse_chunk=True))

    assert len(calls) == 1
