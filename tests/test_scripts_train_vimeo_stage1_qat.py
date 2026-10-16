"""QAT and warm-start wiring in `scripts/train_vimeo_stage1.py`.

Split from `test_scripts_train_vimeo_stage1.py` because this is a distinct
concern: not "does the trainer train" but "is the quantization noise reaching
the forward pass, and is a fine-tune prevented from being mistaken for a
resume".

Why it exists at all, from `outputs/stage1_latent_shape/README_64ch_trained.md`:
the 64-channel transform leads the deployed baseline by a flat +2.450 dB at its
float ceiling, but gives up +0.179/+0.627/+1.710 dB MORE than the baseline to
5/4/3-bit quantization. Subtracting the two reproduces the measured gate
advantage of +2.37/+1.88/+0.79 dB almost exactly, so by 3-bit quantization
fragility takes 70% of what the transform earned. Stage 1 had never seen a
quantized latent in training; the deployed baseline trained with `--qat-enabled`.

The `stage1` fixture and `_args` helper are imported rather than duplicated, so
the toy widths stay defined in one place.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from test_scripts_train_vimeo_stage1 import _args, stage1  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parents[1]


def _calibration(tmp_path, stage1, args, *, bits=4, mode="per_channel"):
    """A structurally real calibration file matching the toy model's width.

    `make_tiny_calibration` writes the exact document shape
    `scripts/calibrate_quantizer.py` produces, so `from_calibration` accepts it
    unchanged - no encoder pass over real data needed.
    """
    from helpers import make_tiny_calibration
    from nvc.training.checkpoint import save_checkpoint

    model = stage1.build_model(args)
    checkpoint = tmp_path / "source.pt"
    save_checkpoint(checkpoint, model=model,
                    optimizer=stage1.build_optimizer(model, None, args), epoch=1,
                    history=[], model_config=model.config_dict(), extra=None)
    path = make_tiny_calibration(
        tmp_path / "calib.json", checkpoint_path=checkpoint, bits=bits, mode=mode,
        num_channels=args.latent_channels)
    return path, checkpoint


# --- the default must not move ---------------------------------------------------


def test_phase_a_attaches_no_quantization_noise_by_default(stage1):
    """Every number measured so far, the +87.9% gate included, came from a run
    without it. Turning it on by default would silently invalidate comparisons."""
    assert stage1.build_quantization_noise(_args(stage1)) is None
    assert stage1.build_model(_args(stage1)).quantization_noise is None


# --- the noise has to reach the forward pass -------------------------------------


def test_qat_enabled_without_a_calibration_says_what_to_do(stage1):
    """The noise scale is fitted to a trained model's own latents, so there is no
    sensible default, and the error has to explain the fine-tune ordering rather
    than just reporting a missing flag."""
    with pytest.raises(SystemExit) as error:
        stage1.build_quantization_noise(
            _args(stage1, qat_enabled=True, qat_calibration=None))

    message = str(error.value)
    assert "--qat-calibration" in message
    assert "calibrate_quantizer.py" in message


def test_qat_attaches_noise_the_model_actually_applies(stage1, tmp_path):
    """Constructing the noise is not the same as applying it.

    `ResidualGDNAutoencoder.forward` applies it itself in training mode only, so
    this checks both halves of that contract: two training forwards differ, and
    two eval forwards do not.
    """
    args = _args(stage1)
    path, _ = _calibration(tmp_path, stage1, args)
    args = _args(stage1, qat_enabled=True, qat_calibration=path)

    noise = stage1.build_quantization_noise(args)
    assert noise is not None
    model = stage1.build_model(args, quantization_noise=noise)
    assert model.quantization_noise is noise

    batch = torch.rand(2, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    model.train()
    torch.manual_seed(1)
    first = model(batch)
    torch.manual_seed(2)
    second = model(batch)
    assert not torch.allclose(first, second), "noise is attached but never applied"

    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch), model(batch)), "noise leaked into eval"


def test_qat_bits_are_asserted_against_the_calibration_file(stage1, tmp_path):
    """The calibration's scale IS the bit depth, so a run aimed at one depth must
    not silently train at another."""
    args = _args(stage1)
    path, _ = _calibration(tmp_path, stage1, args, bits=5)

    with pytest.raises(ValueError, match="5-bit"):
        stage1.build_quantization_noise(
            _args(stage1, qat_enabled=True, qat_calibration=path, qat_bits=4))


def test_qat_mode_is_asserted_against_the_calibration_file(stage1, tmp_path):
    """Training with global noise while the codec quantizes per-channel is the
    mismatch this pair of flags exists to prevent."""
    args = _args(stage1)
    path, _ = _calibration(tmp_path, stage1, args, mode="global")

    with pytest.raises(ValueError, match="mode"):
        stage1.build_quantization_noise(
            _args(stage1, qat_enabled=True, qat_calibration=path,
                  qat_mode="per_channel"))


# --- a QAT checkpoint must be identifiable --------------------------------------


def test_a_qat_checkpoint_records_that_it_is_one(stage1, tmp_path):
    """`quantization_noise` is deliberately absent from `config_dict()`, so a QAT
    checkpoint and a full-precision one of the same width are identical on disk
    without this fingerprint - and the resume guard below has nothing to compare."""
    args = _args(stage1)
    path, _ = _calibration(tmp_path, stage1, args)
    args = _args(stage1, qat_enabled=True, qat_calibration=path)

    noise = stage1.build_quantization_noise(args)
    fingerprint = stage1.qat_fingerprint(args, noise)

    assert fingerprint["qat_bits"] == 4
    assert fingerprint["qat_mode"] == "per_channel"
    assert str(path) in fingerprint["qat_calibration"]
    # Full precision records nothing, so the absence is itself meaningful.
    assert stage1.qat_fingerprint(_args(stage1), None) is None


def test_resuming_across_a_qat_change_is_refused(stage1, tmp_path, monkeypatch):
    """The architecture guard cannot catch this one: a QAT model and a
    full-precision model have identical `model_config` and identical tensor
    shapes. Resuming across the change would continue under a different objective
    while the global best-loss tracker compared the two."""
    def _must_not_download(*_args, **_kwargs):
        raise AssertionError("main() reached the chunk loop - the QAT guard did not fire")

    monkeypatch.setattr(stage1, "prepare_chunk", _must_not_download)

    args = _args(stage1)
    path, _ = _calibration(tmp_path, stage1, args)

    # A full-precision latest.pt, exactly as the +87.9% run produced.
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = stage1.build_model(args)
    from nvc.training.checkpoint import save_checkpoint
    save_checkpoint(checkpoint_dir / "latest.pt", model=model,
                    optimizer=stage1.build_optimizer(model, None, args), epoch=5,
                    history=[], model_config=model.config_dict(), extra=None)

    with pytest.raises(SystemExit) as error:
        stage1.main([
            "--output-dir", str(tmp_path), "--scratch-dir", str(tmp_path / "s"),
            "--chunks", "1", "--qat-enabled", "--qat-calibration", str(path),
            "--latent-channels", str(args.latent_channels),
            "--base-channels", str(args.base_channels), "--residual-blocks", "1",
            "--device", "cpu",
        ])

    message = str(error.value)
    assert "quantization-aware" in message
    assert "--init-from" in message, "the message does not say how to proceed"


# --- a warm start is not a resume ------------------------------------------------


def _stub_chunk_pipeline(stage1, monkeypatch, tmp_path):
    """Everything between "start a chunk" and "train on it", stubbed out.

    These tests are about what main() does to the model and the counters before
    any data is touched, and the real path downloads 9GB from Kaggle.
    """
    monkeypatch.setattr(stage1, "prepare_chunk", lambda *a, **k: tmp_path / "group")
    monkeypatch.setattr(stage1, "build_loaders", lambda *a, **k: (None, None))
    chunks = stage1._load_script("train_vimeo_qat_combined")
    monkeypatch.setattr(chunks, "relink_sequences_to_chunk", lambda *a, **k: None)
    monkeypatch.setattr(chunks, "discover_complete_sequence_ids", lambda *a, **k: ["x"])
    monkeypatch.setattr(chunks, "write_chunk_split_lists", lambda *a, **k: None)
    monkeypatch.setattr(chunks, "build_chunk_manifests", lambda *a, **k: (None, None))
    monkeypatch.setattr(stage1, "_load_script", lambda name: chunks)
    return chunks


def test_init_from_copies_weights_and_nothing_else(stage1, tmp_path, monkeypatch):
    """QAT changes the objective, so carrying the source run's epoch counter,
    optimizer state or best-loss tracker would compare two different objectives.
    A warm start has to copy weights and reset the rest."""
    args = _args(stage1)
    source_model = stage1.build_model(args)
    with torch.no_grad():  # make the weights distinctive
        for parameter in source_model.parameters():
            parameter.add_(0.25)
    source = tmp_path / "warm.pt"
    from nvc.training.checkpoint import save_checkpoint
    save_checkpoint(source, model=source_model,
                    optimizer=stage1.build_optimizer(source_model, None, args),
                    epoch=28, history=[{"epoch": 28}],
                    model_config=source_model.config_dict(), extra=None)

    seen = {}

    def _capture(**kwargs):
        seen["start_epoch"] = kwargs.get("start_epoch")
        seen["best_val_loss"] = kwargs.get("best_val_loss")
        seen["weights"] = {name: parameter.detach().clone()
                           for name, parameter in kwargs["model"].named_parameters()}
        raise SystemExit("stop before training")

    _stub_chunk_pipeline(stage1, monkeypatch, tmp_path)
    monkeypatch.setattr(stage1, "train_one_chunk", _capture)

    with pytest.raises(SystemExit):
        stage1.main([
            "--output-dir", str(tmp_path), "--scratch-dir", str(tmp_path / "s"),
            "--chunks", "1", "--init-from", str(source),
            "--latent-channels", str(args.latent_channels),
            "--base-channels", str(args.base_channels), "--residual-blocks", "1",
            "--device", "cpu",
        ])

    assert seen["start_epoch"] == 1, "a warm start carried the source's epoch counter"
    assert seen["best_val_loss"] == float("inf"), (
        "a warm start carried the source's best-loss tracker")
    source_weights = dict(source_model.named_parameters())
    for name, tensor in seen["weights"].items():
        assert torch.allclose(tensor, source_weights[name]), f"{name} was not warm-started"


def test_init_from_refuses_a_different_architecture(stage1, tmp_path):
    """A warm start copies weights, so the shapes have to agree - and the message
    should say that rather than surfacing a per-tensor shape wall."""
    wide = _args(stage1, latent_channels=8, base_channels=8)
    model = stage1.build_model(wide)
    source = tmp_path / "warm.pt"
    from nvc.training.checkpoint import save_checkpoint
    save_checkpoint(source, model=model,
                    optimizer=stage1.build_optimizer(model, None, wide), epoch=1,
                    history=[], model_config=model.config_dict(), extra=None)

    with pytest.raises(SystemExit, match="different architecture"):
        stage1.main([
            "--output-dir", str(tmp_path), "--scratch-dir", str(tmp_path / "s"),
            "--chunks", "1", "--init-from", str(source),
            "--latent-channels", "4", "--base-channels", "8", "--residual-blocks", "1",
            "--device", "cpu",
        ])


def test_init_from_is_ignored_once_the_run_has_its_own_checkpoint(
        stage1, tmp_path, monkeypatch, capsys):
    """Otherwise every reconnect would discard the run's own progress and restart
    from the warm start - and "Reopen and Run All" is the documented way to resume
    a Colab run after a disconnect."""
    args = _args(stage1)
    from nvc.training.checkpoint import save_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = stage1.build_model(args)
    save_checkpoint(checkpoint_dir / "latest.pt", model=model,
                    optimizer=stage1.build_optimizer(model, None, args), epoch=9,
                    history=[], model_config=model.config_dict(), extra=None)
    source = tmp_path / "warm.pt"
    save_checkpoint(source, model=model,
                    optimizer=stage1.build_optimizer(model, None, args), epoch=1,
                    history=[], model_config=model.config_dict(), extra=None)

    _stub_chunk_pipeline(stage1, monkeypatch, tmp_path)
    monkeypatch.setattr(stage1, "train_one_chunk",
                        lambda **kwargs: (_ for _ in ()).throw(SystemExit("stop")))

    with pytest.raises(SystemExit):
        stage1.main([
            "--output-dir", str(tmp_path), "--scratch-dir", str(tmp_path / "s"),
            "--chunks", "1", "--init-from", str(source),
            "--latent-channels", str(args.latent_channels),
            "--base-channels", str(args.base_channels), "--residual-blocks", "1",
            "--device", "cpu",
        ])

    output = capsys.readouterr().out
    assert "ignoring --init-from" in output
    assert "resume" in output
