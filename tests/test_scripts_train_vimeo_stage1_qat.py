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


# ----------------------------------------------------------------------------------------
# M24 Phase B: the unit-step quantizer
# ----------------------------------------------------------------------------------------


class _UnitStepArgs:
    """The smallest argument set the two builders read."""

    quantizer = "unit_step"
    unit_step = 1.0
    qat_enabled = True
    qat_bits = 4
    qat_mode = "per_channel"
    qat_calibration = None
    rate_enabled = True
    rate_track_scale = False
    rate_scale_momentum = 0.9
    rate_calibration = None


def test_unit_step_needs_no_calibration_file(stage1):
    """The whole point of a declared step: nothing has to be measured first, so
    Phase B no longer depends on a Phase A calibration pass."""
    noise = stage1.build_quantization_noise(_UnitStepArgs())

    assert noise is not None
    assert noise.scale.shape == (1, 1, 1, 1)
    assert float(noise.scale) == 1.0
    assert noise.mode == "global"


def test_the_unit_step_rate_proxy_uses_that_same_step_as_its_bin_width(stage1):
    """The proxy's bin width and the quantizer's step have to be the same number,
    or the rate term is modelling a quantizer that is not the one in the loop."""
    estimator = stage1.build_rate_estimator(_UnitStepArgs())

    assert estimator is not None
    assert float(estimator.bin_width) == 1.0
    assert estimator.track_scale is False


def test_scale_tracking_is_refused_with_a_unit_step(stage1):
    """The inversion that matters. Scale tracking exists because the PERCENTILE
    grid recalibrates to each model's latent range (M9F.5), which makes a frozen
    bin width exploitable. A fixed step does not recalibrate, so the frozen bin
    width is the CORRECT model and tracking would put the invariance back - the
    very thing M24 found caps the codec."""
    args = _UnitStepArgs()
    args.rate_track_scale = True

    with pytest.raises(SystemExit) as error:
        stage1.build_rate_estimator(args)

    message = str(error.value)
    assert "refused with --quantizer unit_step" in message
    assert "M9F.5" in message
    assert "scale-sensitive" in message


@pytest.mark.parametrize("step", [0.0, -1.0, float("inf"), float("nan")])
def test_a_malformed_unit_step_is_refused_by_both_builders(stage1, step):
    args = _UnitStepArgs()
    args.unit_step = step

    with pytest.raises(SystemExit, match="--unit-step must be finite and > 0"):
        stage1.build_quantization_noise(args)
    with pytest.raises(SystemExit, match="--unit-step must be finite and > 0"):
        stage1.build_rate_estimator(args)


def test_the_checkpoint_records_the_quantization_RULE_not_just_its_depth(stage1):
    """A percentile latent and a unit-step latent are not the same object, so a
    resume across that change must be refused. `model_config` cannot tell, which
    is why the fingerprint carries it."""
    args = _UnitStepArgs()
    noise = stage1.build_quantization_noise(args)

    fingerprint = stage1.qat_fingerprint(args, noise)

    assert fingerprint["quantizer"] == "unit_step"
    assert fingerprint["unit_step"] == 1.0


def test_the_percentile_fingerprint_records_no_unit_step(stage1, tmp_path):
    real = _args(stage1, qat_enabled=True, qat_bits=4, qat_mode="per_channel")
    args = _UnitStepArgs()
    args.quantizer = "percentile"
    args.qat_calibration, _ = _calibration(tmp_path, stage1, real)

    noise = stage1.build_quantization_noise(args)
    fingerprint = stage1.qat_fingerprint(args, noise)

    assert fingerprint["quantizer"] == "percentile"
    assert fingerprint["unit_step"] is None


def test_percentile_remains_the_default_and_still_demands_its_calibration(stage1):
    """Every existing Stage 1 command must behave exactly as before."""
    defaults = _args(stage1)

    assert defaults.quantizer == "percentile"
    assert defaults.unit_step == 1.0

    args = _UnitStepArgs()
    args.quantizer = "percentile"
    args.qat_calibration = None
    with pytest.raises(SystemExit, match="--qat-enabled requires --qat-calibration"):
        stage1.build_quantization_noise(args)


def test_the_docstring_explains_why_the_track_scale_advice_inverts(stage1):
    """A reader following the old Phase B instructions would turn on exactly the
    flag that defeats the new quantizer, so the reason has to be written down."""
    doc = stage1.__doc__

    assert "THE UNIT-STEP QUANTIZER" in doc
    assert "must be OFF" in doc
    assert "9.7%" in doc, "the sweep that failed, and why this one should not"
    assert "escape route" in doc


# --- resuming a run under its own fingerprint ----------------------------------------


def _resume_until_training(stage1, tmp_path, monkeypatch, extra, flags):
    """Save a latest.pt with `extra`, then resume with `flags`; return what happened.

    "stop" from the stubbed train_one_chunk means the resume guard let the run
    through; any other SystemExit is the guard's refusal message.
    """
    args = _args(stage1)
    from nvc.training.checkpoint import save_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = stage1.build_model(args)
    save_checkpoint(checkpoint_dir / "latest.pt", model=model,
                    optimizer=stage1.build_optimizer(model, None, args), epoch=3,
                    history=[], model_config=model.config_dict(), extra=extra)
    _stub_chunk_pipeline(stage1, monkeypatch, tmp_path)
    monkeypatch.setattr(stage1, "train_one_chunk",
                        lambda **kwargs: (_ for _ in ()).throw(SystemExit("stop")))
    with pytest.raises(SystemExit) as outcome:
        stage1.main([
            "--output-dir", str(tmp_path), "--scratch-dir", str(tmp_path / "s"),
            "--chunks", "1",
            "--latent-channels", str(args.latent_channels),
            "--base-channels", str(args.base_channels), "--residual-blocks", "1",
            "--device", "cpu", *flags,
        ])
    return str(outcome.value)


UNIT_STEP_FLAGS = ["--quantizer", "unit_step", "--unit-step", "1.0", "--qat-enabled"]


def test_an_interrupted_unit_step_run_resumes(stage1, tmp_path, monkeypatch):
    """The Phase B sweep on JupyterLab died mid-run and could never resume: the guard
    read back only the `qat_*` keys, dropped the `quantizer`/`unit_step` the run had
    itself written, and refused the mismatch. The checkpoint here is saved with
    exactly the fingerprint the same flags produce."""
    noise = stage1.build_quantization_noise(_UnitStepArgs())
    fingerprint = stage1.qat_fingerprint(_UnitStepArgs(), noise)
    assert fingerprint["quantizer"] == "unit_step"

    outcome = _resume_until_training(stage1, tmp_path, monkeypatch, fingerprint,
                                     UNIT_STEP_FLAGS)
    assert outcome == "stop", f"the resume was refused:\n{outcome}"


def test_a_pre_rule_qat_checkpoint_resumes_under_the_percentile_rule(
        stage1, tmp_path, monkeypatch):
    """Checkpoints saved before the rule was recorded (the QAT fine-tune that gated at
    +46.5%) were all trained on the percentile grid, so they resume under it."""
    args = _args(stage1)
    path, _ = _calibration(tmp_path, stage1, args)
    old = {"qat_bits": 4, "qat_mode": "per_channel", "qat_calibration": str(path)}

    outcome = _resume_until_training(
        stage1, tmp_path, monkeypatch, old,
        ["--qat-enabled", "--qat-calibration", str(path)])
    assert outcome == "stop", f"the resume was refused:\n{outcome}"


def test_a_pre_rule_checkpoint_is_still_refused_under_the_unit_step(
        stage1, tmp_path, monkeypatch):
    """Reading an old fingerprint as `percentile` must not open the door to resuming
    it under a different rule - that is the change the guard exists to catch."""
    old = {"qat_bits": 4, "qat_mode": "global", "qat_calibration": "None"}

    outcome = _resume_until_training(stage1, tmp_path, monkeypatch, old, UNIT_STEP_FLAGS)
    assert "quantization-aware" in outcome
