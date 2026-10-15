"""Tests for `colab_train_stage1.ipynb`.

A notebook cannot be run in CI, but the ways this one breaks are all static:
a flag renamed in `scripts/train_vimeo_stage1.py` and not here, a typo in a
cell, or the output directory colliding with the baseline runs'. Each of those
only shows up in Colab after a 6-10GB chunk has already been downloaded, so
they are worth catching here instead.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab_train_stage1.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells(notebook, kind):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == kind]


def test_the_notebook_is_valid(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "no cells"
    assert notebook["metadata"]["colab"]["name"] == "colab_train_stage1.ipynb"


def test_every_code_cell_parses(notebook):
    """A SyntaxError in a cell near the end would surface hours into a run."""
    for index, source in enumerate(_cells(notebook, "code")):
        if source.lstrip().startswith("!"):
            continue  # IPython shell magic, not Python
        try:
            ast.parse(source)
        except SyntaxError as error:
            raise AssertionError(f"code cell {index} does not parse: {error}") from error


def _script_flags(*names: str) -> set[str]:
    import importlib.util

    from nvc.utils.config import load_default_config

    defaults = load_default_config()
    flags: set[str] = set()
    for name in names:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        flags.update(option
                     for action in module.build_arg_parser(defaults)._actions
                     for option in action.option_strings)
    return flags


def test_every_flag_the_notebook_passes_exists_in_a_script_it_calls():
    """The notebook drives both scripts by flag name. A rename on one side and
    not the other fails in Colab only after the first chunk has downloaded.

    Checked against the union of the two parsers rather than per call site: it
    still catches a rename in either script, which is the failure worth
    guarding, without the test having to parse the cells' argument lists.
    """
    known = _script_flags("train_vimeo_stage1", "calibrate_quantizer",
                          "benchmark_intra_gate")

    used = set(re.findall(r"'(--[a-z0-9-]+)'", NOTEBOOK.read_text(encoding="utf-8")))

    assert used, "no flags found - has the notebook stopped driving the scripts?"
    assert used <= known, f"notebook passes flags no script defines: {sorted(used - known)}"


def test_it_trains_the_stage_1_architecture(notebook):
    text = "\n".join(_cells(notebook, "code"))

    assert "ResidualGDNAutoencoder" in text
    assert "train_vimeo_stage1.py" in text
    assert "BaselineAutoencoder" not in text


def test_the_output_directory_cannot_collide_with_an_earlier_run(notebook):
    """Two collisions to avoid, both of which silently train nothing.

    The baseline notebook keeps its progress.json in the Drive project root, so
    sharing that would skip chunks and overwrite the baseline checkpoints.

    And since c38ecc3 there is a second one: the 192-channel Stage 1 run owns
    `stage1_vimeo`, and its progress.json lists all ten chunks as completed even
    though nine of them produced only NaN. Pointing the 64-channel run at that
    directory would skip every chunk. The directory must therefore be under the
    project dir, and must not be the bare `stage1_vimeo`.
    """
    config = _cells(notebook, "code")[0]

    # Every finished run keeps its directory, and every one of them holds a
    # progress.json listing its chunks as completed. Reusing any of these skips
    # every chunk and trains nothing, so each gets retired by name as it lands.
    retired = {
        "stage1_vimeo": "the diverged 192-channel run (NaN latest.pt, 18 NaN epochs)",
        "stage1_vimeo_64ch": "the 64ch/1-block run that gated at +87.9%",
    }

    match = re.search(r"OUTPUT_DIR\s*=\s*DRIVE_PROJECT_DIR\s*/\s*'([^']+)'", config)
    assert match, "OUTPUT_DIR is no longer a name under DRIVE_PROJECT_DIR"
    name = match.group(1)
    assert name not in retired, (
        f"OUTPUT_DIR is {name!r}, which belongs to {retired[name]}; its "
        "progress.json already lists its chunks as completed and they would all "
        "be skipped")
    assert name.startswith("stage1_vimeo"), (
        f"{name!r} does not identify itself as a Stage 1 run")


def test_the_output_directory_names_the_architecture_it_holds(notebook):
    """Checkpoints from different latent widths are mutually unloadable, so the
    directory name is the only thing standing between a future run and a
    confusing shape mismatch. It should say which architecture lives there."""
    config = _cells(notebook, "code")[0]

    output = re.search(r"OUTPUT_DIR\s*=\s*DRIVE_PROJECT_DIR\s*/\s*'([^']+)'", config).group(1)
    channels = re.search(r"^LATENT_CHANNELS\s*=\s*(\d+)", config, re.MULTILINE).group(1)

    assert channels in output, (
        f"OUTPUT_DIR {output!r} does not mention LATENT_CHANNELS={channels}")


def test_the_architecture_settings_are_the_stage_1_ones(notebook):
    """64 latent channels, not 192 - and the width is the only thing that changed.

    `outputs/stage1_latent_shape/README.md` measured the 192-channel latent at
    3.16-3.26x the baseline's coded bytes for +0.43 dB at the float ceiling, with
    the cost decoupled from channel content so that rate training cannot recover
    it. 64 channels restores exact symbol parity with the baseline while keeping
    7,208,899 parameters, because the capacity is in the body - which is why
    BASE_CHANNELS stays at 192.
    """
    config = _cells(notebook, "code")[0]

    # Rate-critical, so pinned hard. These two set the symbol count, and the
    # 192-channel run's +560.6% is what happens when the first one drifts.
    assert re.search(r"^LATENT_CHANNELS\s*=\s*64", config, re.MULTILINE)
    assert re.search(r"^BASE_CHANNELS\s*=\s*192", config, re.MULTILINE)
    # Free in rate, so deliberately a tunable rather than a fixed value: blocks
    # change the parameter count at an identical latent shape, so they cannot
    # affect the symbol count. 1 gated at +87.9%; 2 is the next increment.
    # test_the_latent_width_restores_symbol_parity_with_the_baseline proves the
    # parity claim for whichever value is set here.
    blocks = re.search(r"^RESIDUAL_BLOCKS\s*=\s*(\d+)", config, re.MULTILINE)
    assert blocks, "RESIDUAL_BLOCKS not found"
    assert int(blocks.group(1)) in {1, 2}, (
        f"RESIDUAL_BLOCKS={blocks.group(1)} is outside the range this project has "
        "measured; add the measurement before widening it")


def test_the_latent_width_restores_symbol_parity_with_the_baseline():
    """The reason for the number above, checked against the models rather than
    taken on trust from a comment: at this width the Stage 1 latent carries
    exactly as many symbols as the model it has to beat, so the gate measures the
    transform and not the latent's size."""
    import json as _json

    import torch

    from nvc.models import BaselineAutoencoder, ResidualGDNAutoencoder

    notebook = _json.loads((ROOT / "colab_train_stage1.ipynb").read_text(encoding="utf-8"))
    config = "".join(notebook["cells"][2]["source"])
    settings = {
        key: int(re.search(rf"^{key}\s*=\s*(\d+)", config, re.MULTILINE).group(1))
        for key in ("LATENT_CHANNELS", "BASE_CHANNELS", "RESIDUAL_BLOCKS")
    }

    image = torch.zeros(1, 3, 128, 128)
    stage1 = ResidualGDNAutoencoder(
        latent_channels=settings["LATENT_CHANNELS"],
        base_channels=settings["BASE_CHANNELS"],
        residual_blocks=settings["RESIDUAL_BLOCKS"]).eval()
    baseline = BaselineAutoencoder(latent_channels=64).eval()
    with torch.no_grad():
        assert stage1.encode(image).numel() == baseline.encode(image).numel()
    # And the capacity is still there to make the comparison worth running.
    assert stage1.num_parameters() > 10 * baseline.num_parameters()


def test_phase_b_is_documented_as_needing_a_calibration_first(notebook):
    """Running Phase B without a calibration is the one ordering mistake the
    notebook cannot catch for the user - the script exits, but only after the
    install and mount cells have run."""
    text = "\n".join(_cells(notebook, "markdown"))

    assert "--rate-calibration" in "\n".join(_cells(notebook, "code"))
    assert "calibrat" in text.lower()


def test_it_points_at_the_real_gate_and_its_denominator(notebook):
    """A checkpoint is an input to the Stage 1 gate, not a result. The notebook
    has to say what to do next or the number never gets measured."""
    text = "\n".join(_cells(notebook, "markdown"))

    assert "benchmark_intra_gate.py" in text
    assert "176.2" in text


def test_it_warns_against_the_parity_harness_for_a_stage_1_checkpoint(notebook):
    """`benchmark_parity.py --gop 1` produced the denominator, but it rebuilds the
    deployed stack through prepare_rate_point, and every piece of that is tied to
    the transform it was fitted on. Someone reaching section 7 after a 13-hour run
    should not have to rediscover that.

    The warning must give the reason that still holds - the calibration signature
    over the residual grid, checked by `check_provenance` - and NOT the old
    channel-count argument. Stage 1 ran at 192 channels until 2026-10-14, where the
    harness also died on a shape mismatch in `ChannelContextEntropyModel`'s
    `nn.Embedding(latent_channels, hidden)`. At 64 channels that shape now matches,
    so the loud failure is gone and only provenance stands between the user and a
    plausible-looking number computed with the wrong statistics. A warning still
    citing the shape would read as obsolete and get ignored.
    """
    text = "\n".join(_cells(notebook, "markdown"))

    assert "Do not use `benchmark_parity.py --gop 1`" in text
    assert "calibration signature" in text.lower(), (
        "the warning does not give the reason that currently applies")
    assert "provenance" in text.lower()


def test_it_does_not_claim_the_gate_needs_the_entropy_stack_refitted(notebook):
    """An earlier draft said the intra grids, G16 context model and codebooks all
    had to be recalibrated 'before the number means anything'. That is true of a
    full-video number and false of this gate: the intra path needs none of it."""
    text = "\n".join(_cells(notebook, "markdown"))

    assert "before the number means anything" not in text
    assert "not needed for this gate" in text


# --- the audit fixes: things that would each cost a session or a checkpoint -------


def test_phase_b_is_off_by_default(notebook):
    """"Reopen and Run All" is the documented way to resume after a disconnect. If
    the Phase B cells were unguarded, finishing Phase A during a Run All would
    immediately start a 9GB calibration download and a second full training run
    that nobody asked for."""
    config = _cells(notebook, "code")[0]

    assert re.search(r"^RUN_PHASE_B\s*=\s*False", config, re.MULTILINE)


def test_the_calibration_and_phase_b_cells_are_guarded(notebook):
    code = _cells(notebook, "code")
    calibration = next(c for c in code if "calibrate_quantizer.py" in c)
    phase_b = next(c for c in code if "--rate-enabled" in c)

    for cell in (calibration, phase_b):
        assert "if not RUN_PHASE_B:" in cell
        assert "else:" in cell


def test_phase_b_snapshots_phase_a_before_overwriting_best(notebook):
    """Phase B resets the best-loss tracker - its objective is D + lambda*R, which
    is not comparable to Phase A's plain MSE - so its first epoch overwrites
    best.pt. Confirmed at train_vimeo_stage1.py's `not args.reset_progress` guard.
    Without a snapshot, a badly chosen lambda destroys ~20 hours of Phase A work."""
    phase_b = next(c for c in _cells(notebook, "code") if "--rate-enabled" in c)

    assert "best_phase_a.pt" in phase_b
    assert "shutil.copy" in phase_b
    # idempotent, or a Phase B resume would overwrite the snapshot with Phase B's
    # own weights and lose the thing it exists to protect
    assert "if not phase_a_best.is_file():" in phase_b


def test_the_sanity_cell_raises_on_a_cpu_runtime(notebook):
    """It used to print a warning. Phase A on CPU is ~100x slower, and a printed
    warning scrolls away - a whole session would be wasted before anyone noticed."""
    sanity = next(c for c in _cells(notebook, "code") if "ms/step" in c)

    assert "raise RuntimeError(" in sanity
    assert "if not torch.cuda.is_available():" in sanity


def test_the_progress_cell_survives_a_first_run(notebook):
    """It reads progress.json, which does not exist until Phase A has written to
    Drive. Crashing there would look like a training failure."""
    progress = next(c for c in _cells(notebook, "code") if "Completed chunks" in c)

    assert "is_file()" in progress


def test_the_notebook_sets_gradient_clipping(notebook):
    """The first full run had none and lost the transform at epoch 3 of chunk 1,
    then trained NaN for eighteen epochs across the nine remaining chunks. The
    flag existing in the script is not enough - the notebook has to pass it."""
    import re

    config = _cells(notebook, "code")[0]
    launch = next(c for c in _cells(notebook, "code") if "train_vimeo_stage1.py" in c)

    match = re.search(r"^GRAD_CLIP_NORM\s*=\s*([0-9.e-]+)", config, re.MULTILINE)
    assert match, "GRAD_CLIP_NORM not found in the config cell"
    assert float(match.group(1)) > 0, "clipping disabled - this is what diverged"
    assert "'--grad-clip-norm'" in launch, "configured but never passed to the script"
