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


def test_the_output_directory_cannot_collide_with_the_baseline_runs(notebook):
    """The baseline notebook keeps its progress.json in the Drive project root.
    Sharing it would make this run skip chunks it never trained on, and quietly
    overwrite the baseline checkpoints while it was at it."""
    config = _cells(notebook, "code")[0]

    assert re.search(r"OUTPUT_DIR\s*=\s*DRIVE_PROJECT_DIR\s*/\s*'stage1_vimeo'", config)


def test_the_architecture_settings_are_the_stage_1_ones(notebook):
    config = _cells(notebook, "code")[0]

    assert re.search(r"LATENT_CHANNELS\s*=\s*192", config)
    assert re.search(r"BASE_CHANNELS\s*=\s*192", config)
    assert re.search(r"RESIDUAL_BLOCKS\s*=\s*1", config)


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
    deployed stack through prepare_rate_point, all of which is fitted to a
    64-channel latent - against Stage 1's 192 it raises a provenance error instead
    of producing a number. Someone reaching section 7 after a 13-hour run should
    not have to rediscover that."""
    text = "\n".join(_cells(notebook, "markdown"))

    assert "Do not use `benchmark_parity.py --gop 1`" in text
    assert "64-channel" in text


def test_it_does_not_claim_the_gate_needs_the_entropy_stack_refitted(notebook):
    """An earlier draft said the intra grids, G16 context model and codebooks all
    had to be recalibrated 'before the number means anything'. That is true of a
    full-video number and false of this gate: the intra path needs none of it."""
    text = "\n".join(_cells(notebook, "markdown"))

    assert "before the number means anything" not in text
    assert "not needed for this gate" in text
