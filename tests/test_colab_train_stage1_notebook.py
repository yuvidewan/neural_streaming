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
        "stage1_vimeo_64ch_2blk": "the 2-block run, rejected at +102.8%/+101.5%",
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


# --- JupyterLab / output-folder behaviour: executed, not just parsed ---------------------


def _run_cell(source: str, namespace: dict) -> dict:
    exec(compile(source, "<notebook cell>", "exec"), namespace)
    return namespace


def _code(notebook, marker: str) -> str:
    return next(c for c in _cells(notebook, "code") if marker in c)


def test_outside_colab_everything_is_created_next_to_the_notebook(notebook, tmp_path, monkeypatch):
    """The config cell run in a plain Python process (no google.colab): all paths hang off
    the folder the notebook was opened from, and the DATASET scratch is not inside the output
    folder - so zipping the output folder can never sweep up 10 GB of frames."""
    monkeypatch.chdir(tmp_path)
    ns = _run_cell(_cells(notebook, "code")[0], {})

    assert ns["IN_COLAB"] is False
    assert ns["WORK_DIR"] == tmp_path.resolve()
    assert ns["OUTPUT_DIR"] == tmp_path.resolve() / "outputs"
    assert ns["ZIP_PATH"] == tmp_path.resolve() / "outputs.zip"
    assert ns["SCRATCH_DIR"] == tmp_path.resolve() / "vimeo_scratch"
    assert ns["REPO_DIR"] == tmp_path.resolve() / "neural_streaming"
    assert ns["OUTPUT_DIR"] not in ns["SCRATCH_DIR"].parents
    assert ns["SCRATCH_DIR"] not in ns["OUTPUT_DIR"].parents


def test_rerunning_the_config_cell_after_the_repo_cd_does_not_move_the_output_folder(
        notebook, tmp_path, monkeypatch):
    """A later cell cd's into the cloned repo. If the config cell is run again in that
    kernel, WORK_DIR must stay put - otherwise outputs/ would be created inside the repo."""
    monkeypatch.chdir(tmp_path)
    config = _cells(notebook, "code")[0]
    ns = _run_cell(config, {})
    (tmp_path / "neural_streaming").mkdir()
    monkeypatch.chdir(tmp_path / "neural_streaming")
    _run_cell(config, ns)

    assert ns["OUTPUT_DIR"] == tmp_path.resolve() / "outputs"


def test_google_colab_and_content_paths_are_only_reachable_on_colab(notebook):
    """JupyterLab has no google.colab and no /content. Every import of it, and every
    hard-coded /content path, must sit behind an IN_COLAB check (or an ImportError guard)."""
    for index, source in enumerate(_cells(notebook, "code")):
        if source.lstrip().startswith("!"):
            continue
        tree = ast.parse(source)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

        def guarded(node) -> bool:
            while node in parents:
                node = parents[node]
                if isinstance(node, (ast.If, ast.IfExp)) and "IN_COLAB" in ast.unparse(node.test):
                    return True
                if isinstance(node, ast.Try) and any(
                        h.type is not None and "ImportError" in ast.unparse(h.type) for h in node.handlers):
                    return True
            return False

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("google.colab"):
                assert guarded(node), f"code cell {index}: unguarded google.colab import"
            if isinstance(node, ast.Import) and any(a.name.startswith("google.colab") for a in node.names):
                assert guarded(node), f"code cell {index}: unguarded google.colab import"
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "/content" in node.value:
                assert guarded(node), f"code cell {index}: unguarded /content path {node.value!r}"


def test_the_output_folder_is_created_first_and_holds_everything_generated(notebook):
    """Everything the run generates is passed an OUTPUT_DIR-rooted path - the trainer's
    --output-dir, the calibration file, and calibrate_quantizer's side files, which default to
    the repo's own outputs/ and would otherwise be left out of the zip."""
    code = _cells(notebook, "code")
    text = "\n".join(code)

    assert "'--output-dir', str(OUTPUT_DIR)" in text
    assert "'--metrics-dir', str(OUTPUT_DIR / 'metrics')" in text
    assert "'--visualizations-dir', str(OUTPUT_DIR / 'visualizations')" in text
    assert "CALIBRATION_PATH = OUTPUT_DIR / 'calibration'" in text
    # created before any download or training cell
    mkdir_index = next(i for i, c in enumerate(code) if "OUTPUT_DIR.mkdir(" in c)
    train_index = next(i for i, c in enumerate(code) if "train_vimeo_stage1.py" in c)
    download_index = next(i for i, c in enumerate(code) if "'clone'" in c)
    assert mkdir_index < download_index < train_index


def test_the_kernel_can_import_nvc_without_a_restart(notebook):
    """pip install -e only takes effect in processes started afterwards, and the sanity
    cell imports nvc inside the kernel."""
    clone = _code(notebook, "'clone'")

    assert "sys.path.insert(0, src_dir)" in clone


def _kaggle_namespace(work_dir: Path, in_colab: bool = False) -> dict:
    return {"WORK_DIR": work_dir, "IN_COLAB": in_colab, "DRIVE_PROJECT_DIR": None}


def _isolated_home(monkeypatch, home: Path) -> None:
    for variable in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(variable, str(home))
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)


def test_kaggle_json_next_to_the_notebook_is_picked_up(notebook, tmp_path, monkeypatch):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    _isolated_home(monkeypatch, home)
    (work / "kaggle.json").write_text('{"username": "u", "key": "k"}', encoding="utf-8")

    _run_cell(_code(notebook, "kaggle_json_target"), _kaggle_namespace(work))

    copied = home / ".kaggle" / "kaggle.json"
    assert copied.read_text(encoding="utf-8") == '{"username": "u", "key": "k"}'
    assert (work / "kaggle.json").is_file()          # the user's own copy is left alone


def test_kaggle_credentials_in_the_environment_need_no_file(notebook, tmp_path, monkeypatch):
    _isolated_home(monkeypatch, tmp_path / "home")
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")

    _run_cell(_code(notebook, "kaggle_json_target"), _kaggle_namespace(tmp_path))


def test_missing_kaggle_credentials_stop_with_instructions_instead_of_hanging(notebook, tmp_path, monkeypatch):
    _isolated_home(monkeypatch, tmp_path / "home")

    with pytest.raises(RuntimeError, match="same folder as this notebook"):
        _run_cell(_code(notebook, "kaggle_json_target"), _kaggle_namespace(tmp_path))


def _fake_run(tmp_path: Path) -> dict:
    outputs = tmp_path / "outputs"
    (outputs / "checkpoints").mkdir(parents=True)
    (outputs / "calibration").mkdir()
    (outputs / "checkpoints" / "best.pt").write_bytes(b"best" * 1000)
    (outputs / "checkpoints" / "latest.pt").write_bytes(b"latest" * 1000)
    (outputs / "checkpoints" / "history.json").write_text("[]", encoding="utf-8")
    (outputs / "progress.json").write_text('{"completed_chunks": [1]}', encoding="utf-8")
    (outputs / "calibration" / "stage1_4bit_train.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vimeo_scratch").mkdir()
    (tmp_path / "vimeo_scratch" / "frame.png").write_bytes(b"dataset")       # must NOT be zipped
    (tmp_path / "kaggle.json").write_text("secret", encoding="utf-8")          # must NOT be zipped
    return {"OUTPUT_DIR": outputs, "ZIP_PATH": tmp_path / "outputs.zip", "IN_COLAB": False}


def test_the_last_cell_zips_the_output_folder_and_nothing_else(notebook, tmp_path, capsys):
    import zipfile

    final = _cells(notebook, "code")[-1]
    assert "make_archive" in final

    _run_cell(final, _fake_run(tmp_path))

    archive = tmp_path / "outputs.zip"
    assert archive.is_file()
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(n for n in bundle.namelist() if not n.endswith("/"))
    assert names == ["outputs/calibration/stage1_4bit_train.json", "outputs/checkpoints/best.pt",
                     "outputs/checkpoints/history.json", "outputs/checkpoints/latest.pt",
                     "outputs/progress.json"]
    assert "5 files" in capsys.readouterr().out


def test_running_the_zip_cell_twice_overwrites_instead_of_nesting(notebook, tmp_path):
    import zipfile

    final = _cells(notebook, "code")[-1]
    namespace = _fake_run(tmp_path)
    _run_cell(final, dict(namespace))
    (namespace["OUTPUT_DIR"] / "checkpoints" / "best.pt").write_bytes(b"better" * 1000)
    _run_cell(final, dict(namespace))

    with zipfile.ZipFile(tmp_path / "outputs.zip") as bundle:
        assert not any("outputs.zip" in n for n in bundle.namelist())
        assert bundle.read("outputs/checkpoints/best.pt") == b"better" * 1000


def test_the_zip_cell_works_on_its_own_after_a_kernel_restart(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_run(tmp_path)

    _run_cell(_cells(notebook, "code")[-1], {})      # no OUTPUT_DIR / ZIP_PATH defined

    assert (tmp_path / "outputs.zip").is_file()


def test_the_zip_cell_refuses_to_zip_a_folder_that_does_not_exist(notebook, tmp_path):
    with pytest.raises(RuntimeError, match="Nothing to zip"):
        _run_cell(_cells(notebook, "code")[-1],
                  {"OUTPUT_DIR": tmp_path / "missing", "ZIP_PATH": tmp_path / "missing.zip"})


# --- a headless GPU server has no libxcb: the desktop OpenCV build cannot be imported ---------


def test_a_headless_server_gets_a_working_opencv_before_anything_imports_it(notebook):
    """`pip install` brings in the desktop `opencv-python`, which needs libxcb/libGL. A minimal
    GPU server has neither, so the trainer died on its very first import (`import cv2` ->
    libxcb.so.1: cannot open shared object file) before downloading anything. The notebook
    swaps in the headless build when - and only when - cv2 does not import, and does so
    after the install cell and before any cell that imports nvc or runs the trainer."""
    code = _cells(notebook, "code")
    install = next(i for i, c in enumerate(code) if "pip install -q -r requirements.txt" in c)
    fix = next(i for i, c in enumerate(code) if "opencv-python-headless" in c)
    first_nvc_use = next(i for i, c in enumerate(code) if "import nvc" in c or "train_vimeo_stage1.py" in c)

    assert install < fix < first_nvc_use
    cell = code[fix]
    assert "'import cv2" in cell                                   # probes in a fresh process
    assert "raise RuntimeError(" in cell                            # fails loudly if it did not help
    assert "if ok:" in cell and "nothing to do" in cell             # a working install is left alone


# --- the shared cv2/ folder: why uninstalling ONE OpenCV build is not enough ---------------------


class _FakePip:
    """Just enough of pip and the filesystem to reproduce the failure seen on a real server.

    Every OpenCV distribution writes the SAME `cv2/` folder, but each keeps its own metadata. The
    image started with `opencv-python-headless` registered; `pip install -e .` then installed
    `opencv-python`, whose files overwrote the folder with the desktop build (needs libxcb).
    """

    def __init__(self, registered=("opencv-python", "opencv-python-headless")):
        self.registered = set(registered)
        self.cv2 = "gui"                       # None (folder gone) | "gui" | "headless"
        self.commands: list[list[str]] = []

    def run(self, command, **kwargs):
        import subprocess

        self.commands.append(list(command))
        if command[1] == "-c":                 # the cv2 import probe
            if self.cv2 == "headless":
                return subprocess.CompletedProcess(command, 0, stdout="4.10.0\n", stderr="")
            error = ("ImportError: libxcb.so.1: cannot open shared object file" if self.cv2 == "gui"
                     else "ModuleNotFoundError: No module named 'cv2'")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=f"Traceback...\n{error}\n")
        if command[1:4] == ["-m", "pip", "uninstall"]:
            removed = {name for name in command[5:] if name in self.registered}
            self.registered -= removed
            if removed:
                self.cv2 = None                 # the shared folder is deleted
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command[1:4] == ["-m", "pip", "install"]:
            name = command[-1].split("=")[0].split(">")[0].split("<")[0]
            if name not in self.registered:     # an installed distribution is "already satisfied"
                self.registered.add(name)
                self.cv2 = "headless" if name == "opencv-python-headless" else "gui"
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")


def _run_opencv_cell(notebook, monkeypatch, fake: _FakePip, versions=None):
    import importlib.metadata
    import subprocess

    versions = versions or {}

    def version(name):
        if name in versions:
            return versions[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(subprocess, "run", fake.run)
    monkeypatch.setattr(importlib.metadata, "version", version)
    _run_cell(_code(notebook, "opencv-python-headless"), {})


def test_the_old_uninstall_only_opencv_python_sequence_leaves_no_cv2_at_all():
    """The model reproduces the real failure: uninstalling just `opencv-python` deletes the
    shared folder, pip then calls the already-registered headless build 'satisfied', and cv2 is
    simply gone. This is the negative control for the test below."""
    fake = _FakePip()
    fake.run(["python", "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"])
    fake.run(["python", "-m", "pip", "install", "-q", "opencv-python-headless>=4.9,<5.0"])

    assert fake.cv2 is None
    assert "No module named 'cv2'" in fake.run(["python", "-c", "import cv2"]).stderr


def test_the_opencv_swap_recovers_when_headless_is_already_registered(notebook, monkeypatch, capsys):
    """The exact situation that broke the first version of this cell on a real server."""
    fake = _FakePip()

    _run_opencv_cell(notebook, monkeypatch, fake, versions={"opencv-python": "4.10.0.84"})

    assert fake.cv2 == "headless"
    out = capsys.readouterr().out
    assert "libxcb.so.1" in out and "imports fine now" in out        # says WHY, and that it worked


def test_the_opencv_swap_removes_every_build_before_installing(notebook, monkeypatch):
    fake = _FakePip()

    _run_opencv_cell(notebook, monkeypatch, fake, versions={"opencv-python": "4.10.0.84"})

    uninstall = next(c for c in fake.commands if c[1:4] == ["-m", "pip", "uninstall"])
    assert {"opencv-python", "opencv-python-headless", "opencv-contrib-python",
            "opencv-contrib-python-headless"} <= set(uninstall)
    assert fake.commands.index(uninstall) < fake.commands.index(
        next(c for c in fake.commands if c[1:4] == ["-m", "pip", "install"]))


def test_the_opencv_swap_pins_the_release_pip_already_chose(notebook, monkeypatch):
    """Same release as the desktop build, so the headless wheel has the same numpy requirement
    and installing it cannot upgrade numpy under torch."""
    fake = _FakePip()
    _run_opencv_cell(notebook, monkeypatch, fake, versions={"opencv-python": "4.10.0.84"})
    assert next(c for c in fake.commands if c[1:4] == ["-m", "pip", "install"])[-1] == \
        "opencv-python-headless==4.10.0.84"

    fake = _FakePip(registered=("opencv-python-headless",))        # only headless registered
    _run_opencv_cell(notebook, monkeypatch, fake, versions={"opencv-python-headless": "4.9.0.80"})
    assert next(c for c in fake.commands if c[1:4] == ["-m", "pip", "install"])[-1] == \
        "opencv-python-headless==4.9.0.80"

    fake = _FakePip(registered=())                                  # nothing registered: use the range
    _run_opencv_cell(notebook, monkeypatch, fake)
    assert next(c for c in fake.commands if c[1:4] == ["-m", "pip", "install"])[-1] == \
        "opencv-python-headless>=4.9,<5.0"


def test_the_opencv_swap_is_skipped_when_cv2_already_imports(notebook, monkeypatch, capsys):
    """A healthy environment (Colab, a laptop) must not be touched."""
    fake = _FakePip()
    fake.cv2 = "headless"

    _run_opencv_cell(notebook, monkeypatch, fake)

    assert len(fake.commands) == 1 and fake.commands[0][1] == "-c"
    assert "nothing to do" in capsys.readouterr().out


def test_the_opencv_swap_fails_loudly_if_it_still_cannot_import_cv2(notebook, monkeypatch):
    import subprocess

    fake = _FakePip()
    real_run = fake.run

    def never_fixed(command, **kwargs):
        if command[1] == "-c":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="ImportError: still broken\n")
        return real_run(command, **kwargs)

    fake.run = never_fixed
    with pytest.raises(RuntimeError, match="still does not import"):
        _run_opencv_cell(notebook, monkeypatch, fake)


def test_a_leftover_progress_file_is_called_out_before_anything_downloads(notebook, tmp_path, capsys):
    """A leftover progress.json makes the run skip the chunks it lists. Right after a disconnect
    that is the point; if the folder is from the diverged 192-channel run it would skip chunks that
    never trained. The folder cell says which situation this is, before a 9 GB download starts."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    cell = _code(notebook, "OUTPUT_DIR.mkdir(")
    namespace = {"IN_COLAB": False, "OUTPUT_DIR": outputs, "SCRATCH_DIR": tmp_path / "scratch"}

    _run_cell(cell, dict(namespace))
    assert "NOTE" not in capsys.readouterr().out

    (outputs / "progress.json").write_text("{}", encoding="utf-8")
    _run_cell(cell, dict(namespace))
    out = capsys.readouterr().out
    assert "already exists" in out and "RESUME" in out and "older run" in out


# --- QAT: the fine-tune the notebook is now set up for ---------------------------


def _qat_calibration_path(config):
    """The repo-relative path QAT_CALIBRATION points at.

    Written across two source lines in the notebook, so the literal is rejoined
    rather than matched in one piece.
    """
    match = re.search(r"QAT_CALIBRATION\s*=\s*Path\((.+?)\)", config, re.DOTALL)
    assert match, "QAT_CALIBRATION is not a Path literal any more"
    return "".join(re.findall(r"'([^']*)'", match.group(1)))


def test_qat_is_configured_and_its_flags_are_passed(notebook):
    """QAT_ENABLED in the config cell does nothing unless the launch cell adds the
    flags. Both halves have to be present, or the run silently trains in full
    precision again - which is exactly the asymmetry QAT exists to fix."""
    config = _cells(notebook, "code")[0]
    launch = next(c for c in _cells(notebook, "code") if "train_vimeo_stage1.py" in c)

    assert re.search(r"^QAT_ENABLED\s*=\s*True", config, re.MULTILINE)
    for flag in ("'--qat-enabled'", "'--qat-calibration'", "'--qat-bits'",
                 "'--qat-mode'", "'--init-from'"):
        assert flag in launch, f"{flag} is never passed to the trainer"


def test_qat_warm_starts_rather_than_training_from_scratch(notebook):
    """The noise scale is calibrated from a trained model's own latents, so a cold
    start would apply a scale that does not match the weights. INIT_FROM is not
    optional for a QAT run."""
    config = _cells(notebook, "code")[0]

    assert re.search(r"^\s*INIT_FROM\s*=", config, re.MULTILINE), "no INIT_FROM"
    assert "best.pt" in config, "INIT_FROM does not point at a checkpoint"


def test_the_qat_calibration_it_points_at_exists_in_the_repo(notebook):
    """It is committed precisely so the clone brings it along and the run needs no
    uploads. A path that does not resolve would fail only after the install and
    mount cells have already run."""
    relative = _qat_calibration_path(_cells(notebook, "code")[0])

    assert (ROOT / relative).is_file(), (
        f"QAT_CALIBRATION points at {relative!r}, which is not in the repo")


def test_the_qat_calibration_matches_the_configured_bits_and_mode(notebook):
    """The trainer asserts --qat-bits/--qat-mode against the file rather than
    reading them from it, so a mismatch here fails the run. Better to catch it
    than to discover it after a 9GB download."""
    config = _cells(notebook, "code")[0]
    document = json.loads(
        (ROOT / _qat_calibration_path(config)).read_text(encoding="utf-8"))

    bits = int(re.search(r"^QAT_BITS\s*=\s*(\d+)", config, re.MULTILINE).group(1))
    mode = re.search(r"^QAT_MODE\s*=\s*'([^']+)'", config, re.MULTILINE).group(1)

    assert document["quantization"]["bits"] == bits
    assert document["quantization"]["mode"] == mode


def test_residual_blocks_matches_the_checkpoint_being_warm_started_from(notebook):
    """--init-from copies weights, so the architectures have to agree. The warm
    start is the 1-block run that gated at +87.9%; 2 blocks was measured and
    rejected at +102.8%/+101.5%."""
    config = _cells(notebook, "code")[0]

    assert re.search(r"^RESIDUAL_BLOCKS\s*=\s*1", config, re.MULTILINE), (
        "INIT_FROM points at the 1-block checkpoint, so blocks must be 1")
    assert "stage1_vimeo_64ch" in config, "INIT_FROM is not the 1-block run"


def test_it_records_that_two_blocks_was_rejected(notebook):
    """The next person to reach for capacity should meet the measurement rather
    than repeat the run: 2 blocks was free in rate and still lost at every
    operating point, while training better on Vimeo."""
    config = _cells(notebook, "code")[0]

    assert "REJECTED" in config or "rejected" in config
    assert "102.8" in config
