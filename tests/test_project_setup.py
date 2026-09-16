"""Sanity checks for the repository foundation.

These do not test codec behavior (see the per-module test files) - only that
the package structure, configuration, packaging metadata and CI are importable
and internally consistent.
"""

from pathlib import Path

from nvc.utils.config import Config, load_default_config


def test_subpackages_import() -> None:
    import nvc
    import nvc.compression
    import nvc.data
    import nvc.evaluation
    import nvc.models
    import nvc.utils
    import nvc.video

    assert nvc.__version__


def test_config_defaults_have_expected_types() -> None:
    config = Config()

    assert isinstance(config.frame_width, int)
    assert isinstance(config.frame_height, int)
    assert isinstance(config.latent_dim, int)
    assert isinstance(config.batch_size, int)
    assert isinstance(config.learning_rate, float)
    assert isinstance(config.epochs, int)
    assert isinstance(config.raw_data_dir, Path)
    assert isinstance(config.checkpoint_dir, Path)


def test_load_default_config_reads_json_overrides() -> None:
    config = load_default_config()

    # configs/default.json currently sets these two explicitly.
    assert config.frame_width == 256
    assert config.batch_size == 8


def test_config_from_json_rejects_unknown_keys(tmp_path: Path) -> None:
    bad_config_path = tmp_path / "bad_config.json"
    bad_config_path.write_text('{"not_a_real_field": 1}', encoding="utf-8")

    try:
        Config.from_json(bad_config_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown config key")


# --- BUG-02 / BUG-04: the package must be installable and licensed ------------------------


ROOT = Path(__file__).resolve().parents[1]

# Distribution name for each third-party module src/nvc imports. Kept explicit
# because import name and PyPI name differ often enough to matter (cv2 ->
# opencv-python, pytorch_msssim -> pytorch-msssim).
_DISTRIBUTION_FOR = {
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "pytorch_msssim": "pytorch-msssim",
    "tqdm": "tqdm",
}


def _package_imports() -> set[str]:
    """Every third-party top-level module imported anywhere under src/nvc."""
    import ast
    import sys

    stdlib = set(sys.stdlib_module_names)
    found: set[str] = set()
    for path in (ROOT / "src" / "nvc").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module.split(".")[0]] if node.module and node.level == 0 else []
            else:
                continue
            found.update(n for n in names if n not in stdlib and n != "nvc")
    return found


def _declared_dependencies() -> list[str]:
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def test_every_module_the_package_imports_is_a_declared_dependency():
    """BUG-02: pyproject declared no dependencies at all, so `pip install nvc`
    produced a package that raised ImportError on first use. This asserts the
    declaration keeps matching what the code actually imports."""
    declared = " ".join(_declared_dependencies()).lower()
    missing = [
        name for name in sorted(_package_imports())
        if _DISTRIBUTION_FOR.get(name, name).lower() not in declared
    ]
    assert not missing, (
        f"src/nvc imports {missing} but pyproject.toml does not declare them; "
        "an install would fail at runtime"
    )


def test_declared_dependencies_are_actually_imported():
    """The other direction: a dependency nothing imports is dead weight on every
    install. Pillow was pinned for exactly this reason and used nowhere."""
    imported = {_DISTRIBUTION_FOR.get(n, n).lower() for n in _package_imports()}
    for requirement in _declared_dependencies():
        name = requirement.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        assert name in imported, (
            f"pyproject declares {name!r} but nothing under src/nvc imports it"
        )


def test_dependencies_are_version_bounded():
    """An unbounded pin means a future major release silently breaks installs."""
    for requirement in _declared_dependencies():
        assert any(op in requirement for op in (">=", "==", "~=")), requirement
        assert "<" in requirement, f"{requirement} has no upper bound"


def test_matplotlib_is_an_extra_not_a_runtime_dependency():
    """It is used only by scripts/, so it should not be forced on every install."""
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "matplotlib" not in " ".join(data["project"]["dependencies"])
    extras = data["project"]["optional-dependencies"]
    assert any("matplotlib" in r for r in extras.get("research", []))
    assert any("pytest" in r for r in extras.get("dev", []))


def test_the_native_c_source_ships_with_the_package():
    """The range coder compiles range_coder.c on demand and has no Python
    fallback, so a wheel without the .c file has no arithmetic coder at all."""
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    assert any(
        "*.c" in patterns
        for package, patterns in package_data.items()
        if "_native" in package
    ), "range_coder.c would be omitted from a built wheel"


def test_the_project_has_a_license():
    """BUG-04: without one the work is 'all rights reserved' by default, which
    blocks any public, academic or portfolio use."""
    import tomllib

    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c)" in text
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["license"] == {"file": "LICENSE"}
    assert any("License :: OSI Approved" in c for c in data["project"]["classifiers"])


def test_ci_runs_the_whole_test_suite_on_the_supported_python():
    """BUG-06: a workflow that silently tested a subset, or a Python the package
    does not support, would look green while guarding nothing.

    Read as text rather than parsed, so the check needs no YAML dependency.
    """
    import re
    import tomllib

    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert re.search(r"^on:\s*$", workflow, re.MULTILINE)
    assert re.search(r"^\s+push:\s*\n\s+branches: \[master\]", workflow, re.MULTILINE)
    assert re.search(r"^\s+pull_request:", workflow, re.MULTILINE)

    run_lines = re.findall(r"run: (.+)", workflow)
    pytest_runs = [line for line in run_lines if "pytest" in line]
    assert pytest_runs == ["python -m pytest tests/ -q -rfEs --durations=15"]
    assert any('pip install -e ".[dev,research]"' in line for line in run_lines)

    python = re.search(r'python-version: "([\d.]+)"', workflow).group(1)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    minimum = project["requires-python"].removeprefix(">=")
    assert tuple(map(int, python.split("."))) >= tuple(map(int, minimum.split(".")))
