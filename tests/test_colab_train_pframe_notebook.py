"""Static checks on `colab_train_pframe.ipynb`: the ways it breaks only show up on Colab
after a chunk download, so catch them here."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "colab_train_pframe.ipynb").read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in NOTEBOOK["cells"] if c["cell_type"] == "code"]


def _p1():
    spec = importlib.util.spec_from_file_location("p1_pframe", ROOT / "scripts" / "p1_pframe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_notebook_is_valid_and_every_cell_parses():
    assert NOTEBOOK["nbformat"] == 4
    assert NOTEBOOK["metadata"]["colab"]["name"] == "colab_train_pframe.ipynb"
    for index, source in enumerate(CODE):
        if source.lstrip().startswith("!"):
            continue
        ast.parse(source)


def test_every_flag_the_notebook_passes_exists_in_train_vimeo():
    import argparse
    p1 = _p1()
    seen = {}

    def capture(self, args=None, namespace=None):
        seen["parser"] = self
        raise SystemExit(0)

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        try:
            p1.main(["train-vimeo"])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = original
    sub = next(a for a in seen["parser"]._actions if isinstance(a, argparse._SubParsersAction))
    known = {flag for action in sub.choices["train-vimeo"]._actions for flag in action.option_strings}
    passed = set(re.findall(r"'(--[a-z-]+)'", "\n".join(CODE)))
    assert passed, "no flags found"
    assert passed <= known, f"unknown flags: {passed - known}"


def test_the_output_folder_is_its_own():
    config = CODE[0]
    assert "'pframe_gate2'" in config
    for taken in ("stage1_vimeo_64ch_qat'", "'stage1_vimeo'", "'stage1_vimeo_64ch'"):
        assert f"OUTPUT_DIR = DRIVE_PROJECT_DIR / {taken}" not in config
