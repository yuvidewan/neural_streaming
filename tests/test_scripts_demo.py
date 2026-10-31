"""`scripts/build_demo.py` and `scripts/run_demo.py`.

Both run against a miniature repository in `tmp_path`: synthetic frames named the
way the real ones are (`<sequence>_<index>.png`) and a tiny random-weight
checkpoint. Nothing here needs a dataset, a trained model or a GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from helpers import make_tiny_checkpoint

from nvc.demo import bundles as demo_bundles
from nvc.video import CodecBundle

ROOT = Path(__file__).resolve().parents[1]
SIZE = 64


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sequence(directory: Path, name: str, count: int, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    base = torch.nn.functional.interpolate(torch.rand(1, 3, 8, 8, generator=generator),
                                           size=(SIZE * 2, SIZE * 2), mode="bilinear",
                                           align_corners=False)[0]
    for t in range(count):
        frame = (base[:, t:t + SIZE, 2 * t:2 * t + SIZE].clamp(0, 1) * 255).round().byte()
        cv2.imwrite(str(directory / f"{name}_{t + 1:06d}.png"),
                    np.ascontiguousarray(frame.permute(1, 2, 0).numpy()[:, :, ::-1]))


@pytest.fixture()
def repo(tmp_path):
    _write_sequence(tmp_path / "data" / "frames" / "test", "walk", 7, seed=1)
    _write_sequence(tmp_path / "data" / "frames" / "test", "Road_Trip", 5, seed=2)
    for index, name in enumerate(("a", "b", "c")):
        _write_sequence(tmp_path / "data" / "frames" / "train", name, 4, seed=10 + index)
    make_tiny_checkpoint(tmp_path / "ckpt.pt")
    return tmp_path


def _build_args(repo: Path, *extra: str) -> list[str]:
    return ["--repo-root", str(repo), "--checkpoint", str(repo / "ckpt.pt"), "--size", str(SIZE),
            "--bits", "4", "--no-h264", "--device", "cpu", *extra]


# --- build_demo.py -------------------------------------------------------------------------


def test_build_demo_parser_defaults():
    mod = _load_script("build_demo")
    args = mod.build_arg_parser().parse_args([])
    assert args.sequences == list(mod.DEFAULT_SEQUENCES)
    assert args.max_frames == 60 and args.size == 256 and args.bits == [3, 4, 5]
    assert args.checkpoint is None and args.frames_dir is None
    assert not args.force_standin and not args.all_bundles and not args.no_h264
    assert args.repo_root == ROOT


def test_build_demo_builds_a_standin_and_precomputes_clips(repo, capsys):
    mod = _load_script("build_demo")
    code = mod.main(_build_args(repo, "--sequences", "walk", "Road_Trip", "--max-frames", "6"))
    assert code == 0, capsys.readouterr().err

    bundle_path = repo / demo_bundles.STANDIN_DIR / "standin_4bit.pt"
    bundle = CodecBundle.load(bundle_path)                       # loads and verifies
    assert bundle.provenance["kind"] == "standin"
    assert bundle.provenance["calibration_sequences"] == ["a", "b", "c"]
    assert not (repo / demo_bundles.STANDIN_DIR / "standin_3bit.pt").exists()

    clips = repo / "outputs" / "demo" / "clips"
    walk = json.loads((clips / "walk" / "clip.json").read_text())
    assert walk["frames"] == 6 and walk["source"] == "precomputed" and walk["width"] == SIZE
    road = json.loads((clips / "road-trip" / "clip.json").read_text())
    assert road["name"] == "Road_Trip" and road["frames"] == 5        # only 5 frames exist
    for clip_id, count in (("walk", 6), ("road-trip", 5)):
        result = json.loads((clips / clip_id / "standin_4bit" / "result.json").read_text())
        assert result["nvc"]["decoder_exact"] is True and result["frames"] == count
        assert result["h264"]["available"] is False
        assert len(list((clips / clip_id / "standin_4bit" / "nvc").glob("*.png"))) == count
    out = capsys.readouterr().out
    assert "building stand-ins" in out and "run_demo.py" in out


def test_build_demo_uses_trained_bundles_when_present(repo, capsys):
    mod = _load_script("build_demo")
    assert mod.main(_build_args(repo, "--sequences", "walk", "--max-frames", "4")) == 0
    standin = repo / demo_bundles.STANDIN_DIR / "standin_4bit.pt"
    trained = repo / demo_bundles.TRAINED_DIR / "nvc_m22_4bit.pt"
    trained.parent.mkdir(parents=True)
    trained.write_bytes(standin.read_bytes())
    standin.unlink()
    capsys.readouterr()

    assert mod.main(_build_args(repo, "--sequences", "walk", "--max-frames", "4")) == 0
    assert "building stand-ins" not in capsys.readouterr().out
    assert not standin.exists()                                  # no stand-in was rebuilt
    assert (repo / "outputs/demo/clips/walk/nvc_m22_4bit/result.json").is_file()

    # --force-standin builds them anyway, but the trained family is still the one encoded.
    assert mod.main(_build_args(repo, "--sequences", "walk", "--max-frames", "4",
                                "--force-standin")) == 0
    assert standin.is_file()


@pytest.mark.parametrize("extra, message", [
    (["--frames-dir", "does/not/exist"], "frames folder not found"),
    (["--sequences", "nope"], "no frames for sequence(s) nope"),
    (["--size", "100"], "--size must be"),
    (["--max-frames", "1"], "--max-frames must be"),
    (["--checkpoint", "missing.pt"], "no autoencoder checkpoint found"),
    (["--calibration-dir", "does/not/exist"], "no calibration frames"),
])
def test_build_demo_reports_unusable_inputs(repo, capsys, extra, message):
    mod = _load_script("build_demo")
    arguments = _build_args(repo, "--sequences", "walk")
    code = mod.main(arguments + extra)
    assert code == 1
    assert message in capsys.readouterr().err
    assert not (repo / "outputs" / "demo" / "clips").exists()


def test_build_demo_groups_frames_by_sequence_and_index(tmp_path):
    mod = _load_script("build_demo")
    for name in ("b_000002.png", "b_000001.png", "a-x_000010.png", "notes.png", "c_final.png"):
        (tmp_path / name).write_bytes(b"")
    groups = mod.group_by_sequence(tmp_path)
    assert {k: [p.name for p in v] for k, v in groups.items()} == {
        "a-x": ["a-x_000010.png"], "b": ["b_000001.png", "b_000002.png"]}


# --- run_demo.py ---------------------------------------------------------------------------


def test_run_demo_parser_defaults():
    mod = _load_script("run_demo")
    args = mod.build_arg_parser().parse_args([])
    assert args.host == "127.0.0.1" and args.port == 8000 and args.max_frames == 60
    assert args.device == "auto" and not args.no_browser and not args.allow_remote


@pytest.mark.parametrize("host, expected", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("127.0.0.5", True),
    ("0.0.0.0", False), ("192.168.1.20", False), ("example.com", False)])
def test_run_demo_recognises_loopback_hosts(host, expected):
    assert _load_script("run_demo").is_loopback(host) is expected


def test_run_demo_serves_the_app_on_the_requested_address(tmp_path, capsys):
    pytest.importorskip("fastapi")
    mod = _load_script("run_demo")
    served = {}

    def fake_serve(app, host, port):
        served.update(app=app, host=host, port=port)

    code = mod.main(["--repo-root", str(tmp_path), "--port", "8123", "--no-browser",
                     "--max-frames", "12", "--device", "cpu"], serve=fake_serve)
    assert code == 0
    assert served["host"] == "127.0.0.1" and served["port"] == 8123
    state = served["app"].state.demo
    assert state.max_frames == 12 and state.device == "cpu" and state.repo_root == tmp_path.resolve()
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8123" in out and "no codec bundle found" in out


@pytest.mark.parametrize("extra, message", [
    (["--host", "0.0.0.0"], "--allow-remote"),
    (["--host", "192.168.1.20"], "--allow-remote"),
    (["--port", "0"], "--port must be"),
    (["--repo-root", "does/not/exist"], "repository folder not found"),
])
def test_run_demo_refuses_unsafe_or_unusable_settings(tmp_path, capsys, extra, message):
    mod = _load_script("run_demo")

    def never(*args, **kwargs):
        raise AssertionError("the server must not start")

    arguments = ["--repo-root", str(tmp_path), "--no-browser"] + extra
    assert mod.main(arguments, serve=never) == 1
    assert message in capsys.readouterr().err


def test_run_demo_allows_a_remote_host_only_when_asked(tmp_path):
    pytest.importorskip("fastapi")
    mod = _load_script("run_demo")
    served = {}
    code = mod.main(["--repo-root", str(tmp_path), "--no-browser", "--host", "0.0.0.0",
                     "--allow-remote"], serve=lambda app, host, port: served.update(host=host))
    assert code == 0 and served["host"] == "0.0.0.0"
