"""Tests for FFmpeg discovery and invocation (Milestone 7).

Subprocess and PATH lookups are monkeypatched throughout, so the whole
module runs without FFmpeg installed. The few cases that genuinely need a
real FFmpeg are marked and skipped gracefully when it is unavailable.
"""

from __future__ import annotations

import subprocess

import pytest

from nvc.evaluation import ffmpeg as ffmpeg_module
from nvc.evaluation.ffmpeg import (
    EncoderNotAvailableError,
    FFmpegCommandError,
    FFmpegNotFoundError,
    available_encoders,
    ffmpeg_info,
    ffmpeg_version,
    find_ffmpeg,
    find_ffprobe,
    has_encoders,
    is_ffmpeg_available,
    parse_version_number,
    probe_frame_count,
    require_encoders,
    run_command,
)

_ENCODER_OUTPUT = """Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 D..... = Data
 T..... = Attachments
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC
 V....D libx265              libx265 H.265 / HEVC
 A....D aac                  AAC (Advanced Audio Coding)
"""

_VERSION_OUTPUT = (
    "ffmpeg version 9.0-full_build-www.gyan.dev Copyright (c) 2000-2026\n"
    "built with gcc 16.1.0\n"
)


def _fake_which(mapping):
    return lambda name: mapping.get(name)


def _fake_run(stdout="", stderr="", returncode=0):
    def runner(arguments, **kwargs):
        return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)
    return runner


# --- Executable discovery ---


def test_find_ffmpeg_returns_the_discovered_path(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )

    assert find_ffmpeg() == "/usr/bin/ffmpeg"


def test_find_ffprobe_returns_the_discovered_path(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffprobe": "/usr/bin/ffprobe"})
    )

    assert find_ffprobe() == "/usr/bin/ffprobe"


def test_missing_ffmpeg_raises_with_install_guidance(monkeypatch):
    monkeypatch.setattr(ffmpeg_module.shutil, "which", _fake_which({}))

    with pytest.raises(FFmpegNotFoundError) as info:
        find_ffmpeg()

    message = str(info.value)
    assert "not found on PATH" in message
    # The error has to tell the user how to fix it, not just that it failed.
    assert "winget install" in message or "brew install" in message


def test_missing_ffprobe_raises(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )

    with pytest.raises(FFmpegNotFoundError):
        find_ffprobe()


def test_is_ffmpeg_available_requires_both_executables(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which",
        _fake_which({"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}),
    )
    assert is_ffmpeg_available() is True

    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )
    assert is_ffmpeg_available() is False


# --- Command execution ---


def test_run_command_returns_completed_process_on_success(monkeypatch):
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_run(stdout="ok"))

    completed = run_command(["ffmpeg", "-version"])

    assert completed.returncode == 0
    assert completed.stdout == "ok"


def test_run_command_raises_with_stderr_on_failure(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run",
        _fake_run(stderr="Unknown encoder 'libx265'", returncode=1),
    )

    with pytest.raises(FFmpegCommandError) as info:
        run_command(["ffmpeg", "-c:v", "libx265"])

    # The actual FFmpeg diagnostic must survive, not be swallowed.
    assert "Unknown encoder" in str(info.value)
    assert "exit code 1" in str(info.value)


def test_run_command_maps_missing_executable_to_not_found(monkeypatch):
    def raiser(arguments, **kwargs):
        raise FileNotFoundError(arguments[0])

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", raiser)

    with pytest.raises(FFmpegNotFoundError):
        run_command(["ffmpeg", "-version"])


def test_run_command_maps_timeout_to_command_error(monkeypatch):
    def raiser(arguments, **kwargs):
        raise subprocess.TimeoutExpired(arguments, 5)

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", raiser)

    with pytest.raises(FFmpegCommandError, match="timed out"):
        run_command(["ffmpeg", "-version"], timeout=5)


def test_run_command_never_uses_a_shell(monkeypatch):
    # Paths on Windows routinely contain spaces; shell=True would require
    # quoting and would expose the command to metacharacter interpretation.
    captured = {}

    def runner(arguments, **kwargs):
        captured.update(kwargs)
        captured["arguments"] = arguments
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(ffmpeg_module.subprocess, "run", runner)
    run_command(["ffmpeg", "-i", "C:/a path/in.png"])

    assert captured.get("shell") is not True
    assert isinstance(captured["arguments"], list)


# --- Version reporting ---


def test_ffmpeg_version_returns_the_first_line(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", _fake_run(stdout=_VERSION_OUTPUT)
    )

    version = ffmpeg_version()

    assert version.startswith("ffmpeg version 9.0-full_build")
    assert "\n" not in version


@pytest.mark.parametrize(
    "line, expected",
    [
        ("ffmpeg version 9.0-full_build-www.gyan.dev Copyright", "9.0-full_build-www.gyan.dev"),
        ("ffmpeg version n7.1 Copyright (c) 2000-2024", "n7.1"),
        ("ffmpeg version 6.1.1-3ubuntu5 Copyright", "6.1.1-3ubuntu5"),
    ],
)
def test_parse_version_number_handles_varied_build_strings(line, expected):
    assert parse_version_number(line) == expected


def test_parse_version_number_returns_none_for_unrecognized_text():
    assert parse_version_number("not an ffmpeg banner") is None


def test_ffmpeg_info_collects_paths_and_version(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which",
        _fake_which({"ffmpeg": "/usr/bin/ffmpeg", "ffprobe": "/usr/bin/ffprobe"}),
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", _fake_run(stdout=_VERSION_OUTPUT)
    )

    info = ffmpeg_info()

    assert info.ffmpeg_path == "/usr/bin/ffmpeg"
    assert info.ffprobe_path == "/usr/bin/ffprobe"
    assert set(info.to_dict()) == {"ffmpeg_path", "ffprobe_path", "version"}


# --- Encoder availability ---


def test_available_encoders_parses_the_encoder_listing(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", _fake_run(stdout=_ENCODER_OUTPUT)
    )

    encoders = available_encoders()

    assert "libx264" in encoders
    assert "libx265" in encoders
    assert "aac" in encoders
    # The multi-line flag legend above the "------" separator must never be
    # mistaken for encoder rows (regression guard: each legend line's first
    # token is also 6 characters, the same shape as a real encoder row).
    assert "=" not in encoders
    assert "Video" not in encoders
    assert len(encoders) == 3


def test_require_encoders_passes_when_all_present(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )
    monkeypatch.setattr(
        ffmpeg_module.subprocess, "run", _fake_run(stdout=_ENCODER_OUTPUT)
    )

    require_encoders(["libx264", "libx265"])  # must not raise


def test_require_encoders_raises_for_a_missing_encoder(monkeypatch):
    listing = _ENCODER_OUTPUT.replace(
        " V....D libx265              libx265 H.265 / HEVC\n", ""
    )
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffmpeg": "/usr/bin/ffmpeg"})
    )
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_run(stdout=listing))

    with pytest.raises(EncoderNotAvailableError, match="libx265"):
        require_encoders(["libx264", "libx265"])


def test_has_encoders_is_false_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(ffmpeg_module.shutil, "which", _fake_which({}))

    assert has_encoders(["libx264"]) is False


# --- ffprobe frame counting ---


def test_probe_frame_count_parses_the_count(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffprobe": "/usr/bin/ffprobe"})
    )
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_run(stdout="42\n"))

    assert probe_frame_count("clip.mp4") == 42


def test_probe_frame_count_raises_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(
        ffmpeg_module.shutil, "which", _fake_which({"ffprobe": "/usr/bin/ffprobe"})
    )
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", _fake_run(stdout="N/A\n"))

    with pytest.raises(FFmpegCommandError, match="frame count"):
        probe_frame_count("clip.mp4")


# --- Real FFmpeg, when it happens to be installed ---


@pytest.mark.skipif(not is_ffmpeg_available(), reason="FFmpeg is not installed")
def test_real_ffmpeg_reports_a_version():
    assert "ffmpeg" in ffmpeg_version().lower()


@pytest.mark.skipif(
    not has_encoders(["libx264", "libx265"]),
    reason="FFmpeg build lacks libx264/libx265",
)
def test_real_ffmpeg_provides_both_required_encoders():
    require_encoders(["libx264", "libx265"])  # must not raise
