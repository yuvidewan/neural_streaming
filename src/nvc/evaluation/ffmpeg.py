"""FFmpeg/ffprobe discovery and safe invocation.

Every classical-codec measurement in this project goes through FFmpeg, so
this module exists to make that dependency explicit and its failures
legible: FFmpeg is located on PATH (never a hardcoded install directory),
its presence and encoder support are verified up front, and every failure
raises a typed, actionable error rather than surfacing as a confusing
non-zero exit code deep inside a benchmark loop.

Commands are executed as argument lists, never as shell strings, so paths
containing spaces (very common on Windows) need no quoting and no shell
metacharacter can be interpreted.

Discovery is intentionally not cached: a benchmark run is long-lived, and
`shutil.which` is cheap next to spawning FFmpeg itself. Tests monkeypatch
`shutil.which` and `subprocess.run` directly.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG_EXECUTABLE = "ffmpeg"
FFPROBE_EXECUTABLE = "ffprobe"

_INSTALL_HINT = (
    "Install FFmpeg (with libx264 and libx265 support) and make sure it is on PATH.\n"
    "  Windows:  winget install Gyan.FFmpeg\n"
    "  macOS:    brew install ffmpeg\n"
    "  Linux:    sudo apt install ffmpeg\n"
    "Then open a NEW terminal (an already-running shell keeps its old PATH) and "
    "verify with:  ffmpeg -version"
)


class FFmpegError(RuntimeError):
    """Base class for every FFmpeg-related failure in this project."""


class FFmpegNotFoundError(FFmpegError):
    """ffmpeg or ffprobe could not be found on PATH."""


class FFmpegCommandError(FFmpegError):
    """An FFmpeg/ffprobe invocation exited non-zero."""


class EncoderNotAvailableError(FFmpegError):
    """The installed FFmpeg build lacks a required encoder (e.g. libx265)."""


@dataclass(frozen=True)
class FFmpegInfo:
    ffmpeg_path: str
    ffprobe_path: str
    version: str

    def to_dict(self) -> dict:
        return {
            "ffmpeg_path": self.ffmpeg_path,
            "ffprobe_path": self.ffprobe_path,
            "version": self.version,
        }


def find_ffmpeg() -> str:
    """Absolute path to ffmpeg, or raise FFmpegNotFoundError."""
    return _require_executable(FFMPEG_EXECUTABLE)


def find_ffprobe() -> str:
    """Absolute path to ffprobe, or raise FFmpegNotFoundError."""
    return _require_executable(FFPROBE_EXECUTABLE)


def _require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegNotFoundError(
            f"'{name}' was not found on PATH.\n{_INSTALL_HINT}"
        )
    return path


def is_ffmpeg_available() -> bool:
    """True when both ffmpeg and ffprobe are on PATH.

    Used by tests to skip FFmpeg integration cases gracefully instead of
    failing on machines without it.
    """
    return shutil.which(FFMPEG_EXECUTABLE) is not None and (
        shutil.which(FFPROBE_EXECUTABLE) is not None
    )


def run_command(
    arguments: list[str], *, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """Run a command (argument list, no shell) and return the completed process.

    Raises FFmpegCommandError with captured stderr on a non-zero exit, so
    the actual FFmpeg diagnostic reaches the caller rather than being lost.
    """
    try:
        completed = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(
            f"Could not execute '{arguments[0]}'.\n{_INSTALL_HINT}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegCommandError(
            f"Command timed out after {timeout}s: {' '.join(arguments)}"
        ) from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        # FFmpeg puts its build banner in stderr too; the tail holds the
        # actual error, so show that rather than a wall of configuration.
        tail = "\n".join(stderr.splitlines()[-12:]) if stderr else "(no stderr)"
        raise FFmpegCommandError(
            f"Command failed with exit code {completed.returncode}:\n"
            f"  {' '.join(arguments)}\n--- stderr (tail) ---\n{tail}"
        )
    return completed


def ffmpeg_version() -> str:
    """First line of `ffmpeg -version`, e.g. 'ffmpeg version 9.0-full_build ...'."""
    completed = run_command([find_ffmpeg(), "-hide_banner", "-version"], timeout=30)
    first_line = (completed.stdout or "").strip().splitlines()
    return first_line[0].strip() if first_line else "unknown"


def parse_version_number(version_line: str) -> str | None:
    """Extract just the version number from an `ffmpeg -version` first line.

    Returns None when the line does not look like FFmpeg's version output -
    builds vary wildly ('n7.1', '9.0-full_build', '6.1.1-3ubuntu5'), so the
    full line is what gets recorded in metadata; this is a convenience.
    """
    match = re.search(r"version\s+(\S+)", version_line)
    return match.group(1) if match else None


def ffmpeg_info() -> FFmpegInfo:
    """Locate both executables and capture the version string, or raise."""
    return FFmpegInfo(
        ffmpeg_path=find_ffmpeg(),
        ffprobe_path=find_ffprobe(),
        version=ffmpeg_version(),
    )


def available_encoders() -> set[str]:
    """Names of every encoder the installed FFmpeg build supports."""
    completed = run_command([find_ffmpeg(), "-hide_banner", "-encoders"], timeout=60)
    encoders: set[str] = set()
    # `ffmpeg -encoders` prints a multi-line flag legend first (one line per
    # media type, e.g. " V..... = Video", " A..... = Audio"), then a
    # "------" separator, then the actual encoder rows. Every legend line's
    # first token is also 6 characters, so it must be skipped by position
    # (before the separator), not just by shape - matching it as if it were
    # an encoder row previously added a spurious "=" entry to the result.
    past_separator = False
    for line in (completed.stdout or "").splitlines():
        if line.startswith(" ---"):
            past_separator = True
            continue
        if not past_separator:
            continue
        # Encoder rows look like: " V....D libx264   libx264 H.264 / AVC ..."
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            encoders.add(parts[1])
    return encoders


def require_encoders(names: list[str]) -> None:
    """Raise EncoderNotAvailableError unless every named encoder exists.

    Checked before a benchmark starts rather than discovered mid-run: a
    build without libx265 is a setup problem, and finding that out after
    encoding half a dataset wastes real time.
    """
    available = available_encoders()
    missing = [name for name in names if name not in available]
    if missing:
        raise EncoderNotAvailableError(
            f"The installed FFmpeg build does not provide: {', '.join(missing)}.\n"
            f"A build with GPL codecs enabled is required (libx264 for H.264, "
            f"libx265 for H.265).\n{_INSTALL_HINT}"
        )


def has_encoders(names: list[str]) -> bool:
    """Non-raising counterpart to require_encoders, for test skip guards."""
    if not is_ffmpeg_available():
        return False
    try:
        return set(names) <= available_encoders()
    except FFmpegError:
        return False


def probe_frame_count(video_path: str | Path) -> int:
    """Count decodable frames in a video via ffprobe.

    Used to verify an encode preserved the source frame count before any
    metric is computed against it - a silently dropped or duplicated frame
    would misalign every subsequent per-frame comparison.
    """
    completed = run_command(
        [
            find_ffprobe(), "-v", "error",
            "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        timeout=600,
    )
    text = (completed.stdout or "").strip()
    try:
        return int(text)
    except ValueError as exc:
        raise FFmpegCommandError(
            f"Could not read a frame count from ffprobe output: {text!r}"
        ) from exc
