"""Environment verification for the Neural Video Compression project.

Run this after creating and activating the project virtual environment to
confirm required tools, packages, and the project directory structure are
in place. This script does not train, compress, or download anything - it
only inspects the current environment.

Usage (from the project root, PowerShell):
    .venv\\Scripts\\Activate.ps1
    python scripts\\check_environment.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 11)

REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/frames",
    "data/processed",
    "outputs/checkpoints",
    "outputs/compressed",
    "outputs/reconstructed",
    "outputs/metrics",
    "outputs/visualizations",
    "src/nvc",
    "tests",
]


def _ok(label: str, detail: str) -> None:
    print(f"[OK]      {label}: {detail}")


def _warn(label: str, detail: str) -> None:
    print(f"[WARN]    {label}: {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"[MISSING] {label}: {detail}")


def check_python() -> bool:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) >= MIN_PYTHON:
        _ok("Python", f"{version_str} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required)")
        return True
    _fail("Python", f"{version_str} found, but >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} is required")
    return False


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        _fail("PyTorch", "not installed. Run: pip install -r requirements.txt")
        return False

    _ok("PyTorch", torch.__version__)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        _ok("CUDA", f"available ({gpu_name})")
    else:
        _warn("CUDA", "not available - the project will run on CPU (this is fine)")
    return True


def check_opencv() -> bool:
    try:
        import cv2
    except ImportError:
        _fail("OpenCV", "not installed. Run: pip install -r requirements.txt")
        return False
    _ok("OpenCV", cv2.__version__)
    return True


def check_ffmpeg() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        _fail("FFmpeg", "not found on PATH. Install it separately (see README.md)")
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "version unknown"
        _ok("FFmpeg", first_line)
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        _warn("FFmpeg", f"found at {ffmpeg_path} but could not execute it ({exc})")
        return False


def check_directories() -> bool:
    missing = [d for d in REQUIRED_DIRS if not (PROJECT_ROOT / d).is_dir()]
    if not missing:
        _ok("Project structure", f"all {len(REQUIRED_DIRS)} required directories present")
        return True
    _fail("Project structure", f"missing directories: {', '.join(missing)}")
    return False


def main() -> int:
    print("Neural Video Compression - Environment Check")
    print("=" * 50)

    results = [
        check_python(),
        check_torch(),
        check_opencv(),
        check_ffmpeg(),
        check_directories(),
    ]

    print("=" * 50)
    if all(results):
        print("All checks passed. Environment is ready for development.")
        return 0

    print("One or more checks failed - see [MISSING] lines above for details.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
