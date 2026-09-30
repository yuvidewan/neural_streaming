"""ctypes bridge to the native block-motion search (block_search.c).

Same approach as `nvc.compression._native` (plain ctypes plus a `gcc -shared` call, built
on first use, no CPython C-API extension), with two deliberate differences:

* **This backend is optional.** Motion is estimated only by the encoder and its result is
  transmitted, so nothing about *decoding* depends on how it was estimated. The range coder
  has no fallback because it has no other implementation; motion search does
  (`estimate_block_motion_torch`), so `nvc.video.motion` falls back to it, with a one-time
  warning, when this library cannot be built or loaded. `load()` therefore never raises -
  it returns None and records why in `last_error()`.
* **Builds are atomic.** The library is compiled to a process-private temp file and moved into
  place with `os.replace`, so two processes importing at once (DataLoader workers, pytest-xdist)
  cannot leave a half-written binary for a third to load.

The compiler is chosen by `nvc.compression._native._select_compiler`, which already handles
the 32-bit-vs-64-bit MinGW trap. Build flags are `-O3 -ffp-contract=off`, and never
`-ffast-math`: the kernel's result is only bit-identical to the PyTorch reference because it
adds floats in a fixed order (see block_search.c), and reassociation would silently break that.
The AVX2 code path is selected at run time inside the library, so no `-mavx2` is needed and a
binary built here runs on any x86-64 CPU.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

from nvc.compression._native import _select_compiler

ABI_VERSION = 1
MODE_AUTO, MODE_SCALAR, MODE_AVX2 = 0, 1, 2

_NATIVE_DIR = Path(__file__).resolve().parent
_SOURCE = _NATIVE_DIR / "block_search.c"

_lib = None
_load_attempted = False
_last_error: str | None = None


def _binary_path() -> Path:
    system = platform.system()
    suffix = ".dll" if system == "Windows" else ".dylib" if system == "Darwin" else ".so"
    return _NATIVE_DIR / f"block_search{suffix}"


def last_error() -> str | None:
    """Why `load()` returned None, if it did."""
    return _last_error


def compile_library(source: Path, binary: Path) -> None:
    """Compile `source` into `binary`, atomically. Raises RuntimeError with the compiler's own
    message on failure. Exposed so tests can build a deliberately broken variant of the kernel
    (a negative control) with exactly the flags the real build uses."""
    if not source.is_file():
        raise RuntimeError(f"native source file missing: {source}")
    compiler = _select_compiler()
    if compiler is None:
        raise RuntimeError(
            "no C compiler (gcc/cc/clang) found on PATH - install one (MinGW-w64 on Windows, "
            "build-essential on Linux, Xcode Command Line Tools on Mac) to use the native "
            "motion search")
    temporary = binary.with_name(f"{binary.stem}.{os.getpid()}.tmp{binary.suffix}")
    command = [compiler, "-O3", "-ffp-contract=off", "-shared", "-o", str(temporary), str(source)]
    if platform.system() != "Windows":
        command.insert(1, "-fPIC")
        command.append("-lm")
    try:
        result = subprocess.run(command, capture_output=True, timeout=120, text=True)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"compiling {source.name} timed out after 120s") from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke compiler {compiler!r}: {exc}") from exc
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"compiling {source.name} failed (exit {result.returncode}):\n{result.stderr.strip()}")
    try:
        os.replace(temporary, binary)
    except OSError:
        # On Windows a DLL another process has loaded cannot be replaced. That process built
        # the same source, so use what is there rather than fail.
        temporary.unlink(missing_ok=True)
        if not binary.is_file():
            raise


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Declare every signature, and refuse a library from a different ABI."""
    c_i32, c_f32p, c_i32p = ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
    try:
        lib.nvc_bs_abi_version.argtypes, lib.nvc_bs_abi_version.restype = [], c_i32
        found = lib.nvc_bs_abi_version()
        if found != ABI_VERSION:
            raise RuntimeError(f"native motion library has ABI {found}, expected {ABI_VERSION}")
        lib.nvc_bs_has_avx2.argtypes, lib.nvc_bs_has_avx2.restype = [], c_i32
        lib.nvc_bs_padded_height.argtypes, lib.nvc_bs_padded_height.restype = [c_i32, c_i32], c_i32
        lib.nvc_bs_padded_width.argtypes, lib.nvc_bs_padded_width.restype = [c_i32, c_i32], c_i32
        lib.nvc_bs_pad.argtypes = [c_f32p, c_i32, c_i32, c_i32, c_f32p]
        lib.nvc_bs_pad.restype = c_i32
        lib.nvc_bs_search.argtypes = [c_f32p, c_f32p, c_i32, c_i32, c_i32, c_i32, c_i32, c_i32,
                                      c_i32p, c_i32p, c_i32, c_i32]
        lib.nvc_bs_search.restype = c_i32
    except AttributeError as exc:
        raise RuntimeError(f"native motion library is missing an expected symbol: {exc}") from exc
    return lib


def load_library(source: Path = _SOURCE, binary: Path | None = None) -> ctypes.CDLL:
    """Build (if missing or older than its source) and load a library. Raises RuntimeError."""
    binary = binary if binary is not None else _binary_path()
    stale = source.is_file() and binary.is_file() and source.stat().st_mtime > binary.stat().st_mtime
    if not binary.is_file() or stale:
        compile_library(source, binary)
    try:
        return _bind(ctypes.CDLL(str(binary)))
    except OSError as exc:
        raise RuntimeError(f"found {binary} but ctypes could not load it: {exc}") from exc


def load():
    """The loaded library, building it first if needed, or None (see `last_error()`).
    Never raises. Cached after the first call."""
    global _lib, _load_attempted, _last_error
    if _load_attempted:
        return _lib
    _load_attempted = True
    try:
        _lib = load_library()
    except RuntimeError as exc:
        _last_error = str(exc)
        _lib = None
    return _lib


def info() -> dict:
    """{'available': bool, 'simd': 'avx2' | 'scalar' | None, 'error': str | None}."""
    lib = load()
    if lib is None:
        return {"available": False, "simd": None, "error": _last_error}
    return {"available": True, "simd": "avx2" if lib.nvc_bs_has_avx2() else "scalar", "error": None}
