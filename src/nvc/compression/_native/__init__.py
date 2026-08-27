"""ctypes bridge to the compiled C range coder (range_coder.c).

This build has **no pure-Python fallback** for arithmetic coding - the C
backend is the only implementation `range_coder.py`'s public
`encode_symbols`/`decode_symbols` will ever call (see that module's
docstring; the original Python implementation is kept, commented out, for
reference/teaching purposes only, not as a runtime path). That makes this
module's job different from a typical "accelerate if possible" shim: it
must either load a working native library or fail loudly and specifically,
never silently do nothing and let a caller hit a confusing later error.

`load()` builds the shared library on first import if a compiler is
available and no up-to-date compiled artifact exists yet, then loads it via
ctypes. It never raises itself - it returns `None` on any failure - but it
records *why* in `last_error()`, so the caller (`range_coder.py`'s
`ensure_native_backend()`) can raise a clear, actionable exception instead
of a bare "returned None."

Deliberately plain ctypes + a `gcc -shared` call, not a CPython C-API
extension built via setuptools: the compiled library doesn't link against
Python at all, so the same .dll/.so works unchanged across Python versions
and doesn't need to match a specific build's ABI.
"""

from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parent
_SOURCE = _NATIVE_DIR / "range_coder.c"

_lib = None
_load_attempted = False
_last_error: str | None = None


def _binary_path() -> Path:
    system = platform.system()
    if system == "Windows":
        return _NATIVE_DIR / "range_coder.dll"
    if system == "Darwin":
        return _NATIVE_DIR / "range_coder.dylib"
    return _NATIVE_DIR / "range_coder.so"


def _set_error(message: str) -> None:
    global _last_error
    _last_error = message


def last_error() -> str | None:
    """Why `load()` returned None, if it did. `None` if loading succeeded
    or hasn't been attempted yet."""
    return _last_error


def _build() -> bool:
    """Try to compile the shared library. Returns True on success, never
    raises. Records a specific reason via `_set_error` on failure."""
    if not _SOURCE.is_file():
        _set_error(f"native source file missing: {_SOURCE}")
        return False

    compiler = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if compiler is None:
        _set_error(
            "no C compiler (gcc/cc/clang) found on PATH - install one (e.g. "
            "MinGW-w64 on Windows, build-essential on Linux, Xcode Command Line "
            "Tools on Mac) to build the native range coder"
        )
        return False

    binary = _binary_path()
    cmd = [compiler, "-O3", "-shared", "-o", str(binary), str(_SOURCE)]
    if platform.system() != "Windows":
        cmd.insert(1, "-fPIC")
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=60, text=True)
    except subprocess.TimeoutExpired:
        _set_error(f"compiling {_SOURCE.name} timed out after 60s")
        return False
    except OSError as exc:
        _set_error(f"failed to invoke compiler {compiler!r}: {exc}")
        return False

    if result.returncode != 0:
        _set_error(
            f"compiling {_SOURCE.name} failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
        return False
    if not binary.is_file():
        _set_error(f"compiler reported success but {binary} was not produced")
        return False
    return True


def load():
    """Return the loaded ctypes library (building it first if needed), or
    `None` if no compiler/binary is available - call `last_error()` for
    why. Never raises. Result is cached after the first call.
    """
    global _lib, _load_attempted
    if _load_attempted:
        return _lib
    _load_attempted = True

    binary = _binary_path()
    source_is_newer = _SOURCE.is_file() and (
        not binary.is_file() or _SOURCE.stat().st_mtime > binary.stat().st_mtime
    )
    if not binary.is_file() or source_is_newer:
        if not _build():
            return None

    try:
        lib = ctypes.CDLL(str(binary))
    except OSError as exc:
        _set_error(f"found {binary} but ctypes could not load it: {exc}")
        return None

    c_i64_p = ctypes.POINTER(ctypes.c_int64)
    c_u8_p = ctypes.POINTER(ctypes.c_uint8)

    try:
        lib.rc_encode.argtypes = [
            c_i64_p, ctypes.c_int64,           # symbols, n
            c_i64_p, ctypes.c_int64,           # cumulative, table_width
            c_i64_p,                           # table_index
            ctypes.POINTER(c_u8_p), c_i64_p,   # out_data, out_len
        ]
        lib.rc_encode.restype = ctypes.c_int32

        lib.rc_decode.argtypes = [
            c_u8_p, ctypes.c_int64,            # payload, payload_len
            ctypes.c_int64,                    # n
            c_i64_p, ctypes.c_int64,           # cumulative, table_width
            c_i64_p,                           # table_index
            c_i64_p,                           # out_symbols
        ]
        lib.rc_decode.restype = ctypes.c_int32

        lib.rc_free.argtypes = [c_u8_p]
        lib.rc_free.restype = None
    except AttributeError as exc:
        _set_error(f"{binary} loaded but is missing an expected symbol: {exc}")
        return None

    _lib = lib
    return _lib
