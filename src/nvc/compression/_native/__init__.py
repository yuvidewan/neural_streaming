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

COMPILER SELECTION
------------------
A machine can have more than one compiler on PATH, and `shutil.which` only
ever returns the first match by name - on Windows in particular, a 32-bit-
only MinGW install earlier on PATH than a working 64-bit one is a real
configuration (observed in the wild, not hypothetical), and it fails
silently: the build succeeds, but the resulting DLL is architecture-
mismatched against a 64-bit Python and `ctypes.CDLL` raises a cryptic
`WinError 193 ("%1 is not a valid Win32 application")` at LOAD time, not at
build time. `_select_compiler` below scans every PATH directory (not just
the first hit) and prefers whichever candidate's own `-dumpmachine` output
confirms matches the running Python's bitness, falling back to the old
first-found-by-name behavior only when no candidate's target architecture
can be determined at all.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import struct
import subprocess
from pathlib import Path

_NATIVE_DIR = Path(__file__).resolve().parent
_SOURCE = _NATIVE_DIR / "range_coder.c"

_COMPILER_NAMES = ("gcc", "cc", "clang")

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


def _candidate_compilers() -> list[str]:
    """Every `gcc`/`cc`/`clang` found on PATH, in PATH order, deduplicated.

    `shutil.which(name)` alone only reports the first directory that has a
    match for ONE name - it can't tell you there was a second, later `gcc`
    that might be the one you actually want. This walks PATH directory by
    directory and checks each candidate name in each, so every compiler on
    PATH is considered, not just the first name/directory combination.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        for name in _COMPILER_NAMES:
            found = shutil.which(name, path=directory)
            if found and found not in seen:
                seen.add(found)
                candidates.append(found)
    return candidates


def _compiler_matches_python_bitness(compiler: str) -> bool | None:
    """Does `compiler`'s target architecture match the running Python's
    (32-bit vs 64-bit)? Returns None when that can't be determined (no
    `-dumpmachine` support, an unrecognized triple, or the subprocess call
    itself failing) - callers must treat None as "unknown", not "no"."""
    try:
        result = subprocess.run([compiler, "-dumpmachine"], capture_output=True,
                                 timeout=5, text=True)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None

    triple = result.stdout.strip().lower()
    is_64bit_triple = any(tag in triple for tag in
                          ("x86_64", "amd64", "win64", "aarch64", "arm64",
                           "powerpc64", "mips64", "riscv64"))
    # Classic mingw.org's `gcc -dumpmachine` prints the bare word "mingw32"
    # (not a full target triple) and that toolchain only ever targets
    # 32-bit x86 - handled explicitly since it won't match the general
    # "32" triple tags below.
    is_32bit_triple = triple == "mingw32" or any(
        tag in triple for tag in ("i386", "i486", "i586", "i686", "win32", "arm-"))
    if is_64bit_triple and not is_32bit_triple:
        return struct.calcsize("P") * 8 == 64
    if is_32bit_triple and not is_64bit_triple:
        return struct.calcsize("P") * 8 == 32
    return None  # ambiguous or unrecognized triple - can't tell


def _select_compiler() -> str | None:
    """Pick a C compiler from PATH, preferring one confirmed to match the
    running Python's bitness (see the module docstring's "COMPILER
    SELECTION" section) over merely being first on PATH by name."""
    candidates = _candidate_compilers()
    for candidate in candidates:
        if _compiler_matches_python_bitness(candidate) is True:
            return candidate
    # No candidate could be CONFIRMED to match (every -dumpmachine call
    # failed, or every triple was unrecognized) - fall back to the simple
    # first-found-by-name choice rather than refusing to build at all.
    return candidates[0] if candidates else None


def _build() -> bool:
    """Try to compile the shared library. Returns True on success, never
    raises. Records a specific reason via `_set_error` on failure."""
    if not _SOURCE.is_file():
        _set_error(f"native source file missing: {_SOURCE}")
        return False

    compiler = _select_compiler()
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
            c_i64_p, ctypes.c_int64,                            # symbols, n
            c_i64_p, ctypes.c_int64, ctypes.c_int64,            # cumulative, width, num_tables
            c_i64_p,                                            # table_index
            ctypes.POINTER(c_u8_p), c_i64_p,                    # out_data, out_len
        ]
        lib.rc_encode.restype = ctypes.c_int32

        lib.rc_decode.argtypes = [
            c_u8_p, ctypes.c_int64,                             # payload, payload_len
            ctypes.c_int64,                                     # n
            c_i64_p, ctypes.c_int64, ctypes.c_int64,            # cumulative, width, num_tables
            c_i64_p,                                            # table_index
            c_i64_p,                                            # out_symbols
        ]
        lib.rc_decode.restype = ctypes.c_int32

        lib.rc_free.argtypes = [c_u8_p]
        lib.rc_free.restype = None

        # Milestone 12: resumable decoder - see range_coder.py's
        # ResumableDecoder and range_coder.c's "Resumable decoder" section.
        lib.rc_decoder_open.argtypes = [c_u8_p, ctypes.c_int64]
        lib.rc_decoder_open.restype = ctypes.c_void_p

        lib.rc_decoder_decode.argtypes = [
            ctypes.c_void_p, ctypes.c_int64,                    # handle, n
            c_i64_p, ctypes.c_int64, ctypes.c_int64,            # cumulative, width, num_tables
            c_i64_p,                                            # table_index
            c_i64_p,                                            # out_symbols
        ]
        lib.rc_decoder_decode.restype = ctypes.c_int32

        lib.rc_decoder_close.argtypes = [ctypes.c_void_p]
        lib.rc_decoder_close.restype = None
    except AttributeError as exc:
        _set_error(f"{binary} loaded but is missing an expected symbol: {exc}")
        return None

    _lib = lib
    return _lib
