"""The native block-motion search (`nvc/video/_native/block_search.c`).

The contract: for any input it accepts, `estimate_block_motion_native` returns bit-for-bit the
motion field of `estimate_block_motion_torch`. That is only worth asserting if the tests can
also FAIL, so this file has two halves:

  * equivalence - the kernel against the reference, over shapes, block sizes, search ranges,
    both SIMD paths, early exit on and off, and several thread counts;
  * negative controls - six deliberately broken copies of the C source, each of which the same
    test data must catch.

Random frames alone are NOT enough, and the negative controls are why this file knows it.
Three of the six mutations (a different channel-sum order, a missing (sum / B*B) * B*B step,
row-wise partial sums) were not detected by random, shifted, quantised or flat frames: those
frames essentially never contain candidates whose costs are mathematically equal but differ
in float rounding. `_rounding_sensitive` builds frames that do - candidates holding the SAME
values in a different accumulation order - and only then do those mutations show up.

In CI a missing native library fails these tests instead of skipping them, so a broken build
cannot hide.
"""

from __future__ import annotations

import itertools
import os
import shutil
import warnings

import numpy as np
import pytest
import torch

import nvc.video.motion as motion
from nvc.video import VideoCodec, _native
from test_video_codec import _build_bundle, _frames

GENERAL_CASES = [(64, 64, 16, 4), (96, 144, 12, 6), (96, 144, 24, 8), (128, 96, 8, 3),
                 (64, 128, 32, 8), (64, 64, 16, 0), (80, 80, 20, 5)]      # (H, W, block, range)
GENERAL_KINDS = ("noise", "shift", "quantised", "quantised7", "flat", "smooth")
ROUNDING_CASES = [(48, 48, 8, 4), (96, 144, 12, 6), (64, 64, 16, 8), (80, 80, 20, 5),
                  (96, 144, 24, 8), (64, 128, 32, 8)]
ROUNDING_SEEDS = range(4)


# --- fixtures ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lib():
    library = _native.load()
    if library is None:
        message = f"native motion library unavailable: {_native.last_error()}"
        if os.environ.get("CI"):
            pytest.fail(message)         # a broken build must not be able to hide as a skip
        pytest.skip(message)
    return library


def _has_avx2(library) -> bool:
    return bool(library.nvc_bs_has_avx2())


def _general(kind: str, height: int, width: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(1, 3, height, width, generator=g)
    if kind == "noise":
        return a, torch.rand(1, 3, height, width, generator=g)
    if kind == "shift":
        return a, torch.roll(a, (2, -3), (2, 3)) + 0.01 * torch.rand(1, 3, height, width, generator=g)
    if kind == "quantised":          # many exactly-equal costs
        return torch.round(a * 3) / 3, torch.round(torch.roll(a, (1, 1), (2, 3)) * 3) / 3
    if kind == "quantised7":
        return torch.round(a * 7) / 7, torch.round(torch.roll(a, (1, 2), (2, 3)) * 7) / 7
    if kind == "flat":               # every candidate ties
        return torch.full((1, 3, height, width), 0.25), torch.full((1, 3, height, width), 0.25)
    if kind == "smooth":             # like a decoded reconstruction against the sharp current frame
        low = torch.nn.functional.interpolate(torch.rand(1, 3, height // 8, width // 8, generator=g),
                                              size=(height, width), mode="bilinear", align_corners=False)
        return low, (low + 0.03 * torch.rand(1, 3, height, width, generator=g)).clamp(0, 1)
    raise ValueError(kind)


def _rounding_sensitive(kind: str, height: int, width: int, block: int, seed: int):
    """Frames where many candidates have mathematically EQUAL cost but a different float
    rounding, because they contain the same values in a different accumulation order.

    rowperm: the reference is one row pattern (random 24-bit-mantissa values, shuffled) tiled
             with period `block`, constant along x, against an all-zero current frame. Every
             vertical shift sees each pattern row exactly once, in a rotated order.
    tile:    the reference is one random `block` x `block` tile repeated in both directions.
    """
    g = torch.Generator().manual_seed(1000 + seed)
    if kind == "rowperm":
        base = torch.rand(3, block, generator=g)[:, torch.randperm(block, generator=g)]
        ref = base.repeat(1, height // block + 1)[:, :height][None, :, :, None]
        return ref.expand(1, 3, height, width).contiguous(), torch.zeros(1, 3, height, width)
    tile = torch.rand(1, 3, block, block, generator=g)
    ref = tile.repeat(1, 1, height // block + 1, width // block + 1)[:, :, :height, :width].contiguous()
    return ref, torch.zeros(1, 3, height, width)


@pytest.fixture(scope="module")
def workload():
    """Every input, with the PyTorch reference's answer computed once."""
    items = []
    for index, ((h, w, b, r), kind) in enumerate(itertools.product(GENERAL_CASES, GENERAL_KINDS)):
        ref, cur = _general(kind, h, w, seed=index)
        items.append((f"{kind} {h}x{w} B{b} R{r}", ref, cur, b, r))
    for (h, w, b, r), kind, seed in itertools.product(ROUNDING_CASES, ("rowperm", "tile"), ROUNDING_SEEDS):
        ref, cur = _rounding_sensitive(kind, h, w, b, seed)
        items.append((f"{kind}#{seed} {h}x{w} B{b} R{r}", ref, cur, b, r))
    return [(label, ref, cur, b, r,
             motion.estimate_block_motion_torch(ref, cur, block_size=b, search_range=r))
            for label, ref, cur, b, r in items]


def _run(lib, ref, cur, b, r, *, mode=0, early_exit=True, threads=1):
    return motion._run_native(lib, ref, cur, block_size=b, search_range=r, threads=threads,
                              early_exit=early_exit, mode=mode)


def _mismatches(lib, workload, **kwargs) -> list[str]:
    return [label for label, ref, cur, b, r, expected in workload
            if not torch.equal(expected, _run(lib, ref, cur, b, r, **kwargs))]


# --- equivalence ------------------------------------------------------------------------


@pytest.mark.parametrize("early_exit", [True, False], ids=["early-exit", "full-search"])
@pytest.mark.parametrize("mode,name", [(1, "scalar"), (2, "avx2"), (0, "auto")])
def test_native_matches_the_reference_on_every_case(lib, workload, mode, name, early_exit):
    if mode == 2 and not _has_avx2(lib):
        pytest.skip("this CPU has no AVX2")
    assert _mismatches(lib, workload, mode=mode, early_exit=early_exit) == []


def test_the_result_does_not_depend_on_the_thread_count(lib, workload):
    subset = workload[::5]
    for threads in (1, 2, 3, 7):
        assert _mismatches(lib, subset, threads=threads) == [], f"threads={threads}"


def test_full_size_frames_match_the_reference(lib):
    for kind, seed in (("shift", 1), ("noise", 2), ("smooth", 3)):
        ref, cur = _general(kind, 256, 256, seed)
        expected = motion.estimate_block_motion_torch(ref, cur, block_size=16, search_range=16)
        assert torch.equal(expected, _run(lib, ref, cur, 16, 16, threads=4)), kind


def test_a_known_shift_is_recovered(lib):
    g = torch.Generator().manual_seed(5)
    reference = torch.rand(1, 3, 64, 64, generator=g)
    current = torch.roll(reference, shifts=(3, -2), dims=(2, 3))
    found = _run(lib, reference, current, 16, 6)
    assert (found[0, 1:3, 1:3] == -3).all() and (found[1, 1:3, 1:3] == 2).all()


# --- negative controls: the tests above must be able to fail ----------------------------

# Each mutation is a list of (old, new) source edits, applied to BOTH code paths where the
# construct exists in both. `old` must occur in the source: if the C is refactored so a
# pattern disappears this test fails loudly instead of "passing" against an unmutated kernel.
MUTANTS = {
    "tie-break inverted": [("key < best_key", "key > best_key")],
    "early exit skips ties": [("(acc / bb) * bb > best) break;", "(acc / bb) * bb >= best) break;"),
                              ("_CMP_LE_OQ", "_CMP_LT_OQ")],
    "sum without the (s / B*B) * B*B step": [
        ("(acc / bb) * bb", "acc"),
        ("_mm256_mul_ps(_mm256_div_ps(acc, bbv), bbv)", "acc")],
    "channel sum as a0 + (a1 + a2)": [
        ("acc += (a0 + a1) + a2;", "acc += a0 + (a1 + a2);"),
        ("_mm256_add_ps(_mm256_add_ps(d0, d1), d2)", "_mm256_add_ps(d0, _mm256_add_ps(d1, d2))")],
    "wrong replicate padding at the right edge": [
        ("sx = sx < 0 ? 0 : (sx >= W ? W - 1 : sx);", "sx = sx < 0 ? 0 : (sx >= W ? 0 : sx);")],
    "row-wise partial sums": [
        ("""                    for (int x = 0; x < B; x++) {
                        const float a0 = fabsf(c0[x] - p0[x]);
                        const float a1 = fabsf(c1[x] - p1[x]);
                        const float a2 = fabsf(c2[x] - p2[x]);
                        acc += (a0 + a1) + a2;
                    }""",
         """                    float rowsum = 0.0f;
                    for (int x = 0; x < B; x++) {
                        const float a0 = fabsf(c0[x] - p0[x]);
                        const float a1 = fabsf(c1[x] - p1[x]);
                        const float a2 = fabsf(c2[x] - p2[x]);
                        rowsum += (a0 + a1) + a2;
                    }
                    acc += rowsum;"""),
        ("""                    for (int x = 0; x < B; x++) {
                        __m256 d0""", """                    __m256 rowsum = _mm256_setzero_ps();
                    for (int x = 0; x < B; x++) {
                        __m256 d0"""),
        ("acc = _mm256_add_ps(acc, _mm256_add_ps(_mm256_add_ps(d0, d1), d2));",
         "rowsum = _mm256_add_ps(rowsum, _mm256_add_ps(_mm256_add_ps(d0, d1), d2));"),
        ("""                    if (early_exit && best != INFINITY) {
                        const __m256 scaled""", """                    acc = _mm256_add_ps(acc, rowsum);
                    if (early_exit && best != INFINITY) {
                        const __m256 scaled"""),
    ],
}


@pytest.mark.parametrize("name", list(MUTANTS))
def test_a_broken_kernel_is_caught(lib, workload, tmp_path, name):
    text = _native._SOURCE.read_text(encoding="utf-8")
    for old, new in MUTANTS[name]:
        assert old in text, f"mutation pattern no longer in block_search.c: {old[:60]!r}"
        text = text.replace(old, new)
    source = tmp_path / "mutant.c"
    source.write_text(text, encoding="utf-8")
    mutant = _native.load_library(source=source, binary=tmp_path / "mutant_lib.bin")

    paths = [("scalar", 1)] + ([("avx2", 2)] if _has_avx2(lib) else [])
    for path, mode in paths:
        assert _mismatches(mutant, workload, mode=mode), f"{name!r} was NOT caught on the {path} path"


def test_the_unmutated_kernel_passes_the_same_check(lib, workload, tmp_path):
    """Guards the negative controls themselves: a fresh build of the real source, loaded the
    same way as a mutant, must agree with the reference everywhere."""
    source = tmp_path / "copy.c"
    shutil.copy(_native._SOURCE, source)
    copy = _native.load_library(source=source, binary=tmp_path / "copy_lib.bin")
    assert _mismatches(copy, workload, mode=1) == []


# --- the dispatcher ---------------------------------------------------------------------


@pytest.fixture
def native_calls(monkeypatch):
    calls = []
    real = motion._run_native

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(motion, "_run_native", spy)
    return calls


def _pair(size=64, seed=0):
    return _general("shift", size, size, seed)


def test_auto_uses_the_native_kernel_when_it_can(lib, native_calls, monkeypatch):
    monkeypatch.delenv("NVC_MOTION_BACKEND", raising=False)
    ref, cur = _pair()
    got = motion.estimate_block_motion(ref, cur, block_size=16, search_range=4)
    assert native_calls and torch.equal(
        got, motion.estimate_block_motion_torch(ref, cur, block_size=16, search_range=4))


@pytest.mark.parametrize("make", [
    lambda: tuple(t.double() for t in _pair()),                       # not float32
    lambda: tuple(t[:, :1].contiguous() for t in _pair()),            # not 3 channels
    lambda: (torch.full((1, 3, 32, 32), float("nan")), torch.rand(1, 3, 32, 32)),   # non-finite
], ids=["float64", "one-channel", "nan"])
def test_auto_falls_back_to_the_reference_for_inputs_it_does_not_handle(lib, native_calls, monkeypatch, make):
    monkeypatch.delenv("NVC_MOTION_BACKEND", raising=False)
    ref, cur = make()
    got = motion.estimate_block_motion(ref, cur, block_size=16, search_range=2)
    assert not native_calls
    assert torch.equal(got, motion.estimate_block_motion_torch(ref, cur, block_size=16, search_range=2))


def test_backend_native_is_strict_about_unsupported_input(lib):
    ref, cur = _pair()
    with pytest.raises(ValueError, match="cannot take this input.*float32"):
        motion.estimate_block_motion(ref.double(), cur.double(), block_size=16, search_range=4,
                                     backend="native")


def test_backend_torch_never_touches_the_native_kernel(lib, native_calls):
    ref, cur = _pair()
    motion.estimate_block_motion(ref, cur, block_size=16, search_range=4, backend="torch")
    assert not native_calls


def test_the_environment_variable_selects_the_backend(lib, native_calls, monkeypatch):
    ref, cur = _pair()
    monkeypatch.setenv("NVC_MOTION_BACKEND", "torch")
    motion.estimate_block_motion(ref, cur, block_size=16, search_range=4)
    assert not native_calls
    monkeypatch.setenv("NVC_MOTION_BACKEND", "native")
    motion.estimate_block_motion(ref, cur, block_size=16, search_range=4)
    assert native_calls


def test_an_unknown_backend_is_rejected():
    ref, cur = _pair()
    with pytest.raises(ValueError, match="backend must be one of"):
        motion.estimate_block_motion(ref, cur, backend="cuda")


@pytest.mark.parametrize("backend", ["auto", "native", "torch"])
def test_every_backend_rejects_bad_inputs_the_same_way(lib, backend):
    ref, cur = _pair(64)
    with pytest.raises(ValueError, match="must have the same shape"):
        motion.estimate_block_motion(ref, cur[:, :, :48], block_size=16, backend=backend)
    with pytest.raises(ValueError, match=r"expected \[1, C, H, W\]"):
        motion.estimate_block_motion(ref[0], cur[0], block_size=16, backend=backend)
    with pytest.raises(ValueError, match="must divide evenly"):
        motion.estimate_block_motion(ref, cur, block_size=24, backend=backend)


def test_missing_library_falls_back_once_with_a_warning(lib, monkeypatch, native_calls):
    monkeypatch.delenv("NVC_MOTION_BACKEND", raising=False)
    monkeypatch.setattr(motion._native, "load", lambda: None)
    monkeypatch.setattr(motion._native, "last_error", lambda: "no compiler on PATH")
    monkeypatch.setattr(motion, "_warned_unavailable", False)
    ref, cur = _pair()
    expected = motion.estimate_block_motion_torch(ref, cur, block_size=16, search_range=4)
    with pytest.warns(RuntimeWarning, match="no compiler on PATH"):
        assert torch.equal(motion.estimate_block_motion(ref, cur, block_size=16, search_range=4), expected)
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # the second call must stay silent
        assert torch.equal(motion.estimate_block_motion(ref, cur, block_size=16, search_range=4), expected)
    assert not native_calls
    with pytest.raises(RuntimeError, match="unavailable: no compiler on PATH"):
        motion.estimate_block_motion(ref, cur, block_size=16, search_range=4, backend="native")


# --- through the real codec -------------------------------------------------------------


def test_the_encoded_stream_is_byte_identical_with_either_backend(lib, native_calls, monkeypatch):
    """The end-to-end statement of the contract, on the miniature codec test_video_codec.py
    uses: same bytes, same reconstructions, and the native path really was the one used."""
    codec = VideoCodec(_build_bundle(), device="cpu")
    frames = _frames()

    monkeypatch.setenv("NVC_MOTION_BACKEND", "torch")
    reference, reference_frames = codec.encode(frames, return_reconstructions=True)
    assert not native_calls

    monkeypatch.setenv("NVC_MOTION_BACKEND", "native")
    native, native_frames = codec.encode(frames, return_reconstructions=True)
    assert native_calls

    assert native.data == reference.data
    assert torch.equal(native_frames, reference_frames)
    assert torch.equal(codec.decode(native.data), reference_frames)


# --- the library and its loader ---------------------------------------------------------


def test_the_kernel_refuses_bad_arguments(lib):
    cur = np.zeros((3, 32, 32), dtype=np.float32)
    pad = np.zeros((3, lib.nvc_bs_padded_height(32, 2), lib.nvc_bs_padded_width(32, 2)), dtype=np.float32)
    dy = np.zeros(4, dtype=np.int32)
    dx = np.zeros(4, dtype=np.int32)
    c_float = _native.ctypes.POINTER(_native.ctypes.c_float)
    c_int = _native.ctypes.POINTER(_native.ctypes.c_int32)

    def call(H=32, W=32, R=2, B=16, begin=0, end=4, mode=0, out=dy):
        return lib.nvc_bs_search(cur.ctypes.data_as(c_float), pad.ctypes.data_as(c_float), H, W, R, B,
                                 begin, end, out.ctypes.data_as(c_int), dx.ctypes.data_as(c_int), 1, mode)

    assert call() == 0
    assert call(H=30) == -1                 # not divisible by the block size
    assert call(R=-1) == -1
    assert call(end=5) == -1                # past the last block
    assert call(begin=3, end=2) == -1
    assert call(mode=9) == -1
    assert lib.nvc_bs_search(None, None, 32, 32, 2, 16, 0, 4, None, None, 1, 0) == -1
    if not _has_avx2(lib):
        assert call(mode=2) == -2


def test_an_abi_mismatch_is_rejected(lib, tmp_path):
    text = _native._SOURCE.read_text(encoding="utf-8").replace("#define NVC_ABI_VERSION 1", "#define NVC_ABI_VERSION 99")
    source = tmp_path / "abi.c"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match="ABI 99"):
        _native.load_library(source=source, binary=tmp_path / "abi_lib.bin")


def test_a_failed_build_raises_and_leaves_no_partial_file(tmp_path):
    source = tmp_path / "broken.c"
    source.write_text("this is not C", encoding="utf-8")
    binary = tmp_path / "broken_lib.bin"
    if _native._select_compiler() is None:
        pytest.skip("no compiler")
    with pytest.raises(RuntimeError, match="compiling broken.c failed"):
        _native.compile_library(source, binary)
    assert list(tmp_path.glob("*.tmp*")) == [] and not binary.exists()


def test_info_reports_what_loaded(lib):
    info = _native.info()
    assert info["available"] is True and info["error"] is None
    assert info["simd"] in ("avx2", "scalar")


def test_the_native_source_ships_in_the_package():
    """The BUG-02 lesson: a wheel without the .c file would silently lose the fast path."""
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["setuptools"]["package-data"]["nvc.video._native"] == ["*.c"]
    assert (root / "src/nvc/video/_native/block_search.c").is_file()
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    assert "src/nvc/video/_native/*.dll" in ignored and "src/nvc/video/_native/*.dylib" in ignored
