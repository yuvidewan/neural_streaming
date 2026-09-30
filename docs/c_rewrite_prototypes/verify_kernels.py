"""Verify and time the prototype C kernels against the current PyTorch code.

Run from the repo root with the project's virtualenv:

    python docs/c_rewrite_prototypes/verify_kernels.py

Needs gcc on PATH and a CPU with AVX2 + FMA. Compiles kernels.c into a temp directory.
Real DAVIS frames are used if data/external/DAVIS exists, else synthetic ones (the
adversarial cases - exact ties, flat frames, noise - are synthetic either way).

Checks (all must print 0 mismatches):
  1. block motion search: every motion vector equal to nvc.video.motion.estimate_block_motion
  2. codebook assignment: argmin equal to SharedCodebook.assign_tensor, including EXACT-TIE
     rows (duplicated prototypes), where the lowest-index rule is what is being tested
Also reports (not pass/fail): how often raw float costs differ in the last bit between the
C kernel and torch's matmul, and how that can flip an argmin.
"""
from __future__ import annotations

import ctypes
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nvc.compression.entropy_model import EmpiricalEntropyModel  # noqa: E402
from nvc.video import entropy as ventropy  # noqa: E402
from nvc.video import motion as vmotion  # noqa: E402

H = W = 256
B, R = 16, 16
A, K = 16, 512


def build_library() -> ctypes.CDLL:
    compiler = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if compiler is None:
        raise SystemExit("no C compiler on PATH (need gcc/clang)")
    suffix = ".dll" if platform.system() == "Windows" else ".so"
    out = Path(tempfile.mkdtemp(prefix="nvc_kernels_")) / f"kernels{suffix}"
    cmd = [compiler, "-O3", "-mavx2", "-mfma", "-ffp-contract=off", "-shared", "-o", str(out),
           str(HERE / "kernels.c")]
    if platform.system() != "Windows":
        cmd.insert(1, "-fPIC")
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(str(out))


lib = build_library()
P = ctypes.POINTER
f32p, i32p, i64p = P(ctypes.c_float), P(ctypes.c_int32), P(ctypes.c_int64)
lib.nvc_block_search.argtypes = [f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, i32p, i32p, ctypes.c_int]
lib.nvc_assign_argmin.argtypes = [f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, i64p]
lib.nvc_cost_rows.argtypes = [f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p, ctypes.c_int]


def median_ms(fn, n=7):
    fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1000


# --------------------------------------------------------------------------- 1. motion
def c_motion(ref: torch.Tensor, cur: torch.Tensor, early: int) -> np.ndarray:
    padded = F.pad(ref, (R,) * 4, mode="replicate")[0].contiguous().numpy()
    wp = W + 2 * R + 8
    pad = np.full((3, H + 2 * R, wp), 1e6, dtype=np.float32)
    pad[:, :, :W + 2 * R] = padded
    cur_np = np.ascontiguousarray(cur[0].numpy())
    n = (H // B) * (W // B)
    dy = np.zeros(n, dtype=np.int32)
    dx = np.zeros(n, dtype=np.int32)
    lib.nvc_block_search(cur_np.ctypes.data_as(f32p), pad.ctypes.data_as(f32p), H, W, wp, R, B,
                         dy.ctypes.data_as(i32p), dx.ctypes.data_as(i32p), early)
    return np.stack([dy.reshape(H // B, W // B), dx.reshape(H // B, W // B)]).astype(np.int64)


def davis_pairs():
    import cv2
    root = ROOT / "data/external/DAVIS/JPEGImages/480p"
    if not root.is_dir():
        return [], []
    pairs, labels = [], []
    for seq in sorted(root.iterdir())[:6]:
        paths = sorted(seq.glob("*.jpg"))
        for i in (0, 10, 25):
            if i + 1 >= len(paths):
                continue
            imgs = []
            for p in (paths[i], paths[i + 1]):
                img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                imgs.append(torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)[None].contiguous())
            pairs.append(imgs)
            labels.append(f"{seq.name}[{i}]")
    return pairs, labels


print("=== 1. block motion search: C/AVX2 vs nvc.video.motion.estimate_block_motion ===")
pairs, labels = davis_pairs()
print(f"real DAVIS pairs: {len(pairs)}" + ("" if pairs else "  (DAVIS not found - synthetic cases only)"))
g = torch.Generator().manual_seed(7)
a = torch.rand(1, 3, H, W, generator=g)
synthetic = [
    ([a, a.clone()], "identical noise (all zero motion)"),
    ([torch.full((1, 3, H, W), 0.5)] * 2, "flat gray (every candidate ties)"),
    ([torch.rand(1, 3, H, W, generator=g), torch.rand(1, 3, H, W, generator=g)], "two unrelated noise frames"),
    ([a, torch.roll(a, shifts=(3, -5), dims=(2, 3))], "known shift (3, -5)"),
    ([torch.zeros(1, 3, H, W)] * 2, "all zero"),
    ([torch.round(a * 4) / 4, torch.round(torch.roll(a, (1, 2), (2, 3)) * 4) / 4], "4-level quantised (many exact ties)"),
]
for pair, label in synthetic:
    pairs.append(pair)
    labels.append(label)

mismatches = 0
for (ref, cur), label in zip(pairs, labels):
    expected = vmotion.estimate_block_motion(ref, cur, block_size=B, search_range=R).numpy()
    for early in (0, 1):
        got = c_motion(ref, cur, early)
        if not np.array_equal(expected, got):
            mismatches += 1
            print(f"  MISMATCH {label} early_exit={early}: {(expected != got).any(axis=0).sum()} of 256 blocks")
print(f"pairs tested: {len(pairs)} x 2 modes    mismatching runs: {mismatches}")

ref, cur = pairs[0]
t_torch = median_ms(lambda: vmotion.estimate_block_motion(ref, cur, block_size=B, search_range=R), 3)
t_c0 = median_ms(lambda: c_motion(ref, cur, 0))
t_c1 = median_ms(lambda: c_motion(ref, cur, 1))
print(f"torch (current)          {t_torch:8.1f} ms/frame")
print(f"C AVX2, no early exit    {t_c0:8.1f} ms/frame  ({t_torch / t_c0:5.1f}x)   includes padding + copies")
print(f"C AVX2, with early exit  {t_c1:8.1f} ms/frame  ({t_torch / t_c1:5.1f}x)")

# --------------------------------------------------------------------- 2. codebook assign
print("\n=== 2. codebook assignment: fused C kernel vs SharedCodebook.assign_tensor ===")
rng = np.random.default_rng(12)


def make_book(duplicate_offsets=()):
    freqs = EmpiricalEntropyModel.from_symbols(
        rng.integers(0, A, size=(400, K)), bits=4, num_tables=K).frequencies.copy()
    for off in duplicate_offsets:                    # row k+off becomes an exact copy of row k
        for k in range(0, K - off):
            if (k // off) % 2 == 0:
                freqs[k + off] = freqs[k]
    return ventropy.SharedCodebook(freqs, bits=4)


def transposed_log2(book):
    return np.ascontiguousarray(book._log2_costs.astype(np.float32).T)


def rows(n, sharpness):
    logits = torch.from_numpy(rng.normal(size=(n, A)).astype(np.float32)) * sharpness
    return torch.softmax(logits, dim=1).contiguous()


def c_assign(r: torch.Tensor, lt: np.ndarray) -> np.ndarray:
    rn = r.numpy()
    out = np.empty(rn.shape[0], dtype=np.int64)
    lib.nvc_assign_argmin(rn.ctypes.data_as(f32p), lt.ctypes.data_as(f32p), rn.shape[0], A, K,
                          out.ctypes.data_as(i64p))
    return out


book = make_book()
lt = transposed_log2(book)
total = bad = 0
for sharpness in (0.5, 1, 2, 3, 6, 10):
    for _ in range(4):
        r = rows(16384, sharpness)
        bad += int((c_assign(r, lt) != book.assign_tensor(r)).sum())
        total += r.shape[0]
print(f"random codebook:      argmin mismatches {bad} / {total} rows")

# Exact ties: duplicate prototypes at several offsets (same SIMD lane, different lane, far apart).
#  (a) PASS/FAIL: the kernel's own lowest-index rule, checked against numpy's argmin (first
#      occurrence) of the kernel's own cost matrix.
#  (b) INFO: agreement with torch. Torch is NOT a valid oracle for exact ties: its matmul gives
#      identical prototypes different last-bit costs when one of them sits in the final 8-column
#      tile (k = 504..511), so torch sees no exact tie there. See C_REWRITE_REPORT.md section 5.
tie_bad = tie_total = torch_differs = 0
for offsets in ((1,), (3,), (8,), (5, 256), (1, 2, 4, 8, 16, 32, 64, 128, 256)):
    tie_book = make_book(offsets)
    tie_lt = transposed_log2(tie_book)
    r = rows(8192, 3.0)
    rn = r.numpy()
    costs = np.empty((rn.shape[0], K), dtype=np.float32)
    lib.nvc_cost_rows(rn.ctypes.data_as(f32p), tie_lt.ctypes.data_as(f32p), rn.shape[0], A, K,
                      costs.ctypes.data_as(f32p), 1)
    fused = c_assign(r, tie_lt)
    tie_bad += int((fused != np.argmin(costs, axis=1)).sum())
    torch_differs += int((fused != tie_book.assign_tensor(r)).sum())
    tie_total += rn.shape[0]
print(f"exact-tie codebooks:  kernel vs numpy argmin of its own costs: {tie_bad} / {tie_total} wrong")
print(f"                      kernel vs torch: {torch_differs} / {tie_total} differ   [INFO only, see comment]")

# Bit-exactness experiment: two VALID float32 summation orders (fused multiply-add chain vs
# separate multiply and add) against torch's matmul. Neither reproduces torch's costs bit for
# bit, and a different-but-valid order can flip an argmin near a tie. Swept over how peaked
# the distributions are (sharper distributions make near-ties more likely).
differ = {1: 0, 0: 0}
flips = {1: 0, 0: 0}
entries = rows_seen = 0
for sharpness in (3, 6, 10):
    for _ in range(4):
        r = rows(16384, sharpness)
        rn = r.numpy()
        torch_costs = (r @ torch.from_numpy(book._log2_costs.astype(np.float32)).T).numpy()
        reference = book.assign_tensor(r)
        for use_fma in (1, 0):
            cc = np.empty((rn.shape[0], K), dtype=np.float32)
            lib.nvc_cost_rows(rn.ctypes.data_as(f32p), lt.ctypes.data_as(f32p), rn.shape[0], A, K,
                              cc.ctypes.data_as(f32p), use_fma)
            differ[use_fma] += int((cc.view(np.int32) != torch_costs.view(np.int32)).sum())
            flips[use_fma] += int((np.argmin(cc, axis=1) != reference).sum())
        entries += torch_costs.size
        rows_seen += rn.shape[0]
for use_fma, label in ((1, "fused multiply-add   "), (0, "separate multiply,add")):
    print(f"cost order {label}: {differ[use_fma]:>9} / {entries} cost entries not bit-equal to torch "
          f"({100 * differ[use_fma] / entries:.1f}%); argmin flips vs torch {flips[use_fma]} / {rows_seen} rows")

for n in (16384, 4096):
    r = rows(n, 3.0)
    t_t = median_ms(lambda: book.assign_tensor(r), 10)
    t_c = median_ms(lambda: c_assign(r, lt), 10)
    note = "encoder (whole frame)" if n == 16384 else "decoder (one group of 16 channels)"
    print(f"rows={n:6d}  torch {t_t:6.2f} ms | C {t_c:6.2f} ms | {t_t / t_c:4.1f}x   [{note}]")

print("\nPASS" if mismatches == 0 and bad == 0 and tie_bad == 0 else "\nFAIL - see mismatches above")
