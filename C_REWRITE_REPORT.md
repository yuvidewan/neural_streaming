# Where a C/C++ rewrite would pay: measured profile, two prototypes, and an integration plan

**Status: research only. No production code changed** (`src/nvc/` is untouched). Written
2026-09-30 by Yuvraj (with Claude) as a hand-off to Aditya. Everything below is either
measured by the scripts in [`docs/c_rewrite_prototypes/`](docs/c_rewrite_prototypes/) or
explicitly labelled as an estimate or as read from the code.

The range coder is the one place already rewritten in C (Milestone 8B). The question was:
*where else would a rewrite help?* Answer, in one paragraph: **two places, both in
`nvc.video`, and nothing else is worth it.** Block-motion search is **85% of encode time**;
codebook assignment is **~47% of decode time**. Both were prototyped in AVX2 C and checked
against the current PyTorch code. The range coder that motivated the exercise is now only
0.2-3% of a frame.

> **STATUS UPDATE (branch `c-motion-search`, not merged): step 1 is implemented.** The native
> motion-search kernel below is built into `nvc.video` and tested; the projections in section 3
> are replaced by measurements. **Measured end to end on a CPU-only laptop: encode 463 -> 96
> ms/frame (4.8x)**, motion search 85% -> 15% of encode, **0 mismatches on 720 real DAVIS pairs**
> plus 34 tests and six mutation-based negative controls (see CHANGELOG 2026-09-30 "Native
> block-motion search"). **Still open, and only you can do them:** run
> `scripts/verify_promoted_codec.py` with `NVC_MOTION_BACKEND=torch` and `=native` (both must
> report 54/54 byte-identical; needs the bundles), run the profile on your GPU, and see CI's first
> Linux result. **Step 2 (codebook assignment) is not started**; it is now the top encode cost (44%).

---

## 0. For Aditya: what to do, in order

1. **Reproduce on your machine first (about 30 minutes, no code changes).** From the repo
   root, with the project venv:
   ```
   python docs/c_rewrite_prototypes/verify_kernels.py      # needs gcc + a CPU with AVX2/FMA; prints PASS
   python docs/c_rewrite_prototypes/profile_codec.py       # CPU profile (needs no codec bundles)
   python docs/c_rewrite_prototypes/profile_codec.py --device cuda   # YOUR profile on the GPU
   ```
   The profile in this report is from a **CPU-only laptop**. Yours is different: the
   networks and the assignment matmul are fast on a GPU, and the motion loop is probably
   kernel-launch-bound there instead. **The `--device cuda` path is untested** (no GPU on
   the author's machine); it calls `torch.cuda.synchronize()` around each stage, so check
   that first if the numbers look odd. **If motion search is not the dominant encode cost on
   your GPU, stop here and skip step 2.**
2. **Motion-search kernel** (section 3) - recommended first; encoder-only, so the safest.
   The full checklist is in section 6.
3. **Codebook-assignment kernel** (section 4) only if decode speed matters to you, and only
   after reading section 5. It is riskier: the decoder must reproduce the encoder's
   assignment exactly.
4. **Do not spend time on** anything in section 7's "measured, not worth it" table.

**Priority against the roadmap, stated plainly.** `PARITY_ROADMAP.md` rule 6 asks *what does
this change about the BD-rate against H.264?* Answer: **nothing.** Both kernels are meant to
give byte-identical streams. They shorten experiment turnaround and nothing else. Stage 1
training is the critical path. And the roadmap lists the K=512 codebook and G16 context model
under "Retire", and Stage 3 replaces block search with learned flow, so the assignment kernel
in particular may have a short life. The motion kernel keeps its value for the Stage 0
scoreboard and every baseline you re-run. Do this only when the evaluation loops are slowing
you down.

---

## 1. Where the time goes

Measured on: Intel i5-8350U (4 cores/8 threads, AVX2+FMA, **no GPU**), PyTorch CPU with 4
threads, `nvc.video.VideoCodec`, 256x256 frames (the size the Stage 0 scoreboard uses),
real DAVIS frames (`bear`), 64-channel latent, 4-bit, GOP 10, block 16, search range 16,
channel group 16, 512-entry codebook, **random weights** (timing depends on tensor shapes,
not weight values). 10 frames = 1 I-frame + 9 P-frames. Two full runs:

| Stage | Encode share | Decode share |
|---|---:|---:|
| **Block-motion search** (`estimate_block_motion`) | **84.6-85.1%** | - (decoder does not search) |
| **Codebook assignment** (`SharedCodebook.assign_tensor`) | 7.1-7.5% | **46.3-47.2%** |
| Context-model network (`log_probabilities`) | 0.8% | 22.3-24.3% |
| Autoencoder decode | 3.2-3.3% | 10.8-12.3% |
| Autoencoder encode | 2.3% | 6.7-7.0% |
| Context planes | 0.1% | 3.3-3.7% |
| Range coder (C, already done) | 0.2-0.3% | 3.0-3.2% |
| `warp_blocks` | 0.2-0.9% | 1.4% |
| Everything else (quantiser, payload packing) | under 0.3% | under 0.5% |
| **Absolute time** | **781-1070 ms/frame** | **118-170 ms/frame** |

The **shares are stable** between the two runs (within about 1.5 points); the absolute times
moved by up to 35% with laptop load. Quote the shares, treat milliseconds as a range.

Isolated, single stage (median, ms): `estimate_block_motion` 610-985, `assign_tensor` (16384
rows) 52-84, autoencoder encode 8-12, decode 14-18, context network (64 channels) 8-10,
`encode_symbols` 2.7-3.0, `decode_symbols` 2.8-4.3, `warp_blocks` 1.3-2.4, `latent_to_symbols`
0.5-1.0.

The 8B comment that the coder was the bottleneck is out of date for a reason worth stating:
**the C coder worked so well that it is now ~3 ms of a 118-170 ms decode.**

---

## 2. Method and what was checked

Both kernels are in [`docs/c_rewrite_prototypes/kernels.c`](docs/c_rewrite_prototypes/kernels.c),
built with `-O3 -mavx2 -mfma -ffp-contract=off` (never `-ffast-math`; contraction is turned
off so the compiler cannot silently fuse a multiply and an add). [`verify_kernels.py`](docs/c_rewrite_prototypes/verify_kernels.py)
compares each against the current implementation and prints PASS/FAIL.
[`profile_codec.py`](docs/c_rewrite_prototypes/profile_codec.py) produces the table above.

---

## 3. Candidate 1: block-motion search - `nvc/video/motion.py:58`

**Why it is slow.** `estimate_block_motion` is a Python double loop of **33 x 33 = 1,089
iterations per frame**. Each iteration runs a handful of full-frame tensor operations (subtract,
abs, channel-sum, avg-pool, then compare and four `torch.where` updates over the block grid).
From reading the code that is roughly 15-20 small tensor ops per iteration, i.e. on the order of
16-22 thousand tensor-op dispatches per frame (an estimate from the code, not measured).

**Prototype: `nvc_block_search`.** One SIMD lane per candidate vector (8 candidates at once), and
every lane sums its 16x16 window in exactly the order torch does. Early exit skips a group of
candidates once every lane is strictly worse than the best so far (ties are never skipped).

| | torch (current) | C AVX2, no early exit | C AVX2, early exit |
|---|---:|---:|---:|
| ms per frame (range over 4 runs) | 610-985 | 39-68 | 24-35 |
| speed-up | 1x | 14-16x | **25-31x** |

Single-threaded, and the C figures include the padding and copying done from Python.

**Same answers.** 0 mismatches over **24 frame pairs x 2 modes** (early exit on and off): 18
real DAVIS pairs and 6 adversarial cases - identical frames, a flat gray frame (every candidate
ties), an all-zero frame, two unrelated noise frames, a known (3, -5) shift, and a 4-level
quantised pair with many exact ties. That is a test of 24 pairs, **not** a proof; section 6
gives the gate that would make it one.

**Why this one is the safe one.** Only the *encoder* searches; the vectors are transmitted, so
the decoder never runs this. If a rare tie ever resolved differently, streams would still
decode correctly - the only thing that could change is byte-identity with old streams.

**Effect - projected, then measured.** *Projected* (Amdahl, from the shares and the prototype
speed-ups): 140-190 ms/frame with this kernel alone (~5.5x), and about 100-130 ms/frame (~8x) once
the assignment kernel of section 4 is also in. *Measured after integrating step 1* (branch
`c-motion-search`, idle laptop, 4 threads): **463 -> 96 ms/frame, 4.8x**; the motion stage went from
437 to 16 ms per P-frame (27x) and from 85% to 15% of encode. The measured ratio is a little under the
projection, and both runs were made on a quieter machine than the earlier profile, so compare the
ratio and the shares, not the absolute milliseconds with section 1's. The ~8x figure for both kernels
is still a projection. For scale: a full DAVIS TEST encode has roughly 650
P-frames; at 0.6-1.0 s each that is **about 7-10 minutes of pure motion search per rate point on
this laptop, versus roughly 15-25 seconds** with the kernel. **28 scripts** reference
`estimate_block_motion` through their own frozen copy (`scripts/m10h_motion_compensation.py:182`
and its users), so every research experiment pays this today.

**Not yet in the prototype (needed for production):** scalar fallback + runtime CPU check,
multi-threading, input validation. Section 6.

---

## 4. Candidate 2: codebook assignment - `nvc/video/entropy.py:210` (`assign_tensor`)

**Why it is slow.** For each of N = 16,384 positions it computes a 512-way argmin, and to do so
builds the full **[16384, 512] float matrix (33 MB)** with `probabilities @ log2_costs.T`, then
makes several more full passes over it in `_torch_argmin_lowest_index` (min, argmin, compare,
sum, nonzero). It is memory-bound, not compute-bound. In the decoder it is called once per
channel group (4 times per frame at G=16) - which is why it is **~47% of decode**.

**Prototype: `nvc_assign_argmin`.** Fused: computes each row's 512 costs in registers and keeps a
running (min, lowest-index) - the 33 MB matrix never exists. Four independent multiply-add chains
are interleaved to hide FMA latency (a first version without that was only 2x).

| | torch (current) | C AVX2 | speed-up |
|---|---:|---:|---:|
| encoder-size call (16,384 rows) | 52-84 ms | 12.5-24 ms | **3.4-4.8x** |
| decoder-size call (4,096 rows, one group) | 13-23 ms | 3-4.3 ms | **4.4-5.5x** |

Single-threaded. Row-parallelism (not measured) should add more.

**Same answers:** 0 mismatches over **393,216 rows** (codebook distributions from soft to very
peaked). The kernel's own tie rule (lowest index on exactly equal cost) is checked against
numpy's first-occurrence argmin of the kernel's own costs: **0 wrong of 40,960 rows** across
duplicated-prototype codebooks.

**Projected effect** (estimate): decode goes from 118-170 to about 76-106 ms/frame (**~1.6x**);
encode gets a further ~7% share removed. Combined with candidate 1, encode is **roughly 8x**
faster. If you also fix the redundant network passes (section 7), decode approaches 2x.

**This is the riskier kernel. Read section 5 before touching it.**

---

## 5. Findings about bit-exactness (read before candidate 2)

The decoder must compute *the same* assignment the encoder did, or every later symbol is parsed
with the wrong table. So "same answers on my test data" is not enough; float behaviour matters.
Three measured facts (all reproduced by `verify_kernels.py`):

1. **Two perfectly valid float32 summation orders do not reproduce torch's costs bit for bit.**
   Across 196,608 rows x 512 prototypes, a fused multiply-add chain differs from torch's matmul
   in **0.7%** of cost entries (0 argmin flips in 196,608 rows); a separate multiply-then-add
   order differs in **22.8%** of entries and **flipped 38 of 196,608 argmins (0.019%)**. So a
   different-but-valid arithmetic order *can* change an assignment near a tie. The M20 report's
   own finding (the median position prefers its prototype by only 0.003 bits) is why near-ties
   are not rare enough to ignore.
2. **Torch's own matmul is position-dependent at the last bit.** With duplicated (identical)
   prototypes, torch gives *different* costs for the copies whenever one of them sits in the
   **last 8-column tile (k = 504-511)** - about 47% of rows for the affected pairs - while
   every other position tested (offsets 1, 3, 8 and 256) gives bit-equal costs. So torch has no exact tie there, and
   `_torch_argmin_lowest_index`'s lowest-index rule is never reached for those pairs. This is
   why the harness's "kernel vs torch on exact-tie codebooks" line shows 2,353/40,960 differing:
   **torch is not a valid oracle for exact ties**, so the pass/fail check for ties compares the
   kernel against numpy's argmin of the kernel's own costs instead. (Real trained codebooks
   have no duplicate prototypes; this is an edge case, but it demonstrates the mechanism.)
3. **Implication.** The current assignment depends on which BLAS library and which CPU (or
   GPU) computed it, at the last-bit level. The project has only ever encoded and decoded in
   the same process/machine (`verify_promoted_codec.py`, the tests), so **cross-machine
   encode/decode has not been shown to be safe** - I did not test it (no second BLAS/GPU
   available), it is a hypothesis this evidence makes worth testing. A fixed-order C kernel would
   not depend on the library. That is a possible *benefit* of candidate 2, but it only exists if
   the change is introduced as a proper versioned change (section 6, step 2c).

---

## 6. Exact checklist

### Step 1 - motion-search kernel (recommended first)

a. **Kernel.** New `src/nvc/video/_native/block_search.c`, from `nvc_block_search`. Add (i) a
   **scalar path** with identical semantics (each candidate's window summed in row-major order,
   `((a0+a1)+a2)` per pixel), (ii) a **runtime CPU check** (`__builtin_cpu_supports("avx2")`,
   or `__attribute__((target("avx2")))` on the SIMD function) so a binary built with generic
   flags never executes an unsupported instruction, (iii) argument validation (dimensions
   divisible by the block size, non-null pointers). Keep `-ffp-contract=off`; no `-ffast-math`.
b. **Loader.** `nvc/compression/_native/__init__.py` is hardwired to one library: `_SOURCE =
   range_coder.c`, `_binary_path()`, and a `load()` that sets argtypes for the coder's seven
   functions, compiled with plain `gcc -O3 -shared`. Either generalise it to a small
   `load(name)` registry or add a sibling loader for `nvc/video/_native/`, reusing
   `_select_compiler()` (it already handles the 32-vs-64-bit MinGW trap). The new library needs
   `-mavx2 -ffp-contract=off` (candidate 2 also needs `-mfma`).
c. **Packaging** (the BUG-02 lesson): add the new `.c` to `package-data` in `pyproject.toml` and
   confirm it is in a built wheel; add `src/nvc/video/_native/*.dll` to `.gitignore` (line 15
   covers only `compression/_native`).
d. **Python side.** Keep `estimate_block_motion` (torch) exactly as it is - it is the reference
   for the tests and the fallback. Add a native variant and a dispatcher. Unlike the range
   coder, **correctness of decoding never depends on the native path**, so a fallback is
   legitimate here.
e. **Threads (optional, measure it).** Give the kernel a `[block_start, block_end)` range so
   Python can run chunks in a thread pool: ctypes releases the GIL during the call, and this
   avoids an OpenMP/libgomp runtime dependency on the loaded DLL.
f. **Tests** (negative-controlled: show each fails against a deliberately broken kernel).
   Reuse the cases in `verify_kernels.py` (real-ish pairs, flat, exact ties, quantised ties,
   known shift, all-zero), early exit on and off, scalar path vs SIMD path, dispatcher.
   `tests/test_video_codec.py` already pins `nvc.video` against the frozen research script;
   keep that passing.
g. **Merge gates.** (1) full `pytest` green; (2) **`scripts/verify_promoted_codec.py` reports
   54/54 streams byte-identical on all of DAVIS TEST, run twice** - once with
   `NVC_MOTION_BACKEND=torch` and once with `NVC_MOTION_BACKEND=native` (PowerShell:
   `$env:NVC_MOTION_BACKEND="native"; python scripts/verify_promoted_codec.py`). This is what
   turns the 720-pair check into proof; it needs the codec bundles, which only you have (they are
   gitignored); (3) the profile harness on your machine shows the speed-up; (4) a CHANGELOG
   entry with the numbers.

   **Status on branch `c-motion-search`:** a-f done (kernel with scalar + AVX2 paths and run-time
   dispatch, loader, packaging, dispatcher, threads, tests with mutation controls). Gate (1) is run
   before each commit; (2) is **not** done - it is yours; (3) measured on a CPU-only laptop
   (463 -> 96 ms/frame), GPU not measured; (4) done.

### Step 2 - codebook-assignment kernel (only if decode speed matters)

a. Read section 5. Decide up front how you will treat a single mismatch.
b. **Acceptance test on REAL rows, not synthetic ones.** Dump the `[rows, 16]` probability
   matrices the real context model produces on DAVIS TEST for the six bundles (deployed and M22,
   5/4/3-bit) and compare the kernel's argmin with the current path. **The rule: zero mismatches
   over all rows -> it is a pure implementation swap. Any mismatch -> it is a format change.**
   Synthetic rows (this report) gave 0 in 393,216, but the codebook used there is random.
c. **If it is a format change** (or if you want the cross-machine determinism benefit), fold an
   assignment-version marker into the identity hash (`SharedCodebook.codebook_id` /
   `model_identity`) so an old stream against a new decoder is *refused*, not silently
   mis-decoded - the same pattern as the M14 identity-check fix. Re-verify all 54 streams.
d. Encoder and decoder must switch **together**. The decoder calls it per group (4,096 rows), the
   encoder once per frame (16,384 rows); both sizes are covered by the harness.
e. Exact ties: keep the lowest-index rule, and test it against numpy's argmin of the kernel's own
   costs (as `verify_kernels.py` does) rather than against torch (section 5, fact 2).
f. Weigh it against the roadmap: the K=512 codebook is on the "Retire" list.

### Step 3 - GPU

Profile first (step 0). If the motion loop is launch-bound on your GPU, the CPU kernel does not
help; the equivalent is a CUDA kernel or a single batched torch formulation. That is a separate
investigation and has not been started.

---

## 7. Other findings

**A real cost, but not a C rewrite - redundant network passes in the decoder.**
`decode_residual_frame` (`nvc/video/entropy.py:285`) evaluates the context network for **all 64
channels on every group** and keeps only the 16 it needs. That is ~4x redundant work: the
context network measured 29-46 ms per decoded frame, so roughly 22-34 ms of it is wasted (an
estimate from the structure, not a measurement of a fix). Fixing it needs the model to accept a
channel slice, and it changes the batch shape, and the Stage 1 commit already recorded that
convolutions give last-bit-different results at different batch sizes (~7e-9). Given section 5,
that must be checked with a full round trip before it is trusted.

**GDN layers - no evidence for a custom kernel.** On CPU, one forward+backward of the Stage 1
model (batch 1, 256x256) was ~1.3 s; **convolutions were ~88% of profiled time and all
elementwise ops (the GDN pieces, LowerBound, adds) ~5%**. GPU shares are unmeasured (there is no
GPU here) and will differ; profile on yours before writing anything. If the elementwise share
is large there, try `torch.compile` on Colab/Linux first (it was blocked on the author's
Windows CPU, so untested).

**Measured or read, and not worth rewriting:**

| Component | Evidence |
|---|---|
| Range coder | 2.7-3.0 ms encode, 2.8-4.3 ms decode per frame of 16,384 symbols; already C |
| `warp_blocks` | 1.3-2.4 ms (already vectorised indexing) |
| Quantiser (`latent_to_symbols`) | 0.5-1.0 ms |
| Context planes | 0.9-1.2 ms |
| Autoencoder / context forward passes | already run in oneDNN/MKL; only hardware (GPU) helps |
| Frame/PNG loading | `image_io.py` decodes with OpenCV's native code already; whether training is loader-bound on the RTX 5060 was **not measured** |
| `.nvc/.nvct` header packing loops | tens of microseconds; the per-frame `.nvc` param loop is not on the video path |
| Calibration, entropy-table construction, bundle-identity hashing | offline / one-time |

---

## 8. Limits of this study

- **CPU only, one laptop.** The i5-8350U timings vary ~35% run to run. The GPU profile is
  unknown, and it is what your machine will see.
- **Random weights.** Fine for timing (cost depends on shapes), but symbol statistics are not
  the trained model's; the early-exit rate in motion search depends on the *frames*, not the
  weights, and was measured on real DAVIS frames.
- **One sequence** in the end-to-end profile (`bear`, first 10 frames); 6 sequences in the motion
  equality check.
- **256x256 only.** The Stage 0 README warns nothing was measured at larger resolution; motion
  search cost scales with blocks x candidates.
- **The projections are Amdahl estimates** from measured shares and measured kernel speed-ups,
  not end-to-end measurements of an integrated kernel. Nothing was integrated.
- **Prototypes are single-threaded, AVX2-only, unvalidated inputs.** They are reference code,
  not the deliverable.
- **The "position-dependent BLAS" and "cross-machine risk" points** are demonstrated on this
  machine's torch/MKL only; other libraries were not tested.
