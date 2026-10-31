# Handoff for Aditya: Phase B worked, here is what's left

From Yuvraj's session, 31 Oct 2026. This replaces the 24 Oct note: you ran that QAT gate
(+46.5%, `cd4c35e`), and Yuvraj then ran Phase B. Short version: **rate-aware training on the
unit step works**, the intra gate can now score it, and what's left needs a GPU.

## Where things stand

| Stage 1 model | intra BD-rate vs x264 all-intra (DAVIS TEST, sequence_mean/psnr) |
|---|---|
| deployed 593k codec | +176.2% |
| run2, 64-ch | +87.9% |
| run4, 64-ch QAT (your gate) | +46.5% |
| **run5, Phase B lambda sweep** | **-12.3%** (first time below x264) |

### The official TEST gate (run on Yuvraj's CPU, 31 Oct)

`outputs/stage1_vimeo/colab_run5_phase_b/gate_phase_b_unitstep.json`: 9 sequences, 719 frames,
400 TRAIN calibration frames, `run valid: True`, decode-exact on every point.

| lambda (step 1.0) | bytes | bpp | PSNR | MS-SSIM | clipped values |
|---|---|---|---|---|---|
| 1e-4 | 5,350,268 | 0.908 | 33.431 dB | 0.9869 | 581 |
| 3e-4 | 4,044,930 | 0.687 | 32.830 dB | 0.9840 | 467 |
| 1e-3 | 2,752,733 | 0.467 | 31.410 dB | 0.9751 | 1,468 |

(clipped out of 11.8M latent values per point, at most 0.012%)

| BD-rate of the sweep against | PSNR | MS-SSIM |
|---|---|---|
| x264 all-intra, sequence_mean (primary) | **-12.3%** | -19.4% |
| x264, the other three conventions | -13.3% / -11.7% / -2.4% | -19.5% / -19.4% / -19.5% |
| run4 QAT, same TEST frames | -31.5% | -32.0% |
| bmshj2018-factorized (committed anchor curve) | **+17.6%** (was +90.9%) | +4.2% |
| bmshj2018-hyperprior | +36.8% | +13.6% |

All eight x264 conventions are negative. `global_pooled/psnr` is the weakest at -2.4%, so the
honest headline is "at or below x264 all-intra", not a fixed margin. The overlap with x264 is
only 31.4-32.9 dB from three points, which is the other reason for task 1.

## What Phase B produced

`colab_train_stage1.ipynb` with `RUN_PHASE_B = True`, `QUANTIZER = 'unit_step'`, `UNIT_STEP = 1.0`,
`RATE_LAMBDAS = [1e-4, 3e-4, 1e-3]`, each lambda warm started from run4's `best.pt`, chunks 1-7.
All three finished, no NaN; lambda 1e-4 was interrupted overnight and resumed after the fix in
`c0c4cbd` (the resume guard was dropping the `quantizer`/`unit_step` fingerprint keys).

Checkpoints are in Yuvraj's `outputs (2).zip`; in the repo layout they go to

```
outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda{1e-4,3e-4,1e-3}/checkpoints/{best,latest}.pt
```

(`history.json` for each is committed; `*.pt` is gitignored as usual).

### Real rate, measured on VAL-B (400 TRAIN frames fit, 80 VAL-B frames report, TEST untouched)

Same harness as `scripts/m24_offline_gate.py`, whose run4 numbers it reproduces to the third
decimal. Ideal bits under a fitted per-channel Laplace, decoded-pixel PSNR.

| | bpp | PSNR | MS-SSIM |
|---|---|---|---|
| run4, 3-bit grid (today's codec) | 0.469 | 28.09 dB | 0.953 |
| run4, 4-bit grid | 0.730 | 31.12 dB | 0.982 |
| run4, 5-bit grid | 0.987 | 32.20 dB | 0.990 |
| **lambda 1e-3**, step 1.0 | 0.471 | 30.82 dB | 0.981 |
| **lambda 3e-4**, step 1.0 | 0.692 | 32.10 dB | 0.989 |
| **lambda 1e-4**, step 1.0 | 0.915 | 32.61 dB | 0.991 |

- BD-rate of the sweep against run4's grid curve: **-28.9%**; against run4 with plain unit-step
  rounding: **-21.4%**, so most of it is the training, not the quantizer swap.
- At equal bits (0.47 bpp) lambda 1e-3 is **+2.7 dB** over the 3-bit codec.
- PHASE_B.md's pre-stated failure test "lambda moves the operating point by less than ~10%" is
  passed by a wide margin: real bits 0.92 -> 0.47 bpp (-49%); latent std 9.2 -> 2.2.
- Not settled: the 32.6 dB ceiling. The best point is 32.72 dB, but no lambda was aimed at high
  rate, so the high end is untested (task 1 below).

## New: the intra gate scores unit-step models

`scripts/benchmark_intra_gate.py --quantizer unit_step` codes every `--checkpoint` at every
`--unit-steps` value (default 1.0) and treats them all as ONE curve, which is how PHASE_B.md says
to gate a sweep. Real `.nvct` bytes through the real range coder, decode checked bit-exact.

- The unit step is the codec's own affine grid with `scale = step` and an integer zero point
  (`calibration.unit_step_quantization_params`), so there's no format or coder change. Only the
  alphabet is fitted on TRAIN; no percentile clipping. TEST values beyond the TRAIN span are
  clamped and reported per point as `clipped_values`.
- Tests: `tests/test_unit_step_calibration.py`, plus six new cases in `tests/test_benchmark_intra_gate.py`.
- Smoke result (1 sequence, 5 frames, 40 calibration frames): **-19.4% vs x264 all-intra**,
  decode-exact on every point. Too small to quote; the full TEST run is what counts.

## What you need to do (GPU)

### 1. Extend the sweep at both ends (about 9 hours)

Each lambda took about 4-4.5 hours on JupyterLab. The current curve overlaps x264 only between
~30.8 and ~32.6 dB, and the high end is where the ceiling question lives. In the notebook's
configuration cell:

```python
RUN_PHASE_B  = True
QUANTIZER    = 'unit_step'
UNIT_STEP    = 1.0
RATE_LAMBDAS = [3e-5, 3e-3]     # ONLY the new ones; the three finished runs stay as they are
```

and keep the same `OUTPUT_DIR` contents next to the notebook (run4's `checkpoints/best.pt` is the
warm start; Phase A skips itself). Each lambda lands in its own `phase_b_unitstep_lambda<tag>`
folder. If a run dies, Run All resumes it from its own `latest.pt` since `c0c4cbd`.

### 2. Gate all five lambdas together (about 10 minutes on your GPU)

```bash
.venv/Scripts/python.exe scripts/benchmark_intra_gate.py --quantizer unit_step --checkpoint outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda3e-5/checkpoints/best.pt outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda1e-4/checkpoints/best.pt outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda3e-4/checkpoints/best.pt outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda1e-3/checkpoints/best.pt outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda3e-3/checkpoints/best.pt --output-dir outputs/stage1_vimeo/colab_run5_phase_b --output-name gate_phase_b_5lambda.json
```

Check `run valid: True`, decode-exact on every point, `clipped_values` small relative to the
latent count, and all eight conventions moving the same way, as you did for run2 and run4.
Then compare against bmshj2018 without rebuilding CompressAI: the committed
`outputs/benchmarks/compressai_anchor/compressai_anchor.json` holds its TEST curve, and
`benchmark_parity.bd_rate` on the two curves is the number PHASE_B.md set a target for
(+90.9% before Phase B, +17.6% with the three-lambda sweep; the factorized model sits at
-23.3% vs x264).

### 3. Then decide on the video path

Every number here is intra only. The full-video BD-rate still uses the 593k model, because the
P-frame stack (M10K entropy model, M11-G16 context model, K=512 codebooks, M14 motion table,
M23's scale map) is fitted to the old latent. PHASE_B.md estimates a third refit at about three
hours and says not to spend it until the intra numbers justify it. At -31.5% against run4 on
TEST, they now do. One extra thing this refit has to handle: the residual side
also has to move to the unit step (M24 measured the residual grid saturating the same way).

## Repo state on Yuvraj's side

- Branch `c-motion-search`; the trainer resume fix is also on `master` (`a72508a`).
- Uncommitted on his side, not part of this work: demo code (`src/nvc/demo/`, `scripts/*_demo.py`,
  their tests) and paper material in `docs/`.
- `tests/test_m18_intra_quantizer.py::test_m18_scripts_do_not_modify_any_production_source`
  fails whenever `src/nvc/` has uncommitted edits; it's a working-tree check, not a bug.
