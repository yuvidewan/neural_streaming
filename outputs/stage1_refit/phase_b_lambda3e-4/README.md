# Phase B pilot (lambda 3e-4): video refit

Pilot of `docs/HANDOFF_Aditya_video_refit.md` against
`outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda3e-4/checkpoints/best.pt`
(epoch 24, ResidualGDNAutoencoder).

## Step 1: residual quantizer, decided 2026-10-31: unit step

`m24_residual_gate.json` / `.log`, default settings (400 TRAIN frames, VAL-B 4 sequences,
GOP 10), walk check IDENTICAL, deployed grid verified. Reference step (median 4-bit
residual step) 0.66502, a quarter of run4's 2.548, as expected from a latent ~3x smaller.

| comparison on VAL-B | BD-rate |
|---|---|
| unit_step_laplace vs grid_histogram (deployed) | **-21.03%** |
| unit_step_laplace vs grid_laplace (same density model) | **-24.31%** |
| grid_laplace vs grid_histogram (density model alone) | +4.33% |

On run4 the same gate gave -8.31% for the grid rule and the grid won at 3-bit. Here the unit
step wins across the whole range, so **residuals go to the unit step**. Nearest swept point to
the trained step: 0.99754, 0.449 bpp, 32.093 dB, residual span +-33 (a 7-bit alphabet).

The grid still saturates the way M24 found on run4: 31.0 / 32.0 / 32.4 / 32.5 / 32.5 dB from
3 to 7 bits.

## Consequence for the plumbing

- Intra at step 1.0 needs **8 bits** (`gate_phase_b_unitstep.json`), residuals 7. The `.nvct`
  header has ONE `quantization_bits` for both params blocks (`src/nvc/video/container.py`),
  so the stream's alphabet width is 8 and the residual tables are 256 wide.
- Every refit script is keyed by `bits` and builds `2 ** bits` alphabets through
  `m10h_motion_compensation.calibrate_grids`, which only knows the percentile grid. The unit
  step has to enter there (and in every calibration cache key) before M10K is refitted.

## Steps 3-4: the refit and the full-video number, 2026-10-31

Plumbing: `scripts/unit_step_grids.py` gives every refit script the quantizer the checkpoint
records (`extra.quantizer`), by running `calibrate_grids` unmodified with its fitting primitive
swapped. `m10h_motion_compensation.py` is untouched, so `m22_residual.code_identity()` is too.
Percentile checkpoints get byte-identical calibration (`tests/test_unit_step_grids.py`).
Stream width at step 1.0: 8 bits (intra 8, residual 7).

| step | dir | result | run4 refit |
|---|---|---|---|
| M10K | `m10k_phase_b_lambda3e-4` | **GATE NOT PASSED**: -0.43% vs M10J (1.94256 vs 1.93419 bits/symbol); kept as plumbing, M10L/M11 build on it | +1.14 / +1.35 / +2.22% |
| M10L | `m10l_phase_b_lambda3e-4` | passed, K=512, +0.01% vs M10K, table time -98.8%, 478/512 used | K=512, +0.03-0.05% |
| M11 offline | `m11_offline_phase_b_lambda3e-4` | passed, `neighbourhood_x_channel`, net +6.79%, controls 0.000% | +1.77 / +4.97 / +8.03% |
| M11 trained | `m11_phase_b_lambda3e-4` | G=16, **+3.74%** coded bytes vs M10L, round trip exact (11/11) | +5.35 / +5.95 / +5.32% |
| M13 | `m13_phase_b_lambda3e-4` | +6.32%, meaningful | +0.96 / +2.61 / +4.20% |
| M14 | `m14_phase_b_lambda3e-4` | motion +12.49% meaningful, intra +0.28% weak (rejected again) | motion +11.6-13.0% |

Two harness notes, both in `m10l_offline_gate.py`:
- its zero-loss check required the codebook SEARCH to reproduce M10K's payload byte for byte.
  At 256 symbols, integer frequencies make ~1 position in 16,384 per frame a near-tie
  (another table cheaper by ~1e-6 bits), which can flip one byte. Moves that are near-ties
  (< `NEAR_TIE_BITS` = 1e-4 bits) are now accepted and counted; anything else still stops.
- the L1 metric comparison materialises [16,384 x 512 x 256] floats (8 GB) at 8 bits and ran
  out of GPU memory, so this run used `--metric-comparison-bits 0`: `code_length` only, which
  is what run4 and the deployed codec both selected.

### Full video: `outputs/benchmarks/parity_phase_b_lambda3e-4/`

`benchmark_parity.py`, full DAVIS TEST (9 sequences, 719 frames), GOP 10, decode-exact, run
valid. One model is one point: **0.4644 bpp, 32.699 dB PSNR, 0.98386 MS-SSIM** (sequence_mean).

Bits needed relative to each curve at the SAME quality (`rate_ratio_at_quality`):

| against | sequence_mean PSNR | sequence_mean MS-SSIM | global_pooled PSNR |
|---|---|---|---|
| **run4 refitted codec** | **0.651x (-34.9%)** | 0.722x (-27.8%) | 0.641x (-35.9%) |
| x264 default | 2.29x | 1.97x | 2.99x |
| x265 default | 2.44x | 1.94x | 3.16x |
| x264 low-delay | 1.68x | 1.45x | 2.15x |
| x265 low-delay | 1.83x | 1.41x | 2.35x |

**The handoff's failure condition is not met: the pilot needs a third fewer bits than run4's
refitted curve at matched quality.** The intra gain reaches the stream, and is larger than the
-31.5% intra gate figure, because the unit step also wins on residuals (step 1). Still about
2.3x x264's bits at this quality, so not parity. A point ratio, not a BD-rate: the BD-rate
needs the other two lambdas (step 5, running).

## Step 5: all three lambdas, 2026-11-04

`scripts/run_phase_b_refit.sh 1e-4` and `1e-3`, same chain, same settings. Before the 1e-4
M11-G1 round trip could pass, the residual encoders had to assign codebook prototypes group by
group like the decoder (`assign_by_group`, commit b3444a0); that changes no byte the old encoders
got right, so the lambda 3e-4 numbers above stand.

| step | lambda 1e-3 (7-bit stream) | lambda 3e-4 (8-bit) | lambda 1e-4 (8-bit) |
|---|---|---|---|
| M10K | GATE NOT PASSED, kept | GATE NOT PASSED, kept | GATE NOT PASSED, kept |
| M10L | K=512, -0.04% vs M10K | K=512, +0.01% | K=512, -0.02% |
| M11 offline | net +8.51% | net +6.79% | net +4.39% |
| M11 trained G16 | +4.91% bytes, exact | +3.74%, exact | +3.11%, exact |
| M13 | +7.05% | +6.32% | +5.53% |
| M14 motion / intra | +11.48% / +0.46% weak | +12.49% / +0.28% weak | +13.35% / +0.22% weak |

Full video (`outputs/benchmarks/parity_phase_b_lambda*/`, full DAVIS TEST, GOP 10, all three
runs valid and decode-exact):

| lambda | bpp | PSNR | MS-SSIM |
|---|---|---|---|
| 1e-3 | 0.2955 | 31.375 | 0.97466 |
| 3e-4 | 0.4644 | 32.699 | 0.98386 |
| 1e-4 | 0.6486 | 33.316 | 0.98677 |

3-point BD-rate (`benchmark_parity.bd_rate`; positive = Phase B needs more bits). The run4 row
uses run4's `nvc_deployed` curve from `parity_refit/parity_64ch_qat.json`, which this function
reproduces at +214.2% vs x264.

| against | seq-mean PSNR | seq-mean MS-SSIM | global-pooled PSNR |
|---|---|---|---|
| **run4 refitted codec** | **-31.1%** | -27.7% | -30.4% |
| x264 default | **+127.1%** (run4: +214.2%) | +106.7% | +187.4% |
| x265 default | +141.9% (run4: +233.7%) | +102.2% | +204.0% |
| x264 low-delay | +65.5% | +50.8% | +106.2% |
| x265 low-delay | +80.5% | +47.6% | +125.3% |

**Rate-aware training plus the unit step takes the full video codec from +214.2% to +127.1%
vs x264 (-87 points), a 31% bit saving over run4 at the same quality.** It still needs about
2.3x x264's bits, so this is not parity. The overlap with run4 is 31.4-32.9 dB. M10K failed its
gate on all three (its learned model is about M10J's quality) and was kept as plumbing. M14's
intra table was again too weak to ship.
