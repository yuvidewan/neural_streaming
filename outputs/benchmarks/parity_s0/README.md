# Stage 0 scoreboard - NVC vs H.264/H.265, single pass

`PARITY_ROADMAP.md` Stage 0 gate: *one reproducible BD-rate number against H.264.*
**Met.** Run 2026-09-16, `scripts/benchmark_parity.py`, 32 minutes, `valid: true`.

## Headline

Primary convention, fixed in the script's docstring before any result was seen:
pooled BPP, per-frame quality averaged per sequence and then over the 9 sequences.
Positive BD-rate means NVC needs more bits than the classical codec for the same quality.

| NVC (M22 grid) vs | BD-rate, PSNR | BD-rate, MS-SSIM | bits NVC spends at its 3 / 4 / 5-bit points (PSNR) |
|---|---:|---:|---|
| **H.264, default** | **+450.7%** | **+378.3%** | 4.76x / 5.71x / 7.47x |
| H.265, default | +485.0% | +371.5% | 5.07x / 6.07x / 7.91x |
| H.264, I every 10, no B | +300.5% | +244.7% | 3.49x / 4.14x / 5.40x |
| H.265, I every 10, no B | +336.0% | +244.8% | 3.83x / 4.49x / 5.86x |

The frozen deployed codec (without M22) is +458.5% / +384.3% against default H.264.
M22's residual grid moves the scoreboard by about 8 points of a 450-point deficit.

**What this replaces.** The README's "~4.8x fewer bits" was one point: NVC 3-bit
against H.264 crf33. The single pass confirms that point (4.76x), but the gap is
not constant. It widens with quality, and averaged over NVC's range it is **5.5x
on PSNR (4.8x on MS-SSIM)**.

## Why the gap widens: NVC's curve is flat

| | 3-bit | 4-bit | 5-bit |
|---|---:|---:|---:|
| NVC + M22, BPP | 0.3170 | 0.5043 | 0.7211 |
| PSNR (dB) | 28.31 | 29.46 | 29.82 |
| MS-SSIM | 0.9468 | 0.9676 | 0.9737 |

Across 28.3-29.8 dB, each doubling of rate buys NVC **1.3 dB**, against **2.8 dB**
for H.264 and H.265 over the same quality window. Going from 4 to 5 bits, NVC gains
+0.36 dB for 43% more bits. Finer quantization no longer buys quality: the
decoder is near the ceiling of what this 593k-parameter autoencoder can
reconstruct. That is strong evidence for the roadmap's diagnosis. The binding
constraint is the transform, not the entropy coder, and no amount of entropy
coding moves a ceiling.

Classical reference points (primary convention):

| CRF | H.264 BPP | PSNR | MS-SSIM | H.265 BPP | PSNR | MS-SSIM | H.264 low-delay BPP | PSNR | MS-SSIM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 | 0.1194 | 30.67 | 0.9701 | 0.1323 | 31.29 | 0.9721 | 0.1778 | 30.94 | 0.9719 |
| 30 | 0.0936 | 29.70 | 0.9621 | 0.1032 | 30.31 | 0.9648 | 0.1385 | 29.96 | 0.9643 |
| 32 | 0.0743 | 28.77 | 0.9522 | 0.0805 | 29.34 | 0.9557 | 0.1086 | 29.02 | 0.9548 |
| 34 | 0.0593 | 27.82 | 0.9389 | 0.0634 | 28.36 | 0.9438 | 0.0859 | 28.09 | 0.9428 |
| 36 | 0.0480 | 26.92 | 0.9227 | 0.0499 | 27.46 | 0.9303 | 0.0683 | 27.20 | 0.9281 |

## GOP structure is a minority of the gap

Forcing x264 into NVC's own structure (an I-frame every 10 frames, P-frames only,
no scene-cut keyframes) cuts the PSNR multiple from 5.5x to 4.0x (BD-rate +451% to
+301%). In log terms that is about a fifth of the gap. The remaining **4x at
equal structure** is the transform, motion model and entropy model together.
Longer GOPs and B-frames are worth adopting eventually, but they are not the main
problem.

## Sensitivity: the convention does not change the conclusion

BD-rate of NVC + M22 against default H.264 under all four conventions:

| convention | PSNR | MS-SSIM |
|---|---:|---:|
| **sequence mean** (primary) | **+451%** | **+378%** |
| frame mean (August harness) | +452% | +381% |
| sequence pooled MSE (M21/M22 scripts) | +484% | +378% |
| global pooled MSE | +507% | +381% |

The August comparison mixed two of these: H.264 was scored by frame mean and NVC
by sequence pooled MSE. On NVC's 3-bit point that understated NVC by 0.14 dB. The
effect is real but small next to the gap.

## Why this run can be trusted

- **Same frames, same scorer.** Every arm is scored by one function on uint8
  pixels a real decoder would deliver. NVC's float output is rounded; FFmpeg's
  output already is 8-bit. Bytes are whole files on disk: `.nvct` containers for
  NVC, `.mp4` files for FFmpeg.
- **NVC reproduces M22 exactly.** All six NVC points match M22 Phase 19's recorded
  DAVIS TEST byte totals exactly, and match its float PSNR/MS-SSIM to within
  1e-3 dB / 1e-4. Every stream was decoded back from its container and the
  reconstruction is bit-exact.
- **FFmpeg reproduces August exactly.** H.264 and H.265 at crf 28, the setting
  both runs share, give the same bytes (703,339 / 779,299), the same frame-mean
  PSNR (30.4378 dB) and the same MS-SSIM (0.970296) as
  `vimeo_vs_h264_h265_davis/` from 21 August.
- **BD-rate limits, stated plainly.** NVC has three rate points spanning only
  28.3-29.8 dB (0.947-0.974 MS-SSIM), so every BD-rate here is averaged over that
  window. The classical curves cover it with 13 CRF points each, so none of it is
  extrapolated. BD-rate uses the project's piecewise-linear `_bd_rate_linear`.
- **Resolution caveat.** 256x256 is small, and nothing here was measured at a
  larger resolution. Do not quote these ratios for 1080p.

## A bug this run found and fixed

The first attempt was stopped and discarded: its MS-SSIM on decoded FFmpeg frames
read about +0.0025 too high. The cause was the tensor's memory layout, not the GPU
as such. A frame tensor with identical values but a channels-last layout, which
`permute` produces, went through a different CUDA float32 convolution path and
scored up to +0.005 high. CPU, float64 and contiguous CUDA inputs all agree. The
fix is in `nvc.evaluation.perceptual_metrics.msssim`, which now forces a contiguous
layout, with a CUDA regression test and a negative control. No earlier result is
affected, as far as a search can show: this script is the only file that both
calls `msssim` and permutes frames into NCHW, and nothing calls pytorch-msssim
directly. PSNR was never affected.

## Files

- `parity.json` - every operating point (per-sequence detail, all conventions),
  every comparison, the reproduction checks and the environment.
- `parity_rd.png` - rate-distortion curves (`scripts/plot_parity.py`).

Reproduce with `./.venv/Scripts/python.exe scripts/benchmark_parity.py --no-resume`.
