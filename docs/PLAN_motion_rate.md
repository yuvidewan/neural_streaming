# Plan: bit-aware motion search + predicted motion coding, fixed 2026-11-19 before any result

Roadmap item 5 (`docs/ROADMAP_2026-11-17.md`), moved ahead of item 3 by the user on 2026-11-19.

## Why (`scripts/p8_motion_share.py` → `outputs/p8_motion/share.json`)

The current codec searches block motion by SAD alone and codes every (dy, dx) on its own, with
two fixed tables. Measured with the current codec (DAVIS TEST in full, first 31 frames of
every UVG video):

| λ | UVG: motion share of stream | DAVIS: motion share of stream |
|---|---|---|
| 1e-2 | **51.2%** | 32.1% |
| 3e-3 | 35.7% | 17.8% |
| 1e-3 | 21.1% | 10.0% |
| 3e-4 | 9.5% | 4.8% |
| 1e-4 | 4.3% | 2.8% |

- Motion costs about the same at every λ (UVG 1.47-1.67 MB over 210 P-frames), so it is a
  fixed cost that dominates the low rates. This is the floor behind UVG's +95.6% low-band
  result vs x264 4:4:4.
- With the fields unchanged, predicting each vector from its neighbours would save little
  (UVG 1e-2: 1.59 MB of raw-vector entropy vs 1.61 MB with a median predictor). The fields
  are noisy, because nothing in the search makes a vector pay for its bits. So the search
  has to change too.

## What changes (encoder search and motion coding only; the learned coders are untouched)

1. **Predicted motion coding.** Each block's vector is coded as the difference from the H.264
   median predictor of its left, top and top-right neighbours (top-left at the right edge;
   missing neighbours as in H.264). Each component is coded with a zero-mean discrete
   Laplace table on [-32, 32] (64 log-spaced scales). The encoder picks the best scale per
   frame and per component and sends the two scale indices (2 bytes per P-frame, counted).
2. **Bit-aware search.** Over the same full ±16 search, each block takes the vector with the
   lowest `SAD + λ_m · bits`, where `bits` is the cost of its difference from the predictor
   under the previous P-frame's scales (a default scale for the first P-frame of a GOP).
   Blocks are decided in raster order, so the predictor uses the vectors already chosen.
   Ties go to the vector with the fewest bits, then to the old tie rule.
3. `λ_m = 0` is predicted coding with the old SAD-only fields. It is a grid point, so the
   gain from coding alone is measured too.

The stream header's motion model id names the new coder, so a decoder cannot mix the two.
Every stream must decode bit for bit.

## Choosing λ_m (VAL-B only)

- Grid: λ_m in {0, 0.25, 0.5, 1, 2, 4, 8, 16} (SAD units per bit).
- For each codec λ, the λ_m with the lowest **VAL-B** `MSE + λ · bpp` (real bytes, the
  models' own training objective) is chosen. That is one λ_m per rate point, fixed before
  TEST or UVG are coded.

## Gate (fixed before any result)

The score is real bytes, GOP 30, untuned, all five λ, decode-exact, BD-rate PSNR (sequence
mean, piecewise linear) of the new codec against the current one on the same frames.

- **Primary, UVG 1080p (all 7 videos, 672 frames):** the 5-point curve vs the current 5-point
  curve (`outputs/p4_wide_latent/uvg_a2_wide_gop30.json` + `outputs/p6_low_rate/uvg_low_rate.json`).
- **Guard, DAVIS TEST:** the 5-point curve vs the current 5-point curve
  (`gate3_a2_wide_gop30.json` + `outputs/p6_low_rate/gate_test.json`).
- **PASS:** UVG **≤ −10%** and DAVIS TEST **≤ 0%** (not worse), every stream decode-exact.
- **SMALL WIN:** UVG between −10% and −3%, DAVIS ≤ 0%. The change is kept and reported as small.
- **NOT PASSED:** anything else. The change is dropped from the default.

Always reported: the same against x264/x265 (4:2:0, 4:4:4, default and low-delay), on PSNR and
MS-SSIM, low band and 5-point, per video; the new motion share; λ_m = 0 (coding alone) on VAL-B;
and encode and decode time.
