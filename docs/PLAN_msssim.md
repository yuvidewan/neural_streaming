# Plan: an MS-SSIM model set (fixed 2026-11-19, before any training or result)

Roadmap item 6 (`docs/ROADMAP_2026-11-17.md`): "MS-SSIM barely improved. A fine-tune with an
MS-SSIM loss, reported as a second model set." It is next in the queue now that B-frames closed
(`docs/PLAN_bframes.md`, NOT PASSED).

## Why

Every model was trained on MSE. On MS-SSIM the current codec (bit-aware motion,
`outputs/p8_motion`) is far behind x264, much further than on PSNR:

| BD-rate, 5-point | PSNR | MS-SSIM |
|---|---|---|
| DAVIS vs x264 4:2:0 | +27.6% | +49.9% |
| UVG vs x264 4:2:0 | −18.1% | +48.2% |

Learned codecs are normally reported with a separate MS-SSIM-trained set, and CompressAI does the
same for its anchors.

## What is trained

- **All five λ (1e-2, 3e-3, 1e-3, 3e-4, 1e-4), both the wide I-coder and the A2 P-coder.**
  Each starts from its own current MSE model: `outputs/p4_wide_latent/vimeo` for 1e-3 / 3e-4 /
  1e-4, and `outputs/p6_low_rate/vimeo` for 3e-3 / 1e-2.
- **Loss:** `k · (1 − MS-SSIM) + λ · bpp` in place of `MSE + λ · bpp`, with **k = 1280 / 255²
  ≈ 0.0197** for every λ. This is CompressAI's published pairing (MS-SSIM λ = 1280 × MSE λ,
  e.g. 16.64 ↔ 0.0130), rewritten in this repo's loss form. It is fixed from outside our data,
  so nothing is tuned on results.
- **Recipe:** the same as `outputs/p6_low_rate/run_train.sh`:
  - one Vimeo chunk (chunk 7), deleted after use;
  - I-coder 2,000 steps, P-coder 2,500 steps (unroll 3, batch 8, crop 256, lr 1e-4);
  - P-coder I-frames from the same λ's new I-coder.
- Motion is unchanged: the bit-aware search with the λ_m values chosen in
  `docs/PLAN_motion_rate.md`.

## Gate (fixed before any result)

Real bytes, GOP 30, untuned, all five λ, decode-exact. The score is the BD-rate (sequence mean,
piecewise linear) of the MS-SSIM set against the current MSE set, on the same frames:
`outputs/p8_motion/test_motion.json` (DAVIS TEST) and `outputs/p8_motion/uvg_motion.json`
(UVG 1080p, 7 videos).

- **PASS:** BD-rate **MS-SSIM ≤ −10% on both** DAVIS TEST and UVG, every stream decode-exact.
- **SMALL WIN:** ≤ −3% on both.
- **NOT PASSED:** anything else.

Whatever the verdict, the MS-SSIM set is a second model set. It never replaces the MSE set,
which stays the codec for PSNR results.

Always reported:
- BD-rate PSNR against the MSE set (expected worse).
- Both metrics against x264/x265 (4:2:0, 4:4:4, default and low-delay), low band and 5-point.
- Per video.
- The training curves.
