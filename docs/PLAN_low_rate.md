# Plan: lower-rate models (λ 3e-3 and 1e-2), fixed 2026-11-18 before any result

Roadmap item 2 (`docs/ROADMAP_2026-11-17.md`).

## Why

The codec has three models (λ 1e-3, 3e-4, 1e-4). On DAVIS TEST they span 33.0-37.9 dB, and on
UVG 1080p 36.8-40.9 dB. So every comparison with x264 is a high-quality comparison. The usual
range is about 30-38 dB, and on UVG at 36.8 dB we still spend 1.23x x264 4:4:4's bits. We do not
know how the codec compares with x264 at low rates, where learned codecs are usually strongest
and where the cows/schoolgirls gap was measured.

## What is trained

Two new rate points, **λ 3e-3 and λ 1e-2**, each with both coders of the current codec (A2
wide-latent P-coder + wide I-coder, `scripts/p4_wide_latent.py train-vimeo`):
- **Warm start** from the trained λ 1e-3 Vimeo models (`outputs/p4_wide_latent/vimeo/{i,p}/lambda1e-3`).
  No stage 3 or gate 1 model exists at these λ, so the λ 1e-3 models are the closest start.
- **Data:** Vimeo chunks 4, 5, 6. Each is downloaded, used by all four models, then deleted.
- **Recipe:** gate 2's, unchanged: I-coder 2,000 steps per chunk on single frames, P-coder 2,500
  steps per chunk, unroll 3, batch 8, crop 256, grad clip 1, seed 43, lr 1e-4, x0.1 on the last
  chunk.
- **One change:** the P-coder's training I-frames come from the new wide I-coder of the same λ
  (frozen, as trained so far), not from the old intra model, because no old intra model exists
  at these λ. Within a chunk the I-coder trains first.
- **Evaluation rig:** `p4_gate3.py`'s motion table and stream header come from the old λ 1e-3
  rig for both new λ (same reason). Motion is coded losslessly, so this can cost bits but cannot
  break decoding.
- Output: `outputs/p6_low_rate/vimeo/`. Rough time: about 70 minutes per λ per chunk, so
  **about 8-9 hours** including downloads, on the laptop GPU, one job at a time.

## Gate (DAVIS TEST, real bytes, GOP 30, untuned, every stream must decode exactly)

The new λ are coded by `p4_gate3.py` exactly as gate 3 coded the old ones. The 5-point curve is
the 2 new points plus the 3 points of `gate3_a2_wide_gop30.json`.

- **Validity:** each new point must have lower bpp than the next higher-rate point (1e-2 < 3e-3 <
  1e-3). Otherwise the run is **INVALID** (the λ did not take), and is reported as such.
- **Score:** BD-rate (PSNR, sequence mean, piecewise linear) vs **x264 4:4:4 default** over the
  **low band**: the curve {1e-2, 3e-3, 1e-3} against x264's full-range curve
  (`parity_classical_444_gop30` + `parity_classical_444_hq_gop30`).
- **PASS: ≤ +80.9%**, the untuned 3-point codec's figure over its own range. That is, the low-rate
  models are at least as close to x264 as the high-rate ones.
- **NOT PASSED: above +80.9%.** The models are still kept to extend the curve, but the result says
  that the codec is weakest at low rate.

Always reported:
- The same BD-rate vs x264 4:2:0, x264 low-delay (4:2:0 and 4:4:4) and x265, on PSNR and MS-SSIM.
- The 5-point full-range BD-rate vs every anchor.
- Per-video results, including cows and schoolgirls.
- UVG 1080p with all 5 points (not gated), which then covers about 33-41 dB.

Encode-time tuning is not used here, so the gate measures the models. A tuned run can follow.
