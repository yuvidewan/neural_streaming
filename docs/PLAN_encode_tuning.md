# Plan: encode-time latent tuning (fixed 2026-11-17, before any result)

Status: planned.

## Idea

For every frame, the encoder spends a few gradient steps improving the latents it sends
(y at 1/16 and x at 1/8). Each step lowers that frame's coded cost, `mse + lambda * bits/pixel`,
the same loss the coders were trained on. The decoder, the models, the entropy coder and the
bitstream format all stay unchanged, so every stream must still decode bit for bit.

Published work (Campos et al. 2019, Yang et al. 2020) reports 5-10% savings for learned
image codecs. Encoding gets slower; decoding does not.

## Method

`scripts/p4_gate3.py --tune-steps N --tune-lr L`. With N = 0 the output is byte-identical to
gate 3.
- **Starting point:** the encoder's own latents.
- **Optimizer:** Adam on y and x for N steps.
- **Rounding:** straight-through for both the decoder and the rate, so the loss at every step
  is exactly the frame's coded cost with rounding, using ideal bits.
- **Keep the best:** the encoder keeps the iterate with the lowest coded cost. Step 0, the
  untuned latents, is a candidate, so tuning never raises a frame's coded cost.
- **Scope:** every P-frame (A2 P-coder) and every I-frame (wide I-coder).
- **Greedy:** each frame is tuned on its own. The effect on later frames is ignored.
- **Base codec:** gate 3's best arm (A2 P-coder + wide I-frames, models in
  `outputs/p4_wide_latent/vimeo`, +39.9% vs x264 at GOP 30). Same block16 motion, GOP 30.

## Choosing the settings (VAL-B only)

- **Grid:** lr ∈ {0.01, 0.03, 0.1} × steps ∈ {20, 50}.
- **Data:** the 4 VAL-B videos, GOP 30, real bytes, all 3 lambdas.
- **Score:** BD-rate (PSNR) vs the untuned codec on the same videos.
- **Pick:** the lowest BD-rate. If two settings are within 0.5 points, take the one with
  fewer steps.
- TEST is not touched until the setting is fixed.

## Gate (TEST, real bytes, GOP 30, every stream must decode exactly)

The score is BD-rate (PSNR) of the tuned codec vs the untuned codec (gate 3's
`gate3_a2_wide_gop30.json`).
- **PASS: ≤ -5%.** Encode tuning becomes the default slow encoder mode.
- **SMALL WIN: between -5% and -2%.** It is kept as an optional slow-encode mode, but it is
  not the next big lever.
- **NOT WORTH IT: above -2%.**

Always reported:
- BD-rate vs x264, and vs x264 low-delay at GOP 30.
- Per-video results.
- Encode time per frame.
- MS-SSIM.
