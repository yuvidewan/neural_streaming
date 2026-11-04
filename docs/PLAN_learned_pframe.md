# Plan: a learned P-frame coder

Status: DRAFT, 2026-11-05. Nothing built yet.

## Why

`outputs/benchmarks/bit_split_phase_b/README.md`: on full DAVIS TEST, a P-frame costs
0.59-0.68x an I-frame at slightly lower PSNR, P-frame residuals are 73-81% of the stream,
motion 4-10%, I-frames ~15%. The codec is +127.1% BD-rate vs x264 (2.3x the bits).

Today's P-frame is hand-built around an intra-only network:

    x_warp = Warp(x_hat_{t-1}, mv)          integer 16x16 block motion, SAD search
    dz     = E(x_t) - E(x_warp)             E trained only on single images
    x_hat  = D(E(x_warp) + Q(dz))

No part of that was trained to make `dz` cheap. Phase B's rate term reached I-frames
(-31% vs run4) but not this path. Better intra cannot close the gap: I-frames are 15% of
the bits. Illustration, not a measurement: P at 0.2x an I-frame would make the stream
0.42x its size, about x264's bits at the current 2.3x.

## What gets built

A conditional P-frame autoencoder, trained with a rate term, that uses the warped
previous frame as a CONDITION rather than something to subtract:

    c      = F(x_warp)                       context features (decoder has x_warp)
    y      = E_p(x_t, c)                     P-frame latent
    y_hat  = round(y)                        unit step, like Phase B
    p(y_hat | c) = Laplace(mu(c), b(c))      conditional prior from the context
    x_hat  = D_p(y_hat, c)

- **Kept as is:** the Phase B intra model for I-frames (frozen); integer block motion and its
  coded payload (4-10% of bits, deterministic, already paid for); the `.nvct` container
  layout (a new P-frame payload kind and format version); the arithmetic coder.
- **Entropy coding:** the predicted (mu, b) per position are quantized to a shared codebook
  of prototypes, the same mechanism M10L/M11 already ship, so the coder needs no
  per-symbol float tables. Prototype assignment is per channel group, matching the decoder
  (the `assign_by_group` lesson).
- **Determinism:** the decoder now runs F and D_p, so both sides go through
  `deterministic_kernels()`, and the round trip is checked bit-exact on every benchmark
  sequence as today. Same-machine exactness is the bar; cross-device exactness is out of
  scope, as it is for the current codec.

## Training

- **Data:** Vimeo-90K septuplets (7 frames per clip), the same Kaggle chunks and Colab
  machinery as `train_vimeo_stage1.py`. DAVIS TRAIN is local for the pilot.
- **Stage 1, one step:** `x_hat_{t-1}` = the frozen intra codec's reconstruction of frame
  t-1; train E_p, D_p, F and the prior on D + lambda*R with additive-noise rate proxy.
- **Stage 2, unrolled:** chain 3-4 P-frames on its own reconstructions so it learns to stop
  error growth along a GOP of 10 (the GOP-boundary spike M16 found is exactly this).
- **Lambdas:** one P-coder per intra lambda (1e-3, 3e-4, 1e-4), each paired with its intra
  checkpoint, so each rate point stays one checkpoint pair plus a width.

## Gates (stop at the first failure)

1. **Local pilot (go/no-go before any Colab time).** A small P-coder at lambda 3e-4 trained
   for a few hours on DAVIS TRAIN, judged on VAL-B with the rate PROXY. Passes if its
   P-frames cost <= 0.45x an I-frame at matched PSNR (today 0.64x). If it cannot beat the
   hand-built path on its own training domain, Colab will not save it.
2. **Colab training**, three lambdas on Vimeo, stages 1 and 2. Passes if the VAL-B proxy
   P/I ratio holds out of domain.
3. **Real bitstream.** Codebook fit, `.nvct` integration, decode-exact on all 54 sequence x
   rate-point streams, then `benchmark_parity` and `bit_split` on full DAVIS TEST.
   Passes if BD-rate vs x264 beats +127.1% by more than 10 points. Reported against x264,
   x265 and both low-delay arms, PSNR and MS-SSIM, as now.

## Cost and risks

- **Build:** model, trainer, codebook fit and container work, about 2-3 days of coding before
  gate 1. Gate 1 itself is an afternoon on the 8 GB laptop GPU at 256x256 crops.
- **Colab:** comparable to Phase B per lambda for stage 1, plus stage 2.
- **Risks:**
  - The P-coder could learn to lean on the intra-quality reference and fail at low rate.
    Stage 2 and the 1e-3 point test this.
  - Integer block motion may cap the gain; dense learned motion is the follow-up, not part
    of this plan.
  - Decoder cost rises, since every P-frame runs a network on decode.
  - The local VAL-B proxy can overstate the real gain, as it did for M21. Gate 3 is the only
    number that counts.
