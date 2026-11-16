# Wider latent for the P-coder, gate 1 (2026-11-14)

`scripts/p4_wide_latent.py`, gate 1 of `docs/PLAN_wider_latent.md`. Run by `run_gate1.sh`.

**Setup:**
- Three copies of the stage 3 P-coder per lambda get the same fine-tune: 5,000 steps, batch
  8, base lr 5e-5, seed 7, the same DAVIS TRAIN block16 pairs in the same order. New layers
  use lr 5e-4.
  - **plain:** today's latent, 64 channels at 1/16. Reused from `outputs/p3_hyperprior/finetune_plain`.
  - **A1:** +128 channels at 1/16.
  - **A2:** +32 detail channels at 1/8, coded given the decoded 1/16 latent.
- At step 0, A1 and A2 reconstruct exactly like plain, because the new decoder branch starts
  at zero.
- **Rate** is scored with the motion oracle's closed loop: VAL-B, GOP 10, block16 motion
  paid, ideal bits including every latent.
- **Ceiling** is P-frame PSNR with no rounding, one P-frame at a time from the intra
  reconstruction, over the first 20 frames.

| lambda | coder | bits/P | P PSNR | ceiling on car-roundabout |
|---|---|---|---|---|
| 1e-3 | plain | 17,807 | 32.08 | 30.90 |
| 1e-3 | A1 | 6,920 | 23.58 | 24.90 |
| 1e-3 | A2 | 20,184 | 32.54 | 31.94 |
| 3e-4 | plain | 28,131 | 33.09 | 31.23 |
| 3e-4 | A1 | 35,197 | 34.26 | 32.77 |
| 3e-4 | A2 | 45,096 | 36.03 | 36.02 |
| 1e-4 | plain | 40,186 | 33.62 | 31.44 |
| 1e-4 | A1 | 63,448 | 36.34 | 34.60 |
| 1e-4 | A2 | 77,105 | 38.26 | 37.58 |

The judged video is **car-roundabout**, which has the lowest plain ceiling. The ceilings on
all four VAL-B videos are in `gate1.json`.

| arm | BD-rate vs plain | ceiling lift (mean over lambdas) | verdict |
|---|---|---|---|
| A1 | +10.4% | -0.43 dB | NOT PASSED |
| **A2** | **-20.6%** | **+3.99 dB** | **PASSED** |

The bar, fixed in the plan before the run, was BD-rate ≤ -10% and a ceiling lift ≥ +1.5 dB.

## What it means

- **The diagnosis was right.** Plain's ceiling barely moves with the rate: 30.9 → 31.4 dB
  from λ=1e-3 to 1e-4. A2's moves a lot: 31.9 → 37.6 dB. More room in the latent is what
  lets quality keep climbing.
- **The gain grows with quality.** A2 vs plain at the same P-frame PSNR, interpolated in
  log-rate:

  | PSNR | plain bits/P | A2 bits/P | saving |
  |---|---|---|---|
  | 32.54 dB | 21,943 | 20,184 | -8.0% |
  | 33.09 dB | 28,093 | 22,908 | -18.5% |
  | 33.62 dB | 40,159 | 25,879 | -35.6% |

  This is the regime where we lose to x264: detailed videos stuck near 28.5 dB.
- **A2 goes far past plain's range.** A2 reaches 38.3 dB, where plain tops out at 33.6 dB.

## Caveats

- **Narrow overlap.** The BD-rate is computed over only the 1.1 dB where the two curves
  overlap (32.54–33.62 dB). Most of A2's curve lies above anything plain can reach.
- **Low rates.** At low rates the saving is small, about -8%.
- **A1's verdict is contaminated.** A1's λ=1e-3 run diverged on its very last step
  (31.27 dB at step 4,750, then 25.07 dB at step 5,000), and the saved model is the broken
  one. At its other two lambdas A1 lifted the car-roundabout ceiling by +1.5 and +3.2 dB.
  A1 still lost to A2 at every lambda, so this does not change which arm goes on.
- **This is P-frames only, with ideal bits, after a short DAVIS fine-tune.** The I-frame
  model keeps its own ceiling (27.6 dB on cows without rounding). Gate 2 must widen it
  too, as the plan says.

## Next

Gate 2: retrain A2 on Vimeo, plus a wider I-frame model with the learned per-value prior.
Gate 3: real bytes at GOP 30. It passes if it beats +69.1% vs x264 by more than 8 points.

# Gate 2 and gate 3 (2026-11-16): PASSED

Gate 2 is a Vimeo retrain, run by `run_gate2.sh`. Gate 3 is real bytes on DAVIS TEST at GOP 30
(`scripts/p4_gate3.py`). The recipe and the bar were fixed in `docs/PLAN_wider_latent.md`
before any result.

## Gate 3: real bytes, TEST, GOP 30, every stream decodes exactly

| codec | vs x264 (PSNR) | vs x264 low-delay, GOP 30 | vs x264 (MS-SSIM) | vs x265 (PSNR) |
|---|---|---|---|---|
| stage 3 P-coder (before) | +69.1% | +40.4% | +55.9% | +80.7% |
| A2 P-coder + old I-frames | **+40.4%** | +16.5% | +52.2% | +52.6% |
| A2 P-coder + wide I-frames | **+39.9%** | **+16.1%** | +51.6% | +52.0% |

**Bar: either arm at or below +61.1%. Both arms pass, by about 21 points.** `gate3_verdict_gop30.json`.
The low-delay column uses the fair GOP-30 classical runs (`gate3_vs_classical_gop30.json`).
`p4_gate3.py` also prints a "lowdelay" line of about +4.5%, but that is against x264 forced to
GOP 10, which is not a fair match.

| lambda | arm | bpp | PSNR | I PSNR | P PSNR | I-frame share of bytes |
|---|---|---|---|---|---|---|
| 1e-3 | stage 3 | 0.258 | 32.26 | 30.17 | 31.05 | 6.7% |
| 1e-3 | A2 + wide I | 0.298 | 33.05 | 30.85 | 32.05 | 6.6% |
| 3e-4 | stage 3 | 0.410 | 33.26 | 31.20 | 31.76 | 6.3% |
| 3e-4 | A2 + wide I | 0.616 | 35.25 | 33.51 | 34.45 | 6.0% |
| 1e-4 | stage 3 | 0.580 | 33.87 | 31.58 | 32.15 | 5.9% |
| 1e-4 | A2 + wide I | 1.028 | 37.92 | 35.44 | 36.99 | 5.6% |

The ideal-to-real coder overhead is +0.06%. On 719 frames, encoding takes 200-220 s and
decoding 7-9 s (stage 3: 243-269 s and 7-9 s).

## Per video, vs x264 (PSNR BD-rate)

| video | stage 3 | A2 + old I | A2 + wide I | A2 + wide I vs x264 low-delay |
|---|---|---|---|---|
| surf | -5.3% | -20.0% | **-20.5%** | -29.6% |
| cat-girl | +25.3% | -4.4% | **-4.7%** | -15.0% |
| bmx-bumps | +71.6% | +52.2% | +50.9% | +28.9% |
| drift-chicane | +101.2% | +64.4% | +62.8% | +9.0% |
| car-turn | +139.6% | +85.8% | +84.5% | +23.4% |
| cows | +292.1% | +184.1% | +181.3% | +67.3% |
| schoolgirls | +292.3% | +220.0% | +219.2% | +66.0% |
| drone | -20.6% | beats x264 | beats x264 | |
| gold-fish | above range | beats x264 | beats x264 | |

- On drone and gold-fish, the new codec's lowest point already beats x264's best point. Drone
  is 35.69 dB at 0.263 bpp against x264's 35.41 dB at 0.343 bpp. Gold-fish is 34.28 dB at
  0.220 bpp against 33.46 dB at 0.304 bpp. The curves overlap too little for a BD-rate, so
  these are listed as wins, not numbers. `gate3_per_video.json`.
- **We now beat x264 on 4 of 9 test videos** (stage 3: 2, plus gold-fish out of range).
- Every video improved. The two detailed videos improved the most in points (cows -111,
  schoolgirls -73), but they are still the gap.

## Gate 2 checks (VAL-B, reported, not pass/fail)

`gate2_check.json`. The motion oracle loop, GOP 10.

| lambda | stage 3 P | A2 P (Vimeo) | old I: rounded / ceiling | wide I: rounded / ceiling |
|---|---|---|---|---|
| 1e-3 | 0.254 bpp, 31.91 dB | 0.298 bpp, 32.89 dB | 30.82 / 32.10 dB | 31.50 / 33.47 dB |
| 3e-4 | 0.403 bpp, 32.93 dB | 0.603 bpp, 35.35 dB | 32.11 / 32.75 dB | 34.37 / 36.48 dB |
| 1e-4 | 0.571 bpp, 33.51 dB | 1.016 bpp, 38.25 dB | 32.62 / 32.88 dB | 36.56 / 38.03 dB |

- **A2 P-coder vs stage 3: -31.2% P-frame BD-rate.** Gate 1 measured -20.6% after only a DAVIS
  fine-tune.
- **The wide I-coder lifts the I-frame ceiling from 32.9 to 38.0 dB.** The old intra model's
  ceiling barely moves with lambda.

## Caveats (must be read with the result)

- **Extra training.** The A2 coders got one more pass over Vimeo chunks 4-10 than stage 3 did.
  Some of the gain may be extra training rather than the wider latent. Gate 1 isolated the
  latent's effect at -20.6% with equal training.
- **Best of two arms.** Two arms were run and the bar needed only one. Both pass, and they
  differ by only 0.5 points, so this choice does not matter here.
- **The wide I-coder at lambda 1e-4 diverged** at the end of chunk 6: its 250-step training
  average fell from 39.7 to 27.4 dB. No earlier save was kept. It recovered to 38.5 dB by the
  end of chunk 7 and finished at 39.0 dB. The released model is the recovered one.
- **Rate range.** The new codec runs at higher rates (up to 1.03 bpp vs 0.58). The BD-rate vs
  x264 is computed where the curves overlap, which is the low-rate end, where gate 1 found
  the wider latent's gain smallest. The gain at matched high quality is larger than the
  headline.
- **MS-SSIM gained much less than PSNR** (+55.9% to +51.6%). The training loss is MSE.
- **One crash and one code fix during the run.** A Windows file lock, most likely OneDrive,
  failed a checkpoint rename. The save now retries, and a duplicate final save was removed. No
  training step was lost or repeated. The training recipe is unchanged.
- All results are on 9 DAVIS TEST videos, resized and centre-cropped to 256x256 (719 frames). This is a small test set at a low resolution.

## Next

We are still +39.9% vs x264 default and +16.1% vs x264 low-delay at GOP 30. The remaining gap
is cows and schoolgirls. A wider latent (128 channels at 1/8, or 4 values per pixel, as
DCVC-RT uses), B-frames, and a rate-aware quantizer are the candidates.
