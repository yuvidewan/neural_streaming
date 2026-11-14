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
