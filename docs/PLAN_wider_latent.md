# Plan: a wider latent for the learned P-frame coder (Plan A)

Status: gate 1 PASSED 2026-11-14 by A2 (+32 detail channels at 1/8): -20.6% P-frame BD-rate vs plain, and the car-roundabout ceiling is +3.99 dB higher. A1 NOT PASSED (+10.4%; its λ=1e-3 run diverged on the last step). Next is gate 2. See outputs/p4_wide_latent/README.md.

## Why

`docs/RESEARCH_2026-11-14.md`, Part 0, measured on our codec:
- The gap to x264 is a **quality ceiling** on detailed videos. *cows* and *schoolgirls* are
  +292% vs x264 each and stuck near 28.5 dB whatever the bitrate. We already beat x264 on
  *drone* and *surf*.
- **The cause is capacity.** Without rounding, the intra model reaches only 27.6 / 28.6 dB on
  those videos. A search for the best possible latent adds just +0.6-0.8 dB, and the
  P-coder without rounding gains only +0.1 dB. The latent is 64 channels at 1/16
  resolution, i.e. 0.25 values per pixel. DCVC-RT judged 2 values per pixel too few and
  chose 4.

## Arms (all start from the stage 3 P-coder; same latent budget for A1 and A2)

- **plain:** the stage 3 coder with the standard fine-tune. Reused from the hyperprior gate
  (`outputs/p3_hyperprior/finetune_plain`): the same recipe, data order and seed.
- **A1: +128 channels at 1/16** (192 in total, 0.75 values per pixel).
- **A2: +32-channel detail latent at 1/8** beside the 64 at 1/16 (also 0.75 values per pixel).

For A1 and A2:
- The new latent comes from the encoder's stride-8 features.
- Its prior sees the reference context and the already-decoded 1/16 latent (decoded first),
  so it is coded "given the coarse picture".
- Its decoder branch is added into the synthesis and starts at zero, so the reconstruction
  at step 0 is exactly the stage 3 coder's.
- The new encoder branch starts small and random (latent std ~0.7), so it is not stuck at
  zero.
- The rate of every latent counts.

## Gate 1 (fixed before any result)

- **Fine-tune:** every arm gets 5,000 steps at batch 8, the same DAVIS TRAIN block16 pairs in
  the same order, seed 7, base lr 5e-5. New layers use lr 5e-4.
- **Score 1, rate:** P-frame BD-rate vs plain with the motion oracle loop (VAL-B, GOP 10,
  block16 motion paid, ideal bits).
- **Score 2, ceiling:** the P-coder's quality with **no rounding**, one P-frame at a time
  from the intra reconstruction of the previous frame, on VAL-B. It is judged on the VAL-B
  video where plain's ceiling is lowest.
- **Pass:** an arm passes if **BD-rate ≤ -10% vs plain AND its ceiling on that video is at
  least 1.5 dB above plain's.**
- **Near-miss rule:** if an arm lifts the ceiling by ≥ 1.5 dB but misses the BD-rate bar,
  the reading is "the capacity works, it needs full training". It goes on to gate 2 only
  if the BD-rate is at least break-even (≤ 0%).

## Gate 2 and 3

- **Gate 2:** a Vimeo retrain of the winning arm, plus a wider I-frame model with the learned
  per-value prior. The old fixed entropy stack charged unused channels, which is why the
  2026-10 192-channel run failed.
- **Gate 3:** real bytes at GOP 30. **Passes if it beats +69.1% vs x264 by more than 8
  points.**
