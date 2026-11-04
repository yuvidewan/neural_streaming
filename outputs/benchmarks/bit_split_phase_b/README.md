# Where the bits go: Phase B video codec, split by frame type (2026-11-05)

`scripts/bit_split.py` runs `benchmark_parity`'s `nvc_deployed` arm unchanged (same rig,
encoder and decode-exact check) on full DAVIS TEST (9 sequences, 719 frames, GOP 10:
75 I-frames, 644 P-frames) and reads every `.nvct` before it is deleted. Totals reproduce
`parity_phase_b_lambda*` byte for byte (1,740,415 / 2,735,515 / 3,820,310), all decode-exact.

| lambda | I-frames | P motion | P residual | framing | bytes per I | bytes per P | P / I | PSNR I / P |
|---|---|---|---|---|---|---|---|---|
| 1e-3 (7-bit) | 16.3% | 9.5% | **73.2%** | 0.9% | 3,794 | 2,236 | **0.59** | 31.20 / 31.08 |
| 3e-4 (8-bit) | 15.3% | 5.9% | **78.2%** | 0.6% | 5,592 | 3,571 | **0.64** | 32.62 / 32.37 |
| 1e-4 (8-bit) | 14.5% | 4.2% | **80.8%** | 0.4% | 7,409 | 5,044 | **0.68** | 33.22 / 32.99 |

**A P-frame costs 59-68% of an I-frame and comes out slightly worse.** Temporal prediction
saves only about a third of an intra frame. P-frame residuals are 73-81% of the stream and
grow with rate; motion is 4-10%; I-frames are ~15%. Phase B's rate-aware training reached the
I-frame transform, but the P-frame residual is still `dz = E(x_t) - E(warp(x_hat_{t-1}))`
through an encoder trained only on single images, with integer 16x16 block motion
(`m10h_motion_compensation.py`).

Illustration only, not a measurement: if P-frames cost 0.2x an I-frame at the same quality,
the stream would shrink to (75 + 0.2*644) / (75 + 0.64*644) = 0.42x, which at the current
2.3x x264 is about x264's bits. No amount of intra improvement can do that: I-frames are 15%.

Next: a learned P-frame coder (conditional/residual coding trained end to end with a rate
term), not further tuning of the hand-built latent differencing.
