# Lower-rate models (λ 3e-3 and 1e-2): DAVIS gate PASSED, but UVG shows a low-rate weakness (2026-11-19)

Plan, gate fixed before any result: [`docs/PLAN_low_rate.md`](../../docs/PLAN_low_rate.md).
All numbers below come from `report.py` → `results.json`.

## What was trained

- Two new rate points, λ 3e-3 and λ 1e-2, each with both coders of the current codec
  (A2 wide-latent P-coder and wide I-coder).
- Warm-started from the λ 1e-3 Vimeo models, then trained on Vimeo chunks 4-6 with gate 2's
  recipe (`run_train.sh`; log `train.log`).
- The P-coder's training I-frames came from the new wide I-coder of the same λ.
- The evaluation rig borrows λ 1e-3's motion table and stream header.
- Training ran 2026-11-18 to 2026-11-19, about 9 GPU hours on the laptop RTX 5060. It was
  stopped twice and resumed from its checkpoints.

## Gate: DAVIS TEST, real bytes, GOP 30, untuned (`gate_test.json`)

Every stream decoded exactly, and the bits drop as λ rises (1e-2 < 3e-3 < 1e-3), so the run
is **valid**.

| λ | bpp | PSNR (dB) | MS-SSIM |
|---|---|---|---|
| **1e-2 (new)** | 0.101 | 28.60 | 0.9400 |
| **3e-3 (new)** | 0.174 | 31.06 | 0.9679 |
| 1e-3 | 0.298 | 33.05 | 0.9821 |
| 3e-4 | 0.616 | 35.25 | 0.9899 |
| 1e-4 | 1.028 | 37.92 | 0.9949 |

The codec now covers **28.6-37.9 dB** on DAVIS. Before, it covered 33.0-37.9 dB.

**Score:** the low band {1e-2, 3e-3, 1e-3} against x264 4:4:4 (full-range anchor), BD-rate
PSNR, is **+47.1%**. The bar was +80.9%, so the gate **PASSES**. The codec is closer to x264
at low rate than at high rate.

BD-rate (positive = we spend more bits; "old 3-pt" is the 33.0-37.9 dB curve from before):

| anchor | low band, PSNR | 5-point, PSNR | old 3-pt, PSNR | low band, MS-SSIM | 5-point, MS-SSIM |
|---|---|---|---|---|---|
| x264 4:2:0 | +35.1% | +34.0% | +33.0% | +62.9% | +58.7% |
| x264 4:2:0 low-delay | +14.1% | +12.9% | +11.8% | +37.6% | +33.6% |
| **x264 4:4:4** | **+47.1%** | +63.9% | +80.9% | +66.2% | +65.7% |
| x264 4:4:4 low-delay | +25.0% | +38.2% | +51.5% | +40.5% | +39.4% |
| x265 4:2:0 | +43.5% | +43.0% | +42.7% | +60.9% | +57.0% |
| x265 4:2:0 low-delay | +19.7% | +18.6% | +17.7% | +31.5% | +27.2% |
| x265 4:4:4 | +47.6% | +67.8% | +88.5% | +55.3% | +54.9% |
| x265 4:4:4 low-delay | +22.2% | +37.3% | +52.7% | +25.8% | +24.5% |

Every comparison overlaps fully: 28.6-33.0 dB for the low band, 28.6-37.9 dB for 5 points.

What this says:
- **Against the 4:4:4 anchors, low rate is much better** than high rate: +47% instead of
  +81% vs x264.
- **Against the 4:2:0 anchors, the gap is about the same at every rate** (+33-35% vs x264).
  That is expected: 4:2:0 throws away colour detail, which costs RGB PSNR mostly at high
  quality. So the 4:4:4 improvement is most likely x264 4:4:4 losing its colour advantage
  at low rate, not our codec getting better. This is an inference from the two tables, not
  a separate measurement.
- **We still lose to x264 everywhere** on the average. This does not beat H.264.
- **MS-SSIM is worse than PSNR** at every rate, as before. The models are trained on MSE.

### Per video vs x264 4:4:4 (BD-rate PSNR)

| video | low band | 5-point |
|---|---|---|
| gold-fish | **−5.8%** | +1.9% |
| drone | **−5.1%** | +2.9% |
| cat-girl | +1.0% | +7.9% |
| surf | +7.9% | +5.3% |
| bmx-bumps | +60.0% | +75.3% |
| drift-chicane | +83.2% | +91.7% |
| car-turn | +85.6% | +99.0% |
| schoolgirls | +139.0% | +208.7% |
| cows | +148.9% | +188.9% |

- We beat x264 4:4:4 on two videos at low rate, and are close on two more.
- **cows and schoolgirls are still the problem.** At λ 1e-2 they sit at about 25.2 dB, about
  3.4 dB below the average. The quality ceiling on detailed videos is still there at low rate.

Encode-time tuning was not used; a tuned run would add roughly another −6%.

## UVG 1080p, all 5 points (not gated, `uvg_low_rate.json`)

7 sequences, 672 frames, the same 1920x1072 crop as the anchors. Every stream decoded exactly
and the bits are in order.

| λ | bpp | PSNR (dB) | MS-SSIM |
|---|---|---|---|
| **1e-2 (new)** | 0.058 | 32.41 | 0.9338 |
| **3e-3 (new)** | 0.078 | 34.78 | 0.9558 |
| 1e-3 | 0.125 | 36.81 | 0.9702 |
| 3e-4 | 0.273 | 38.72 | 0.9815 |
| 1e-4 | 0.594 | 40.92 | 0.9897 |

BD-rate PSNR (low band = 32.4-36.8 dB; 5-point = 32.4-40.9 dB; old 3-pt = 36.8-40.9 dB):

| anchor | low band | 5-point | old 3-pt |
|---|---|---|---|
| x264 4:2:0 | +65.0% | −12.4% | −57.2% |
| x264 4:2:0 low-delay | +27.1% | −29.6% | −62.6% |
| **x264 4:4:4** | **+95.6%** | **+27.9%** | −18.9% |
| x264 4:4:4 low-delay | +47.7% | +4.7% | −27.6% |
| x265 4:2:0 | +141.1% | +29.4% | −42.2% |
| x265 4:4:4 | +175.3% | +82.2% | +17.2% |
| x265 4:4:4 low-delay | +86.2% | +30.8% | −10.3% |

**The UVG win from 2026-11-18 was a high-quality win only.** Over 32.4-36.8 dB we spend about
twice x264 4:4:4's bits (+95.6%), and over the full 5-point range we are +27.9% vs x264 4:4:4.
So on 1080p the codec is the opposite of DAVIS: strong at high quality, weak at low rate.

A likely reason, not yet measured: the new points barely lower the bits on UVG (0.078 → 0.058
bpp from 3e-3 to 1e-2, while PSNR drops 2.4 dB). On Beauty, for example, 1e-2 costs 0.058 bpp
against 0.065 at 3e-3. That looks like a fixed cost per frame that does not shrink with λ. The
borrowed λ 1e-3 motion table and header, the I-frames, and the hyperprior side information are
the candidates. A breakdown of the bytes per stream part would tell which.

Per video, BD-rate PSNR vs x264 4:4:4:

| video | low band | 5-point |
|---|---|---|
| YachtRide | +12.0% | **−11.0%** |
| Beauty | +21.2% | **−21.9%** |
| ShakeNDry | +52.4% | +10.0% |
| ReadySetGo | +107.4% | +60.3% |
| Bosphorus | +133.7% | +38.2% |
| Jockey | +211.7% | +96.7% |
| HoneyBee | +486.1% | +197.7% |
