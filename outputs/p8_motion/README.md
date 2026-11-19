# Motion: bit-aware search + predicted motion coding — PASS (2026-11-19)

Plan and gate, fixed before any result: [`docs/PLAN_motion_rate.md`](../../docs/PLAN_motion_rate.md).
Code: [`scripts/motion_pred.py`](../../scripts/motion_pred.py), `--motion-lambda` in
[`scripts/p4_gate3.py`](../../scripts/p4_gate3.py). Scoring: [`report.py`](report.py) → [`results.json`](results.json).

## What changed

Only the encoder's block-motion search and the way motion vectors are coded. The learned I- and
P-coders are untouched.

- Before: each 16×16 block took the lowest-SAD vector, and every (dy, dx) was coded on its own
  with two fixed tables.
- Now: each vector is coded as its difference from the H.264 median predictor, using one of 64
  Laplace tables (2 bytes per P-frame pick the tables). The search picks `SAD + λ_m · bits`, so
  a vector only moves away from its neighbours when that saves enough residual.

## Gate result (real bytes, GOP 30, untuned, 5 λ, decode-exact)

| | BD-rate PSNR vs current codec | Bar | |
|---|---|---|---|
| **UVG 1080p** (7 videos, primary) | **−10.68%** | ≤ −10% | pass |
| **DAVIS TEST** (9 videos, guard) | **−4.75%** | ≤ 0% | pass |
| Every stream decode-exact | yes | | |

**Verdict: PASS.** Bit-aware motion with predicted coding becomes the default.

Also measured:
- MS-SSIM BD-rate vs current: UVG −10.42%, DAVIS −5.54%.
- Low band (λ 1e-2 / 3e-3 / 1e-3), PSNR: UVG **−18.48%**, DAVIS −7.63%.

## λ_m picked on VAL-B (lowest MSE + λ·bpp)

| λ | λ_m | VAL-B bpp at λ_m = 0 (coding only) → chosen | VAL-B motion bytes |
|---|---|---|---|
| 1e-2 | 0.25 | 0.1055 → 0.0947 | 62,156 → 35,537 |
| 3e-3 | 0.25 | 0.1792 → 0.1709 | 55,671 → 32,873 |
| 1e-3 | 0.125 | 0.3023 → 0.2963 | 51,047 → 35,420 |
| 3e-4 | 0.125 | 0.6086 → 0.6037 | 48,331 → 34,445 |
| 1e-4 | 0.0625 | 1.0196 → 1.0154 | 46,376 → 35,663 |

λ_m = 0 (prediction with the old SAD-only fields) is not cheaper than the old tables: the
measurement before the plan already showed this (1.59 vs 1.61 MB at UVG 1e-2). The whole gain
comes from making the search pay for bits.

## Per rate point

UVG 1080p:

| λ | bpp old → new | PSNR old → new | motion share of stream (was) |
|---|---|---|---|
| 1e-2 | 0.0580 → **0.0386** (−33%) | 32.41 → 32.01 | 24.8% (51.2%) |
| 3e-3 | 0.0777 → 0.0612 | 34.78 → 34.60 | 14.2% (35.7%) |
| 1e-3 | 0.1248 → 0.1152 | 36.81 → 36.73 | 10.2% (21.1%) |
| 3e-4 | 0.2729 → 0.2673 | 38.72 → 38.70 | 5.0% (9.5%) |
| 1e-4 | 0.5940 → 0.5921 | 40.93 → 40.92 | 2.7% (4.3%) |

DAVIS TEST:

| λ | bpp old → new | PSNR old → new | motion share (was) |
|---|---|---|---|
| 1e-2 | 0.1011 → 0.0874 | 28.60 → 28.51 | 19.0% (32.1%) |
| 3e-3 | 0.1739 → 0.1602 | 31.06 → 31.03 | 9.4% (17.8%) |
| 1e-3 | 0.2983 → 0.2858 | 33.05 → 33.03 | 5.7% (10.0%) |
| 3e-4 | 0.6160 → 0.6035 | 35.25 → 35.24 | 2.6% (4.8%) |
| 1e-4 | 1.0276 → 1.0158 | 37.92 → 37.92 | 1.6% (2.8%) |

## Against x264 / x265 (BD-rate PSNR, + = we need more bits)

| | old 5-pt | **new 5-pt** | old low band | **new low band** |
|---|---|---|---|---|
| UVG vs x264 4:2:0 | −12.4% | **−18.1%** | +65.0% | **+42.4%** |
| UVG vs x264 4:4:4 | +27.9% | **+17.9%** | +95.6% | **+65.8%** |
| UVG vs x264 4:4:4 low-delay | +4.7% | **−3.7%** | +47.7% | +25.5% |
| UVG vs x265 4:4:4 | +82.2% | +68.1% | +175.3% | +133.9% |
| DAVIS vs x264 4:2:0 | +34.0% | **+27.6%** | +35.1% | **+24.8%** |
| DAVIS vs x264 4:4:4 | +63.9% | **+55.8%** | +47.1% | **+35.6%** |
| DAVIS vs x264 low-delay | +12.9% | **+7.6%** | +14.1% | +5.5% |
| DAVIS vs x265 4:2:0 | +43.0% | +36.2% | +43.5% | +32.5% |

MS-SSIM against every anchor is in `results.json` (`vs_anchors`). It moves the same way, but
less on UVG vs x265 (+85.3% → +86.0%, 5-point).

## Per video (BD-rate PSNR)

UVG, new vs current: Bosphorus −20.5%, HoneyBee −20.2%, ReadySetGo −15.0%, YachtRide −9.7%,
Jockey −8.2%, Beauty −6.3%, ShakeNDry −0.1%. Versus x264 4:4:4 (old → new):
- Beauty −21.9 → −20.3, YachtRide −11.0 → **−18.5**: we win on both.
- Bosphorus +38.2 → +14.6, ReadySetGo +60.3 → +37.7, ShakeNDry +10.0 → +12.4.
- HoneyBee +197.7 → +147.6, Jockey +96.7 → +85.9: still far behind.

DAVIS, new vs current: every video improves (−3.3% schoolgirls to −9.7% drift-chicane).
Versus x264 4:4:4 we now win on **drone (−2.4%), gold-fish (−1.7%) and surf (−4.5%)**. cat-girl
is at +3.1%, and cows (+176.7%) and schoolgirls (+197.8%) are still the big gap.

## Time

- Encode, UVG: 1.19-1.20 s/frame (about the same as before; the raster-order search runs on CPU).
- Encode, DAVIS: 0.035 s/frame.
- Decode: UVG about 0.40 s/frame, DAVIS 0.011 s/frame (the sequential motion decode adds very
  little).

## What this leaves

- Motion is now 25% of the UVG stream at λ 1e-2 (it was 51%). The UVG low band is still
  +65.8% vs x264 4:4:4, so the low-rate floor is now mostly residual and I-frame bytes.
- ShakeNDry gained nothing. HoneyBee and Jockey are still the worst UVG videos.
- Next on the roadmap: B-frames (item 3).
