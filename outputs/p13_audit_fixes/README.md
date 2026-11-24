# Fixing the second audit's two findings (2026-11-24/25, done)

Plan, fixed before any run: [`docs/PLAN_audit2_fixes.md`](../../docs/PLAN_audit2_fixes.md)
(with amendment 1, also fixed before the run it changed). The audit that found both problems:
[`outputs/p12_audit2/README.md`](../p12_audit2/README.md). Chain:
[`run_all.sh`](run_all.sh), log [`gate.log`](gate.log). Scoring: [`report.py`](report.py) →
[`results.json`](results.json).

No model was trained or changed, and no setting was tuned.

## The two fixes

1. **Drop the 1 KB of unused header.** Streams with wide I-frames (every current headline) no
   longer write the two old quantization blocks (64 channels × 8 B × 2 = 1,024 B). The header
   declares 0 parameters, so this is still `.nvct` version 2 and old streams still read. An
   old-I-frame decode of a block-less stream raises an error instead of guessing.
   Code: `scripts/m10h_motion_compensation.py`, `scripts/p4_gate3.py`. Tests:
   `tests/test_nvct_no_quant_blocks.py`.
2. **Feed the x264/x265 anchors at the real frame rate.** New option `--real-fps` in
   `scripts/benchmark_parity.py` reads each video's fps from its manifest. UVG is 120 fps,
   MCL-JCV is 24/25/30 per video. **DAVIS records no frame rate, so DAVIS anchors stay at
   30 fps.** New anchor folders: `outputs/benchmarks/{uvg,mcljcv}_classical_{420,444}_gop30_realfps`.
   The old ones are kept.

## Checks (all passed)

| Check | Result |
|---|---|
| Unit tests (block-less round trip, old streams still read, writer refuses wrong counts, old-I decode raises) | pass |
| Proof 2: the 12 audit streams re-encoded = old file with the counts zeroed and bytes 56–1,079 removed; fresh-process decode identical | **12/12 pass** |
| Real re-run, DAVIS TEST MSE set: every video exactly 1,024 B smaller, same PSNR and MS-SSIM, decode-exact | pass |
| Real re-run, DAVIS TEST MS-SSIM set: same | pass |
| Real re-run, UVG MSE set: same | pass |
| MCL-JCV videos that really are 30 fps give byte-identical anchors (only the frame rate changed) | **16 videos, 2,432 rows, all identical** |
| Coverage (amendment 1): the 120 fps x264 anchors reach above our top PSNR on every UVG video | **7/7**, none short |

The plan said 18 MCL-JCV videos are 30 fps. The manifest has 16 (plus 7 at 25 and 7 at 24).
That was a counting slip in the plan. The check ran on all 30 fps videos there are.

## Re-quoted headlines

BD-rate vs the anchor (negative = we use fewer bits). "Real" rows were re-coded under the new
container. "Derived" rows take the old streams minus 1,024 B per video, which the checks above
show is exact.

| Headline | Fix 1 | Old | After fix 1 | **After both** | Per-video mean, old → after both |
|---|---|---|---|---|---|
| DAVIS MSE set vs x264 4:2:0, PSNR | real | +27.6 | +26.72 | **+26.72** | |
| DAVIS MSE set vs x264 4:4:4, PSNR | real | +55.8 | +54.73 | **+54.73** | |
| UVG MSE set vs x264 4:2:0, PSNR | real | −18.09 | −18.13 | **−20.22** | −5.83 → −14.14 |
| UVG MSE set vs x264 4:4:4, PSNR | real | +17.88 | +17.83 | **+17.96** | +37.05 → +37.06 |
| UVG MSE set vs x264 4:2:0 low-delay, PSNR | real | −34.13 | −34.16 | **−34.23** | −29.33 → −31.69 |
| UVG low band vs x264 4:4:4, PSNR | real | +65.82 | +65.70 | **+65.89** | +99.05 → +98.94 |
| MS-SSIM set vs MSE set, DAVIS, MS-SSIM | real | −22.43 | −22.60 | **−22.60** | |
| MS-SSIM set vs MSE set, UVG, MS-SSIM | real/derived | −9.66 | −9.66 | **−9.66** | |
| MS-SSIM set, DAVIS vs x264 4:2:0, MS-SSIM | real | +16.27 | +14.88 | **+14.88** | |
| MS-SSIM set, UVG vs x264 4:2:0, MS-SSIM | derived | +0.35 | +0.32 | **+0.43** | +20.69 → +19.04 |
| MCL-JCV MSE set vs x264 4:2:0, PSNR | derived | −1.41 | −1.44 | **−2.33** | +19.81 → +19.32 |
| MCL-JCV MSE set vs x264 4:2:0 low-delay, PSNR | derived | −12.35 | −12.38 | **−13.12** | −1.01 → −1.37 |
| MCL-JCV MSE set vs x264 4:4:4, PSNR | derived | +40.09 | +40.04 | **+38.68** | +62.07 → +62.12 |
| MCL-JCV MS-SSIM set vs x264 4:2:0, MS-SSIM | derived | −0.42 | −0.46 | **−0.62** | +19.41 → +19.21 |

DAVIS rows have no "both" change because DAVIS anchors stay at 30 fps.

What moved:
- **Fix 1** is worth about 1 point on DAVIS (small streams, so 1 KB matters) and about 0.05 on
  the 1080p sets, as the audit predicted.
- **Fix 2** moves UVG vs x264 4:2:0 by 2.1 points in our favour (−18.1 → −20.2). The 4:4:4 and
  low-band rows barely move. One row moves against us by 0.1 (UVG MS-SSIM vs x264 4:2:0, +0.32 →
  +0.43). MCL-JCV moves by 0.2–1.4 points in our favour.
- The UVG **per-video mean** vs x264 4:2:0 moves much more (−5.8 → −14.1). At 30 fps the
  anchor curve stopped below our top point on 6 of 7 videos, so each video's comparison window
  was cut short. At 120 fps with CRF 0–6 every video is fully covered. The pooled headline was
  much less affected.

Not a headline, but it moved: UVG vs **x265 4:2:0** goes from +20.9% to +8.6% for the same
reason (the 30 fps x265 curve also stopped short).

## Coverage (amendment 1): top PSNR per UVG video, dB

| Video | Ours (top) | x264 4:2:0 @30 | x264 4:2:0 @120 | x264 4:4:4 @30 | x264 4:4:4 @120 |
|---|---|---|---|---|---|
| Beauty | 38.14 | 40.65 | 42.51 | 42.82 | 52.14 |
| Bosphorus | 42.95 | 41.99 | 44.28 | 43.89 | 52.02 |
| HoneyBee | 40.28 | 39.76 | 41.71 | 42.33 | 52.11 |
| Jockey | 41.20 | 40.78 | 42.98 | 43.04 | 52.13 |
| ReadySetGo | 41.15 | 40.27 | 42.61 | 42.41 | 52.13 |
| ShakeNDry | 40.37 | 40.12 | 42.75 | 42.05 | 52.12 |
| YachtRide | 42.35 | 41.34 | 43.85 | 43.37 | 52.09 |

The 120 fps columns include CRF 0–6. Without them, the same CRF 8 lands about 1.4 dB lower at
120 fps than at 30 fps.

## Lesson: the −1.9% that was not quoted

The first UVG 4:2:0 run at 120 fps used the old CRF list (8–44). Per video, x264 was exactly as
efficient as at 30 fps (BD within ±0.6%). But CRF 8 now topped out below our top point on
every video, so BD-rate only compared the curves at lower quality, where we are weaker. The
pooled number came out −1.9% instead of −18.1%. The plan's claim that both fixes "can only
lower our BD-rate" was wrong. The audit had measured one video, and BD-rate depends on where
the curves overlap. Amendment 1 (CRF 0–6, agreed before any further run) fixed the coverage.
The −1.9% is kept here as a record, not quoted.

**Rule we take from this:** whenever anchors change, check that they cover our quality range
on every video before reading the BD-rate.
