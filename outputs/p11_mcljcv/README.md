# MCL-JCV, a second 1080p test set — measured (2026-11-23)

Plan, fixed before any download or result: [`docs/PLAN_mcljcv.md`](../../docs/PLAN_mcljcv.md).
This is a **measurement, not a gate**: nothing was trained or tuned, and MCL-JCV chose no setting.
Data: [`scripts/prepare_mcljcv.py`](../../scripts/prepare_mcljcv.py) (range-read of the 30
1080p sources from HF `uscmcl/MCL-JCV_Dataset`, first 96 frames, BT.709 → RGB, cropped to
1920×1072, same protocol as UVG). Chain: [`run_all.sh`](run_all.sh), report:
[`report.py`](report.py) → [`results.json`](results.json).

Codec: the current MSE set (`outputs/p6_low_rate/vimeo` + `p4_wide_latent`) and the MS-SSIM set
(`outputs/p10_msssim/vimeo`), A2 wide-latent P-coder, GOP 30, bit-aware motion, untuned,
real bytes. **Every stream of both sets decodes bit for bit** (300 streams).
Anchors: x264/x265 medium, GOP 30, CRF 8–44, 4:2:0 and 4:4:4, default and low-delay, scored on
the same RGB frames (`outputs/benchmarks/mcljcv_classical_{420,444}_gop30/`).

## Headline (BD-rate, 5-point; low band λ 1e-2/3e-3/1e-3 in brackets; − = fewer bits than the anchor)

| Anchor | MSE set, PSNR | MS-SSIM set, MS-SSIM | UVG MSE set, PSNR (for reference) |
|---|---|---|---|
| x264 4:2:0 | **−1.4%** (+37.6%) | **−0.4%** (+16.2%) | −18.1% |
| x264 4:4:4 | +40.1% (+58.1%) | +10.3% (+24.6%) | +17.9% |
| x264 4:2:0 low-delay | **−12.3%** (+17.7%) | **−12.3%** (+0.5%) | |
| x264 4:4:4 low-delay | +23.6% (+34.8%) | **−3.2%** (+7.1%) | −3.7% |
| x265 4:2:0 | +26.6% (+85.9%) | +10.0% (+35.0%) | |
| x265 4:4:4 | +82.0% (+108.1%) | +15.5% (+36.6%) | +68.1% |
| x265 4:2:0 low-delay | +6.3% (+47.8%) | **−8.5%** (+8.8%) | |
| x265 4:4:4 low-delay | +48.6% (+64.8%) | **−5.1%** (+9.3%) | |

**UVG consistency check (rule fixed in the plan):** MCL-JCV vs x264 4:2:0, PSNR, MSE set, 5-point
is −1.4%, against UVG's −18.1%: **16.7 points worse, inside the 20-point limit, so the UVG
result is not called UVG-specific.** But the lead over x264 4:2:0 shrinks to about parity on
these 30 videos, and the low band is clearly behind (+37.6%).

In short: on a fresh 1080p set the codec **ties x264 4:2:0** (PSNR with the MSE set, MS-SSIM with
the MS-SSIM set), **beats x264 4:2:0 low-delay by 12%** on both metrics, and on MS-SSIM beats x265
low-delay. It is still behind x264 in full colour (4:4:4) on PSNR (+40.1%), behind x265 default,
and weak at low rate.

## Per video vs x264 4:4:4 (BD-rate of the headline metric)

- **MSE set, PSNR: 6 wins / 24 losses.** Wins: SRC17 −35.3, SRC14 −32.3, SRC16 −24.4, SRC04
  −9.1, SRC07 −5.9, SRC06 −5.9. Worst: SRC20 +422.6, SRC13 +289.7, SRC30 +174.6, SRC18 +172.8,
  SRC26 +147.2, SRC29 +135.8, SRC12 +102.2.
- **MS-SSIM set, MS-SSIM: 11 wins / 19 losses.** Wins: SRC06 −51.2, SRC17 −40.8, SRC14 −33.2,
  SRC04 −32.6, SRC01 −31.5, SRC16 −30.4, SRC08 −24.5, SRC07 −24.2, SRC05 −11.4, SRC11 −8.6,
  SRC21 −2.9. Worst: SRC20 +361.2, SRC13 +207.1, SRC26 +115.7.

As on DAVIS (cows, schoolgirls), the average hides a long tail: a handful of videos (SRC20,
SRC13, SRC18, SRC26, SRC29, SRC30) cost 2–5× x264's bits and dominate the mean. That tail, not
the typical video, is the remaining gap. Full per-video numbers for both metrics are in
`results.json`.

## Per point

| λ | MSE set bpp / PSNR / MS-SSIM | MS-SSIM set bpp / PSNR / MS-SSIM | motion share (MSE) |
|---|---|---|---|
| 1e-2 | 0.053 / 32.23 / 0.9400 | 0.057 / 30.23 / 0.9530 | 28% |
| 3e-3 | 0.081 / 34.83 / 0.9627 | 0.086 / 32.71 / 0.9693 | 17% |
| 1e-3 | 0.136 / 36.70 / 0.9748 | 0.192 / 35.51 / 0.9824 | 12% |
| 3e-4 | 0.312 / 38.09 / 0.9830 | 0.418 / 37.41 / 0.9893 | 5% |
| 1e-4 | 0.589 / 40.71 / 0.9905 | 0.754 / 39.16 / 0.9929 | 3% |

Speed (RTX 5060 laptop, 1920×1072): encode 1.2–1.3 s/frame, decode 0.41–0.48 s/frame. Not real time.

## Notes and limits

- First 96 frames per video only (the plan's choice, as for UVG); MCL-JCV sources run 24/25/30 fps.
- The anchor loop was rewritten mid-run to load each video once (`benchmark_parity.py
  --sequence-major`). Bytes are identical to the old path and scores agree to about 1e-7 (the
  /255 runs on the GPU); all anchor points come from the new path.
- MCL-JCV is now a reported test set. It must not be used to choose anything; if it ever is, a
  further fresh set is needed. Whether any MCL-JCV clip also appears in the Vimeo training data
  has not been checked (different sources, so unlikely) — queued for the item-8 audit.
