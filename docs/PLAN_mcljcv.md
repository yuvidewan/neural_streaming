# Plan: MCL-JCV, a second 1080p test set (fixed 2026-11-21, before any download or result)

> **Status 2026-11-23: measured** ([`outputs/p11_mcljcv/README.md`](../outputs/p11_mcljcv/README.md)). MSE set −1.4% vs x264 4:2:0 (PSNR), MS-SSIM set −0.4% (MS-SSIM); UVG consistency: 16.7 points worse than UVG, inside the 20-point rule, so not UVG-specific. All 300 streams decode-exact.

Roadmap item 7 (`docs/ROADMAP_2026-11-17.md`): "Few test sets. UVG (done), then MCL-JCV."
DAVIS TEST is 256×256 and UVG is only 7 videos, so one more standard 1080p set guards against
results that only hold on those videos. MCL-JCV is the third set that learned video codecs
(DVC, DCVC and later) report next to UVG and HEVC Class B.

This is a **measurement, not a test of a change**. Nothing is trained or tuned, and MCL-JCV is
never used to choose any setting.

## Data

- **Source:** USC MCL, `uscmcl/MCL-JCV_Dataset` on Hugging Face, linked from
  mcl.usc.edu/mcl-jcv-dataset. The licence is MIT-style (free use, with the copyright notice
  kept).
- **All 30 videos**, 1920×1080, 8-bit YUV 4:2:0, 24/25/30 fps, 120–150 frames. They are the
  `MCL_JCV/1080P/YUV_source/*.yuv` members of zips 003–010.
- **Download only what is used.** HTTP range requests read just these members out of the zips,
  not the 79 GB repository (which is mostly H.264 test clips). The decompression stops after
  the first 96 frames. That is about 9 GB raw, deleted once converted.
- **Protocol:** the same as UVG (`scripts/prepare_uvg.py`) and the literature:
  - the first 96 frames of each video;
  - BT.709 to RGB;
  - the height centre-cropped from 1080 to 1072;
  - every video in split "test".
  A new `scripts/prepare_mcljcv.py` writes the manifest. The PNGs stay in `C:/mcljcv_scratch`.

## What is coded (one job at a time)

1. **Anchors:** x264/x265 with `scripts/benchmark_parity.py --skip-nvc`, GOP 30, the same CRF
   list as UVG (8–44).
   - 4:2:0 and 4:4:4, default and low-delay.
   - Scoring runs on the GPU. It ran on the CPU for UVG, and that took most of the time.
2. **The MSE set:** the current codec (`outputs/p8_motion` settings: A2 + wide I, GOP 30,
   bit-aware motion λ_m), all five λ, untuned, real bytes, decode-exact.
3. **The MS-SSIM set** (`outputs/p10_msssim/vimeo`): the same settings.

## Reported (no pass/fail; fixed before any result)

- For each set: BD-rate PSNR and MS-SSIM against x264/x265, for every anchor (4:2:0 / 4:4:4 ×
  default / low-delay), 5-point and low band, sequence mean.
- PSNR is quoted for the MSE set, and MS-SSIM for the MS-SSIM set.
- Per video, with the wins and losses against x264 4:4:4 counted.
- **A consistency check against UVG,** stated now: if MCL-JCV's BD-rate against x264 4:2:0
  (PSNR, MSE set, 5-point) is more than 20 points worse than UVG's −18.1%, the UVG result is
  called UVG-specific everywhere it is quoted.
- Decode-exactness, bytes, and encode/decode time.

## Time (estimates)

| Step | Time |
|---|---|
| Download and convert | about 30 min |
| Anchors | about 5–6 h per colour format |
| MSE set | about 8–9 h (4.3× UVG's frames) |
| MS-SSIM set | about 8–9 h |
| **Total** | **about 1.2 days of machine time** |
