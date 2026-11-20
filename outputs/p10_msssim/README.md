# MS-SSIM model set — SMALL WIN (2026-11-21)

Plan and gate, fixed before any training or result: [`docs/PLAN_msssim.md`](../../docs/PLAN_msssim.md).
Code: `--distortion msssim`, `--init-dir`, `--init-lambda` in
[`scripts/p4_wide_latent.py`](../../scripts/p4_wide_latent.py). Chain: [`run_all.sh`](run_all.sh),
report: [`report.py`](report.py) → [`results.json`](results.json).

All 10 models (wide I-coder + A2 P-coder at λ 1e-2 … 1e-4) were fine-tuned from their own MSE
models on Vimeo chunk 7 (deleted after use) with `k·(1 − MS-SSIM) + λ·bpp`, k = 1280/255².
Coded with real bytes, GOP 30, untuned, bit-aware motion. Every stream decodes bit for bit.

## Gate

BD-rate MS-SSIM (sequence mean) of the MS-SSIM set against the current MSE set
(`outputs/p8_motion`), same frames:

| | BD-rate MS-SSIM | PASS bar | SMALL WIN bar |
|---|---|---|---|
| DAVIS TEST (9 videos) | **−22.43%** | ≤ −10% ✓ | ≤ −3% ✓ |
| UVG 1080p (7 videos) | **−9.66%** | ≤ −10% ✗ | ≤ −3% ✓ |

**Verdict: SMALL WIN.** UVG misses the PASS bar by 0.34 points. The bars were fixed before
training and are not moved. As planned, this is a **second model set**: the MSE set stays the
codec for PSNR results.

BD-rate PSNR against the MSE set (expected worse): DAVIS +79.2%, UVG +110.8%.

## Against x264 / x265, MS-SSIM BD-rate (5-point; low band in brackets)

| Anchor | DAVIS before | DAVIS now | UVG before | UVG now |
|---|---|---|---|---|
| x264 4:2:0 | +49.9% (+51.9%) | **+16.3%** (+21.4%) | +48.2% (+86.4%) | **+0.4%** (+20.9%) |
| x264 4:4:4 | +56.1% (+54.6%) | **+21.2%** (+24.0%) | +57.9% (+91.4%) | **+11.0%** (+31.7%) |
| x264 4:2:0 low-delay | +26.4% (+28.6%) | **−2.1%** (+2.6%) | +21.0% (+47.7%) | **−16.0%** (−2.1%) |
| x264 4:4:4 low-delay | +31.5% (+30.8%) | **+2.1%** (+4.9%) | +28.3% (+50.7%) | **−7.8%** (+5.5%) |
| x265 4:2:0 | +48.6% (+50.4%) | **+15.1%** (+20.0%) | +86.0% (+155.1%) | **+17.9%** (+45.9%) |
| x265 4:4:4 | +46.3% (+45.0%) | **+13.5%** (+16.0%) | +84.0% (+142.3%) | **+20.8%** (+48.2%) |
| x265 4:2:0 low-delay | +20.6% (+23.1%) | **−6.7%** (−2.0%) | +36.2% (+77.5%) | **−12.4%** (+7.1%) |
| x265 4:4:4 low-delay | +17.7% (+17.5%) | **−8.7%** (−6.2%) | +34.9% (+68.2%) | **−8.3%** (+8.8%) |

On MS-SSIM the codec now **ties x264 4:2:0 on UVG (+0.4%)** and **beats both low-delay x264
and x265 on UVG and DAVIS** (4:2:0, 5-point). On PSNR this set is far behind (e.g. +118.6% vs
x264 4:2:0 on UVG), which is why it does not replace the MSE set.

## Per video, BD-rate MS-SSIM vs the MSE set

- DAVIS: bmx-bumps −18.4, car-turn −23.2, cat-girl −17.8, **cows −30.3**, drift-chicane −5.4,
  drone −18.1, gold-fish −9.2, **schoolgirls −27.6**, surf −15.6.
- UVG: Beauty −13.7, Bosphorus −3.9, **HoneyBee +12.7** (worse), Jockey −9.7, ReadySetGo
  −16.3, ShakeNDry −0.5, YachtRide −23.8.

The two worst DAVIS videos against x264 (cows, schoolgirls) gained the most. HoneyBee is the
one video that got worse, and it is what keeps UVG just above −10%.

Per video vs x264 4:4:4 (MS-SSIM, before → now): DAVIS cat-girl, drone, gold-fish and surf are
now wins; UVG Beauty (−31.9%) and YachtRide (−24.8%) are wins, ShakeNDry is near parity
(+4.9%).

## Per point (UVG)

| λ | bpp MSE → MS-SSIM | MS-SSIM MSE → MS-SSIM | PSNR MSE → MS-SSIM |
|---|---|---|---|
| 1e-2 | 0.0386 → 0.0567 | 0.9245 → 0.9477 | 32.01 → 30.14 |
| 3e-3 | 0.0612 → 0.0884 | 0.9529 → 0.9657 | 34.60 → 33.00 |
| 1e-3 | 0.1152 → 0.2244 | 0.9692 → 0.9822 | 36.73 → 36.32 |
| 3e-4 | 0.2673 → 0.4919 | 0.9814 → 0.9897 | 38.70 → 38.15 |
| 1e-4 | 0.5921 → 0.8591 | 0.9896 → 0.9938 | 40.92 → 40.47 |

On UVG the same λ lands at a higher rate (1.4–1.9× the bits), so the two curves overlap only
partly; on DAVIS it lands at a lower rate (0.79–0.90×). The BD-rate uses only the overlap.

## Training (chunk 7, first → last logged window; PSNR is reported, the loss is MS-SSIM)

| model | 1e-2 | 3e-3 | 1e-3 | 3e-4 | 1e-4 |
|---|---|---|---|---|---|
| I: dB / bpp | 28.38/0.237 → 27.23/0.232 | 31.07/0.318 → 30.11/0.306 | 33.01/0.437 → 32.37/0.423 | 33.83/0.704 → 34.58/0.670 | 37.19/1.065 → 34.70/1.014 |
| P: dB / bpp | 29.15/0.175 → 28.51/0.176 | 32.06/0.210 → 30.94/0.209 | 34.42/0.270 → 33.08/0.260 | 36.56/0.423 → 35.95/0.393 | 37.88/0.662 → 37.31/0.632 |

Training was stopped by the user once (2026-11-20, at P 1e-3 step 2000) and resumed from
`latest.pt`; the chunk was downloaded again on resume.

## Speed

Unchanged architecture: UVG 1.19–1.53 s/frame encode, 0.41–0.59 s/frame decode; DAVIS
0.035 / 0.012 s/frame.
