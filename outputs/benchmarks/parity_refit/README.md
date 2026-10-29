# The refitted codec's full-video scoreboard: +214.2% against x264

2026-10-25. `scripts/benchmark_parity.py`, GOP 10, full DAVIS TEST (9 sequences,
719 frames), `decode-exact=True` at every rate point, `run valid: True`.

This is the number the six-milestone P-frame refit exists to produce: the
64-channel GDN/residual transform with QAT, carrying the **refitted** M10K →
M10L → M11-G16 → M13 → M14 stack (`outputs/stage1_refit/`), measured on the same
single-pass harness and the same primary convention as the Stage 0 scoreboard.

## Result

| anchor | deployed codec | 64ch+QAT, fixed stack | **refitted** | **refitted + M22** |
|---|---|---|---|---|
| x264 default, PSNR | +450.7% | +248.1% | **+214.17%** | +214.12% |
| x264 I-10 no-B, PSNR | +300.5% | +152.4% | **+127.79%** | +127.72% |
| x265 default, PSNR | +485.0% | +269.8% | **+233.74%** | +233.66% |
| x265 I-10 no-B, PSNR | +336.0% | +175.2% | **+148.38%** | +148.30% |
| x264 default, MS-SSIM | +378.3% | +237.0% | **+203.87%** | +204.60% |
| x265 default, MS-SSIM | +371.5% | n/a | **+196.61%** | +197.35% |

Both NVC arms are quoted to two decimals for one reason: at one decimal they are
the same number. See "M22 is dead on this transform" below. The headline arm is
`nvc_deployed`, the simpler of the two.

Against the codec this project actually shipped, the transform plus its refitted
stack is worth **−236.5 BD-rate points** on full video: +450.7% → +214.2%.

## The refit's own contribution, cleanly isolated

The refit changes entropy coding only. Reconstruction is untouched, and the
measurement says so: PSNR and MS-SSIM agree with the fixed-stack run to every
digit reported, independently, through a different script. So the byte column
below is the whole effect.

| bits | fixed-stack bpp | refitted bpp | bytes | PSNR (both) | MS-SSIM (both) |
|---|---|---|---|---|---|
| 5 | 0.8500 | **0.7854** | **−7.60%** | 32.858 | 0.9863 |
| 4 | 0.6136 | **0.5569** | **−9.25%** | 32.289 | 0.9821 |
| 3 | 0.4016 | **0.3570** | **−11.10%** | 30.560 | 0.9661 |

The gain grows as the quantizer coarsens, which is the direction M11 and M13
both predicted on VAL-B.

## The stack transfers at about half its old value

Apples to apples, the `nvc_deployed` arm on both sides:

| transform | fixed stack | + learned stack | the stack is worth |
|---|---|---|---|
| 593k baseline | +528.5% | +458.5% | **−70.0 points** |
| 64ch + QAT | +248.1% | **+214.2%** | **−33.9 points** |

**48%.** The learned entropy stack earns less than half as much on the better
transform, and that is the expected direction rather than a disappointment: a
transform that decorrelates more leaves less redundancy for an entropy model to
collect. Every component said this on the way through — M11 down 1.1–1.5 points
of its own gain, M13's 5-bit point falling just under the 1% bar, M10K alone
roughly unchanged.

### It corrects a projection I had flagged twice

`outputs/benchmarks/fullvideo_transform/README.md` estimated **~+170%** by
assuming the stack would be worth the same 70–78 points on the new transform as
on the old one. Measured: **+214.2%**. The projection was optimistic by 44
points, for exactly the reason the M11 refit note gave when it walked the
estimate back. The estimate is now superseded by this measurement and should not
be quoted.

## M22 is dead on this transform

M22's `symmetric_p01` residual grid was the project's first real coded-rate win
since M13/M14: on the deployed transform it shipped +1.06-1.76% of bytes and
**-7.78 BD-rate points** against x264. Refitted here through its own two-arm
sweep (`outputs/stage1_refit/m22_64ch_qat/`), measured on the same DAVIS TEST:

| bits | plain-grid bpp | M22 bpp | bytes | PSNR plain | PSNR M22 | dPSNR |
|---|---|---|---|---|---|---|
| 5 | 0.7854 | 0.7802 | -0.67% | 32.858 | 32.863 | +0.005 |
| 4 | 0.5569 | 0.5535 | -0.61% | 32.289 | 32.272 | -0.017 |
| 3 | 0.3570 | 0.3533 | -1.05% | 30.560 | 30.504 | -0.056 |

| anchor | M22 is worth, old transform | M22 is worth, here |
|---|---|---|
| x264, PSNR | **-7.78 points** | **-0.05 points** |
| x265, PSNR | **-8.23 points** | **-0.08 points** |
| x264, MS-SSIM | - | **+0.73 points (worse)** |

**The rate gain survives; the BD-rate gain does not.** M22 still saves
0.61-1.05% of bytes, and its VAL-B gate still passes the distortion guard. But a
grid re-centering is a rate/distortion move by construction, and here it pays
for those bytes with -0.017 to -0.056 dB. BD-rate prices both axes, so the two
nearly cancel: 0.05 points on PSNR, and on MS-SSIM the arm is net WORSE by 0.73.

So the two arms are the same codec to within measurement noise, and the simpler
one is the right one to carry forward. **This is the result that justified
running the arm rather than assuming it.** Had the old -7.78 points been carried
over by analogy, the paper's Table I arm would have been credited with a gain it
no longer has.

### Its VAL-B gate agrees, and says it more precisely

| bits | stale arm | refit arm | verdict |
|---|---|---|---|
| 5 | -4.6575% | +0.6029% | marginal |
| 4 | -17.8409% | +0.6242% | marginal |
| 3 | -53.7939% | +0.7913% | marginal |

M22's own pre-registered bars are weak below 0.5% and meaningful above 1.0%. At
its screening point (3-bit) it was **+1.5158%, meaningful** on the old transform
and is **+0.7913%, marginal** here: a **52% transfer**, within a few points of
the 48% the rest of the stack showed. The whole entropy stack - learned models
and recalibrations alike - is worth about half as much once the transform does
more of the work.

The stale arm is the reason M22 needs a sweep rather than a flag: swapping the
grid and leaving the stack alone costs **-4.66 / -17.84 / -53.79%** of the
stream (the old transform's 3-bit figure was -75.16%). The coupling is milder on
this transform but still catastrophic.

## Superseded on 2026-10-26: +210.70% with the M23 scale map

`outputs/m23_hyperprior/README.md` measures a transmitted per-position scale map
in a real `.nvct` v3 container on this same split and convention: **+214.17% ->
+210.70%** against x264, **-3.46 BD-rate points**, with PSNR unchanged and every
round trip exact. The `nvc_deployed` baseline there reproduces this report's
totals to the byte (4,626,280 / 3,279,938 / 2,102,902 B), which is what makes the
two directly comparable.

The +214.17% below remains the correct figure for the codec WITHOUT that side
channel.

## Still not parity, and by how much

| bits | bpp | PSNR | bits vs x264 at the same quality |
|---|---|---|---|
| 3 | 0.3570 | 30.560 | **3.07×** |
| 4 | 0.5569 | 32.289 | **3.06×** |
| 5 | 0.7854 | 32.858 | **3.71×** |

Down from the deployed codec's ~5.5×, but the goal is ≤1.0×. The gap is still
architectural: the 5-bit point costs proportionally more than 3- and 4-bit,
which is the quantization-fragility signature the Stage 1 diagnosis identified,
not an entropy-coding shortfall.

## Read this correctly

- **The M22 arm is present and measured**, as its own two-arm sweep, because
  M22's `symmetric_p01` grid cannot be appended to the chain - it invalidates
  M10K, M11-G16, the K=512 codebook and M13. See `outputs/stage1_refit/README.md`.
  Each `nvc_m22` point records the sha256 of the refit checkpoint it was coded
  with, which pins the arm's provenance per point rather than by directory path.
- **M22's recorded DAVIS totals are not a target here** and the reproduction
  check says so per point rather than failing: they belong to a different
  transform. `reproduces=None` with a stated reason, not `False`.
- **The classical arms are unchanged** from the Stage 0 scoreboard — same
  encoders, presets, CRF ladder and frames — so the three columns in the first
  table are directly comparable.

## Files

- `parity_64ch_qat.json` — the report (52 classical points + 3 NVC points)
- `run.log` — the run
