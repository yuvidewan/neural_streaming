# The refitted codec's full-video scoreboard: +214.2% against x264

2026-10-25. `scripts/benchmark_parity.py`, GOP 10, full DAVIS TEST (9 sequences,
719 frames), `decode-exact=True` at every rate point, `run valid: True`.

This is the number the six-milestone P-frame refit exists to produce: the
64-channel GDN/residual transform with QAT, carrying the **refitted** M10K →
M10L → M11-G16 → M13 → M14 stack (`outputs/stage1_refit/`), measured on the same
single-pass harness and the same primary convention as the Stage 0 scoreboard.

## Result

| anchor | deployed codec | 64ch+QAT, fixed stack | **64ch+QAT, refitted** |
|---|---|---|---|
| x264 default, PSNR | +450.7% | +248.1% | **+214.2%** |
| x264 I-10 no-B, PSNR | +300.5% | +152.4% | **+127.8%** |
| x265 default, PSNR | +485.0% | +269.8% | **+233.7%** |
| x265 I-10 no-B, PSNR | +336.0% | +175.2% | **+148.4%** |
| x264 default, MS-SSIM | +378.3% | +237.0% | **+203.9%** |
| x265 default, MS-SSIM | +371.5% | — | **+196.6%** |

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

- **The M22 arm is absent.** This is the `nvc_deployed` arm: `calibrate_grids`'
  plain percentile residual grid with the refitted stack on top. M22's
  `symmetric_p01` grid is a *different grid* that invalidates M10K, M11-G16, the
  K=512 codebook and M13, so it cannot be appended — see
  `outputs/stage1_refit/README.md`. On the old transform M22 was worth −7.8
  points (+458.5% → +450.7%).
- **M22's recorded DAVIS totals are not a target here** and the reproduction
  check says so per point rather than failing: they belong to a different
  transform. `reproduces=None` with a stated reason, not `False`.
- **The classical arms are unchanged** from the Stage 0 scoreboard — same
  encoders, presets, CRF ladder and frames — so the three columns in the first
  table are directly comparable.

## Files

- `parity_64ch_qat.json` — the report (52 classical points + 3 NVC points)
- `run.log` — the run
