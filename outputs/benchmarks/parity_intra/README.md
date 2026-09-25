# Intra-only scoreboard - the Stage 1 gate's denominator

`PARITY_ROADMAP.md` Stage 1 is gated on *intra-only BD-rate vs the current intra
codec*. That number did not exist. This run measures it, so a trained Stage 1
transform has something to be compared against.

Run 2026-09-25, `scripts/benchmark_parity.py --gop 1`, 25 minutes, `valid: true`,
same 9 DAVIS TEST sequences and 719 frames as the Stage 0 scoreboard, same scorer,
same primary convention (pooled BPP; per-frame quality averaged per sequence, then
over sequences).

```
./.venv/Scripts/python.exe scripts/benchmark_parity.py --gop 1 \
    --output-dir outputs/benchmarks/parity_intra --output-name parity_intra.json --no-resume
```

## Headline

**The current codec, all-intra, is +176.2% BD-rate against x264 all-intra
(+167.6% on MS-SSIM).** Positive means NVC needs more bits for the same quality.

| | 3-bit | 4-bit | 5-bit |
|---|---:|---:|---:|
| NVC BPP | 0.4480 | 0.7012 | 0.9555 |
| NVC PSNR (dB) | 26.228 | 28.692 | 29.601 |
| bits vs x264 all-intra at that quality | 3.16x | 2.53x | 2.72x |

Across the four reporting conventions the figure ranges **+127.5% to +176.2%** on
PSNR and +150.3% to +167.6% on MS-SSIM. The primary convention is the one quoted.

## What this says about the roadmap

The point of measuring it was to split the Stage 0 deficit into a transform part
and a temporal part for the first time. Comparing like with like - NVC against
x264 forced into NVC's own structure, in both runs:

| structure | NVC vs x264, BD-rate (PSNR) |
|---|---:|
| all-intra (this run) | **+176.2%** |
| I every 10, no B (Stage 0) | +306.2% |

The gap nearly doubles once temporal prediction is in play, which says NVC's
motion compensation is the weaker half. Measured directly as what each codec
loses when forced all-intra, at matched PSNR:

| | bpp @ 28.0 dB | bpp @ 29.0 dB | bpp @ 29.5 dB | cost of all-intra |
|---|---:|---:|---:|---:|
| x264, I every 10 | 0.0842 | 0.1082 | 0.1239 | - |
| x264, all-intra | 0.2318 | 0.3026 | 0.3444 | **2.8x** |
| NVC, GOP 10 | n/a | 0.4343 | 0.5379 | - |
| NVC, all-intra | 0.6301 | 0.7874 | 0.9273 | **1.8x** |

x264's temporal tools buy it 2.8x. NVC's buy it 1.8x. Both codecs get worse
without P-frames; x264 gets worse *faster*, which is why the intra-only gap is the
smaller of the two. The transform is still the single largest deficit (+176% with
no motion involved at all, which is what Stage 1 targets), but this run puts a
number on the temporal half rather than leaving it implied, and it is not small.

## The x265 all-intra arm is broken - do not use it

The report contains `nvc_*_vs_h265_intra` at +31.3%. **That number is not usable.**
x265 configured all-intra is performing worse than x264 configured all-intra,
which is backwards for any correctly-configured encoder:

| | bpp @ 29.0 dB |
|---|---:|
| x264, I every 10 (Stage 0) | 0.1082 |
| x265, I every 10 (Stage 0) | 0.0999 |
| x264, all-intra (this run) | 0.3026 |
| x265, all-intra (this run) | **0.5326** |

At GOP 10 x265 beats x264, as it should. At `keyint=1` it needs 76% *more* bits
than x264 for the same quality, and its rate curve flattens out at high CRF
(0.3772 bpp at crf38 to 0.3388 bpp at crf44, while PSNR falls 25.4 to 23.4 dB) -
a rate floor that a healthy encoder does not have. So the flattering +31.3% is a
statement about this x265 configuration, not about NVC being near H.265 intra.

The cause has not been diagnosed. `--x265-params keyint=1:min-keyint=1:scenecut=0:bframes=0`
is the suspect. This does **not** affect Stage 0: its x265 arms behave normally.

## Method notes

- **Calibration runs at GOP 10, coding at GOP 1.** The deployed codec's residual
  grid, context model, codebooks and motion table are all fitted on P-frame data;
  at GOP 1 there are none and `calibrate_grids` raises on an empty collection. The
  intra quantization params and entropy model this run uses are fitted on I-frame
  latents and do not depend on the GOP. Calibrating at the deployed GOP is also
  what makes this *the current intra codec* rather than a re-tuned one.
- **`--gop` does not reach the `h264` / `h265` arms.** They keep the encoder's own
  GOP and B-frames whatever `--gop` says, so `nvc_*_vs_h264` in this report
  (+920.3%) compares all-intra NVC against H.264 *with* temporal prediction. It is
  context, not the gate. The gate is `nvc_deployed_vs_h264_intra`.
- **The M22 reproduction check does not apply here.** M22 recorded its DAVIS totals
  at the deployed GOP; at any other GOP there is nothing to reproduce, so the check
  reports `null` with a reason rather than a failure.
- **`nvc_deployed` and `nvc_m22` are byte-identical** at every rate point
  (2,638,574 / 4,130,011 / 5,628,186 bytes). M22's residual re-centering only
  touches P-frame residuals, so it is inert with no P-frames - an expected result
  that also confirms the intra path is free of it.
- **Decode is bit-exact** at every point (`decode_reconstruction_exact: true`).
