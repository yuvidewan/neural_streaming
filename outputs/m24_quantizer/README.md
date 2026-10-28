# M24 Phase A - the quantizer at frozen weights, and the ceiling it imposes

2026-10-28. `scripts/m24_offline_gate.py`, 400 TRAIN frames fitting every grid and
every density, 80 VAL-B frames reporting, TEST never opened.

The CompressAI anchor found `bmshj2018-factorized` at about half our all-intra
bits on a design that is ours minus the quantizer. That gap mixes three causes -
the grid rule, the density model, and rate-aware lambda training - which need
different work. This phase separates them on the latents the current transform
already produces, so nothing is retrained and nothing is assumed.

## The decomposition

| cause | BD-rate at frozen weights |
|---|---|
| density model (per-channel histogram -> discretised Laplace) | **+0.37%** (slightly worse) |
| grid rule (percentile depth -> unit step) | **-8.30%** |
| both together | -7.96% |

**The density model is worth nothing.** A per-channel histogram on a bounded
alphabet, with 400 frames behind it, already beats a parametric Laplace. So the
"learned per-channel density" the anchor named is not where the gap is.

**The grid rule is worth -8.30%**, available with no retraining at all.

## But the saving is strongly rate-dependent, and that is the real finding

| grid point | PSNR | grid bpp | unit-step bpp at the same PSNR | saving |
|---|---|---|---|---|
| 3-bit | 28.087 dB | 0.46884 | 0.44186 | **-5.8%** |
| 4-bit | 31.124 dB | 0.72985 | 0.69706 | **-4.5%** |
| 5-bit | 32.199 dB | 0.98707 | 0.92230 | **-6.6%** |
| 6-bit | 32.500 dB | 1.24104 | 1.07154 | -13.7% |
| 7-bit | 32.573 dB | 1.49311 | 1.14904 | **-23.0%** |

At the operating points the codec actually ships - 3, 4 and 5 bits - the grid
rule is worth **4.5 to 6.6%**. The headline -8.30% is the BD-rate integrated over
the whole overlap, and it is pulled up by the high-rate end.

**Why the high-rate end diverges: the percentile grid saturates.**

| bits | grid PSNR | gain over the previous depth |
|---|---|---|
| 3 | 28.087 dB | - |
| 4 | 31.124 dB | +3.04 |
| 5 | 32.199 dB | +1.08 |
| 6 | 32.500 dB | +0.30 |
| 7 | 32.573 dB | **+0.07** |

Doubling the levels from 64 to 128 buys **0.07 dB for 0.25 bpp**. The grid clips
everything outside the 0.1/99.9 percentile bounds, and no number of levels INSIDE
the range repairs a clipped value, so the distortion approaches a floor the depth
cannot cross. The asymptote sits near **32.6 dB on this transform**.

Unit-step rounding has no bounds, and at 1.49 bpp it reaches **32.707 dB** -
already past the grid's ceiling - and was still improving when the sweep stopped.

**So our intra quantizer caps reconstruction at about 32.6 dB regardless of bit
depth.** For a codec chasing H.264 parity, that is a hard ceiling rather than an
inefficiency, and it is the strongest argument in this phase for replacing the
quantizer rather than tuning it.

## A correction I owe the record

An early smoke configuration of this gate - 40 TRAIN frames, 4 VAL frames per
sequence, two grid points and three step points - reported the grid rule at
**-23.08%**, and that number was relayed as a preliminary signal. The full
measurement says **-8.30%**. The sparse curves had almost no overlap, so the
BD-rate was integrating over two or three points at the high-rate end, which is
exactly where the gain is largest.

This is the fourth preliminary or projected figure in this line of work to come in
high (the others: the learned stack's +170%, the hyperprior's ~5-6 BD-rate points
twice over). The pattern is consistent and worth naming: an estimate taken from a
narrow or sparse measurement of this codec has been optimistic every time, and the
honest move is to quote the full measurement or nothing.

## What this does and does not say about the anchor's +90.9%

It says **the quantizer and the density together explain about 8 points of it at
frozen weights, and the density explains none of them.** The remaining bulk must
come from rate-aware training - weights trained against a real rate term at a
given lambda - which is consistent with the anchor's own note that a lambda sweep
on the old codec moved the 4-bit point by at most 9.7% in rate while the quantizer
stayed fixed.

It cannot price rate-aware training, because that needs Vimeo and a GPU. That is
Phase B.

## Recommendation for Phase B

1. **Replace the quantizer before training anything.** Unit-step rounding with an
   unbounded integer alphabet, per-channel density, no percentile clipping. It is
   worth 4.5-6.6% at the shipped operating points on its own, it removes the
   32.6 dB ceiling, and every rate-aware training run afterwards would otherwise
   be optimising against a quantizer that caps its own quality.
2. **Then add the rate term and train per lambda.** With the ceiling gone, the
   lambda sweep the anchor found nearly useless becomes meaningful - its own note
   says per-lambda training only helps after the quantizer changes.
3. **Do not spend effort on the density model.** Measured at +0.37%: the
   histogram already wins.

## Files

- `m24_offline_gate.json` - 20 points, three arms, every pairwise BD-rate
- `gate.log` - the run (not committed; `*.log` is ignored repo-wide)
