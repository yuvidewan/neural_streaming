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

---

# M24 Phase A2 - the residual quantizer: the hypothesis did not survive

2026-10-28. `scripts/m24_residual_gate.py`, 358 TRAIN P-frames fitting, 72 VAL-B
P-frames reporting, TEST never opened. `m24_residual_gate.json`.

Phase A found the INTRA grid capping near 32.6 dB. The residual path uses the
same scheme and is 79.6-84.2% of the stream, so the question was whether the
ceiling binds there too - and, if it did, whether it was the common cause behind
five consecutive negative milestones on that path (M17-M21).

**It is not.** The precondition holds, the conclusion does not.

## What held: the residual grid does saturate

| bits | PSNR | gain | bpp | latent residual MSE |
|---|---|---|---|---|
| 3 | 30.890 dB | - | 0.29859 | 2.096099 |
| 4 | 32.062 dB | +1.17 | 0.54333 | 0.626874 |
| 5 | 32.466 dB | +0.40 | 0.79268 | 0.217979 |
| 6 | 32.569 dB | +0.10 | 1.04386 | 0.118911 |
| 7 | 32.595 dB | **+0.03** | 1.29520 | 0.094760 |

Same shape as intra, same plateau near 32.6 dB. Unit-step rounding gets to
32.704 dB at the same rate with **11x less latent error** (MSE 0.0084 against
0.0948), so the plateau is not a limit of what the data allows.

## What did not hold: it is worth no more here than on intra

| | intra (Phase A) | residual (Phase A2) |
|---|---|---|
| density model | +0.37% | **+3.51%** (worse) |
| grid rule | -8.30% | **-8.31%** |
| both together | -7.96% | **-5.09%** |

The grid rule is worth the same -8.3% on the path that carries 80% of the stream
as on the path that carries 12%. And because a parametric Laplace is *worse* on
residuals than the deployed histogram, the two changes together come to only
**-5.09%**.

## And at the shipped operating points it is mostly not a win at all

| matched quality | grid bpp | unit-step bpp | saving |
|---|---|---|---|
| 30.890 dB (~3-bit) | 0.29859 | 0.31765 | **+6.4% - the grid WINS** |
| 32.062 dB (~4-bit) | 0.54333 | 0.52523 | -3.3% |
| 32.466 dB (~5-bit) | 0.79268 | 0.70753 | -10.7% |
| 32.569 dB (~6-bit) | 1.04386 | 0.79259 | -24.1% |
| 32.595 dB (~7-bit) | 1.29520 | 0.83353 | -35.6% |

The crossover sits between 3- and 4-bit, inside the range the codec ships.
**At 3-bit the percentile grid is better by 6.4%.**

That is explicable rather than odd: the grid fits a step per channel to that
channel's own range, which is a crude form of per-channel rate allocation and
genuinely helps when the step is coarse. Its clipping only starts to dominate
once the step is fine enough that the tails are what is left to code. So the two
rules win at opposite ends, and our operating points sit near the crossing.

## The reinterpretation of M17-M21 is refused

The hypothesis was that a clipping ceiling in the residual quantizer was the
common cause behind M17's "unimplementable" oracle gap, M18's "aggregate MSE does
not transfer to byte savings", M19, M20's rejected hysteresis and M21's "local
optimum". Set the two side by side:

| | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| M17's oracle gap (total-stream) | +2.0% | +6.7% | +14.9% |
| grid rule worth here | -10.7% | -3.3% | **+6.4%** |

M17's gap GROWS as the rate falls; the grid's cost SHRINKS and then reverses.
They move in opposite directions. The grid rule is not what those five
milestones were hitting, and this line is closed rather than reopened.

Worth having run: it was hours, it used machinery that already existed, and a
plausible explanation for five negative results is worth an afternoon to refute.

## A fifth optimistic preliminary, and the same lesson

The smoke configuration of this gate - 60 TRAIN frames, two grid points, two step
points - reported the grid rule at **-35.16%**. The full measurement says
**-8.31%**. The sparse curves overlapped only at the top, which is exactly where
the -35.6% matched-quality figure above lives: the smoke was reading the
high-rate end and calling it the average.

That is the fifth preliminary or projected figure in this line of work to come in
high. The pattern is now well enough established to treat as a rule rather than a
coincidence: on this codec, a number from a narrow measurement is optimistic, and
only the full curve is quotable.

## What this changes for Phase B

The anchor's +90.9% is now **overwhelmingly attributable to rate-aware training**,
not to the quantizer. Quantizer and density together are worth -5.09% on the
residual path and -7.96% on intra, at frozen weights, with the residual side
turning negative at the lowest operating point.

So the Phase A recommendation needs amending. **Replacing the quantizer is not a
standalone win to bank first.** It is a prerequisite: unit-step rounding with an
unbounded alphabet is what makes a differentiable rate term meaningful, and the
pairing is what bmshj2018 actually does. Changing the quantizer without the rate
term buys a few percent on intra and roughly nothing on residuals at the rates
this codec uses.

Phase B is therefore one piece of work, not two: **unit-step quantizer plus a
real rate term, trained per lambda.** Needs Vimeo and a GPU.

## Files

- `m24_residual_gate.json` - 20 points, three arms, the walk check and both identity controls
- `residual_gate.log` - the run (not committed; `*.log` is ignored repo-wide)

## The walk was verified, not trusted

Pixel-domain scoring needs the reference latent for every residual, which
`m22_residual.collect_train_residuals` does not return, so this module walks the
closed loop itself. Its docstring records that two details there are load-bearing
and that one was gotten wrong once before `verify_deployed_grid` caught it. So
the walk is checked against m22's own: **identical, max absolute difference
0.0**, on both the smoke and the full TRAIN set, and `verify_deployed_grid`
confirms the grid rebuilt from it is the deployed one. The run exits non-zero if
either check fails.
