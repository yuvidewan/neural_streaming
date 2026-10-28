# M24 Phase B - the brief, and how to run it

Prepared 2026-10-28. Phase A and A2 are measured and committed; this is the work
they point at. It needs Vimeo-90K and a GPU, so it runs on Colab.

## What Phase A and A2 established, and what they did not

| | |
|---|---|
| both quantizers saturate | intra and residual both plateau near **32.6 dB** whatever the bit depth, because the percentile grid clips the tails |
| the grid rule, at frozen weights | **-8.30%** intra, **-8.31%** residual; **+6.4% WORSE at 3-bit** on residuals |
| the density model | **+0.37%** intra, **+3.51%** residual - worth nothing, do not work on it |
| the anchor's +90.9% | therefore **overwhelmingly rate-aware training**, not the quantizer |

So the quantizer is **not** a standalone win to bank. It is a prerequisite: a
fixed step is what makes a differentiable rate term mean anything, and the
pairing is what `bmshj2018` actually does. Phase B is one job, not two.

## Why this lambda sweep should bite where the last one did not

The anchor measured a lambda sweep on the old codec moving the 4-bit point by at
most **9.7%** in rate. Two milestones found the same reason from opposite sides:

- **M9F.5** (recorded in `src/nvc/training/rate_estimator.py`'s own docstring):
  the real quantizer recalibrates its grid to each model's latent range, so a
  model that shrinks its latent uniformly pays almost nothing extra in real
  coded bits - the grid shrinks with it.
- **M24** (this milestone): the same scale-invariance, measured from the coding
  side, is why cost is decoupled from content and why the grid caps quality.

Under the percentile grid, lambda had almost nothing to push against: the
encoder's cheapest response to a rate penalty was to shrink the latent, and the
grid followed. **A fixed step removes that escape route.** That is the mechanism
Phase B is betting on, and it is why the bet is different from the one that
already failed.

## The one inversion to be careful about

`--rate-track-scale` is **recommended** on the percentile quantizer and
**refused** on the unit step, and the trainer raises with an explanation rather
than letting it through.

Scale tracking existed because the percentile grid recalibrates, which made a
frozen bin width mis-specified and exploitable. A fixed step does not
recalibrate, so the codec is genuinely scale-sensitive and the frozen bin width
is now the **correct** model. Turning tracking on here would put back the
invariance the unit step is there to remove - i.e. it would reproduce the failure
Phase B exists to avoid.

Anyone following the older Phase B instructions in `train_vimeo_stage1.py`'s
docstring would reach for that flag, which is why it raises instead of warning.

## How to run it

In `colab_train_stage1.ipynb`'s configuration cell:

```python
RUN_PHASE_B   = True
QUANTIZER     = 'unit_step'
UNIT_STEP     = 1.0                      # bmshj2018's own choice
RATE_LAMBDAS  = [1e-4, 3e-4, 1e-3]       # one run each: this IS the curve
```

Section 5 (calibration) skips itself - a declared step needs no calibration
file, which also means Phase B no longer depends on a Phase A calibration pass.

Section 6 then runs one training per lambda, each into its own folder
(`..._unitstep_lambda1e-4`, and so on), each warm started from Phase A's
`best.pt` with `--init-from`.

**Each lambda needs its own folder, and a resume across the change is refused on
purpose.** The checkpoint now records the quantization *rule*, not just its bit
depth, and `model_config` cannot tell a percentile latent from a unit-step one.
`--init-from` copies weights only and resets the epoch counter, optimizer,
history and chunk list, which is what a change of objective requires.

Equivalent command line for one point:

```
python scripts/train_vimeo_stage1.py --output-dir <dir>_unitstep_lambda3e-4 \
    --init-from <phase-a>/checkpoints/best.pt \
    --quantizer unit_step --unit-step 1.0 \
    --qat-enabled --rate-enabled --rate-lambda 3e-4
```

## What to measure when it finishes, and against what

**Gate the sweep as a curve, never a single run.** Each lambda is one operating
point, and the thing being compared against - `bmshj2018`, qualities 1-8 - is
itself one model per quality. A single lambda cannot be compared to a curve.

1. `scripts/benchmark_compressai_anchor.py` for the all-intra comparison against
   `bmshj2018-factorized`. That is the number with a target: **+90.9%** today,
   and the factorized model sits at **-23.3%** against x264 all-intra.
2. `scripts/m24_offline_gate.py` with the new checkpoints, to check the ceiling
   is actually gone rather than moved - the saturation table is the test, not the
   BD-rate.
3. Only then the video path, which needs the whole entropy stack refitted for a
   third time (M10K -> M10L -> M11 -> M13 -> M14, plus M23's scale map). That is
   roughly the three hours the second refit took, and it should not be spent
   until the intra numbers say the transform is worth it.

## What would make this a failure, stated in advance

- the all-intra gap against `bmshj2018-factorized` does not close materially
  below +90.9%, in which case the remaining gap is in transform capacity or
  training scale rather than the objective, and the next question is a bigger
  transform or longer training rather than another quantizer;
- the saturation table still plateaus, which would mean something other than
  percentile clipping is capping reconstruction;
- lambda still moves the operating point by less than ~10%, which would mean the
  escape route was not the grid after all and M9F.5's diagnosis needs revisiting.

Recording these now because this project's own history says preliminary numbers
on this codec read optimistic - five times so far - and a pre-stated failure
condition is cheaper than an argument afterwards.

## What is already in place

- `--quantizer unit_step` and `--unit-step` in `scripts/train_vimeo_stage1.py`,
  with the percentile path unchanged as the default
- the noise width and the rate proxy's bin width both taken from that step,
  with no calibration file
- `--rate-track-scale` refused in unit-step mode, with the reason in the error
- the quantization rule recorded in the checkpoint so a resume across it is
  refused
- the lambda sweep in `colab_train_stage1.ipynb` section 6, with section 5
  skipping itself
- 11 tests covering all of the above, plus the 44 notebook tests still passing
