# Milestone 9 Plan: Rate-Distortion Optimization

**Status of this document**: covers the full M9 objective and scope boundaries; the **implementation
section below (M9A) is complete, tested, and smoke-tested**. M9's actual training experiments
(lambda selection from measured D/R scale, multi-epoch QAT+rate training, recalibration, full
DAVIS RD benchmarking) are **explicitly not started** - approval for M9A stopped before that
point, per the brief. See "What is NOT done yet" at the end.

## 1. Objective

Determine whether explicitly optimizing rate during training - `L = D + lambda * R` - improves the
rate-distortion operating points beyond the M8 QAT result (`MILESTONE_8_RESULTS.md`), which already
improved on the control model at every bit depth without any rate-aware loss. M8 QAT is the
baseline this milestone must beat; M7 and M8 control remain secondary comparisons.

## 2. Scope boundaries (unchanged for all of M9)

Not touched, and not planned to be touched, anywhere in M9: `BaselineAutoencoder`, `Encoder`,
`Decoder`, `BaselineAutoencoder.forward()`, hard quantization (`quantization.py`),
`EmpiricalEntropyModel`, the arithmetic/range coder, the `.nvc` format, `benchmark_rd.py`'s
implementation, H.264/H.265, temporal coding. No hyperpriors, no autoregressive context models. M9
is a training-objective milestone only.

## 3. M9A - implemented, tested, smoke-tested

### 3.1 What "M9A" means here

The approved scope merged what Phase 1's plan had labeled M9A (rate-estimation infrastructure) and
M9B (the `D + lambda*R` loss) into one implementation pass, since a rate estimator cannot be
smoke-tested or unit-tested for the required properties (lambda=0 equivalence, lambda>0 gradient
contribution, different-lambda-different-loss) without the loss combination existing. What remains
separate and **not started** is M9C's actual training experiments (see §5).

### 3.2 The rate estimator: exact formulation

**Density**: one **Laplace(loc, scale)** per latent channel, `loc`/`log_scale` genuinely
**learnable** `nn.Parameter`s (128 scalars total for the 64-channel production model - trained
jointly via the rate loss itself, the standard "factorized prior" recipe simplified to a single
mode per channel).

**Bin width**: the exact same per-channel `scale` tensor `QuantizationNoise` already loads from an
existing train-split calibration file - never a second, independently derived bin width. When QAT
is also enabled, the training script constructs the rate estimator directly from
`quantization_noise.scale` (the literal same tensor object, not a second load of the same file);
`RateEstimator.from_calibration()` exists only for the rate-without-QAT case.

**Probability mass** (exact CDF difference, not density x width):

```
F(loc + t) = 0.5 + sign(t) * (1 - exp(-|t| / scale)) / 2
P(z) = F(z + w/2) - F(z - w/2)        (w = bin width)
     = G(z + w/2 - loc) - G(z - w/2 - loc),   G(t) = sign(t) * (1 - exp(-|t|/scale)) / 2
```

computed via `torch.expm1` so the "0.5 +" term cancels **algebraically in the code**, never as a
floating-point subtraction of two near-1 (or near-0) values - see
`src/nvc/training/rate_estimator.py`'s module docstring for the full derivation, including why
`torch.sign`'s zero gradient does not break backprop here (the two `sign(t)` factors - one
explicit, one from `d|t|/dt` - combine to `sign(t)^2 == 1` almost everywhere, recovering the true
Laplace density as the gradient).

`probability = clamp(probability, min=1e-9)` before `-log2(...)` guarantees a finite rate for
every input, by construction, not by luck.

**Rate**: `R = -sum(log2(P(symbol)))` over the whole latent, summed in bits, normalized to
**bits-per-pixel by the INPUT IMAGE's pixel count** (`height * width`, no channel factor) - matches
`nvc.evaluation.rd_benchmark`'s `aggregate_bpp` convention exactly (`sum(bytes)*8/sum(width*height)`),
so this training-time number is unit-comparable to the real measured payload BPP, even though the
two are not expected to match exactly (see §3.3).

**Loss**: `total_loss = distortion (plain MSE, unchanged) + lambda_rate * rate`. `lambda_rate=0.0`
reproduces the distortion-only objective exactly (proven, not assumed - see §4).

### 3.3 Training-time proxy vs. deployed entropy model - the distinction, stated plainly

| | TRAINING | DEPLOYMENT |
|---|---|---|
| What | Learned, per-channel **Laplace** density (this milestone) | Static, per-channel **integer histogram** (`EmpiricalEntropyModel`, unchanged) |
| Differentiable | Yes - the whole point | No - a discrete table lookup, no gradient path |
| Fit to | Whatever gradient descent pushes it toward during training | Real symbol counts from calibration, post-hoc |
| Used by the real coder | **Never** - not imported by, or wired into, the `.nvc` encode/decode path | Yes - this is what `encode_symbols`/`decode_symbols` actually use |

The rate proxy is expected to correlate with, but not numerically match, the real coded payload
size. Measuring how well it correlates is explicitly part of M9's evaluation phase (§5), not
assumed here.

### 3.4 Files created

- `src/nvc/training/rate_estimator.py` - `RateEstimator(nn.Module)`, per §3.2.
- `tests/test_rate_estimator.py` - 61 tests (see §4).
- This file.

### 3.5 Files modified

- `src/nvc/training/trainer.py` - **added** `train_one_epoch_with_rate` / `validate_one_epoch_with_rate`,
  alongside the pre-existing `train_one_epoch` / `validate_one_epoch`, which are **byte-for-byte
  untouched**. The new functions do not call `model.forward()` (which never returns the latent);
  they replicate its `encode -> [QAT noise, if attached] -> decode` dispatch explicitly, using only
  `BaselineAutoencoder`'s existing public `encode()`/`decode()`/`quantization_noise` - so
  `models/autoencoder.py` needed no changes at all.
- `src/nvc/training/checkpoint.py` - `save_checkpoint()` gained an optional `extra: dict | None =
  None` kwarg, merged under a single `"extra"` key only when provided. Omitted (the default): the
  saved dict is missing the key entirely, byte-identical to every checkpoint this function produced
  before M9A. `load_checkpoint`/`load_model_from_checkpoint`/`resume_training_state` were **not
  modified** and never look for this key, so old and new checkpoints both load through them
  unchanged either way.
- `src/nvc/training/__init__.py` - exports `RateEstimator`, `train_one_epoch_with_rate`,
  `validate_one_epoch_with_rate`.
- `src/nvc/utils/config.py` - `rate_enabled: bool = False`, `rate_lambda: float = 0.0`,
  `rate_calibration_path: Path | None = None`, mirroring the existing `qat_*` fields' exact pattern
  (including nullable-path handling in `from_json`). Not added to `configs/default.json` - the
  existing `qat_*` fields aren't listed there either; both rely on the dataclass default.
- `scripts/train_autoencoder.py` - `--rate-enabled` / `--rate-lambda` / `--rate-calibration` CLI
  flags; validation (`--rate-lambda` must be finite and >= 0; `--rate-calibration` required only
  when rate is enabled without QAT, and **rejected** if passed together with `--qat-enabled`, so
  there is never an ambiguous "which scale wins" situation); the rate estimator's parameters are
  added to the optimizer's parameter list; per-epoch history gains `rate_enabled` / `rate_lambda` /
  `train_distortion` / `train_rate_bpp` / `val_distortion` / `val_rate_bpp` fields (`None` when
  unused, mirroring the existing `qat_*` history fields' pattern); checkpoint saves pass
  `extra={"rate_estimator_state_dict": ..., "rate_lambda": ...}` when rate training is active,
  **recomputed fresh every epoch** (a real bug caught and fixed during this implementation - see
  §6).

### 3.6 Design decisions made (flagging for visibility, not asking permission post-hoc)

- **`RateEstimator` is a real `nn.Module`**, unlike its sibling `QuantizationNoise` - the one
  deliberate architectural difference, justified by needing genuinely learnable parameters in the
  optimizer's parameter list.
- **Rejecting `--rate-calibration` together with `--qat-enabled`**, rather than silently ignoring
  one or the other - the brief's "do not silently create a second independent quantization scale"
  read as license to make an inconsistent flag combination a hard error, not a silent override.
- **`configs/default.json` left untouched** - matches the existing `qat_*` precedent exactly rather
  than the Phase-1 plan's original assumption that it would need new keys.
- **CLI-level tests for M9A live in `tests/test_rate_estimator.py`**, not a separate
  `test_scripts_training.py` addition - matches how `test_quantization_aware_training.py` already
  keeps QAT's own CLI tests alongside its unit tests, discovered by inspecting that file's own
  structure before writing M9A's tests.

## 4. Test-suite result

```
tests/test_rate_estimator.py: 61 passed
Full suite: 528 passed in 55.56s (0:00:55)
```
528 = 467 pre-M9A + 61 new. Zero failures, zero regressions, zero skips beyond the usual
FFmpeg-guarded ones. Every item on the brief's M9A validation checklist has a dedicated test -
finiteness (normal and extreme latent values, extreme learned scale, very small/large bin width),
non-negativity (parametrized + 20-case random sweep), BPP normalization (hand-computed reference
value), gradient existence/nonzero-ness for `loc`/`log_scale`/latent input, changing
loc/log_scale/latent changes the estimate, invalid bin width/bits/calibration-split/bit-depth/mode
all rejected, negative/NaN/inf lambda rejected and zero accepted, QAT-without-rate and
rate-without-QAT and QAT-with-rate all exercised, four explicit checkpoint-compatibility tests plus
the three **real** M7/M8 checkpoint files on disk confirmed still loadable unchanged.

The single most load-bearing test, run as specified in the brief (not simplified to "totals
match"): `test_lambda_zero_matches_distortion_only_trainer_exactly` proves, with identical initial
weights (`copy.deepcopy`) and no QAT randomness in the mix, that `train_one_epoch` (A) and
`train_one_epoch_with_rate(lambda_rate=0.0)` (B) produce identical reconstructions, identical
distortion, identical total loss, identical gradients on every model parameter, and identical
parameters after one optimizer step - both via a manual replication of each path's logic and via
the actual public trainer functions end-to-end.

## 5. Smoke test result

Run against a **real** DAVIS train-split batch and the **real** M8 QAT calibration
(`outputs/calibration/qat_combined_noise.json`, 8-bit), production-scale model
(`latent_channels=64`), QAT + rate combined:

| | Value |
|---|---|
| Input shape | `(4, 3, 256, 256)` |
| Latent shape | `(4, 64, 16, 16)` |
| QAT enabled | True (8-bit / per_channel) |
| lambda | 0.01 |
| Distortion (MSE) | 0.050302 |
| Estimated rate | 0.834373 bpp |
| Total loss | 0.058646 (finite: True) |
| Encoder params with gradient | 8/8, mean grad norm 6.37e-05 |
| Decoder params with gradient | 8/8, mean grad norm 1.65e-03 |
| `rate_estimator.loc` gradient | exists, norm 8.77e-05 |
| `rate_estimator.log_scale` gradient | exists, norm 4.17e-04 |
| Rate estimator params received gradient | True |
| `optimizer.step()` | completed without error |
| `train_one_epoch_with_rate` (2 batches) | `{loss: 0.0610, distortion: 0.0527, rate: 0.8339}` |
| `validate_one_epoch_with_rate` (2 batches) | `{loss: 0.0582, distortion: 0.0500, rate: 0.8271, psnr: 13.01 dB}` |

The low PSNR (~13 dB) is expected and not a concern - this run used a **freshly, randomly
initialized** model (no pretrained weights loaded), so it tests mechanics (gradients flow, losses
combine, no errors, real data works end-to-end), not trained quality. Not to be confused with a
trained-model result anywhere in this document.

## 6. Bugs found and fixed during this implementation

- **Stale `checkpoint_extra`**: an early draft computed `rate_estimator.state_dict()` once before
  the epoch loop and reused that same dict for every subsequent checkpoint save - silently
  checkpointing epoch-1's rate parameters forever after, since `loc`/`log_scale` are updated by
  `optimizer.step()` every epoch just like the model's own parameters. Caught before any test ran
  against it, by re-reading the diff; fixed to recompute fresh immediately before each save.
- **Test design flaw, not a code bug**: an early version of the lambda=0 equivalence test reused
  one `DataLoader` object across two separate `for batch in loader` consumptions (one via
  `train_one_epoch`, one via `train_one_epoch_with_rate`). `create_train_loader`'s shuffling
  `torch.Generator` advances its own state on every iteration (correct, ordinary epoch-to-epoch
  behavior), so the loader's second pass legitimately returns a different batch order than its
  first - the two training calls were silently training on different data, not exercising the same
  input. Fixed by constructing two separate loader instances from the same seed.

## 7. Numerical concerns considered and resolved

- `torch.sign`'s zero gradient does not silently zero out the rate estimator's gradient - see
  §3.2's derivation; confirmed both analytically and by the passing gradient tests.
- `torch.expm1` bounds every exponential argument to `<= 0` in this formulation (since `|t| >= 0`
  and `scale > 0`), so there is no overflow path to guard against, only underflow - which resolves
  cleanly to `0` and is handled correctly on both sides of the CDF difference.
- The one open, unquantified question: **how well the Laplace proxy's rate estimate will track the
  real payload BPP once compared** - by design not measured yet (that's M9C's job), and flagged
  honestly rather than assumed favorable.

## 8. What is NOT done yet (explicitly, per the approval)

- No multi-epoch training of any kind.
- No lambda selection (needs measuring real D/R scale first, per the brief's own sequencing).
- No recalibration of any M9 checkpoint.
- No `benchmark_rd.py` runs, no RD plots, no comparison tables against M8 QAT/control or M7.
- No modification of any M7 or M8 artifact - confirmed unaffected (checkpoint compatibility tests
  above load the real M7/M8 files directly).
- `MILESTONE_9_RESULTS.md` does not exist yet - correctly, since it requires the above.

Waiting for approval before proceeding to M9C's actual experiments (pilot run to measure D/R scale,
justified lambda set, controlled training, recalibration, full DAVIS benchmarking).

---

# M9C - Rate-distortion pilot and lambda selection

**Status**: complete. Pilot only - no final training, no recalibration, no `.nvc` benchmarking was
performed (see 9C.10). Sections 3-8 above are M9A's historical record and are **not** rewritten here.

## 9C.1 Setup

Every number below comes from the real DAVIS manifest (`data/processed/manifest.json`: 4826 train /
663 val / 719 test frames at 256x256) and the real M8 QAT checkpoint
(`outputs/qat_combined/checkpoints_qat_noise/best.pt`, epoch 40, 4-bit / per_channel QAT).

| | Value |
|---|---|
| Start model (every arm) | `outputs/qat_combined/checkpoints_qat_noise/best.pt` |
| Calibration (QAT noise **and** rate bin width) | `outputs/calibration/vimeo_epoch17_4bit.json`, 4-bit / per_channel, train split |
| Seed / batch size / LR / optimizer | 42 / 8 / 1e-4 / Adam - all `configs/default.json` defaults |
| Crop | none (full 256x256 frames) |
| Device | CUDA (RTX 5060) |

The calibration is the one M8's **QAT training** itself used (`train_vimeo_qat_combined.py`'s own
default is `vimeo_epoch17_<bits>bit.json`), so M9C Rule 7's "same QAT calibration" holds exactly.
Note this is *not* the calibration M8's **benchmark** used - see 9C.7.

## 9C.2 Bug found and fixed in M9A (phase 1)

M9A could not start rate training from any pre-M9 checkpoint - i.e. it could not do the one thing
M9C Rule 6 requires.

`--rate-enabled` puts the rate estimator's `loc`/`log_scale` into the optimizer, so it holds **18**
parameters where every M7/M8 checkpoint's saved optimizer state has **16**.
`optimizer.load_state_dict` therefore raised `ValueError: loaded state dict contains a parameter
group that doesn't match the size of optimizer's group`, and `scripts/train_autoencoder.py` caught
only `RuntimeError` - so it surfaced as an uncaught traceback.

Fixed with:

- `nvc.training.checkpoint.resume_model_only()` - restores model weights + epoch/history, not
  optimizer state. Exported from `nvc.training`; `resume_training_state` is unchanged.
- `--resume-model-only` on `scripts/train_autoencoder.py`, plus a `ValueError` handler that names
  the cause and the remedy instead of surfacing PyTorch's bare message.
- `tests/test_m9c_resume_model_only.py` - 10 regression tests, including one that pins the original
  failure so it cannot silently return.

Starting each arm with a **fresh** optimizer is also the right experimental choice here: all six
arms then begin from byte-identical weights *and* identical (empty) optimizer state, so lambda is
the only difference between them.

Also noted, not fixed (documentation only, no behavioural impact): `train_one_epoch_with_rate`'s
docstring claims the rate estimator "keep[s] receiving a gradient signal even at lambda=0". It does
not - `0.0 * rate` contributes exactly zero gradient. The observable consequence is that the
control arm's rate estimator never trains, which is why its R is not comparable to the other arms'
(9C.6).

## 9C.3 Measured D and R scale (phase 2)

`scripts/m9c_rd_diagnostic.py`, 60 real train batches (480 frames), model weights frozen and
verified bit-identical afterwards (`max |drift| = 0.0`). Measured in the training regime
(`model.train()`, QAT noise applied), matching `train_one_epoch_with_rate` step for step.

| | mean | median | std | min | max |
|---|---|---|---|---|---|
| **D** (MSE, [0,1] pixels) | 1.277731e-03 | 1.252583e-03 | 2.424073e-04 | 8.528195e-04 | 2.274611e-03 |
| **R** at estimator init (bpp) | 2.7943 | 2.7945 | 0.0996 | 2.5692 | 3.0406 |
| **R** after warm fit (bpp) | 1.4079 | 1.4085 | 0.0220 | 1.3572 | 1.4587 |
| PSNR | 29.01 dB | | | 26.43 | 30.69 |

Per-batch `D/R` (fitted): mean 9.075416e-04, std 1.700e-04. `mean R / mean D` = 1101.85.

**R was measured twice on purpose.** `RateEstimator` initializes to `loc=0, log_scale=0`
(scale=1) - a prior that has not seen the latent. The rate it reports then is dominated by prior
misfit, and choosing lambda against it would pick a lambda too small by exactly that factor. The
warm fit trains **only** `loc`/`log_scale` against the frozen latent (600 steps, lr 1e-2) and
plateaus at ~1.41 bpp; that is the magnitude that persists, so it is what lambda must balance.

## 9C.4 How the balancing lambda was calculated

`lambda_balance` is the value at which the two loss terms contribute equally:

```
lambda * mean(R) = mean(D)
lambda_balance   = mean(D) / mean(R) = 1.277731e-03 / 1.4079 = 9.075672e-04
```

(The same figure from the unfitted R would have been 4.572578e-04, ~2x smaller - the misfit factor
described above.) No lambda from any external paper was used; a published value would refer to
different units for both D and R and would be meaningless here.

## 9C.5 Rate-estimator sanity checks (phase 3)

All six pass on the **real** M8 QAT model and real DAVIS batches (`sanity_checks` in
`dr_diagnostic.json`). Checks D and E run under forced-deterministic cuDNN kernels, so their
"difference is exactly zero" assertions mean what they say.

| Check | Result |
|---|---|
| A - more probability mass => less rate | mode 1.28 bits vs tail 29.9 bits; 6-point sweep monotone non-decreasing |
| A2 - stabilized CDF vs `torch.distributions.Laplace` at float64 | max abs probability error 1.1e-16 |
| B - finite and non-negative | all finite; min 0.0 bits/element |
| C - gradients reach latent / `loc` / `log_scale` | all three nonzero |
| D - lambda=0 **is** the distortion-only path | identical loss, **identical gradients on all 16 model params**, **identical params after one real Adam step** |
| E - lambda>0 adds a rate gradient | encoder gradient changes by 4.98e-03; decoder changes by **exactly 0** (correct - the decoder is downstream of the latent the rate is measured on) |

Check D was run as the brief specifies - gradients and a real optimizer step, not just the scalar
loss. On GPU with default (autotuned, non-deterministic) cuDNN kernels the same comparison bottoms
out at a ~1e-9 noise floor and cannot distinguish "no rate contribution" from "a little"; forcing
deterministic kernels is what makes the zero meaningful.

## 9C.6 Lambda pilot (phases 4-6)

**Grid** (`scripts/m9c_lambda_pilot.py`, read mechanically out of `dr_diagnostic.json` - not typed
in): a half-decade log grid one decade either side of the measured balance,
`lambda_balance * [1/10, 1/sqrt(10), 1, sqrt(10), 10]`. At `k * lambda_balance` the rate term is
`k/(1+k)` of the total loss, which is what maps the grid onto the five required regimes.

**Control**: a `lambda=0` arm runs alongside. Without it, D drift over a few hundred steps is
unattributable to lambda. It is a control, not a sixth operating point.

**Schedule**: 5 epochs x 100 batches = **500 optimizer steps per arm**, via the repository's
existing `--max-batches` smoke mechanism. Short on purpose.

| lambda | x balance | regime | init D | final D | init R | final R | init total | final total | val D | val R | val PSNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000e+00 | 0 | control | 1.6480e-03 | 1.2635e-03 | 2.8076 | 2.8306 | 1.6480e-03 | 1.2635e-03 | 1.2938e-03 | 2.9847* | 29.58 |
| 9.0757e-05 | 0.1 | quality-oriented | 1.6459e-03 | 1.2665e-03 | 2.7829 | 2.5650 | 1.8985e-03 | 1.4993e-03 | 1.2981e-03 | 2.6768 | 29.57 |
| 2.8700e-04 | 0.316 | mildly rate-aware | 1.6336e-03 | 1.2823e-03 | 2.7463 | 2.2443 | 2.4217e-03 | 1.9264e-03 | 1.3182e-03 | 2.3066 | 29.49 |
| 9.0757e-04 | 1 | balanced | 1.5320e-03 | 1.3716e-03 | 2.6314 | 1.6156 | 3.9202e-03 | 2.8379e-03 | 1.4918e-03 | 1.6237 | 28.86 |
| 2.8700e-03 | 3.16 | strongly rate-aware | 1.4749e-03 | 1.6049e-03 | 2.3480 | 1.0367 | 8.2135e-03 | 4.5801e-03 | 1.6925e-03 | 1.0376 | 28.31 |
| 9.0757e-03 | 10 | rate-oriented | 1.7883e-03 | 2.0992e-03 | 2.0312 | 0.7699 | 2.0222e-02 | 9.0867e-03 | 2.0958e-03 | 0.7685 | 27.34 |

\* The control's R is **not comparable** to the other arms': at lambda=0 its rate estimator receives
exactly zero gradient (9C.2) and stays at initialization, so 2.9847 is the unfitted-prior value.
The control is a control for **D only**.

D and PSNR here are **float-latent** values - eval mode, no quantization applied - not
quantized-codec quality.

`init` is the first epoch's average (which includes the fresh-optimizer transient at step 1);
`final` is the fifth epoch's. The control shows the transient is mild and gone by epoch 2
(1.6480e-03 -> 1.2139e-03 -> settling at ~1.26e-03, against the 1.2777e-03 the diagnostic measured).

**The trend is exactly the expected one, and strictly monotone**: R falls and D rises with lambda,
across all five arms, with no exceptions. Rate/distortion vs lambda plots are in
`outputs/m9c_lambda_pilot/lambda_pilot_rd.png`.

Gradient norms and estimator statistics (from `lambda_pilot_summary.json`):

| lambda | encoder grad | decoder grad | `loc` grad | `log_scale` grad | fitted `loc` (mean/std) | fitted `scale` (mean) |
|---|---|---|---|---|---|---|
| 0 | 3.86e-03 | 5.23e-03 | **0** | **0** | 0.000 / 0.000 | 1.000 |
| 9.0757e-05 | 7.59e-03 | 8.05e-03 | 1.29e-06 | 1.66e-05 | 0.002 / 0.031 | 1.050 |
| 2.8700e-04 | 7.45e-03 | 1.04e-02 | 4.13e-06 | 4.76e-05 | 0.002 / 0.033 | 1.048 |
| 9.0757e-04 | 6.70e-02 | 3.60e-02 | 1.37e-05 | 1.06e-04 | 0.003 / 0.036 | 1.042 |
| 2.8700e-03 | 5.71e-02 | 1.86e-02 | 4.57e-05 | 1.82e-04 | 0.004 / 0.040 | 1.029 |
| 9.0757e-03 | 5.53e-02 | 1.10e-02 | 1.28e-04 | 2.02e-04 | 0.005 / 0.040 | 1.016 |

## 9C.7 Failure-mode analysis (phase 7)

`scripts/m9c_failure_modes.py`. The pilot's headline R comes from each arm's **own** trained
estimator, which can fall either because the latent genuinely got cheaper or because the density
merely got better at describing an unchanged latent. Two independent latent-side measures separate
these:

- **R_refit** - a **fresh** estimator fitted to each arm's final latent with identical init, steps
  (600) and LR (1e-2). Every arm scored by equal capacity fitted with equal effort.
- **H_symbol** - plug-in empirical entropy of the latent quantized on the **existing frozen** 4-bit
  grid, bits per input pixel. Independent of the Laplace family entirely. Read-only: nothing was
  recalibrated, and `EmpiricalEntropyModel` / the range coder / `.nvc` were not touched or run.

| lambda | val D | R_own | **R_refit** | H_symbol | **clipped on frozen grid** | latent abs-mean |
|---|---|---|---|---|---|---|
| 0 (control) | 1.2938e-03 | 2.9847* | 1.4591 | 0.7616 | 50.57% | 8.1786 |
| 9.0757e-05 | 1.2981e-03 | 2.6768 | 1.4191 | 0.7938 | 46.14% | 7.3065 |
| 2.8700e-04 | 1.3182e-03 | 2.3066 | 1.3471 | 0.8441 | 38.31% | 5.9740 |
| 9.0757e-04 | 1.4918e-03 | 1.6237 | 1.1790 | 0.9074 | 21.26% | 3.6827 |
| 2.8700e-03 | 1.6925e-03 | 1.0376 | 0.9482 | 0.8554 | 6.81% | 1.8588 |
| 9.0757e-03 | 2.0958e-03 | 0.7685 | 0.7541 | 0.7223 | 1.94% | 1.0496 |

| Failure mode | Verdict |
|---|---|
| Rate collapse | **No** - no arm below 0.75 bpp; fitted scales all within [0.996, 1.058] |
| Rate explosion | **No** - no NaN/Inf anywhere; no rate above 3 bpp |
| Decoder-only compensation | **No** - R_refit falls monotonically (1.4591 -> 0.7541), so the latent itself got cheaper |
| Estimator gaming | **No** - R_refit moved by 0.705 bpp between control and strongest arm; latent abs-mean fell 8.18 -> 1.05 |
| Lambda insensitivity | **No** - R_refit spread 0.665 bpp across the arms |
| Lambda over-dominance | **No** - even the strongest arm holds 27.34 dB (float latent) |
| NaNs / Infs | **None**, any arm, any measurement |

Three things this analysis surfaced that matter more than the pass/fail verdicts:

**(a) The frozen grid is badly mismatched to the M8 QAT model - a pre-existing M8 condition, not
something M9C caused.** The unmodified M8 QAT checkpoint clips **48.91%** of its latent against
`vimeo_epoch17_4bit.json` (max abs-z = 72.52); M7 on the same grid clips 0.10% (max abs-z = 14.32).
The M8 QAT latent drifted ~5x outside the M7-derived grid its own QAT noise was scaled from, so
that noise (+/-0.34) was negligible against the real latent spread for most of M8's training. M8's
*benchmark* numbers are unaffected - `MILESTONE_8_RESULTS.md` section 4 shows it used fresh
`qat_combined_noise_*.json` calibrations at 0.05-0.08% clipping (those files are not on disk now).

**Consequence**: `H_symbol` is **not comparable across arms** - it is measured through a grid whose
clipping rate varies from 50.6% to 1.9% between them, and heavy clipping *lowers* measured entropy
by piling values onto the two extreme symbols while destroying information. That, not any property
of lambda, is why H_symbol rises and then falls. The verdicts above are therefore gated on R_refit,
which has no such confound. `failure_mode_analysis.json` records the clipping rate per arm and flags
`frozen_grid_mismatched: true` / `symbol_entropy_comparable_across_arms: false`.

**(b) The rate estimator is learning-rate starved during the pilot.** It shares the model's
optimizer and LR (1e-4). Fitting this latent needs `loc`/`log_scale` to move by O(1)-O(5) - the
diagnostic's warm fit at lr 1e-2 reached `scale` ~6.45, `loc` std 1.56 - but 500 Adam steps at
1e-4 can move a parameter at most ~0.05, and the arms indeed end at `scale` 1.016-1.050, `loc` std
0.031-0.040, i.e. essentially still at initialization.

**Consequence**: with `loc~0, scale~1` and a bin width of 0.68, `-log2 P(z)` is very nearly
proportional to `abs(z)`, so the rate term the pilot actually applied behaves largely as an **L1
magnitude penalty on the latent** rather than a learned entropy penalty. This is consistent with
R_own tracking latent abs-mean closely (ratio 0.365 -> 0.732 across the sweep). The directional
result still stands - R_refit confirms the latent became genuinely cheaper under a matched-capacity
model - but the mechanism is not yet the one M9 intends.

**(c) Most of the apparent rate gain at low lambda is estimator fitting, not latent change.** At
lambda=9.0757e-05, R_own falls 2.98 -> 2.68 while R_refit moves only 1.4591 -> 1.4191 (2.7%). The
two converge only at high lambda (9.0757e-03: 0.7685 vs 0.7541). The training log's R therefore
overstates the real latent-side gain at the quality-oriented end.

## 9C.8 Selected and rejected lambdas

**Selected for final M9 training: `9.0757e-04` (balance) and `2.8700e-03` (3.16x balance).**

They give the clearest genuine latent-side rate reduction per dB of quality: measured on R_refit
against the control, `9.0757e-04` buys -19.2% rate for -0.72 dB, and `2.8700e-03` buys -35.0% rate
for -1.27 dB. Both are stable, both bracket the measured balance point, and both substantially
reduce clipping against the existing grid (21.3% and 6.8%, from the control's 50.6%) - which
matters for whatever the deployed operating point turns out to be. Two arms, not one, because a
single short pilot cannot rank them and they sit either side of the balance.

**Rejected:**

- `9.0757e-05` and `2.8700e-04` - too weak to justify a long run. Their R_refit gains are 2.7% and
  7.7% respectively, and most of their apparent R_own improvement is the estimator fitting itself
  (9C.7c) rather than a cheaper latent.
- `9.0757e-03` - largest rate reduction (-48.3% R_refit) but the worst distortion cost
  (D +62%, PSNR -2.24 dB float-latent), and returns are already diminishing: going from
  `2.8700e-03` to it buys 0.19 bpp for a further ~1 dB. Worth revisiting only if a low-rate
  operating point is explicitly wanted.

**These lambda values are tied to this bin width.** The balance point is `mean(D)/mean(R)`, and R
scales with the calibration's bin width. If the quantizer is recalibrated (9C.10, a later phase),
`lambda_balance` must be re-derived from a fresh diagnostic run - the numbers above do not transfer.

## 9C.9 Files created / modified

Created:
- `scripts/m9c_rd_diagnostic.py`, `scripts/m9c_lambda_pilot.py`, `scripts/m9c_failure_modes.py`
- `tests/test_m9c_resume_model_only.py` (10 tests), `tests/test_scripts_m9c_pilot.py` (18 tests)
- `outputs/m9c_lambda_pilot/` - `dr_diagnostic.json`, `lambda_pilot_summary.json` / `.csv`,
  `lambda_pilot_table.md`, `lambda_pilot_rd.png`, `failure_mode_analysis.json`, and six per-arm
  checkpoint directories

Modified:
- `src/nvc/training/checkpoint.py` - added `resume_model_only`; `resume_training_state` unchanged
  apart from a docstring note
- `src/nvc/training/__init__.py` - exports `resume_model_only`
- `scripts/train_autoencoder.py` - `--resume-model-only`, its validation, and the `ValueError` handler
- this file

Not touched, and verified so: `BaselineAutoencoder`/`Encoder`/`Decoder`, `quantization.py`,
`EmpiricalEntropyModel`, the range coder, `.nvc`, `rd_benchmark.py`, `train_one_epoch`,
`validate_one_epoch`, `rate_estimator.py`, and every M7/M8 output and checkpoint.

Test suite: **556 passed, 0 failed** (528 pre-M9C + 28 new). No regressions.

## 9C.10 What was NOT done (explicitly)

- **Actual `.nvc` bitrate was NOT validated in M9C.** Every R in this section is the training-time
  Laplace proxy in bits per input pixel. `H_symbol` is a plug-in entropy on a frozen grid, not a
  coded payload size. No claim is made or implied that either equals real `.nvc` bitrate.
- No final/long training run. No fresh final calibration or recalibration of anything.
- No `benchmark_rd.py` run, no H.264/H.265 comparison, no 719-frame x 3-bit benchmark.
- No architecture change, hyperprior, autoregressive context, temporal prediction or motion
  compensation.
- The pilot's D/PSNR figures are float-latent (eval mode, unquantized), so they are **not**
  comparable to M8's quantized `.nvc` benchmark numbers.

---

# M9C.1 - Separate learning rate for the rate estimator, and re-pilot

**Status**: complete. Pilot only - no final training, no calibration, no `.nvc` benchmarking
(see 9C1.9). The M9A and M9C sections above are historical record and are **not** rewritten.

## 9C1.1 Why a separate learning rate was introduced

M9C (9C.7b) measured that the rate estimator was effectively frozen. It shared the model's
optimizer and learning rate of 1e-4, but fitting this latent needs `loc`/`log_scale` to move by
O(1)-O(5). Adam's step is approximately the learning rate per parameter, so 500 steps at 1e-4 can
move a parameter by at most ~0.05 - and the M9C arms indeed ended at `scale` 1.016-1.050 against an
initialization of exactly 1.0, with `loc` std 0.031-0.040 against 0.

The consequence was not cosmetic. With `loc~0, scale~1` and a bin width of 0.68,
`-log2 P(z)` is very nearly proportional to `abs(z)`, so the rate term M9C actually applied behaved
substantially as an **L1 penalty on latent magnitude** rather than as a fitted entropy model. M9C's
directional result was still valid - `R_refit` showed the latent genuinely became cheaper - but the
mechanism was not the one M9 intends.

M9C.1 puts the estimator's two parameters into their **own optimizer parameter group** at a
separate `--rate-lr`. The model's group keeps `--learning-rate` untouched.

| | Old (M9C) | New (M9C.1) |
|---|---|---|
| Model LR | 1e-4 | 1e-4 (unchanged) |
| Rate estimator LR | 1e-4 (shared) | **1e-2** (`--rate-lr`, `Config.rate_lr`) |
| Optimizer structure with rate on | one group of 18 | two groups: 16 model + 2 rate |
| Optimizer structure with rate off | one group of 16 | one group of 16 (unchanged) |

Validation: `--rate-lr` must be finite and > 0, and is **never silently replaced** by the model's
LR - an invalid value aborts the run before any checkpoint is written, since a silent fallback
would quietly reintroduce the exact starvation this flag exists to remove.

`rate_lr` is recorded per epoch in `history.json` and in each checkpoint's `extra`, alongside
`rate_lambda`.

## 9C1.2 Checkpoint compatibility

Three vintages now exist, and all still load:

| Checkpoint | Optimizer groups | Into an M9C.1 rate run |
|---|---|---|
| M7 / M8 (pre-rate) | `[16]` | needs `--resume-model-only` |
| M9A / M9C (single-group rate) | `[18]` | needs `--resume-model-only` |
| M9C.1 (two-group rate) | `[16, 2]` | ordinary `--resume` restores the optimizer |

`resume_training_state` and `resume_model_only` are both unchanged. What M9C.1 adds is
`_describe_optimizer_mismatch()` in `scripts/train_autoencoder.py`, so the two failing vintages are
told apart in the error message rather than both being reported as "predates --rate-enabled" - which
would now be wrong for an M9A/M9C checkpoint. Every path prints explicitly what was restored, and
the `[RATE]` block prints both learning rates so a log can be audited after the fact.

Inference is untouched: a two-group rate checkpoint still rebuilds through
`load_model_from_checkpoint` with no knowledge of optimizers or rate.

## 9C1.3 Phase 2 - does the estimator actually adapt?

`scripts/m9c1_adaptation_check.py`. The identical short pilot (500 steps, lambda = M9C's balance
point 9.0757e-04, same seed / batch / model LR / QAT config / calibration / start checkpoint) run
twice, changing only the rate LR.

| Metric | M9C (rate_lr = 1e-4) | M9C.1 (rate_lr = 1e-2) |
|---|---|---|
| `loc` std | 0.0361 | **1.6598** |
| `loc` abs max | 0.0564 | **3.1041** |
| `scale` mean | 1.0422 | **4.4812** |
| `scale` range | [1.0256, 1.0555] | **[1.8992, 12.3966]** |
| val D | 1.4917e-03 | **1.3156e-03** |
| val PSNR | 28.86 dB | **29.50 dB** |
| estimator R | 1.6236 | 1.3014 |
| matched-refit R | 1.1592 | 1.2985 |
| **misfit (own - refit)** | **+0.4644** | **+0.0029** |

`loc` movement ratio 55x, `scale` movement ratio 205x, all values finite, verdict `adapted: true`.

The load-bearing row is the last one. **Misfit** is how much worse the estimator's own density is
than an equally-sized one fitted properly to the same latent - it is the estimator's error, measured
independently of the latent. It collapsed by a factor of ~160, from 0.46 bpp to 0.003 bpp. The
training signal is now an entropy estimate rather than a stand-in for one.

Note the 1e-4 arm reproduced M9C's balance arm to four significant figures (val D 1.4917e-03 vs
1.4918e-03, val R 1.6236 vs 1.6237, PSNR 28.86 vs 28.86), confirming the parameter-group refactor is
behaviour-preserving at the old learning rate.

`scale` mean 4.48 is **not** a target hit. The M9C diagnostic's rate-only warm fit reached ~6.4,
which was only evidence that substantial movement is possible; the joint objective's optimum need
not coincide with a rate-only fit's, and does not.

## 9C1.4 Phase 3-4 - the re-run lambda pilot

Same five lambdas as M9C (deliberately, so the two are directly comparable), same control arm, same
500 steps, same start checkpoint for every arm, `--rate-lr 1e-2`. New directory
`outputs/m9c1_rate_lr_pilot/`; no M9C checkpoint was reused or overwritten.

| lambda | val D | val PSNR | estimator R | R_refit | final `scale` (mean / range) | `loc` std | clip* | observation |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | 1.2938e-03 | 29.58 | 2.9847** | 1.4591 | 1.000 / [1.00, 1.00] | 0.000 | 50.57% | estimator never trains at lambda=0 |
| 9.0757e-05 | 1.2943e-03 | 29.58 | 1.4370 | 1.4435 | 5.919 / [2.32, 14.30] | 1.403 | 48.86% | rate -1.1%, quality unchanged |
| 2.8700e-04 | 1.2970e-03 | 29.57 | 1.4029 | 1.4124 | 5.536 / [2.21, 13.88] | 1.524 | 45.48% | rate -3.2% for -0.01 dB |
| 9.0757e-04 | 1.3155e-03 | 29.50 | 1.3014 | 1.3164 | 4.481 / [1.90, 12.40] | 1.660 | 35.25% | rate -9.8% for -0.08 dB |
| 2.8700e-03 | 1.4278e-03 | 29.10 | 1.0906 | 1.1088 | 2.718 / [1.19, 8.22] | 1.620 | 15.66% | rate -24.0% for -0.49 dB |
| 9.0757e-03 | 1.7361e-03 | 28.17 | 0.8546 | 0.8697 | 1.517 / [0.68, 4.24] | 1.294 | 3.90% | rate -40.4% for -1.41 dB |

\* clipping against the **M9C diagnostic grid** (`vimeo_epoch17_4bit.json`), which M9C established
is mismatched to this model (the unmodified M8 QAT checkpoint clips 48.9% on it). Reported for
continuity with M9C only; it is **not** a deployment figure and no recalibration was done.

\*\* the control's estimator R remains the unfitted-prior value and is not comparable to the other
arms - at lambda=0 the estimator receives exactly zero gradient regardless of its learning rate
(`0.0 * rate`), which M9C.1 tests explicitly pin.

Rate reduction and distortion are **strictly monotone** in lambda across all five arms, on both the
estimator's own R and the independent `R_refit`.

## 9C1.5 M9C.1 versus M9C

**Misfit, per arm** - the estimator's own error:

| lambda | M9C | M9C.1 |
|---|---|---|
| 9.0757e-05 | +1.2577 | -0.0065 |
| 2.8700e-04 | +0.9595 | -0.0095 |
| 9.0757e-04 | +0.4448 | -0.0150 |
| 2.8700e-03 | +0.0893 | -0.0182 |
| 9.0757e-03 | +0.0144 | -0.0151 |

M9C's estimator only approached its own refit at the highest lambda, where the latent had been
squeezed small enough for a `Laplace(0, 1)` to describe it by accident. M9C.1's agrees everywhere,
to ~0.015 bpp. (The small negative sign is a protocol artefact - the arm's estimator trains on the
noised training latent for 500 steps while the refit sees the clean eval latent for 600 - not a
meaningful difference.)

**Rate-distortion curve** - M9C.1's PSNR minus M9C's, interpolated at the *same* `R_refit`:

| R_refit | M9C.1 PSNR | M9C PSNR (interpolated) | delta |
|---|---|---|---|
| 1.4435 | 29.58 | 29.58 | +0.004 dB |
| 1.4124 | 29.57 | 29.56 | +0.010 dB |
| 1.3164 | 29.50 | 29.38 | **+0.124 dB** |
| 1.1088 | 29.10 | 28.70 | **+0.401 dB** |
| 0.8697 | 28.17 | 27.92 | **+0.252 dB** |

M9C.1 **dominates** the M9C operating curve at every matched rate, by up to +0.40 dB. So the fix is
not only a mechanism correction - it produces a measurably better model at equal rate.

Mechanically, at any given lambda a well-fitted density reports a *lower* rate for the same latent,
so the gradient pushing latent magnitude down is weaker; the encoder is squeezed less hard for the
same nominal lambda, which is why M9C.1's arms show both lower D and (at matched lambda) higher
`R_refit` than M9C's, while still landing on a better curve.

## 9C1.6 Is the rate term still an L1 penalty in disguise?

No. Three independent pieces of evidence:

1. **Misfit ~0.01 bpp** (was 0.46). The estimator's density is now within 1% of a properly fitted
   one, so its output is an entropy estimate, not a proxy for magnitude.
2. **The fitted scale is genuinely per-channel**: at the balance lambda it spans [1.90, 12.40]
   across the 64 channels, a 6.5x spread, with `loc` std 1.66. An L1 penalty is channel-agnostic
   and centred at zero by construction; this is not.
3. **The fitted scale tracks the latent**: it falls 5.92 -> 1.52 as lambda rises and the latent
   shrinks (`abs`-mean 7.84 -> 1.80). The density is following the distribution rather than sitting
   at its initialization.

## 9C1.7 Stability

No NaN or Inf in any arm, any measurement, either phase. No rate collapse (lowest `R_refit` 0.8697;
no fitted scale below 0.68). No rate explosion (highest scale 14.30, well inside the plausible band;
no rate above 3 bpp). `estimator_gaming: false` and `decoder_only_compensation: false`, both gated
on `R_refit`, which is unaffected by the frozen grid's clipping. `lambda_insensitivity: false`
(`R_refit` spread 0.574 bpp). Focused tests cover rate LRs of 1e-3, 1e-2 and 1e-1 over 20 steps with
no divergence.

## 9C1.8 Lambda re-evaluation

The balance point was **re-derived from the new data**, not carried over: the least-pressured arm of
the M9C.1 sweep gives `D/R = 1.294318e-03 / 1.4370 = 9.007e-04`, within 0.8% of M9C's diagnostic
value of 9.0757e-04. It holds because that diagnostic already measured R from a *warm-fitted*
estimator at lr 1e-2 - the same regime training now reaches. The M9C balance figure was correct; it
simply was not being realized in training until this fix.

Rate reduction per dB of quality, against the control:

| lambda | R_refit reduction | PSNR cost | efficiency |
|---|---|---|---|
| 9.0757e-05 | -1.07% | -0.00 dB | 794 %/dB |
| 2.8700e-04 | -3.20% | -0.01 dB | 271 %/dB |
| 9.0757e-04 | -9.78% | -0.08 dB | 122 %/dB |
| 2.8700e-03 | -24.01% | -0.49 dB | 50 %/dB |
| 9.0757e-03 | -40.40% | -1.41 dB | 29 %/dB |

**Recommended for final M9 training: `9.0757e-04`, `2.8700e-03`, `9.0757e-03`** - three points
spanning quality-preserving (-0.08 dB), balanced (-0.49 dB) and aggressive (-1.41 dB), which yields
an operating *curve* rather than a single point and brackets the efficiency knee between the last
two.

**Rejected: `9.0757e-05` and `2.8700e-04`.** Both buy 3% or less rate reduction. That is not worth a
long run, and it is close enough to the control to be indistinguishable from run-to-run variation.

**This changes M9C's recommendation.** M9C proposed 9.0757e-04 and 2.8700e-03 and rejected
9.0757e-03 as too damaging (27.34 dB). Under the corrected LR that same lambda reaches 28.17 dB, so
it is now the most useful point of the three for a low-rate operating point. The M9C selection was
made against a starved estimator and should not be carried forward.

**Still tied to this bin width.** These lambdas balance against R computed on the frozen
`vimeo_epoch17_4bit.json` grid. Fresh calibration after final training is mandatory, and any
recalibration changes R's scale and therefore requires re-deriving the balance.

## 9C1.9 What was NOT done

- **Actual `.nvc` bitrate is still unvalidated.** Every rate figure in this section is either the
  training-time Laplace proxy (`R_estimator`), an independent refit of the same family (`R_refit`),
  or a plug-in symbol entropy on a stated grid (`H_symbol`). None of them is a measured `.nvc`
  payload size, and no claim of a bitrate improvement is made.
- No final or long training run; no final checkpoint; no quantizer calibration or recalibration.
- No `benchmark_rd.py`, no `.nvc` encode/decode, no H.264/H.265, no full DAVIS RD benchmark.
- No change to the architecture, quantizer, entropy coder, `.nvc` format, or `EmpiricalEntropyModel`;
  no hyperprior, autoregressive context, or temporal coding.
- Pilot D/PSNR are float-latent (eval mode, unquantized) and are not comparable to M8's quantized
  `.nvc` benchmark numbers.

## 9C1.10 Files

Created: `scripts/m9c1_adaptation_check.py`, `tests/test_m9c1_rate_lr.py` (24 tests),
`outputs/m9c1_rate_lr_pilot/` (adaptation check, pilot summary JSON/CSV/MD, RD plot, failure-mode
analysis, per-arm checkpoints).

Modified: `src/nvc/utils/config.py` (`rate_lr`), `scripts/train_autoencoder.py` (`--rate-lr`,
validation, two parameter groups, `_describe_optimizer_mismatch`, logging, history/extra fields),
`scripts/m9c_lambda_pilot.py` (`--rate-lr` pass-through), `tests/test_scripts_m9c_pilot.py`,
`TESTING.md`, this file.

Unchanged and verified: models, `quantization.py`, `EmpiricalEntropyModel`, range coder, `.nvc`,
`rd_benchmark.py`, `rate_estimator.py`, `trainer.py`, `checkpoint.py`, and all M7/M8/M9C outputs.

---

# M9 FINAL - training, fresh calibration, and real `.nvc` validation

**Status**: complete. This is the section where the milestone's actual question is answered against
the deployed codec. The M9A, M9C and M9C.1 sections above are historical record and are **not**
rewritten. **Headline: M9-L is a genuine, unambiguous win; the higher lambdas are not, and the
training proxy did not track real bitrate.**

## 9F.1 Pre-flight finding: best-checkpoint selection was silently disabled

Required before launching, and it found a real defect.

The *criterion* was already correct - for a rate-enabled run `val_metrics["loss"]` is
`D + lambda*R`, exactly the M9 objective - but the *seed value* was not.
`best_val_loss` was initialised by minimising `val_loss` over the whole resumed history, and
`val_loss` means different things in different runs: plain MSE for a distortion-only run,
`D + lambda*R` for a rate-aware one. Resuming the M8 QAT checkpoint seeded it with M8's pure-MSE
minimum of **4.554493e-04** (measured on Vimeo), which no `D + lambda*R` epoch can beat. So
`best.pt` was **never written for the entire run** - confirmed retrospectively: every M9C and M9C.1
pilot arm on disk contains only `latest.pt`.

Fixed minimally: seed only from history records produced under the same objective (same
`rate_enabled`, and for rate runs the same `rate_lambda`), and say so in the log when prior history
is ignored. Distortion-only runs are unaffected by construction, including pre-M8A records that
carry no `rate_enabled` key at all. Nine regression tests in
`tests/test_m9_checkpoint_selection.py`, including the exact original failure and both
backward-compatibility cases. The `[BEST]` log line now names the objective it is minimising rather
than always saying "MSE".

This was a training-script fix, not an architectural change, so it did not trigger the brief's STOP
condition.

## 9F.2 Training setup

Four independent runs, each from the same M8 QAT checkpoint
(sha256 `90d51157356953db...`) with `--resume-model-only`, differing only in lambda.

| | Value |
|---|---|
| Budget | 30 full DAVIS-train epochs = 604 batches x 30 = **18,120 steps per model** |
| Model LR / rate LR | 1e-4 / 1e-2 (M9C.1's separate parameter groups) |
| Seed / batch / QAT | 42 / 8 / 4-bit per_channel, `vimeo_epoch17_4bit.json` |
| Epoch numbering | 41-70 (continues M8's 40) |

Identical for every arm; nothing was stopped early or shortened.

**A lambda=0 control was added as a fourth run.** The three deliverable models are the three
approved lambdas; the control is a measurement instrument, not a fourth operating point. Without it
an M8 -> M9 delta is unattributable, because these runs fine-tune on **DAVIS** while the M8 QAT
checkpoint was trained on **Vimeo** - so any improvement could be the rate term *or* simply 30 more
epochs of in-domain data. The control receives identical treatment minus the rate term and
therefore measures that confound alone. It turns out to be worth **-8.98% BD-rate on its own**, so
this was not a hypothetical concern.

### Training results (validation, float latent, proxy R)

| model | lambda | best epoch | val D | val R (proxy) | val total | PSNR |
|---|---|---|---|---|---|---|
| M9-L | 9.0757e-04 | 70 | 1.2217e-03 | 0.6752 | 1.8344e-03 | 29.80 dB |
| M9-M | 2.8700e-03 | 69 | 1.3787e-03 | 0.4006 | 2.5285e-03 | 29.22 dB |
| M9-H | 9.0757e-03 | 70 | 2.0499e-03 | 0.1811 | 3.6935e-03 | 27.37 dB |
| CTRL | 0 | 68 | 1.1920e-03 | 2.9485* | 1.1920e-03 | 29.94 dB |

\* the control's estimator receives exactly zero gradient at lambda=0 and stays at initialization,
so its R is the unfitted-prior value and is not comparable.

Best epochs land at 68-70 of 70, i.e. all four were still improving when the budget ended - the
runs are budget-limited, not converged. No NaN/Inf; every arm finite throughout.

Latent magnitude fell monotonically with lambda - abs-mean **7.98 -> 1.90 -> 1.16 -> 0.77** (CTRL,
L, M, H), abs-max 96.2 -> 32.2 -> 29.1 -> 26.3 - and the fitted estimator scale tracked it
(1.000 -> 1.018 -> 0.506 -> 0.253). This becomes the central fact of section 9F.5.

## 9F.3 Fresh calibration

Every model calibrated separately at 8/6/4-bit from the **train split only**, using the established
methodology unchanged: `per_channel_percentile`, 0.1/99.9 bounds, 50 batches x 8 frames, seed 42.
Nothing was reused - not M7's, not M8's, not the M9C diagnostic grid, not another lambda's. M7 and
M8-QAT were recalibrated with the identical procedure too, so all six models pass through one
comparison rather than citing numbers from a different invocation. Their historical calibration
files are untouched; these live under `outputs/m9_final/calibration/`.

**All 18 calibrations pass**, far inside the project's 2% clipping guard:

| model | 8-bit | 6-bit | 4-bit |
|---|---|---|---|
| M7 | 0.192% | 0.170% | 0.106% |
| M8-QAT | 0.193% | 0.174% | 0.117% |
| M9-CTRL | 0.193% | 0.173% | 0.119% |
| M9-L | 0.195% | 0.180% | 0.134% |
| M9-M | 0.195% | 0.181% | 0.134% |
| M9-H | 0.195% | 0.181% | 0.135% |

This also settles the M9C clipping observation: M8-QAT clips **0.193%** against its *own* fresh
grid, versus the 48.9% M9C measured against the M7-derived diagnostic grid. The M9C figure was a
grid-mismatch artefact exactly as reported there, and it never had any bearing on M8's own
benchmark.

## 9F.4 Actual `.nvc` benchmark (DAVIS test split, 719 frames)

Real encode/decode, measured payload bytes. 162 measurements, 0 failures.

| model | bits | BPP | PSNR dB | MS-SSIM | bytes/frame | enc s/fr | dec s/fr |
|---|---|---|---|---|---|---|---|
| M7 | 8 | 1.9146 | 27.582 | 0.9512 | 15684.2 | 0.0102 | 0.0103 |
| M7 | 6 | 1.4128 | 27.514 | 0.9493 | 11574.0 | 0.0065 | 0.0076 |
| M7 | 4 | 0.9000 | 26.529 | 0.9184 | 7372.9 | 0.0058 | 0.0062 |
| M8-QAT | 8 | 1.8587 | 29.748 | 0.9730 | 15226.9 | 0.0089 | 0.0100 |
| M8-QAT | 6 | 1.3566 | 29.613 | 0.9710 | 11112.9 | 0.0066 | 0.0075 |
| M8-QAT | 4 | 0.8447 | 27.855 | 0.9375 | 6919.4 | 0.0057 | 0.0060 |
| M9-CTRL | 8 | 1.8455 | 30.074 | 0.9739 | 15118.2 | 0.0070 | 0.0079 |
| M9-CTRL | 6 | 1.3431 | 29.925 | 0.9718 | 11002.8 | 0.0051 | 0.0057 |
| M9-CTRL | 4 | 0.8321 | 28.073 | 0.9388 | 6816.6 | 0.0043 | 0.0044 |
| **M9-L** | 8 | **1.7438** | **29.873** | **0.9753** | 14285.4 | 0.0055 | 0.0064 |
| **M9-L** | 6 | **1.2415** | **29.801** | **0.9740** | 10170.7 | 0.0040 | 0.0046 |
| **M9-L** | 4 | **0.7365** | **28.489** | **0.9529** | 6033.6 | 0.0034 | 0.0037 |
| M9-M | 8 | 1.7391 | 29.254 | 0.9684 | 14246.8 | 0.0054 | 0.0063 |
| M9-M | 6 | 1.2369 | 29.144 | 0.9668 | 10132.9 | 0.0041 | 0.0045 |
| M9-M | 4 | 0.7337 | 27.789 | 0.9438 | 6010.5 | 0.0035 | 0.0037 |
| M9-H | 8 | 1.7463 | 27.346 | 0.9412 | 14305.4 | 0.0060 | 0.0068 |
| M9-H | 6 | 1.2447 | 27.256 | 0.9395 | 10196.2 | 0.0046 | 0.0050 |
| M9-H | 4 | 0.7416 | 26.245 | 0.9164 | 6075.4 | 0.0040 | 0.0041 |

**M8-QAT reproduces its published M8 numbers exactly** (1.8587 / 29.748 / 0.9730 at 8-bit;
0.8447 / 27.855 / 0.9375 at 4-bit, versus MILESTONE_8_RESULTS.md's 1.8587 / 29.748 / 0.9730 and
0.8446 / 27.855 / 0.9375). The pipeline is validated end to end by that agreement.

### M8-QAT -> M9, and the control-isolated rate-term effect

| model | bits | BPP change | dPSNR | dMS-SSIM | vs CONTROL: BPP | dPSNR |
|---|---|---|---|---|---|---|
| M9-CTRL | 8 | -0.71% | +0.326 | +0.0009 | - | - |
| M9-CTRL | 6 | -0.99% | +0.311 | +0.0009 | - | - |
| M9-CTRL | 4 | -1.49% | +0.218 | +0.0013 | - | - |
| **M9-L** | 8 | **-6.18%** | **+0.125** | **+0.0023** | -5.51% | -0.201 |
| **M9-L** | 6 | **-8.48%** | **+0.188** | **+0.0030** | -7.56% | -0.123 |
| **M9-L** | 4 | **-12.80%** | **+0.634** | **+0.0154** | -11.49% | **+0.416** |
| M9-M | 8 | -6.44% | -0.494 | -0.0046 | -5.76% | -0.820 |
| M9-M | 6 | -8.82% | -0.469 | -0.0041 | -7.91% | -0.780 |
| M9-M | 4 | -13.13% | -0.066 | +0.0062 | -11.82% | -0.284 |
| M9-H | 8 | -6.05% | -2.402 | -0.0318 | -5.38% | -2.728 |
| M9-H | 6 | -8.25% | -2.357 | -0.0315 | -7.33% | -2.668 |
| M9-H | 4 | -12.20% | -1.610 | -0.0211 | -10.87% | -1.828 |

**M9-L is strictly dominant over M8-QAT at every bit depth**: lower BPP *and* higher PSNR *and*
higher MS-SSIM, simultaneously, three times over. That is the strongest form this result could take
and it does not depend on any interpolation or single operating point.

### BD-rate (whole curve, negative = fewer bits at equal quality)

| model | vs M8-QAT (linear) | vs M8-QAT (quadratic) | vs CONTROL (linear) | MS-SSIM vs M8-QAT (linear) |
|---|---|---|---|---|
| M9-CTRL | -8.98% | -9.68% | +0.00% | -3.25% |
| **M9-L** | **-21.49%** | -46.33% | **-11.76%** | **-23.16%** |
| M9-M | -2.24% | +6.01% | +7.49% | -10.85% |
| M9-H | n/a (no PSNR overlap) | n/a | n/a | +53.45% |

Two fit orders are reported deliberately. With only three operating points a quadratic is an
**exact** fit (3 points, 3 coefficients) and does no smoothing, so it can swing far on curvature the
data does not really pin down - here the two disagree by more than 2x for M9-L. The
piecewise-linear figure never extrapolates curvature and is the conservative floor. **The defensible
claim is M9-L >= 21% BD-rate better than M8-QAT**, of which about 9 points are DAVIS fine-tuning
(the control) and about **12 points are the rate term itself**.

## 9F.5 Proxy versus actual - the central negative result

| model | R_proxy | BPP 8-bit | BPP 6-bit | BPP 4-bit |
|---|---|---|---|---|
| M9-CTRL | 2.9485 | 1.8455 | 1.3431 | 0.8321 |
| M9-L | 0.6752 | 1.7438 | 1.2415 | 0.7365 |
| M9-M | 0.4006 | 1.7391 | 1.2369 | 0.7337 |
| M9-H | 0.1811 | 1.7463 | 1.2447 | 0.7416 |

**The proxy did not track actual bitrate.** Rank agreement is `False` at all three bit depths.
Between M9-L and M9-H the proxy fell **3.7x** (0.6752 -> 0.1811) while measured BPP moved by
**+0.7%** - very slightly *worse*. The proxy ranked M9-H best; the codec ranks it worst.

**The mechanism, measured not guessed.** The deployed quantizer calibrates its grid to each model's
own latent range, so the 4-bit bin width shrinks in step with the latent:

| model | 4-bit calibration scale | latent abs-mean | ratio |
|---|---|---|---|
| M9-CTRL | 4.0020 | 7.9766 | 0.502 |
| M9-L | 0.7658 | 1.9004 | 0.403 |
| M9-M | 0.3802 | 1.1609 | 0.328 |
| M9-H | 0.1840 | 0.7655 | 0.240 |

A uniform shrinkage of the latent is therefore **invisible to the real codec** - the symbol
histogram, and hence its entropy and the coded size, are essentially unchanged. But
`RateEstimator` measures bits against a **fixed** bin width (0.68, frozen from the M7-derived
calibration), so the same shrinkage reads to it as a large rate reduction. Past M9-L, that is
almost all the rate term buys: latent magnitude, which the codec normalises away, paid for with
real distortion, which it does not.

This is a genuine design limitation of the M9A proxy, not a bug in it: the proxy is
scale-sensitive where the deployed pipeline is scale-invariant. It is exactly the mismatch M9C Rule
9 and M9A section 7 flagged as unquantified, now quantified.

It also explains why M9-L works at all. At low lambda the encoder cannot buy much by shrinking
alone, so the pressure goes into genuinely reshaping the latent distribution - which the codec
*does* see. At high lambda shrinking is the cheaper way to satisfy the objective, and the model
takes it.

## 9F.6 Conclusions

1. **Yes - rate-aware training improved actual deployed RD, at the lowest lambda only.** M9-L beats
   M8-QAT on BPP, PSNR and MS-SSIM simultaneously at 8, 6 and 4 bits, for >=21% BD-rate.
2. **At all three bit depths**, with the largest gain at **4-bit** (-12.80% BPP with +0.634 dB and
   +0.0154 MS-SSIM) - the depth M8 identified as weakest, and the one that matters most for
   low-bitrate operation.
3. **M9-L (lambda 9.0757e-04) is the practical operating point.** It is the only one of the three
   that improves on the baseline without qualification.
4. **No - the highest-pressure models provided no useful additional compression.** M9-M and M9-H
   deliver the same real BPP as M9-L (within 1%) while losing up to 2.4 dB. Increasing lambda past
   the measured balance point bought nothing real.
5. **Yes - fresh calibration is healthy.** 18/18 within 0.106-0.195%, an order of magnitude inside
   the 2% guard.
6. **Partly.** The codec validated the *objective* - training against a rate term did improve real
   RD - but not the *proxy*: R_estimator's ordering of the models disagrees with the codec's, for
   the measured reason in 9F.5. A lower proxy R is confirmed to be insufficient evidence, exactly as
   the brief anticipated.

## 9F.7 What was NOT done

- No architecture, quantizer, entropy-coder or `.nvc` change; no hyperprior, autoregressive context
  or temporal coding. `benchmark_rd.py` was used unmodified.
- No M7 or M8 artifact was overwritten - checkpoints and historical calibration files retain their
  original August timestamps, and M8-QAT's re-measured numbers match its published ones.
- H.264/H.265 were not re-run: they do not depend on any M9 model, so M8's published comparison
  still stands unchanged.
- The runs are budget-limited, not converged (best epoch 68-70 of 70). A longer budget could move
  all of these numbers.

## 9F.8 Files

Created: `scripts/m9_final_train.py`, `scripts/m9_final_calibrate_benchmark.py`,
`scripts/m9_final_report.py`, `tests/test_m9_checkpoint_selection.py` (9),
`tests/test_scripts_m9_final.py` (10), `tests/test_scripts_m9_final_eval.py` (9),
`outputs/m9_final/` (4 model directories with `best.pt`/`latest.pt`/`history.json`, 18 calibration
files, 6 benchmark run directories, `training_summary.json`/`.csv`, `calibration_report.json`,
`benchmark_aggregate.json`/`.csv`, `rd_analysis.json`/`.csv`/`.txt`, `m9_final_rd.png`).

Modified: `scripts/train_autoencoder.py` (best-checkpoint seeding fix, objective-aware `[BEST]`
log), `TESTING.md`, this file.

Test suite: **613 passed, 0 failed**.
