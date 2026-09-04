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
