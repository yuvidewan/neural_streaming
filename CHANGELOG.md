# Changelog

This file is the running record of fixes and changes made to this codec,
newest entry first. It exists so anyone (Aditya included) can catch up on
what changed and why without reconstructing it from commit messages.

**Maintenance rule:** when we ship something new, add a new dated entry at
the top. If a later fix changes or supersedes an earlier entry's numbers or
conclusions, that later entry says so explicitly and links back — we don't
silently rewrite history here, but we do keep this file accurate as the
single current picture, not just an append-only log of what we believed at
the time.

---

## 2026-09-05 — Calibration crash fix, stream header overhead, RD-proxy exploit fix, per-channel bit allocation, companding

**Commit:** [`380c4b3`](https://github.com/yuvidewan/neural_streaming/commit/380c4b3a1df1fc1499ee640f4e32acedac0d4fd0)
**Source:** [OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md) audit (items B1, Q1, Q2, Q3) plus the
9F.5 rate-distortion-proxy bug documented in [MILESTONE_9_PLAN.md](MILESTONE_9_PLAN.md).
**Tests:** 662 passing (full suite), zero regressions. 91 new tests added across this change.

Five fixes, all additive — every new field on `QuantizationParams` / `RateEstimator` defaults to
off, and every calibration file or checkpoint already on disk loads unchanged.

### 1. Fixed: `torch.quantile` crash above 16.7M elements (B1)

**The bug:** `torch.quantile` hard-refuses any input above 2²⁴ = 16,777,216 elements
(`RuntimeError: quantile() input tensor is too large`). `calibrate_quantization_params`
(`src/nvc/compression/calibration.py`) called it once per channel on the full calibration set.
With the default 64×16×16 latent this crashes at:

| Mode | Crashes above |
|---|---|
| `global` | 1,024 calibration frames |
| `per_channel` (default) | 65,536 calibration frames |

Not firing today only because `scripts/calibrate_quantizer.py` defaults to 400 frames — a latent
crash waiting for anyone who raises `--max-batches`.

**The fix:** `_quantile_safe()` — below a 10M-element limit it is byte-for-byte identical to
`torch.quantile`; above it, a deterministic random subsample of exactly that many values is used
instead of crashing.

**Result:** no crash above the limit; a subsampled 1st/99th percentile estimate on a 12,800-value
synthetic population agreed with the full-population estimate within 5% relative error across two
independent draws. Below the limit, output is unchanged (bit-exact, verified by test).

### 2. Fixed: per-frame header overhead (Q1)

**The bug (really: waste):** the `.nvc` header re-sends all per-channel `scale`/`zero_point`
values on *every single frame* — 512 of the header's 549 bytes, identical from frame to frame in
a sequence since calibration is fixed. On a ~13KB frame payload that's roughly 4% of every frame
spent re-transmitting data the decoder already has after frame 1.

**The fix:** an additive `.nvcs` stream container — `NVCStreamHeader` / `NVCStreamWriter` /
`NVCStreamReader` (`src/nvc/compression/nvc_format.py`) plus `encode_frame_payload` /
`decode_frame_payload` / `build_stream_header` (`src/nvc/compression/codec.py`). One 33-byte fixed
header + the parameter block once per stream, then a 4-byte length prefix per frame. The original
single-frame `.nvc` format (`NVCHeader`/`NVCWriter`/`NVCReader`/`encode_frame`/`decode_frame`) is
completely unchanged and still works exactly as before — both formats live side by side.

**Result:** stream output is bit-exact against the existing per-frame path (same payload bytes,
same decoded pixels — verified by test) and measurably smaller for any sequence of more than one
frame. **Not yet wired into `benchmark_rd.py`'s actual encoding path** — the primitive is done and
tested; plugging it into the benchmark harness is the natural next step, not done here.

### 3. Fixed: the RD-proxy could be cheated by shrinking the latent uniformly (9F.5)

**The bug:** `RateEstimator`'s differentiable proxy rate loss used a *frozen* bin width fixed at
calibration time. During QAT, the model could reduce the proxy's reported rate simply by shrinking
its latent's dynamic range uniformly — with a frozen bin width, a smaller latent maps to fewer
occupied bins in the proxy's eyes, but the *actual* rate paid by the real quantizer + entropy coder
does not improve, because real calibration would just re-derive a smaller `scale` for the smaller
range. The proxy and the real coder disagreed, and gradient descent found the exploit.

**The fix:** `RateEstimator.update_bin_width()` — an EMA (momentum 0.99 default) that tracks the
*actual* observed dynamic range during training and updates the proxy's bin width to match, so
shrinking the latent no longer fools the proxy. Wired into `train_one_epoch_with_rate` (called
after every optimizer step); deliberately **not** called during validation, so validation always
measures against the calibration-time bin width. CLI: `--rate-track-scale` / `--rate-scale-momentum`
in `scripts/train_autoencoder.py`.

**Result, measured on a real M8 checkpoint:** artificially shrinking the latent's dynamic range and
checking the reported proxy-rate reward for doing so —

| | Reward for pure latent shrinkage |
|---|---|
| Frozen bin width (before) | **+1.4592 bpp** |
| Tracked bin width (after) | **+0.5667 bpp** |

A 61% reduction in the exploit's reward. **What this does *not* establish:** this proves the
mechanism closes, not that M9-M/M9-H (the two milestone configurations that previously showed no
real improvement) will now train better — that needs an actual GPU retraining run to confirm, which
was out of scope here.

### 4. Added: per-channel bit allocation (Q3) — mechanism works, real-data payoff didn't show up

**The idea:** spend more quantization levels on channels carrying more information, fewer on
channels that barely vary (the same axis JPEG's quantization matrix exploits). Half the
infrastructure (per-channel calibration and statistics) already existed — this is "wire it
through," not "build it."

**What was built:**
- `QuantizationParams.bits_per_channel` + a new `effective_q_max` property
  (`src/nvc/compression/quantization.py`) — `bits` stays the entropy table's alphabet depth
  everywhere it already was (so `.nvc` headers, the entropy model, and the range coder need **zero**
  changes); `bits_per_channel` only ever *narrows* an individual channel's clamp bound below that
  shared depth. Unused high symbol values are handled for free by the entropy model's existing
  Laplace smoothing.
- `calibrate_quantization_params(bits_per_channel=...)` — gives a restricted channel a
  proportionally *coarser scale* over the same calibrated range, not just a harsher clamp on an
  unchanged fine step (`src/nvc/compression/calibration.py`).
- `allocate_bits_per_channel()` — the classical water-filling formula from calibration-latent
  variance: `bits_c = avg + 0.5·log2(var_c / geomean(var))`, integer-rounded and rebalanced to hit
  the exact requested average bit budget.
- CLI: `--allocate-bits-per-channel` / `--allocate-average-bits` /
  `--min-bits-per-channel` / `--max-bits-per-channel` in `scripts/calibrate_quantizer.py`.

**Real-data validation (`outputs/checkpoints/vimeo_qat_noise_best.pt`, 400 calibration frames,
64 channels, per-channel variance ratio 26× across channels):** allocating an average of 6
bits/channel (range came out 5–7 bits, inside an 8-bit table) against a **plain uniform 6-bit
baseline at the same average bit budget** —

| | MSE | Mean per-channel empirical entropy |
|---|---|---|
| Uniform 6-bit (baseline) | 0.05635 | 5.216 bits/symbol |
| Water-filled, avg 6-bit | 0.05591 (0.8% better) | 5.249 bits/symbol (0.6% *worse*) |

Essentially a wash, both directions within noise. **Likely reason:** the existing per-channel
percentile-range calibration already adapts each channel's *scale* to its own spread — a
high-variance channel's step is already coarser and a low-variance channel's already finer before
any bit reallocation happens, so water-filling on top of that captures little marginal signal the
scale adaptation hadn't already captured.

**Bottom line:** the mechanism is implemented correctly and verified end to end (a channel given
fewer allocated bits genuinely produces measurably fewer distinct symbols and a coarser step, not
merely a clamp) — but on this codec's actual latent statistics, it is not the free win the
classical formula promises. Anyone using this needs to measure per-checkpoint, not assume a win.

### 5. Added: non-uniform quantization via companding (Q2) — small real win, gamma-sensitive

**The idea:** the latent is documented (`calibration.py`'s own docstring) as "sharply peaked with
long tails" — equal-width bins waste most of their resolution on nearly-empty tail regions. A
power-law companding transform `y = sign(x)·|x|^γ` (mu-law-style, chosen over an iterative
Lloyd-Max fit for simplicity) compresses large magnitudes and expands small ones before the
existing uniform grid, buying finer steps near zero where the density actually is.

**What was built:** `QuantizationParams.companding_gamma` — calibration companies the latent before
computing percentiles; `UniformQuantizer` companies before quantizing and expands (`x =
sign(y)·|y|^(1/γ)`) after dequantizing. `scale`/`zero_point` are computed *in the companded domain*
and stored exactly as before, so — same as bit allocation — **zero changes** to `nvc_format.py`,
`entropy_model.py`, or the range coder; only the calibration file gains one extra recorded number.
CLI: `--companding-gamma` in `scripts/calibrate_quantizer.py`.

**Real-data validation (same checkpoint, 32 calibration frames, 8-bit per-channel), latent MAE/MSE
at a *fixed* bit depth:**

| γ | MAE | MSE |
|---|---|---|
| 1.0 (no companding) | 0.0433 | 0.02316 |
| 0.7 | **0.0387 (−11%)** | **0.02285 (−1.3%)** |
| 0.5 | 0.0413 (roughly a wash) | 0.02343 (slightly worse) |
| 0.3 | 0.0532 (**worse than uncompanded**) | 0.02597 (worse) |

**Bottom line:** real, but modest, and **not monotonic in γ** — mild companding (γ≈0.7) helps,
aggressive companding hurts. This latent is peaked, but not peaked enough to reward the strong
companding its qualitative description might suggest. Needs a small per-checkpoint gamma sweep
before use, not a fixed default.

### Also fixed along the way

While adding tests for per-channel bit allocation, a real bug surfaced: `torch.clamp` in this
project's PyTorch version (2.13.0) refuses a call mixing a scalar `min` with a tensor `max` — which
is exactly what `UniformQuantizer.quantize()` did the moment `effective_q_max` became a per-channel
tensor. Fixed by branching on `effective_q_max`'s type (`torch.clamp` for the scalar case,
`torch.minimum` for the tensor case). Caught by the new tests before this ever shipped.

### Files changed

`MILESTONE_9_PLAN.md` · `OPTIMIZATION_ANALYSIS.md` · `scripts/calibrate_quantizer.py` ·
`scripts/train_autoencoder.py` · `src/nvc/compression/__init__.py` ·
`src/nvc/compression/calibration.py` · `src/nvc/compression/codec.py` ·
`src/nvc/compression/nvc_format.py` · `src/nvc/compression/quantization.py` ·
`src/nvc/training/rate_estimator.py` · `src/nvc/training/trainer.py` · `src/nvc/utils/config.py` ·
`tests/test_entropy_coding.py` · `tests/test_rate_estimator.py`
