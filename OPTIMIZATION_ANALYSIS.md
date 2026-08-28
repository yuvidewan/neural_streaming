# Optimization Analysis

A code-level audit of this codec for two separate goals - **speed** (latency
and throughput) and **quality** (compression efficiency at a given
distortion) - covering both what is already built and what is still
available. Companion to the README, which documents *what the project does*;
this documents *where it could do it better*.

## How to read this

Every claim below is tagged:

- **[measured]** - a number produced by running code on this machine during
  the audit. Reproducible; the command is given.
- **[verified]** - a factual statement about the code, confirmed by reading
  or grepping the file named. Not a performance claim.
- **[estimated]** - an engineering judgement, with the reasoning stated.
  Not measured. Treat accordingly.

Absolute timings are from a CPU-only laptop (Intel i5-8350U, 4 cores, no
CUDA GPU). Absolute milliseconds will differ on other machines; the
*proportions* and the *ranking* are what transfer.

---

## Part 0: The bottleneck has moved (read this first)

Before the Milestone 8B C migration, the pure-Python arithmetic coder
dominated per-frame cost, and the README said - correctly at the time -
that hardware-accelerating the neural network "buys comparatively little
while the coder alone accounts for nearly all of the measured per-frame
time."

**That is no longer true, and this audit measured the reversal.** Full
encode+decode round trip of one 256x256 frame, component by component:

| Stage | Time | Share |
|---|---|---|
| CNN encoder forward | 6.4 ms | 31% |
| quantize -> symbols | 0.5 ms | 2% |
| arithmetic encode (C) | 1.6 ms | 8% |
| arithmetic decode (C) | 2.4 ms | 12% |
| CNN decoder forward | 9.6 ms | 47% |
| **Round trip total** | **20.5 ms (~49 fps)** | |

**[measured]** - 20-40 reps per stage, repeated across three trials; the
neural share held at 76-82% and the coder at 18-24% each time.

Two consequences that change the project's own stated priorities:

1. **The neural forward pass is now the dominant cost (~78%), not the
   entropy coder (~20%).** The hardware-acceleration option the README
   currently files as "buys comparatively little" is now the single
   biggest remaining speed lever. That note should be read as historical.
2. **The codec is already near real-time at 256x256 on a CPU with no GPU
   at all** - ~49 fps round trip. Encode-only is ~118 fps. This does not
   close the gap to H.264/H.265 (still hardware-accelerated, still
   handling full-resolution video rather than 256x256 crops), but "not
   real-time" is no longer an accurate description of the codec at this
   resolution.

---

## Part 1: What is already done right

Auditing for weaknesses also means recording what does not need changing.
These were checked and found correct - listing them prevents a future pass
from "fixing" something that is already deliberate.

| Area | Finding |
|---|---|
| **Transposed-conv artifacts** | `Decoder` uses `ConvTranspose2d(kernel_size=4, stride=2)`. Kernel size is **evenly divisible by stride**, which is precisely the condition that avoids the checkerboard artifacts transposed convolutions are known for (Odena et al., 2016). A naive audit would flag `ConvTranspose2d` on sight; this configuration is already correct. **[verified]** `src/nvc/models/decoder.py` |
| **Percentile calibration** | Calibration clips at the 0.1/99.9 percentiles rather than absolute min/max, because the latent distribution is sharply peaked with long tails. This is the right call and the reasoning is documented in the module itself. **[verified]** `src/nvc/compression/calibration.py` |
| **Per-channel everything** | Both the quantizer grid and the entropy model's frequency tables are per-channel, not global - the latent's channels genuinely differ in distribution. **[verified]** `quantization.py`, `entropy_model.py` |
| **Integer-only coder arithmetic** | The arithmetic coder uses integer cumulative frequencies, not floats, so encoder and decoder compute bit-identical intervals. Using floats here is a classic way to produce a coder that desynchronizes on another platform. **[verified]** `range_coder.py` |
| **Zero-probability safety** | Laplace smoothing plus a `MIN_FREQUENCY = 1` floor guarantees no symbol is ever unencodable, which would otherwise be a hard crash on rare data. **[verified]** `entropy_model.py` |
| **Leakage discipline** | Calibration is train-split-only and enforced in code (QAT rejects a calibration file not marked `calibration_split: train`); Vimeo's official train/test lists are asserted disjoint. **[verified]** `quantization_noise.py`, `vimeo.py` |
| **QAT gating** | Training-time quantization noise is gated on `nn.Module.training`, so it is structurally impossible for it to fire during eval or inference. **[verified]** `models/autoencoder.py` |
| **C coder equivalence** | The C port was validated byte-identical against the Python reference before the Python path was retired. **[measured]** - see README Milestone 8B |

---

## Part 2: Speed findings, ranked by effort vs. payoff

Ranked by payoff per unit of effort, not by raw payoff.

### S1. Mixed-precision training (AMP) - **best ratio**

- **Effort:** Very low. ~5 lines: wrap the forward/loss in `torch.autocast`, add a `GradScaler`.
- **Payoff:** High on GPU. Typically **1.5-3x faster training** on Tensor-Core hardware, plus roughly halved activation memory (which also allows a larger batch).
- **Status:** **[verified]** zero matches for `autocast`/`GradScaler` anywhere in `src/` or `scripts/`. Training is fp32 throughout.
- **Risk:** Low, but not zero - fp16 can underflow small gradients, which is exactly what `GradScaler` exists to handle. Worth a short A/B on one chunk to confirm loss curves match.
- **CPU-only note:** No benefit on the audit machine; this one is for the GPU box.

### S2. `pin_memory=True` in the DataLoader factories

- **Effort:** Trivial. A default change in `loaders.py` (or pass-through from the training scripts).
- **Payoff:** Small but genuinely free - page-locked host memory enables faster asynchronous host-to-device copies. Only helps when a GPU is in play.
- **Status:** **[verified]** every factory in `src/nvc/data/loaders.py` defaults to `pin_memory=False`.
- **Risk:** Essentially none. Unlike `num_workers`, it has no Windows-specific failure mode.

### S3. `cv2.setNumThreads(0)` when using DataLoader workers

- **Effort:** Trivial. One line at import time.
- **Payoff:** Moderate, and specifically prevents a *regression*: OpenCV defaults to multi-threaded decode, so `num_workers=N` gives N processes each spawning their own OpenCV threads, oversubscribing the CPU and often running **slower** than `num_workers=0`. This is a well-known interaction, and it will bite exactly when someone raises `--num-workers` to speed training up.
- **Status:** **[verified]** no `setNumThreads` call anywhere in `src/`.
- **Risk:** None.

### S4. `torch.compile` on the model

- **Effort:** Low. One line, but needs a version/platform check and a warmup allowance.
- **Payoff:** Moderate. PyTorch 2.x graph compilation and kernel fusion - the gain is real but variable, and this is a small model with few layers, which limits fusion opportunity.
- **Status:** **[verified]** no `torch.compile` anywhere.
- **Risk:** Low-moderate. First-call compile latency; occasional backend quirks on Windows.

### S5. Hardware-accelerated inference (ONNX -> GPU/NPU) - **highest raw payoff, highest effort**

- **Effort:** High. Export path, runtime integration, per-platform validation, plus keeping exported and training models in sync.
- **Payoff:** **Now the largest single speed lever available** - it targets the ~78% of round-trip time the neural forward passes occupy (Part 0). Convolutions are exactly what this hardware exists to run.
- **Status:** **[verified]** not started; the README lists it as a deliberate unscheduled future target.
- **Note:** This item's ranking *moved up* as a direct result of the C migration succeeding. It was correctly deprioritized before; that reasoning has now expired.

### S6. Decoded-frame cache for repeated epochs

- **Effort:** Moderate. Cache decoded/cropped tensors to disk, plus invalidation logic.
- **Payoff:** Situational. Trades disk for repeated PNG decode cost across epochs. The project explicitly rejected this once already (to avoid duplicating an 82 GB dataset), and that reasoning still holds for full Vimeo - but the per-chunk training flow only holds one ~9 GB chunk at a time, where a cache is far more tractable.
- **Status:** **[verified]** noted as a known limitation in the README; not implemented.
- **Risk:** Moderate - a stale cache silently training on wrong data is a nasty failure mode. Needs a content hash, not just a path check.

---

## Part 3: Quality findings, ranked by effort vs. payoff

"Quality" here means **compression efficiency** - fewer bits at the same
distortion, or less distortion at the same bits. None of these are speed
optimizations, and several trade *against* speed (see Part 4).

### Q1. Per-frame header overhead - **best ratio, and it is nearly free**

- **Effort:** Low. Send quantization parameters once per stream instead of per frame; a format-version bump and a decoder branch.
- **Payoff:** Direct and immediate. **[verified]** the header is 549 bytes, of which **512 are the per-channel scales and zero-points** (64 channels x 2 float32 x 4 bytes) - re-sent identically in every single frame despite being fixed at calibration time. On a 13 KB frame payload that is roughly a **4% total-size reduction** for no quality loss whatsoever.
- **Why it ranks first:** it is pure waste, not a trade-off. Every other item here costs something.
- **Status:** Already flagged in the README's Milestone 6 limitations and in the gap analysis; still open.

### Q2. Non-uniform quantization matched to the latent distribution

- **Effort:** Moderate. The calibration file format already carries per-channel parameters; this changes what those parameters mean and requires re-calibration, but not retraining.
- **Payoff:** Likely significant. **[verified]** `quantization.py` implements a strictly *uniform* affine grid: `scale = (x_max - x_min) / (q_max - q_min)`, equal-width bins everywhere. But the project's own calibration module documents the latent as *"sharply peaked with long tails (range about [-16.5, 13.7] while the standard deviation is only 2.24)"*. Equal-width bins on a peaked distribution spend most of their resolution on nearly-empty regions. Companding (or Lloyd-Max style bin placement) directly attacks that mismatch.
- **Nuance worth knowing:** the 0.1/99.9 percentile clipping already recovers *part* of this gain by refusing to let outliers stretch the grid. Non-uniform bins are the next step past that, not a replacement for it.

### Q3. Per-channel bit allocation

- **Effort:** Moderate. Bit depth is currently a single scalar for the whole tensor; making it per-channel touches the quantizer, the entropy model's table sizing, and the `.nvc` header.
- **Payoff:** Real. **[verified]** `UniformQuantizer(bits, mode)` applies one `bits` value to every channel - even in `per_channel` mode, which varies *scale and zero-point* per channel but not the *number of levels*. This is precisely the axis JPEG's quantization matrix exploits: spend more levels on channels carrying more information, fewer on channels that barely vary. Half the infrastructure (per-channel calibration and statistics) already exists.

### Q4. Colour-space transform + chroma subsampling

- **Effort:** Moderate-high. A YCbCr transform, subsampled chroma planes, and every metric/shape assumption in the pipeline updated to match.
- **Payoff:** Potentially large - this is one of the oldest and most reliable wins in image/video coding, exploiting the eye's much lower sensitivity to colour detail than luminance.
- **Status:** **[verified]** the pipeline is full-resolution RGB end to end (`image_io.py` reads RGB; the model's `in_channels=3` RGB). Milestone 7's own methodology notes already flag that H.264/H.265 default to `yuv420p` while NVC codes full-resolution RGB - so this is a known, documented asymmetry that currently counts *against* NVC in every benchmark run.
- **Caveat:** this also makes the existing benchmark comparison *more* apples-to-apples, so expect it to change reported numbers on both sides of the comparison.

### Q5. GDN activations instead of ReLU

- **Effort:** Moderate, and **requires full retraining** - it changes the architecture, so every existing checkpoint and calibration is invalidated.
- **Payoff:** Well-supported in the literature. **[verified]** the encoder/decoder use plain `ReLU` with no normalization layers of any kind. Ballé et al. (2017) - already reference [1] in the project's own literature review, and the source of the QAT noise technique this project implements - specifically use Generalized Divisive Normalization because it is designed for exactly this density-modelling role, and report it outperforming conventional activations for compression.
- **Why it ranks below Q1-Q4:** it invalidates the current checkpoint, so it cannot be A/B'd cheaply against existing results.

### Q6. Learning-rate schedule

- **Effort:** Trivial. A few lines; PyTorch ships the schedulers.
- **Payoff:** Small-moderate and indirect - it improves the *trained model*, not the codec design. Typically reaches a better final loss within the same epoch budget.
- **Status:** **[verified]** no `lr_scheduler` anywhere; both training scripts use a fixed Adam learning rate for the entire run.
- **Listed last** deliberately: it is the easiest item here, but it is a training-hygiene improvement rather than a codec improvement, and its payoff is the least predictable.

---

## Part 4: The combined trade-off

The two lists are not independent. Some items help both, some trade
directly against each other, and one pair is actively self-cancelling.

### They point in opposite directions on model size

Every quality item that adds capacity (Q5's GDN layers, and the
transformer/temporal options discussed in the README) makes the neural
forward pass *slower* - and Part 0 shows that forward pass is now **78% of
round-trip time**. Quality work now costs speed in the place where speed
currently hurts most. The reverse was true before the C migration.

This is the central tension: **the project's two goals now compete for the
same budget**, where previously they did not.

### The one item that is pure win

**Q1 (header overhead)** costs nothing on either axis. It removes 512
wasted bytes per frame with no quality change, no speed change, and no
retraining. Nothing else on either list is free.

### The pairing that resolves the tension

**S5 (hardware inference) + any quality item that adds capacity.** S5 buys
back headroom in exactly the component that quality work makes heavier.
Doing quality work *first* and hardware acceleration *later* means
measuring a codec that looks slower than it needs to; doing S5 first makes
every subsequent quality experiment cheaper to evaluate.

### Sequencing recommendation

1. **Q1** - free, immediate, no trade-off.
2. **S1, S2, S3** - hours of work, no architecture risk, and they make
   every subsequent training run cheaper (which compounds, given this
   project's documented compute constraints).
3. **B1** (below) - a latent bug; fix before it fires.
4. **Q2, Q3** - real compression gains, no retraining required.
5. **S5** - buy the speed headroom before spending it.
6. **Q4, Q5** - the expensive, retraining-required items, once there is
   headroom and a fast measurement loop to evaluate them in.

### What none of this changes

None of these items close the structural gaps documented elsewhere -
intra-only coding (no temporal prediction), a static entropy model with no
spatial context, and no rate term in the training objective. Those remain
the dominant reasons H.264/H.265 outperform this codec, and they are
milestone-scale work, not optimizations. **This document is about doing the
current design better, not about changing what the design is.**

---

## Part 5: One real bug found during this audit

### B1. `torch.quantile` size limit will break calibration on a large set

**[measured]** `torch.quantile` refuses inputs above **16,777,216 elements**
(confirmed by binary search on this machine, PyTorch 2.13.0):

```
RuntimeError: quantile() input tensor is too large
```

`calibrate_quantization_params` (`src/nvc/compression/calibration.py`) calls
it on one flattened tensor per channel - or, in `global` mode, on the
entire flattened latent set at once. With the default 64x16x16 latent:

| Mode | Crashes above |
|---|---|
| `global` | **1,024 calibration frames** |
| `per_channel` (default) | 65,536 calibration frames |

**Not currently firing:** `scripts/calibrate_quantizer.py` defaults to 50
batches at batch-size 8 = 400 frames, safely under both limits. But 1,024
frames is a completely reasonable calibration set - anyone raising
`--max-batches` past 128 in global mode hits a hard crash with an error
message that says nothing about calibration.

**Fix:** chunked/streaming quantile estimation, or deterministic
subsampling before the quantile call. Low effort, and worth doing before it
fires on someone mid-run rather than after.

Not fixed in this pass - this document is an audit, and the fix is a code
change that deserves its own review and test.

---

## Reproducing the measurements

Component timing breakdown (Part 0) and the `torch.quantile` limit (B1)
were both produced by ad-hoc scripts during the audit rather than committed
tooling. The committed benchmark covering the arithmetic coder specifically
is:

```bash
python scripts/benchmark_range_coder.py --reps 30 --label baseline
```

Results land in `outputs/benchmarks/range_coder_<label>.json`.
