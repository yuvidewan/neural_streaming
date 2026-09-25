# Roadmap: Reaching H.264 / H.265 Parity

**Objective.** Beat `libx264` at equal quality on the DAVIS test split, measured
by BD-rate. Then `libx265`.

**Where we start** (Stage 0 scoreboard, 2026-09-16,
[`outputs/benchmarks/parity_s0/`](outputs/benchmarks/parity_s0/README.md)). Against
default `libx264` on DAVIS TEST, the M22 codec has a BD-rate of **+450.7% on PSNR**
and **+378.3% on MS-SSIM**: H.264 needs ~5.5x fewer bits for the same PSNR, and
leads at every measured point. The gap widens with quality (4.8x at the 3-bit
point, 7.5x at 5-bit), because each rate doubling buys this codec 1.3 dB against
H.264's 2.8 dB.

This document is the plan to close that. It is deliberately explicit about what
will be thrown away, because the honest answer is "most of the model, none of the
infrastructure."

---

## 1. Why the gap exists

Not a mystery, and not an entropy-coding problem. Milestones 13-22 optimized the
entropy and quantization layers for gains of 1-4% each; even perfectly closing
M17's measured oracle gap (+2.5% / +8.5% / +19.6% of the residual channel) leaves
the codec far behind H.264. The missing factor is not stored in that layer.

The model is:

```
Encoder: Conv(3->32->64->128->64), stride 2, ReLU     593,411 parameters
Decoder: ConvTranspose(64->128->64->32->3), Sigmoid
```

A textbook autoencoder. Against what the literature uses for this task:

| | this codec | codecs that beat H.264/H.265 |
|---|---|---|
| parameters | 593 K | 10-30 M (**~34x more**) |
| nonlinearity | ReLU | GDN / IGDN |
| blocks | plain conv | residual, often attention |
| entropy model | learned context (M11-G16) | hyperprior **+** context |
| motion | 16x16 block search, hand-coded payload | learned optical flow, **learned** MV compression |
| motion compensation | block warp | warp **+** refinement network |
| training | staged, each component frozen in turn | single end-to-end RD objective |

Five of those seven rows are missing capability, not tuning. That is the gap.

**The encouraging part:** this is well-trodden ground. DVC (Lu et al., CVPR 2019)
reached roughly H.264 parity with exactly this class of architecture; the
DCVC line (2021-2023) passed H.265 and then H.266. The target is known to be
reachable, and the route is published.

---

## 2. What survives, and what gets rebuilt

Roughly 40% of the work already done carries forward untouched. That is the
reason this is a 4-6 month plan and not a restart.

### Keep — no changes needed

- **`range_coder.py`** - the from-scratch arithmetic coder. Verified lossless,
  within ~0.03% of its own theoretical optimum, with a C backend ~40x faster than
  the Python original. A learned codec needs exactly this and nothing more.
- **`.nvc` / `.nvct` containers** - header validation, identity fields, stream
  framing. The `entropy_model_id` mismatch detection is the right pattern and
  already generalizes to new models.
- **The evaluation harness** - `rd_benchmark.py`, `codecs.py`, the FFmpeg
  H.264/H.265 arms, BD-rate, MS-SSIM, the DAVIS split definitions.
- **The Vimeo-90k septuplet pipeline** - already on disk. This is the standard
  training set for learned video compression; DVC and its successors all use it.
- **The research method** - pre-registered candidates, VAL-B gates, TEST locked
  until candidate lock, provenance hashes, byte-exact reproduction. This is better
  than most published work and transfers unchanged.
- **The test suite** - 1,584 tests. Most concern the coder, container and data
  pipeline, all of which survive.

### Rebuild

- **The analysis/synthesis transforms** - GDN, residual blocks, ~10M parameters.
- **The entropy model** - add a hyperprior; keep M11-G16's context model as the
  autoregressive half (this is already most of Minnen 2018's joint model).
- **Motion** - learned flow, a motion autoencoder, a compensation network.
- **The training loop** - joint end-to-end RD optimization instead of the current
  staged freeze-and-fit.

### Retire

The M13-M22 research scripts stay in the repository as a record, but the
mechanisms they tuned (codebook routing, residual grid placement, hysteresis)
are properties of the current quantizer and will not survive the rebuild. M22's
`symmetric_p01` result should be ported forward only if the new quantizer turns
out to have the same misalignment - re-measure, do not assume.

---

## 3. The staged plan

Each stage has a gate measured against H.264, not against the previous stage.
A stage that does not move the BD-rate against `libx264` has failed, however good
its internal numbers look - that is the specific failure mode that produced eight
consecutive milestones with no shipped gain.

### Stage 0 - Establish the scoreboard (~1 week, no new modelling)

- [x] Re-run H.264/H.265 vs the current codec in **one pass**, three rate points,
  BD-rate on both metrics - `scripts/benchmark_parity.py`, 2026-09-16. It found
  that the earlier comparison scored H.264 and NVC with different PSNR
  definitions, and that `msssim()` read high on channels-last CUDA tensors
  (fixed).
- [x] Promote the video codec from `scripts/` into `src/nvc/` so there is one codec,
  not two. Add `encode()` / `decode()` - `nvc.video` with frozen, self-verifying
  codec bundles, proven byte-identical to the research code on all of DAVIS TEST
  (`scripts/verify_promoted_codec.py`), 2026-09-16.
- [x] Fix the decoder hang on malformed containers (unbounded `table_index`) - `33844a6`.
- [x] Declare dependencies in `pyproject.toml`, add a LICENSE - `9f419a2`.
- [x] CI - `.github/workflows/tests.yml`: the full suite on Ubuntu, Python 3.11, CPU
  PyTorch and FFmpeg, on every push to master and every pull request.

**Gate:** a single reproducible BD-rate number vs H.264. Everything after this is
measured against it. **Met:** +450.7% (PSNR) / +378.3% (MS-SSIM) against default
`libx264`, reproducible with `scripts/benchmark_parity.py --no-resume`. Every later
stage reports its BD-rate against this same harness.

### Stage 1 - A real analysis/synthesis transform (~3 weeks)

Replace ReLU with GDN/IGDN, add residual blocks, scale to ~8-12M parameters
(192-256 channels). Train on full Vimeo-90k with the existing RD objective.

- [x] The architecture - `nvc.models.gdn` (GDN/IGDN with the bounded
  reparameterization) and `nvc.models.residual_transform`
  (`AnalysisTransform`, `SynthesisTransform`, `ResidualGDNAutoencoder`).
  8,437,827 parameters at the default 192 channels, against the baseline's
  593,411. Stride 16, the `[0, 1]` output range and
  `encode`/`decode`/`config_dict`/`num_parameters` are all unchanged, so the
  existing training, checkpoint and evaluation paths take it as-is.
- [x] The gate's denominator - `scripts/benchmark_parity.py --gop 1`,
  `outputs/benchmarks/parity_intra/`, 2026-09-25. The current codec all-intra is
  **+176.2% BD-rate (PSNR) / +167.6% (MS-SSIM) against x264 all-intra**. A trained
  Stage 1 transform is measured against that number.
- [ ] Train it on full Vimeo-90k (~91,701 sequences; the deployed checkpoints
  saw 10 chunks). This is the long pole of the stage and needs the GPU budget
  in section 4.
- [ ] Re-run the intra-only scoreboard with the trained transform and compare.

**Gate:** intra-only BD-rate vs the current intra codec. Expect the largest
single jump of the whole plan here.

**What measuring the denominator already changed.** It split the Stage 0 deficit
into a transform half and a temporal half for the first time. Against x264 forced
into NVC's own structure: **+176.2% all-intra, +306.2% at GOP 10.** Forced
all-intra, x264 needs 2.8x the bits it needed with P-frames; NVC needs only 1.8x.
Both get worse without temporal prediction - x264 gets worse faster, which is the
whole reason the intra-only gap is the smaller number. The transform remains the
largest single deficit and Stage 1 is still the right next move, but NVC's motion
compensation is measurably the weaker half, and Stage 3 is carrying more of the
remaining gap than this plan assumed when it was written.

One arm from that run is unusable: x265 configured all-intra performs *worse* than
x264 all-intra (0.5326 vs 0.3026 bpp at 29 dB), which is backwards. Its rate curve
also has a floor. Undiagnosed; Stage 0's x265 arms are unaffected. See the run's
README before citing anything `h265_intra`.

### Stage 2 - Hyperprior (~2-3 weeks)

Add a scale hyperprior over the latent (Balle 2018). Keep M11-G16 as the
autoregressive context component - combined, this is the joint
autoregressive-plus-hierarchical model of Minnen 2018, which is what first beat
BPG/HEVC-intra.

**Gate:** BD-rate vs Stage 1. This is where intra parity with H.264 should
become plausible.

### Stage 3 - Learned motion (~4 weeks)

Replace the 16x16 block search with a flow network. Compress the flow field with
its own small autoencoder rather than the hand-designed motion payload. Add a
compensation network that refines the warped frame instead of using it directly.

**Gate:** P-frame BD-rate. The GOP-boundary anomaly M16-M22 kept finding and never
exploited should be re-measured here - learned flow may simply dissolve it.

### Stage 4 - End-to-end joint training (~4 weeks)

Unfreeze everything and train the full loop under one RD objective, across
several lambda values to produce a real rate-distortion curve rather than three
bit-depth points.

**Gate:** full BD-rate vs H.264. This stage historically produces a large gain on
its own, because the staged-and-frozen structure the current codec uses leaves
every interface un-co-adapted.

### Stage 5 - Close on H.265 (open-ended)

Feature-space temporal propagation instead of pixel-space warping (the DCVC
idea), variable-rate in a single model, longer training.

**Gate:** BD-rate vs `libx265`.

---

## 4. Compute

This is the binding practical constraint and should be planned for, not
discovered.

- Stages 1-2 are trainable on the local RTX 5060 (8 GB) at 256x256 crops with a
  small batch - slow but workable.
- Stages 3-5 involve multiple networks in one training graph and realistically
  need a rented 24 GB+ GPU. Budget for it rather than trying to fit around it.
- Full Vimeo-90k is ~91,701 sequences. Current checkpoints saw 10 chunks. Stage 1
  should be the first run that uses the whole set.

---

## 5. Honest timeline

**4-6 months of focused work, roughly 10-14 milestones**, with GPU budget for the
later stages.

Parity with H.264 is a realistic target at Stage 4. Parity with H.265 is Stage 5
and should be treated as a stretch goal, not a commitment.

What is *not* realistic: reaching parity by continuing to optimize the current
architecture. At the 1-4% per milestone the recent milestones delivered, closing
a 5.5x gap takes ~85 milestones. The plan above is shorter precisely because it
changes the thing that is actually limiting the result.

---

## 6. The rule that keeps this on track

Every milestone brief from here starts by answering one question:

> **What does this change about the BD-rate against H.264?**

A milestone that cannot answer it is measuring the wrong thing. Applied
retroactively, M15 through M21 would each have failed this before any code was
written - and the capacity problem would have surfaced a week earlier.
