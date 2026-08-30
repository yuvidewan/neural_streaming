# Research Notes: RD Loss, Temporal Coding, INT8 Validation

Three threads requested after Milestone 8 / the NVC-ACCEL hardware architecture: (A) literature
on joint rate-distortion training, (B) literature on lightweight temporal coding, (C) empirical
validation of the accelerator's INT8 assumption. C is a completed measurement against this
project's real checkpoint and full DAVIS test split; A and B are literature findings with a
concrete, minimally-invasive proposal for this specific codebase, not just a survey.

## TL;DR — recommended order

1. **Fix the INT8 activation scheme** (C found a real, non-trivial cost — cheap fix, do first).
2. **A (RD loss term)** — already named as the literal next milestone in
   [MILESTONE_8_RESULTS.md](MILESTONE_8_RESULTS.md) §20; a training-loss-only change, no
   architecture/codec-format change, consistent with every milestone's constraints so far.
3. **B (temporal coding)** — the bigger lever, but architecturally invasive; sequence after A
   using A's improved model as the temporal codec's starting point, not before.

---

## C. INT8 activation quantization — measured [MEASURED]

Closes [hardware/ARCHITECTURE.md](hardware/ARCHITECTURE.md) §10 risk #1. Ran
[hardware/int8_activation_validation.py](hardware/int8_activation_validation.py): per-output-channel
INT8 weights (symmetric, standard TFLite/TensorRT convention) + per-tensor INT8 activations
(percentile-calibrated, 0.1/99.9 — this project's own existing convention), both fake-quantized in
float so the numerics match real INT8 hardware. Real QAT checkpoint, real 8-bit calibration, full
real `.nvc` round trip (actual `encode_symbols`/`decode_symbols`, not simulated), full 719-frame
DAVIS test split — the same rigor M8 used for QAT.

| Metric | float32 (M8) | INT8-simulated | Δ |
|---|---|---|---|
| Mean PSNR | 29.748 dB | 28.751 dB | **−0.997 dB** |
| Mean MS-SSIM | 0.9730 | 0.9611 | **−0.0119** |

Existing calibration still fits the INT8-simulated encoder's latents (0.1225% clipped, threshold
2%) — no recalibration needed for that part. Entropy coding stayed lossless throughout (asserted
every frame).

**Verdict: this is a real cost, not noise.** For scale: M8 measured QAT's 8-bit→6-bit PSNR drop at
only 0.134 dB; INT8 activation quantization alone costs **~7.4× that**, roughly half of the
6-bit→4-bit drop (1.759 dB). Before committing this precision choice to RTL, one of:

- **Per-channel activation quantization** (not per-tensor) — the standard first fix; this codec's
  own latent quantization is already per-channel, so there's a direct precedent to follow, and
  it's the most likely single biggest win here since activation ranges vary far more across the
  32/64/128 channels of the middle layers than across the whole tensor.
- **INT16 instead of INT8** for activations, matching what DCVC-RT (§B below) reports actually
  shipping for cross-platform-deterministic inference, if the resulting SRAM budget still fits
  §7's on-chip constraints.

Either is a bounded, well-understood change to `int8_activation_validation.py`'s calibration
function, not a new experiment design.

---

## A. Rate-distortion loss term

**The situation this codebase is actually in**, which matters for which technique applies: most
learned image/video codecs ([Ballé et al., *End-to-end Optimized Image Compression*](https://www.cns.nyu.edu/pub/lcv/balle17a-submitted-revised.pdf))
train a single differentiable entropy model *jointly* with the autoencoder, and deploy that same
model. This project's entropy model is different — a **static per-channel histogram, calibrated
once, after training, from held-out train-split latents** (`calibrate_quantizer.py`). There is
currently no rate signal in the training loss at all; MSE reconstruction is the only term.

**How the literature makes quantization differentiable for training** (needed for any rate term
to have a gradient): additive uniform noise, `z̃ = z + Uniform(-1/2, 1/2)`, is the standard
technique — and **this project already has exactly this**, as `QuantizationNoise` from Milestone
8A (`z̃ = z + Uniform(-scale/2, scale/2)`), already measured to help real RD performance in M8.
The missing piece is not the noise mechanism — it's using the noised latent to estimate *rate*,
not just feeding it through the decoder for *distortion*.

**The standard rate estimate**: a differentiable density model over the noised latent (a
factorized/parametric density, e.g. one Gaussian or Laplacian per channel) gives a continuous rate
proxy via its negative log-likelihood, summed over the latent — `R̂ = -Σ log₂ p(z̃)` — added to the
loss as `L = D + λR̂`. Critically, **this proxy density does not have to be the deployed entropy
model.** The concrete, minimally-invasive proposal for this codebase:

1. During training, after `QuantizationNoise` (already exists), fit a per-channel Gaussian or
   Laplacian to the batch's noised latent (a few summary statistics, no new learned parameters
   needed, or a small learned scale per channel if a fixed-form fit undershoots).
2. Add `λ · (-log₂ p(z̃))`, summed over the latent, to the existing MSE loss. `λ` sweeps out an
   R-D curve exactly the way `--qat-bits`/`--qat-mode` already sweep operating points in the
   existing QAT CLI.
3. **Leave `calibrate_quantizer.py` and the deployed static histogram entropy model completely
   unchanged.** The Gaussian/Laplacian proxy only ever influences gradients during training; the
   real entropy coder stays exactly what it is today. No codec-format change, no architecture
   change — a loss-function-only addition, matching every milestone's stated constraints so far.

**One real risk, and a known fix for it**: [*Soft then Hard: Rethinking the Quantization in
Neural Image Compression*](https://arxiv.org/abs/2104.05168) (Guo et al., ICML 2021) documents
that additive-uniform-noise training has a **train-test mismatch** — the network optimizes against
soft/noised latents but is evaluated against hard/rounded ones, which "hurts rate-distortion
performance since the latent representation ability is weakened." Their fix, directly applicable
here: train with the soft (noised) relaxation early, then transition to hard quantization later in
training to close the gap before the finishing calibration pass. Worth building into the schedule
from the start rather than discovering the mismatch empirically the way M8 discovered the
best-checkpoint/final-checkpoint gap.

**Sources:** [Ballé et al. 2017](https://www.cns.nyu.edu/pub/lcv/balle17a-submitted-revised.pdf) ·
[Guo et al., *Soft then Hard*, ICML 2021](https://arxiv.org/abs/2104.05168) ·
[*A Differentiable Entropy Model for Learned Image Compression*](https://link.springer.com/chapter/10.1007/978-3-031-43148-7_28)

---

## B. Temporal / inter-frame coding

Every RD benchmark this project has run flags the same structural gap: NVC is intra-only,
H.264/H.265 exploit temporal redundancy. Two real design families exist, and they trade off very
differently against this project's actual constraints (593K params, CPU-first, and now a hardware
accelerator design already committed to a *fixed, small* datapath):

**Classic approach (DVC-style, [Lu et al. 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rippel_Learned_Video_Compression_ICCV_2019_paper.pdf),
FVC):** an explicit motion-estimation subnetwork (usually optical flow) predicts a dense per-pixel
motion field, warps the reference frame/features, then codes the residual. Effective, but adds a
whole second network (flow estimation) plus a dense warping operation with gather-style memory
access — expensive relative to this project's current 593K-param budget, and a poor match for
NVC-ACCEL's fixed, small PE array (§ARCHITECTURE.md §6), which has no warping/interpolation unit.

**Motion-free / implicit approach — the better fit for this codebase**, exemplified by
[**DCVC-RT** (Jia & Li, CVPR 2025)](https://arxiv.org/abs/2502.20762): discards the entire motion
branch — no motion estimation, no motion compression, no warping. Temporal context comes from a
**lightweight depth-wise-convolution feature extractor operating on a cached reference feature**,
concatenated with the current latent along the channel dimension before the encoder/decoder
process them jointly. Reported: 21% BD-rate savings vs. H.266/VTM at 125/113 fps (1080p). This
maps almost directly onto this project's existing shape: the "cached reference feature" DCVC-RT
uses is exactly the kind of thing this codec already produces every frame — its own 64×16×16
latent. A small depth-wise-conv block consuming *the previous frame's latent* alongside the current
one, before quantization, is a bounded architectural addition (a handful of extra depth-wise
conv layers, not a second network) rather than a redesign.

DCVC-RT also reports **model integerization to INT16** for deterministic cross-platform
inference — worth cross-referencing against this project's own §C finding above (INT8 alone cost
~1 dB; DCVC-RT's own team apparently found INT8 insufficient and went INT16) as independent
evidence for the INT16-over-INT8 fallback already suggested in §C.

**A middle-ground alternative worth knowing about, if real motion vectors are ever wanted (e.g.
because NVC-ACCEL grows a hardware warping unit later):** [**MobileNVC**
(WACV 2024)](https://arxiv.org/abs/2310.01258) uses **block-based** motion compensation running on
a mobile accelerator's existing warping core (not dense per-pixel flow), reporting up to 48%
BD-rate savings and a 10× MAC reduction on the decode side vs. prior on-device codecs. This is a
better match for classical-codec-style block motion vectors than DVC/FVC's dense flow fields, and
notably pairs a *neural* codec with the exact kind of fixed-function hardware unit NVC-ACCEL is
already being designed as. Not the recommended starting point (it assumes hardware that doesn't
exist yet), but the natural pairing if NVC-ACCEL grows a warping block in a later hardware
iteration.

**Recommendation**: DCVC-RT's motion-free, depth-wise-conv, cached-latent approach is the right
template for this project's actual scale and hardware trajectory — smaller parameter/compute
addition than DVC-style motion estimation, and (unlike MobileNVC) doesn't presuppose hardware this
project hasn't built yet.

**Sources:** [Lu et al., *DVC*, CVPR 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rippel_Learned_Video_Compression_ICCV_2019_paper.pdf) ·
[Jia & Li, *Towards Practical Real-Time Neural Video Compression* (DCVC-RT), CVPR 2025](https://arxiv.org/abs/2502.20762) ·
[*MobileNVC*, WACV 2024](https://arxiv.org/abs/2310.01258) · [microsoft/DCVC](https://github.com/microsoft/DCVC)

---

## What I did not do

No code changes were made to `src/nvc/` or any training script based on A or B — these are
research findings and a concrete proposal, not an implementation. C is the one thread that
produced a real measurement, using an already-built validation script; it changed no checkpoints,
calibrations, or existing files (see [hardware/int8_activation_validation.py](hardware/int8_activation_validation.py),
newly added, not a modification of anything). No claim here should be read as "already
implemented" beyond what's explicitly marked [MEASURED].
