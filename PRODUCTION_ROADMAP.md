# From Minor Project to Adopted Technology: A Production Roadmap

This document picks up where `README.md`'s 12-week academic roadmap leaves off. It's organized in four parts, roughly in order of both difficulty and time horizon:

1. **[Where the project actually stands today](#1-where-the-project-actually-stands-today)** — an honest baseline, so everything below is calibrated against reality, not aspiration.
2. **[Closing the codec's own gaps](#2-closing-the-codecs-own-gaps)** — the technical work already implied by this project's own scope that isn't done yet.
3. **[Making it publicly usable](#3-making-it-publicly-usable)** — a genuinely achievable near/medium-term goal: something a stranger could install and use without you standing over their shoulder.
4. **[Optimization techniques](#4-optimization-techniques)** — concrete, specific techniques, organized by where the bottleneck actually is.
5. **[The much longer road: becoming an adopted technology](#5-the-much-longer-road-becoming-an-adopted-technology)** — what "on the way to replacing H.264/H.265" actually requires, stated plainly rather than optimistically.

Read section 5 before getting attached to the framing in this document's title. It's there on purpose.

---

## 1. Where the project actually stands today

Being precise about this matters — everything that follows is calibrated against it, not against what the project could plausibly become.

**What exists and is measured, real, and verified:**
- A deterministic (non-variational) convolutional autoencoder, 593,411 parameters — **not** a VAE, no KL term, trained with plain MSE.
- Post-hoc uniform affine quantization (global and per-channel modes), calibrated from a training split via percentile clipping — **not** quantization-aware training; the network has never seen quantized latents during training.
- A static, per-channel empirical entropy model (order-0, no learned/context/hyperprior model) plus a from-scratch integer arithmetic coder.
- A fully specified `.nvc` binary container format with header validation.
- Real, measured results on DAVIS's held-out test split: **26.6–27.2 dB PSNR at 12x–28x compression versus raw uint8 RGB**, depending on bit depth. Entropy coding verified exactly lossless; the coder itself runs within ~0.03% of its own theoretical optimum.
- A cross-dataset generalization result: a checkpoint trained only on Vimeo-90K loses only ~0.3–0.5 dB versus a checkpoint trained directly on DAVIS, when both are tested on DAVIS (see `results/test1_2026-08-19.md`).
- Two Colab notebooks that automate chunked large-dataset training and dataset-swappable evaluation, with Drive-backed checkpointing.
- 259 automated tests across the data pipeline, model, training, and compression code.

**What this is not, as of today:**
- **Not a video codec.** It compresses individual frames independently. No motion compensation, no inter-frame prediction, no temporal model of any kind. Every frame pays the full "I-frame" cost — the single biggest structural difference from H.264/H.265.
- **Not real-time.** ~40–100ms per frame to encode or decode, in pure Python, on a still image. H.264/H.265 hardware decoders run this many times faster, in silicon, at a tiny fraction of the power draw.
- **Not benchmarked against H.264/H.265 at all.** Every number so far is against raw uint8 RGB or against an earlier version of itself. There is currently no evidence this codec beats, matches, or even comes close to a real video codec at a comparable bitrate.
- **Not quality-aware in the way that matters for video.** Only MSE/PSNR are implemented; no MS-SSIM, no perceptual/VMAF-style metric — the metrics that actually predict how a human perceives quality, and the metrics every real codec comparison in this space is judged on.

Keep this list next to any claim made about this project going forward.

---

## 2. Closing the codec's own gaps

These are the technical items your own README already lists as "not implemented yet" — the honest next layer of the *existing* scope, before anything about public usability or adoption makes sense.

### 2.1 Turn it into an actual video codec
This is the single most important item on this list. Right now, "video compression" means "run the image codec once per frame." Real gains come from:
- **Motion-compensated / predictive coding**: predict this frame's latent from the previous frame's (optical flow, learned warping, or a simple frame-differencing baseline to start), and only encode the residual.
- **GOP structure**: periodic full-quality "I-frames" with predicted "P-frames" (and eventually bidirectional "B-frames") between them, mirroring how every mainstream codec amortizes cost across a group of pictures.
- Start with the simplest possible version (frame-differencing + reusing the existing quantizer/entropy coder on the residual) before reaching for learned optical flow — get a real, measured temporal-compression number before building anything more elaborate.

### 2.2 Learned / context entropy models
Your own Milestone 6 analysis already identifies this as "the largest remaining win": the current entropy model is static and order-0 per channel, so it captures no spatial correlation between neighboring latent positions. A hyperprior or small autoregressive context model (à la Ballé et al., Minnen et al.) would close a meaningful chunk of the gap between "fixed width" and the true achievable entropy — and this is well-trodden research ground, not a novel risk.

### 2.3 Quantization-aware training
The network currently trains on float latents and only meets quantization at inference. Folding a straight-through estimator (or a soft/differentiable quantization relaxation) into training — so the model actually learns to be robust to its own quantization grid — is standard practice and should recover meaningful PSNR, especially at low bit depths where the current 4-bit numbers show the most degradation.

### 2.4 A rate-distortion loss
Training currently optimizes MSE alone; nothing in the loss function knows or cares about bitrate. A proper `distortion + λ · rate` objective (estimated bitrate from the entropy model, differentiable enough to backprop through) is what actually lets you *choose* an operating point on the quality/size curve at training time, rather than only after the fact via bit-depth selection.

### 2.5 Perceptual/quality metrics that matter
Implement MS-SSIM at minimum; ideally also LPIPS or VMAF. PSNR is what you have today, and it's a poor proxy for perceived quality, especially once you're comparing against H.264/H.265 — every serious codec comparison in this field reports MS-SSIM/VMAF because PSNR can favor blur over the kind of quality loss that actually bothers viewers.

### 2.6 The real H.264/H.265 comparison
Encode the *same* test sequences with FFmpeg at matched bitrates (`libx264`/`libx265`, several CRF/bitrate targets) and plot rate-distortion curves side by side. Until this exists, there is no evidence-based claim to make about how this codec compares to what it's meant to eventually replace — this should happen before any of the "adoption" work in Section 5 is worth pursuing further.

### 2.7 Full-scale training
The current checkpoint was trained on 10 chunks of Vimeo-90K via Colab, not the complete ~91,701-sequence dataset, and not with quantization or a rate-distortion term in the loop. Once 2.3 and 2.4 exist, a full training run (ideally on rented GPU time with persistent storage — see the earlier discussion on Vast.ai/RunPod for exactly why) is what actually establishes a ceiling for this architecture.

---

## 3. Making it publicly usable

This is the most tractable, valuable near-term goal, and it's a genuinely different (and much more achievable) target than Section 5. "Publicly usable" means: a stranger who has never spoken to you can install it, encode an image, and get a sensible result or a clear error — with no manual setup, no tribal knowledge, no risk to their machine.

**Packaging and API**
- Ship it as a real installable package (`pip install nvc` or similar) with a clean, documented Python API — `encode(image) -> bytes`, `decode(bytes) -> image` — separate from the CLI scripts, which are currently the only interface.
- Pin dependency versions properly and verify the package installs cleanly on a fresh machine (not just your dev environment).
- A license file. There currently isn't one — this blocks any legitimate public/open-source use, and matters even more given the Vimeo-90K Kaggle mirror used in the training notebook has an unclear license itself (flag that separately if any derived checkpoint is ever redistributed).

**Distribution of trained weights**
- Host checkpoints somewhere a stranger can actually get them (Hugging Face Hub is the natural fit for this) rather than requiring everyone to retrain from scratch on Colab. Retraining should be an option, not a prerequisite.
- Version checkpoints against the exact architecture/calibration they pair with — your `.nvc` header's `entropy_model_id` mismatch-detection is a good pattern; extend that discipline to checkpoint distribution too.

**Robustness for untrusted input**
- Right now the codec has been tested against well-behaved data from your own pipeline. A public tool needs to gracefully handle: malformed/corrupted `.nvc` files, images of arbitrary sizes not divisible by 16, non-RGB images, truncated files, and adversarially crafted headers. `NVCFormatError` already exists for a lot of this — audit that every failure mode raises it cleanly rather than crashing or (worse) silently producing garbage.
- If this is ever exposed as a network service (an upload-and-decode demo, for instance), think about resource exhaustion explicitly: file size limits, decode timeouts, memory bounds on the header's declared dimensions before you allocate based on them.

**Documentation and demo**
- A real docs site (even just a well-organized GitHub Pages site generated from docstrings) beyond the current README, once the API stabilizes.
- A hosted, no-install demo — a small web app or a one-click Colab notebook — where someone can upload an image and see the compressed result and quality numbers without cloning the repo. This is the single highest-leverage thing for getting anyone outside your team to actually try it.

**Engineering hygiene**
- CI (GitHub Actions) running the 259 tests on every push/PR, on Linux at minimum — right now correctness is verified locally and on Colab, not automatically on every change.
- Cross-platform verification: this project's dev history is Windows + Colab; confirm it actually works cleanly on Linux and macOS too, since that's most of your eventual user base.

---

## 4. Optimization techniques

Organized by where the actual bottleneck is — don't optimize the wrong layer.

### 4.1 The network itself (compute per frame)
- **Efficient architectures**: depthwise-separable convolutions, MobileNet/EfficientNet-style blocks in the encoder/decoder to cut FLOPs per frame without a proportional quality loss.
- **Network quantization** (a different axis from the *latent* quantization you've already built): INT8 weights/activations for the encoder/decoder itself, via PyTorch's quantization toolkit or ONNX Runtime's quantization tools — this is a distinct, additional win on top of what's already implemented.
- **Pruning**: remove channels/filters that contribute little, especially once you know which parts of the network matter most for the loss you actually care about (rate-distortion, once 2.4 exists).
- **Knowledge distillation**: train a small, fast "student" network to match a larger, higher-quality "teacher" — a standard way to get most of the quality at a fraction of the compute.

### 4.2 The entropy coder (currently your biggest measured bottleneck)
Your own benchmark already quantifies this: ~40–100ms/frame in pure Python is explicitly flagged as "not real-time." This is the most concrete, well-scoped optimization on this whole list, because you already know exactly where the time goes:
- **Reimplement the arithmetic coder's inner loop in C/Rust/Cython** — the algorithm is already fully specified and tested in Python; porting the hot loop (not the whole codebase) is a contained, low-risk project with a large expected payoff.
- **Batched/vectorized range coding** — research prior art here (e.g. `torchac`, used in several learned-compression codebases) shows this can move entropy coding onto the GPU and process a batch of symbols in parallel rather than one at a time.

### 4.3 Inference systems
- **Export to ONNX / TensorRT** for the encoder and decoder graphs — meaningful latency wins from graph optimization and fused kernels, independent of any architecture change.
- **Mixed precision (FP16/BF16)** inference on GPU.
- **Batch encoding** where the use case allows it (offline transcoding, not live streaming).
- **Hardware accelerators**: mobile NPUs, Apple Neural Engine, Qualcomm Hexagon, or Tensor Cores — but see Section 5 on why this only gets you so far against dedicated video-decode ASICs.

### 4.4 Video-specific (once Section 2.1 exists)
- Motion-compensated latent prediction is itself the biggest *optimization*, not just a feature — it's how real codecs avoid paying the full frame cost 24-60 times a second.
- GOP-level parallelism: I-frames can be encoded/decoded independently of each other, which is a natural unit for parallelization across frames.

---

## 5. The much longer road: becoming an adopted technology

Read this section before treating "replace H.264/H.265" as a near-term or medium-term goal. It isn't one, for reasons that have nothing to do with the quality of the code — the same reasons apply to every neural codec research effort, including ones backed by large industrial labs.

**Why this is a much bigger undertaking than the codec itself:**

- **H.264/H.265 aren't just algorithms — they're standards with silicon behind them.** Every phone, laptop, TV, browser, and streaming device sold in the last decade has a dedicated hardware decoder ASIC for these formats. That hardware decodes video at a tiny fraction of the power draw of running a neural network on a GPU or NPU. A software neural decoder competing against a purpose-built decode chip is not a fair fight on power or latency, and won't be until neural codecs get their own silicon — which is a hardware industry decision, not something a codebase change achieves.
- **Standardization is its own multi-year, multi-organization process.** H.264/H.265/VVC/AV1 came out of ITU-T/MPEG (and, for AV1, the Alliance for Open Media) — bodies made up of dozens of companies negotiating over years. "Becoming adopted" in the way H.265 is adopted means going through something like that process, not shipping a better encoder.
- **Patents are a real, non-trivial risk in this exact space.** H.264/H.265 famously carry heavy patent licensing (MPEG-LA, Access Advance). Neural compression is not a patent-free escape from this — Google, Qualcomm, InterDigital, Disney, and others are actively filing and holding patents on learned compression techniques. Any path toward real commercial adoption needs a patent landscape review, not an assumption that "neural" means "unencumbered."
- **The competitive field is not empty.** JPEG AI (a learned still-image compression standard) has already been finalized by the JPEG committee. MPEG has active exploration into neural video coding. Google, NVIDIA, and academic labs publish in this space continuously (the CLIC — Challenge on Learned Image Compression — is a running venue for exactly this comparison). "Adoption" competes against well-funded, well-published existing efforts, not a blank field.
- **PSNR-only results don't survive contact with this field.** As soon as any real comparison happens, it will be judged on MS-SSIM/VMAF and rate-distortion curves against H.265/AV1 at matched bitrates — not raw-RGB compression ratios. Section 2.5 and 2.6 are prerequisites for even being a credible participant in this conversation.

**What a realistic path actually looks like:**

1. **Finish Section 2 first.** Without a real H.264/H.265 comparison on a proper perceptual metric, there is nothing yet to make an adoption case *for*.
2. **Aim at a niche before a universal replacement.** Neural codecs already show genuine promise in specific regimes: very-low-bitrate video conferencing, satellite/drone footage where storage cost dominates over decode speed, surveillance archives, or medical imaging. Winning one of these narrow, well-defined battles is a real, achievable goal; "replace H.265 for general-purpose video" is not a goal you aim at directly.
3. **Publish.** Treat this as a research contribution, not a product launch. A well-documented, honestly-benchmarked writeup (exactly the kind of rigor already present in this project's README) submitted somewhere the learned-compression community will actually see it is how a project like this gets real feedback and, eventually, real credibility.
4. **Open-source it properly** (Section 3) and let the actual research and open-source communities decide if it's worth building on — adoption, in the sense that matters, is something that happens *to* a project from outside, not something a roadmap document can schedule.
5. **Treat hardware acceleration as a hard dependency, not a nice-to-have**, for any claim about eventually competing with H.264/H.265 on real devices. Software-only neural decoding stays a research/offline tool until that changes.

None of this means the ambition is wrong — genuine, credible neural video compression research is exactly this kind of long game, and every real advance in the field (including the ones now standardized as JPEG AI) started as somebody's much smaller proof-of-concept. It means the honest next milestone is "a rigorously benchmarked, publicly usable research artifact people can build on," not "H.265's replacement" — and that the former is a real, achievable, valuable thing to aim for on its own terms.
