# Stage 1, first Vimeo run — diverged, and what the salvaged checkpoint measures

Phase A (distortion-only) on Colab, 2026-10-12. Ten chunks requested.
**The run diverged at epoch 3 and the architecture has not had a fair trial yet.**

## What happened

| epoch | chunk | train | val | val PSNR |
|---|---|---|---|---|
| 1 | 1 | 0.002608 | 0.001295 | 29.73 dB |
| **2** | **1** | **0.000661** | **0.000459** | **34.91 dB**  ← best.pt |
| 3 | 1 | 0.102310 | 0.217292 | 7.18 dB |
| 4 | 1 | 0.464133 | 0.461670 | 3.55 dB |
| 5–22 | 2–10 | NaN | NaN | NaN |

Eighteen of twenty-two epochs were NaN. `progress.json` nevertheless recorded all
ten chunks as `completed` and `latest.pt` has NaN in all 44 tensors, so a plain
resume would have skipped every chunk and restarted from NaN.

`best.pt` (epoch 2) is intact — all 8,437,827 parameters finite — and is the only
usable output. It saw **one chunk of ten, for two epochs**: roughly 9,000 of the
dataset's 91,701 sequences.

Root cause: **no gradient clipping.** GDN normalizes by
`sqrt(beta + sum_j gamma_ij x_j^2)`, whose gradient is unbounded as the
denominator approaches the `LowerBound` floor, so a single bad batch can take the
transform out. Fixed in `scripts/train_vimeo_stage1.py` (`--grad-clip-norm`,
default 1.0) with a divergence abort and a NaN-resume guard; regression tests in
`tests/test_scripts_train_vimeo_stage1.py`.

## The gate on the salvaged checkpoint

`scripts/benchmark_intra_gate.py`, full DAVIS TEST, 9 sequences / 719 frames,
`decode-exact=True` at every point.

| bit depth | Stage 1 bytes | bpp | PSNR | MS-SSIM | deployed bytes | PSNR | ratio |
|---|---|---|---|---|---|---|---|
| 5 | 17,719,355 | 3.0084 | 29.991 | 0.9753 | 5,628,186 | 29.601 | **3.15×** |
| 4 | 13,175,843 | 2.2370 | 29.502 | 0.9665 | 4,130,011 | 28.692 | **3.19×** |
| 3 | 8,567,270 | 1.4545 | 27.940 | 0.9373 | 2,638,574 | 26.228 | **3.25×** |

**BD-rate vs `h264_intra`: +560.6% PSNR / +456.1% MS-SSIM**, against the deployed
codec's +176.2% / +167.6% (`outputs/benchmarks/parity_intra/`). Worse, not better.

## Reading it

The transform reconstructs **better** at every operating point — +0.39, +0.81 and
+1.71 dB — and the margin widens as the quantizer coarsens, which is where extra
capacity should help most. So the architecture is doing its job in distortion.

It loses on rate, by a factor that is not a mystery:

```
BaselineAutoencoder      latent (1, 64, 16, 16)   514,560 symbols/frame @1080p
ResidualGDNAutoencoder   latent (1, 192, 16, 16) 1,543,680 symbols/frame @1080p
```

Same stride 16, three times the channels, **exactly three times the symbols** —
and the measured byte ratio is 3.15–3.25×. The rate penalty is the channel count,
almost to the digit.

Phase A's objective is plain MSE. Nothing in it asks the transform to make those
192 channels cheap to code, so it spreads information across all of them, which is
what minimizes MSE. Phase B (`D + lambda*R`, lambda = 3e-4) is the step that
applies rate pressure and would concentrate energy into few channels; **it has
never run.** `RUN_PHASE_B` is `False` in the notebook by design, and this run died
before it mattered.

So +560.6% is the expected reading for a distortion-only model scored on a rate
metric, not a verdict on the architecture. What it does establish, and what the
roadmap did not account for, is the size of the hole: **Phase B has to win back
3.2× in rate before Stage 1 breaks even with the model it is replacing**, and then
beat it on top of that. That is a much harder ask than "train the transform".

Worth noting separately: this project's intra entropy stack prices the latent
without a learned cross-channel prior, so 192 channels cost roughly 3× of
whatever 64 cost. Reference designs at 192 channels pair them with a hyperprior
that exploits exactly that redundancy. Either Phase B produces real sparsity, or
the latent shape — not the transform depth — is the thing to change.

## Files

- `best.pt` — epoch 2, finite, the only usable checkpoint
- `history.json` — all 22 epochs, divergence included
- `gate_stage1_colab_run1` — gate report (JSON)
- `gate_full.log` — the run

`latest.pt` is not kept here; it was NaN in every tensor.
