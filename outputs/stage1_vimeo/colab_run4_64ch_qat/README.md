# QAT fine-tune — the intra deficit halves again, +87.9% → +46.5%

Quantization-aware fine-tune of the 64-channel transform, Colab, 2026-10-24,
`e660405` config. Warm-started from the +87.9% checkpoint via `--init-from`,
4-bit per-channel noise, 7 chunks, 27 epochs, **zero NaN**.
`ResidualGDNAutoencoder`, 7,208,899 parameters, latent `(1, 64, 16, 16)` —
unchanged, so this costs nothing in rate.

## The gate

| convention | deployed 593k | 64ch pre-QAT | **64ch + QAT** |
|---|---|---|---|
| sequence_mean/psnr | +176.2% | +87.9% | **+46.5%** |
| sequence_mean/msssim | +167.6% | +110.1% | **+55.8%** |
| frame_mean/psnr | +164.2% | +79.6% | **+41.8%** |
| frame_mean/msssim | +150.3% | +97.4% | **+48.6%** |
| sequence_pooled/psnr | +164.0% | +81.0% | **+42.6%** |
| global_pooled/psnr | +127.5% | +62.3% | **+35.5%** |

Improves on **all eight** conventions. Against the deployed codec's +176.2%, the
intra deficit is now **about a quarter of what it was**. Full DAVIS TEST,
`decode-exact=True` at all three points, `run valid: True`.

## Where the gain came from — and it is not all QAT

At matched bit depth the stream is the same size (+0.4%, +0.5%, +0.7%) for
**+0.86, +0.95 and +1.16 dB** at 5/4/3-bit. But this run also trained for 27
more epochs on top of the warm start, so the result is QAT *plus* more training.
The float-ceiling measurement separates them:

| | float ceiling | loss at 5-bit | 4-bit | 3-bit |
|---|---|---|---|---|
| deployed 593k | 30.139 dB | −0.831 | −1.699 | −4.072 |
| 64ch pre-QAT | 32.589 dB | −1.010 | −2.327 | −5.782 |
| **64ch + QAT** | **33.092 dB** | **−0.678** | **−1.936** | **−5.149** |

So the gain decomposes cleanly:

| bits | ceiling rose | fragility fell | sum | measured |
|---|---|---|---|---|
| 5 | +0.503 | +0.332 | +0.835 | +0.86 |
| 4 | +0.503 | +0.391 | +0.894 | +0.95 |
| 3 | +0.503 | +0.633 | +1.136 | +1.16 |

**The decomposition closes to within 0.06 dB at every point.** Roughly 58% of the
5-bit gain is the extra training and 42% is QAT; at 3-bit it is 43% training and
57% QAT. QAT's share grows as the quantizer coarsens — exactly the signature the
fragility diagnosis predicted
(`outputs/stage1_latent_shape/README_64ch_trained.md`), and the first prediction
in this project to be confirmed by a subsequent run rather than retrofitted.

Stated honestly: **this run does not isolate QAT.** A no-QAT control trained for
the same 27 further epochs would, and was not run. What is isolated is the
fragility component, via the ceiling measurement above, and it is +0.33/+0.39/
+0.63 dB.

## QAT has more to give

The transform still loses **5.149 dB** to 3-bit quantization against the
baseline's 4.072 — an excess of **1.08 dB**, down from 1.71 but not closed. QAT
recovered 37% of the original excess. A longer or lower-LR QAT schedule, or QAT
at 3-bit rather than 4-bit noise, are the obvious things to try.

## A side effect worth noting

Per-channel bit allocation, worth **0.00%** on the pre-QAT model (zero idle
channels), is now worth **−3.34% of bytes for −0.11 dB**: QAT shifted the energy
distribution enough that 6 of 64 channels drop to 3 bits. Small, real, and free —
it needs no training, only wiring `bits_per_channel` into the intra path.

## Training

| | pre-QAT | QAT |
|---|---|---|
| epochs | 40 (10 chunks) | 27 (7 chunks) |
| NaN | 0 | 0 |
| best val loss | 0.0002246 | **0.0001856** |
| best val PSNR | 38.15 dB | **38.95 dB** |
| best at | epoch 28, chunk 7 | epoch 23, chunk 6 |

Epoch 1 opened at 0.0003005 / 36.80 dB, close to the pre-QAT level rather than
from scratch, confirming the warm start loaded. Validation is measured in
`eval()` mode, where the noise is not applied, so these numbers are comparable
across both runs. Chunk 7 again produced the higher PSNR (39.07 dB) while chunk 6
produced the lower loss — the same cross-chunk validation-split artefact noted
in `../colab_run3_64ch_2blk/README.md`.

## Files

- `best.pt` — epoch 23, 7,208,899 params, all finite, QAT fingerprint in `extra`
- `epoch27.pt` — epoch 27, chunk 7, higher val PSNR. **Not gated**
- `history.json`, `gate_qat_epoch23`, `gate_epoch23.log`

## What this does not tell us

Every number here is **intra-only**. The G16 context model, the codebooks and the
motion table are all fitted to the deployed 64-channel latent of the *old*
transform and are refused by the provenance check against this one, so the
full-video consequence is still unmeasured. That is now the largest open question
in the project: the intra deficit has gone from +176.2% to +46.5%, and nobody
knows what that is worth against the +450.7% full-video figure the project is
actually judged on.
