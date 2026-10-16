# Stage 1, 64-channel run — the intra deficit halves

Phase A (distortion-only) on Colab, 2026-10-15, `eb205c5` config. Ten chunks,
40 epochs, **zero NaN**. `ResidualGDNAutoencoder`, 7,208,899 parameters,
latent `(1, 64, 16, 16)` — exact symbol parity with the deployed 593k model.

## The gate

`scripts/benchmark_intra_gate.py`, full DAVIS TEST (9 sequences, 719 frames),
`decode-exact=True` at every point, `run valid: True`.

| convention | deployed codec | Stage 1 @192ch | **Stage 1 @64ch** |
|---|---|---|---|
| sequence_mean/psnr | +176.2% | +560.6% | **+87.9%** |
| sequence_mean/msssim | +167.6% | +456.1% | **+110.1%** |
| frame_mean/psnr | +164.2% | +534.1% | **+79.6%** |
| frame_mean/msssim | +150.3% | +430.2% | **+97.4%** |
| sequence_pooled/psnr | +164.0% | +537.0% | **+81.0%** |
| sequence_pooled/msssim | +167.6% | +456.1% | **+110.1%** |
| global_pooled/psnr | +127.5% | +476.8% | **+62.3%** |
| global_pooled/msssim | +150.3% | +430.2% | **+97.4%** |

**The PSNR BD-rate deficit against x264 all-intra roughly halves on every
convention** (−50.1%, −51.5%, −50.6%, −51.1%). MS-SSIM improves by about a third
(−34%, −35%). Every convention moves the same way, so this is not an artifact of
which one is quoted.

MS-SSIM gaining less than PSNR is expected and worth stating: Phase A's objective
is plain MSE, which is PSNR by construction. Nothing in this run optimized
perceptual quality.

## Why, in one table

| bits | Stage 1 bytes | deployed bytes | Δ bytes | Stage 1 PSNR | deployed PSNR | gain |
|---|---|---|---|---|---|---|
| 5 | 5,723,946 | 5,628,186 | **+1.7%** | 31.973 | 29.601 | **+2.37 dB** |
| 4 | 4,213,490 | 4,130,011 | **+2.0%** | 30.567 | 28.692 | **+1.88 dB** |
| 3 | 2,692,210 | 2,638,574 | **+2.0%** | 27.013 | 26.228 | **+0.79 dB** |

Essentially the same bitstream size — within 2% — for up to **+2.37 dB**. That is
exactly what symbol parity was supposed to buy, and it is the direct contrast with
the 192-channel run, which paid 3.2x the bytes for +0.43 dB at the float ceiling.

The gain shrinks as the quantizer coarsens (+2.37 → +1.88 → +0.79 dB). The extra
capacity is worth most where there are bits to spend, which is the opposite of the
192-channel run's pattern and is consistent with capacity now being spent on
representation rather than on paying for its own width.

## Training

| | run 1 (192ch) | run 2 (64ch) |
|---|---|---|
| epochs | 22 | **40** |
| NaN epochs | **18** | **0** |
| best val loss | 0.0004595 | **0.0002246** |
| best val PSNR | 34.91 dB | **38.15 dB** |
| best at | epoch 2, chunk 1 | epoch 28, chunk 7 |

The gradient clipping added in `8d46bf5` held for all 40 epochs — no divergence,
no NaN, nothing aborted. Validation loss is **2.05x lower** and **+3.24 dB**.

Per-chunk best validation loss:

| chunk | 1 | 2 | 3 | 4 | 5 | 6 | **7** | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| val loss (e-4) | 4.38 | 2.67 | 3.15 | 2.91 | 2.53 | 2.34 | **2.25** | 2.55 | 2.41 | 2.49 |
| PSNR (dB) | 35.22 | 37.25 | 36.59 | 36.81 | 37.83 | 37.79 | **38.15** | 37.63 | 37.92 | 37.65 |

**Chunks 8-10 did not improve on chunk 7** — roughly 27,000 sequences and a
quarter of the run's wall-clock for nothing, including both decayed-LR chunks.
Global-best tracking kept epoch 28's weights, so nothing was lost, but the next
run should treat this as evidence that either the data saturates around chunk 7 or
the final decay arrives too late to help. Worth a cheap experiment before paying
for ten chunks again.

## Where this leaves Stage 1

Halving the deficit is the first real architectural win in the project — M13
through M22 moved coded rate by 1-4% each, and this is 88 BD-rate points. But it
is **+87.9%, not parity**: x264 all-intra still codes equal quality in about
half the bits, and this is an *intra-only* number.

Still unspent, roughly in order of expected value:

1. **Phase B (`D + lambda*R`)**, which has still never run. Note the caveat from
   `outputs/stage1_latent_shape/README.md`: per-channel scale normalization means
   quiet channels keep paying ~68% of full depth, so rate pressure has less to
   cash in here than it would in a design with a learned prior. Expect a real but
   modest gain, not another halving.
2. **Per-channel bit allocation**, currently a dormant no-op worth ~7% of bytes
   for ~0.06 dB (same source). Cheap, and it helps the baseline too.
3. **`RESIDUAL_BLOCKS = 2`** — 9,864,643 parameters at *identical* symbol count,
   so free in rate. The obvious next capacity increment now that width is settled.
4. **Stage 2's hyperprior**, which is what would make conditional pricing possible
   and is the only item here that could plausibly deliver another step change.

A full-video number still needs the G16 context model, codebooks and motion table
refitted to this transform — all P-frame machinery that a new transform
invalidates. Not needed for this gate; worth scoping now that the gate has cleared.

## Files

- `best.pt` — epoch 28, 7,208,899 params, all finite
- `history.json` — all 40 epochs
- `gate_stage1_64ch` — gate report (JSON)
- `gate.log` — the run
