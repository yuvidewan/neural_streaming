# Stage 1, 2 residual blocks — regressed. More capacity is not the answer.

Phase A on Colab, 2026-10-18, `8b5baf6` config. Seven chunks, 28 epochs,
**zero NaN**. `ResidualGDNAutoencoder`, **9,864,643 parameters** (vs 7,208,899),
latent `(1, 64, 16, 16)` — identical symbol count, so this was free in rate.

**Status: complete. Both checkpoints gated, 2026-10-19. 2 blocks is rejected.**

## The gate, so far

| | 1 block (`colab_run2_64ch`) | 2 blocks, epoch 23 | 2 blocks, epoch 28 |
|---|---|---|---|
| parameters | 7,208,899 | 9,864,643 | 9,864,643 |
| BD-rate vs `h264_intra` (sequence_mean/psnr) | **+87.9%** | +102.8% | **+101.5%** |

**Worse by ~14 BD-rate points on the better of the two.** Not neutral — a
regression, and both checkpoints agree.

Epoch 28 had 0.16 dB more Vimeo val PSNR than epoch 23 and that bought 1.3
BD-rate points — real, consistent with the val-split wrinkle below, and nowhere
near the 14-point gap to the 1-block model.

| bits | 2-block bytes | PSNR | 1-block bytes | PSNR |
|---|---|---|---|---|
| 5 | 5,593,408 | 31.642 | 5,723,946 | 31.973 |
| 4 | 4,086,540 | 30.124 | 4,213,490 | 30.567 |
| 3 | 2,572,231 | 26.527 | 2,692,210 | 27.013 |

The 2-block stream is consistently *cheaper* (−2.3%, −3.0%, −4.5%) and
consistently *lower quality* (−0.33, −0.44, −0.49 dB). Net: worse BD-rate at
every point.

## It trained better and generalized worse

| | 1 block | 2 blocks |
|---|---|---|
| best val loss | 0.0002246 | **0.0001886** |
| best val PSNR | 38.15 dB | **39.03 dB** (chunk 7) |
| DAVIS gate | **+87.9%** | +102.8% |

Validation loss on Vimeo improved by 1.19x and val PSNR by nearly 0.9 dB, while
DAVIS BD-rate got worse. That is a **generalization gap**, not a training
failure: the extra capacity fit Vimeo better and transferred less. Worth
remembering before reaching for capacity again.

It also confirms the diagnostic's reading
(`outputs/stage1_latent_shape/README_64ch_trained.md`): quantization, not
capacity, is the binding constraint. The prediction there was that blocks=2 would
widen the high-rate advantage and not fix the low-rate end. It did neither.

## The chunk-7 question, answered

The 1-block run flattened after chunk 7, so this run used seven chunks to test
whether a bigger model keeps improving. It does:

| chunk | 5 | 6 | 7 |
|---|---|---|---|
| best val loss (e-4) | 2.13 | **1.89** | 1.92 |
| best val PSNR (dB) | 38.66 | 38.88 | **39.03** |

PSNR was still rising at chunk 7, so the data had **not** saturated for this
model — a ten-chunk run would have gone further on Vimeo. Given the
generalization gap above, that is not obviously worth buying.

### A methodological wrinkle found here

`best_val_loss` is tracked globally across chunks, but **each chunk has its own
validation split**, so those numbers are measured on different data and are not
strictly comparable. It shows up directly in this run: epoch 23 (chunk 6) wins on
val loss at 0.0001886 while epoch 28 (chunk 7) wins on val PSNR at 39.033 dB.
`best.pt` therefore holds epoch 23 for a reason that is partly which chunk's val
set was easier. Worth fixing — a fixed held-out split would make
best-checkpoint selection mean what it claims.

## Tomorrow

1. **Gate `epoch28.pt`** (chunk 7's last epoch, staged here alongside `best.pt`).
   It has 0.16 dB more val PSNR and may gate better; DAVIS is the tiebreaker
   between the two, since their val splits differ. Command:
   ```
   python scripts/benchmark_intra_gate.py \
       --checkpoint outputs/stage1_vimeo/colab_run3_64ch_2blk/epoch28.pt \
       --output-dir outputs/stage1_vimeo/colab_run3_64ch_2blk \
       --output-name gate_2blk_epoch28
   ```
2. **QAT from the 1-block checkpoint**, unless epoch 28 beats +87.9%. Wired and
   ready in `9750238`; the calibration is committed at
   `outputs/stage1_vimeo/colab_run2_64ch/calibration/stage1_64ch_4bit_train.json`.
   This is the lead aimed at the +1.710 dB the transform loses to 3-bit
   quantization, which is where the BD-rate is weakest.
3. **Revert the notebook to `RESIDUAL_BLOCKS = 1`** for any further Phase A work.
   2 blocks is measured and rejected.

## Files

- `best.pt` — epoch 23, chunk 6, lowest val loss. Gated: **+102.8%**
- `epoch28.pt` — epoch 28, chunk 7, highest val PSNR. Gated: **+101.5%**
- `history.json` — all 28 epochs
- `gate_2blk_epoch23`, `gate_2blk_epoch28` — gate reports (JSON)
- `gate_epoch23.log`, `gate_epoch28.log` — the runs
