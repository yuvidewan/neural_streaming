# Is 192 channels at stride 16 the right latent shape?

**No. It buys +0.43 dB and costs 3.2x the rate, and the cost does not depend on
what the channels carry — so training cannot recover it.**

`scripts/stage1_latent_shape.py`, 2026-10-14. Full DAVIS TEST (9 sequences,
719 frames), 400 TRAIN calibration frames, both checkpoints measured by the same
code on the same frames. Report: `latent_shape.json`, log: `full.log`.

## 1. The transform is barely better, with no quantizer in the way at all

| | float PSNR | float MS-SSIM | parameters |
|---|---|---|---|
| Stage 1 (192ch) | **30.564 dB** | 0.9802 | 8,437,827 |
| deployed baseline (64ch) | 30.139 dB | 0.9773 | 593,411 |
| difference | **+0.425 dB** | +0.0029 | 14x |

This is the ceiling: encode, decode, no quantization, no entropy coding. 14x the
parameters is buying **0.43 dB**. Some of that is the 2-epoch training, but it
bounds what the rest of the pipeline is being asked to pay for.

## 2. The rate penalty is the channel count, at every operating point

| bits | Stage 1 bytes | PSNR | baseline bytes | PSNR | byte ratio |
|---|---|---|---|---|---|
| 5 | 8,801,577 | 29.704 | 2,787,612 | 29.308 | **3.16x** |
| 4 | 6,527,836 | 29.204 | 2,039,694 | 28.440 | **3.20x** |
| 3 | 4,228,319 | 27.704 | 1,296,122 | 26.067 | **3.26x** |

Both models are stride 16. 192 vs 64 channels is 1,543,680 vs 514,560 symbols per
1080p frame — exactly 3x — and the coded bytes land at 3.16-3.26x. This matches
the gate's 3.15-3.25x independently, through different code.

## 3. A hypothesis that turned out to be wrong

Stage 1 trained with **no QAT** (`build_model` is called with no
`quantization_noise`, in both phases), while the deployed baseline trained with
`--qat-enabled`. That asymmetry looked like a strong candidate for part of the
gap. Measured, it is not:

| | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| Stage 1 loss vs its own ceiling | 0.860 dB | **1.359 dB** | **2.859 dB** |
| baseline loss vs its own ceiling | 0.831 dB | 1.699 dB | 4.072 dB |

Stage 1 is *more* robust to quantization than the QAT-trained baseline at 4 and
3 bit, despite never having seen a quantized latent. The QAT gap is not the
explanation and is not worth spending a run on.

(An earlier reading of this, on 20-30 calibration frames from a single TRAIN
sequence, showed PSNR *rising* as bit depth fell. That was a badly fitted
quantizer grid, not a real effect — latent MSE stayed correctly monotone
throughout. At the production 400 frames everything is monotone. Worth recording
because a thin calibration is silently misleading rather than obviously broken.)

## 4. Why training cannot fix this: cost is decoupled from content

| | energy ratio busiest/quietest | cost ratio | quietest quartile pays | Spearman |
|---|---|---|---|---|
| Stage 1 | **20.8x** | **1.32x** | 68.2% of full depth | +0.215 |
| baseline | **45.8x** | **1.25x** | 62.0% of full depth | +0.359 |

A channel carrying 1/20th the energy of its busiest sibling still pays about
two-thirds of full price. **58 of Stage 1's 192 channels sit below 1% of the
strongest channel's energy** and the cheapest decile still costs 2.39 of 4 bits.

The mechanism is the quantizer, not the entropy coder. `EmpiricalEntropyModel`
keeps one static frequency table per channel, so a degenerate channel *could* be
nearly free. But the intra quantizer runs in `per_channel` mode and fits each
channel's scale to that channel's own spread — so a near-idle channel gets a tiny
scale that re-spreads its tiny values across the whole alphabet. Sparsity is
normalized away before the coder ever sees it.

**This is why Phase B cannot rescue the 3.2x.** Rate pressure works by driving
channels toward zero; per-channel scaling then amplifies them straight back to
~68% of full cost. There is very little for `D + lambda*R` to cash in.

## 5. A dormant optimization, and why it is not enough

`nvc.compression.calibration.allocate_bits_per_channel` implements reverse
water-filling, is documented and tested, is accepted by
`calibrate_quantization_params` as `bits_per_channel` — and **is passed nowhere
in production or in the gate path.**

It is also **a no-op by construction**: `max_bits` defaults to `average_bits`
while the realized average is rebalanced back to `average_bits`, and the only
allocation satisfying both is uniform. Measured, both models: `{'4': 192}` and
`{'4': 64}`, byte-identical to uniform. Pinned in
`tests/test_stage1_latent_shape.py`.

Dropping the rebalance makes it real:

| arm | avg bits | Stage 1 bytes | ΔPSNR | baseline bytes | ΔPSNR |
|---|---|---|---|---|---|
| uniform (ships today) | 4.00 | 742,664 | — | 232,686 | — |
| library, as provided | 4.00 | 742,664 (+0.00%) | +0.00 | 232,686 (+0.00%) | +0.00 |
| water-filled | ~3.8 | **693,313 (-6.65%)** | -0.06 | **213,305 (-8.33%)** | -0.16 |
| uniform one bit lower | 3.00 | 485,075 (-34.68%) | -1.30 | 148,870 (-36.02%) | -2.39 |

Water-filling trades ~7-8% of bytes for ~0.06-0.16 dB — roughly **4x more
bit-efficient per dB than simply lowering the bit depth**, which is the control.
A real win, worth taking, and nowhere near 3.2x.

## 6. One experiment that did not work

The channel-pruning sweep (keep the top-k channels by energy, zero the rest) is
**uninformative**, and its own control proves it: the baseline keeps 94% of its
energy at 32/64 channels and still falls from 29.32 to 19.70 dB. Zeroing channels
is input the decoder never saw in training, so the sweep measures decoder
brittleness, not channel importance. It cannot be used to argue a channel count,
and is kept in the report only so nobody re-runs it expecting an answer.

## What to do

**Narrow the latent to 64 channels and keep the depth.** The capacity is in the
body, not the latent projection:

| latent | base | blocks | parameters | symbols/frame @1080p |
|---|---|---|---|---|
| 192 | 192 | 1 | 8,437,827 | 1,543,680 |
| **64** | **192** | **1** | **7,208,899** | **514,560** |
| **64** | **192** | **2** | **9,864,643** | **514,560** |
| 64 | 32 | — | 593,411 (baseline) | 514,560 |

At 64 channels the model keeps 7.2M parameters — 85% of the capacity, 12x the
baseline — and codes **exactly the baseline's symbol count**. That makes the next
gate an apples-to-apples test of the transform rather than a test of latent
width, and removes the entire 3.2x penalty up front. `--residual-blocks 2`
restores the full 8.4M budget and more, still at baseline rate.

Then, in order:

1. **Re-run Phase A at `--latent-channels 64 --base-channels 192`**, with the
   clipping fix from `8d46bf5`. Same cost as the run that just failed.
2. **Fix `allocate_bits_per_channel`'s default** (raise `max_bits`, or expose a
   no-rebalance mode) and pass it in the intra path. ~7% for ~0.06 dB, and it
   helps the baseline too.
3. **Then** Phase B. With symbol parity restored, `D + lambda*R` is tuning an
   operating point rather than trying to claw back 3.2x.

Stride 32 would also cut symbols 4x, but 1080/32 is not an integer and
`DOWNSAMPLE_FACTOR` enforces divisibility, so it needs padding and a fifth
stride-2 stage — a bigger change than narrowing the latent, for the same effect.

The deeper point for Stage 2: this codec prices every channel at nearly full
freight regardless of content. Reference designs afford 192 channels because a
learned hyperprior conditions each channel on the others. Until that exists,
**latent width is close to pure cost here**, and the roadmap's assumption that a
wider latent is a free upgrade does not hold.
