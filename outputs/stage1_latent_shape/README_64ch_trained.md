# The trained 64-channel model: quantization is now the binding constraint

Same diagnostic (`scripts/stage1_latent_shape.py`) re-run on the checkpoint that
gated at +87.9% (`outputs/stage1_vimeo/colab_run2_64ch/`), 2026-10-15. Report:
`latent_shape_64ch_trained.json`, log: `full_64ch.log`.

It overturns two of the recommendations the 192-channel run produced. Both are
corrected below.

## 1. The transform is now genuinely much better

| | float ceiling PSNR |
|---|---|
| Stage 1, 64ch, trained | **32.589 dB** |
| deployed baseline | 30.139 dB |
| **ceiling advantage** | **+2.450 dB** |

With no quantizer in the way, the transform is **+2.45 dB** ahead — against the
untrained 192-channel model's +0.43 dB. Full training plus the right latent shape
turned a marginal transform into a decisively better one.

## 2. But quantization eats the advantage, progressively

Loss against each model's *own* ceiling:

| bits | Stage 1 | baseline | Stage 1 loses |
|---|---|---|---|
| 5 | −1.010 dB | −0.831 dB | **+0.179 dB more** |
| 4 | −2.327 dB | −1.699 dB | **+0.627 dB more** |
| 3 | −5.782 dB | −4.072 dB | **+1.710 dB more** |

This decomposes the gate almost exactly. Gate gain = ceiling advantage − excess
quantization loss:

| bits | 2.450 − excess | gate measured |
|---|---|---|
| 5 | +2.27 dB | **+2.37 dB** |
| 4 | +1.82 dB | **+1.88 dB** |
| 3 | +0.74 dB | **+0.79 dB** |

So the collapsing advantage at low rate is not a capacity limit. **The transform
holds a flat +2.45 dB everywhere; quantization fragility takes progressively more
of it away.** At 3-bit it takes 70% of it.

**Stage 1 has never had QAT.** `build_model` is called with no
`quantization_noise`, in both phases, so the decoder has never seen a quantized
latent. The deployed baseline trained with `--qat-enabled`. That asymmetry is the
obvious candidate and it is now pointed at the right end of the curve.

One honest caveat against over-reading it: the *untrained* 192-channel model was
measured as **more** robust than the baseline (`README.md`, section 3), also
without QAT. Robustness is therefore not purely QAT-determined — a barely-trained
model is blunt and tolerant, and this fully-trained one is sharper and more
fragile. QAT is the strongest available lead, not a certainty.

## 3. Correction: bit allocation is worth nothing on this model

The 192-channel work recommended wiring up per-channel bit allocation as "the
cheapest remaining BD-rate point". **On this checkpoint it is worth exactly 0%.**

| | idle channels (<1% of peak) | busiest/quietest energy | water-filled allocation | bytes |
|---|---|---|---|---|
| Stage 1, 64ch, trained | **0** | 11.8x | `{'4': 64}` | **+0.00%** |
| deployed baseline | 29 | 45.8x | `{'3': 15, '4': 49}` | −8.33% |

The trained 64-channel model **uses all 64 channels** — not one falls below 1% of
the strongest, and the spread is 11.8x against the baseline's 45.8x. Water-filling
has no quiet channels to take levels from, so it allocates 4 bits to all 64 and
changes nothing. Three channels still carry 50% of the energy, but the tail is
full rather than idle.

This is itself a good sign: narrowing from 192 to 64 forced the transform to
pack information into every channel, which is exactly what should happen. It just
also means there is no sparsity left to harvest.

(The −8.33% on the deployed baseline remains real and unclaimed, and
`allocate_bits_per_channel` remains a no-op by construction. That is a separate,
smaller opportunity against the *current* codec, not against Stage 1.)

## 4. Correction: more capacity is not where the loss is

`RESIDUAL_BLOCKS = 2` is still free in rate and still worth measuring, but the
case for it is weaker than it looked:

- at 5-bit the coded result is already within ~1 dB of the ceiling, so there is
  little headroom for a higher ceiling to deliver;
- at 3-bit the gap to the ceiling is **5.78 dB**, and that is quantization, not
  capacity. Raising the ceiling by a few tenths moves the 3-bit point by less.

A higher ceiling helps the high-rate end, where the advantage is already largest.
The weak end of the curve needs the quantization path fixed, not more parameters.

## What this says to do

1. **Wire QAT into the Stage 1 trainer** and retrain. It targets the +1.710 dB
   excess at 3-bit directly — the end of the curve BD-rate is most sensitive to.
   The plumbing partly exists (`build_model(args, quantization_noise=...)` and
   `QuantizationNoise.from_calibration`); `main()` simply never passes it, so this
   is wiring plus a calibration file, mirroring what `m10f_lambda_boundary.py`
   does for the baseline.
2. **`RESIDUAL_BLOCKS = 2`** — still free in rate, still worth a number, but
   expect it to widen the high-rate advantage rather than fix the low-rate one.
3. **Phase B** after QAT rather than before: both touch the quantization path, and
   a rate objective fitted on a QAT-robust decoder is the more meaningful one.
4. Per-channel bit allocation **only against the deployed baseline**, where the
   −8.33% is real. Not a Stage 1 item.
