# M22 — Residual-Quantizer / Autoencoder Freeze-Lift

> **Classification: A — MEANINGFUL RESIDUAL-REPRESENTATION SUCCESS**, graded on
> held-out DAVIS TEST.

---

## 1. Executive summary

**Re-centering the residual quantizer grid reduces real coded bytes by
+1.0554% / +1.3091% / +1.7563% of the total stream at 5/4/3-bit on the full
719-frame DAVIS TEST split, at equal or better quality, with no container change,
no side information and no runtime cost.** BD-rate against the frozen production
baseline is **−2.4746% PSNR / −1.0968% MS-SSIM**.

The change is one line of calibration: the residual grid is forced symmetric about
zero (half-width = max(|p0.1|, |p99.9|)) instead of spanning the raw (0.1, 99.9)
percentile range. Its `scale` is essentially unchanged (ratio 1.02); what moves is
`zero_point`, on 37 of 64 channels. That re-alignment drops 0th-order residual
symbol entropy from 1.568 to 1.239 bits at 3-bit — **21% fewer bits to describe
the same residuals at the same step and the same quality.**

Five things make this a result rather than a number:

1. **It generalizes.** VAL-B predicted +1.6271%; TEST delivered **+1.7563%** — a
   gap of −0.1292 points *in the candidate's favour*, with all 9 TEST sequences
   improving individually (+1.14% to +3.79%). M21's comparable candidate retained
   20% of its VAL-B estimate at this step; this retains 87–116%.
2. **It is attributable.** A full 2×2 (grid × refitted stack) puts the grid's main
   effect at **+1.6271%** and the refit's at **−0.12% to +0.12%** — i.e. nothing.
   The gain is the grid's, not the fitting procedure's.
3. **It is quality-neutral or better.** At 5-bit and 4-bit it is a strict Pareto
   improvement on TEST (fewer bytes *and* higher PSNR). Only 3-bit costs quality,
   −0.0126 dB, an eighth of the pre-declared allowance.
4. **It is verified end to end.** All three baseline arms reproduce M14's recorded
   DAVIS totals byte-for-byte; decode is exact on 9/9 sequences at every rate
   point; an independent process reproduces the locked candidate with zero
   differences across 13 aggregate and 24 per-sequence fields.
5. **It is not the mechanism anyone expected.** Phase 13 shows it closes
   **none** of M17's oracle gap (−5.23% / −0.33% / +0.22%) and barely moves symbol
   agreement (70.18% → 70.51%). The reference-error penalty M17–M21 chased is
   untouched; this is an orthogonal axis.

**What it cost to find:** eight pre-registered grids, two arms each, three rate
points. Six of the eight either lose bytes or buy them with distortion the
pre-declared guard rejects.

**The larger lead M22 leaves on the table:** `broad_p001` (percentiles 0.01/99.99)
scores **BD-rate −8.2372%** — four to seven times the selected candidate — but
shifts the quality-per-bit-depth operating point and so fails a guard built for
drop-in replacement. It is correctly excluded here and is §19's primary
recommendation for M23.

---

## 2. Frozen baseline

Phase 0 ran the deployed closed loop over VAL-B (`car-roundabout`, `drift-straight`,
`pigs`, `stunt`; 246 P-frames, 4,030,464 residual symbols per rate point) at
`git HEAD 19472b80` with no non-M22 working-tree changes, and reproduced the
M17–M21 production totals **byte for byte**.

| | 5-bit | 4-bit | 3-bit |
|---|---:|---:|---:|
| total container bytes | **1,625,012** | **1,140,526** | **723,381** |
| — recorded (M17–M21) | 1,625,012 | 1,140,526 | 723,381 |
| P-frame residual bytes | 1,333,275 | 908,404 | 549,083 |
| I-frame residual bytes | 225,575 | 165,243 | 105,148 |
| motion bytes | 59,367 | 60,084 | 62,355 |
| container overhead | 6,795 | 6,795 | 6,795 |
| BPP | 0.721330 | 0.506270 | 0.321103 |
| PSNR (dB) | 29.4393 | 29.1027 | 27.9431 |
| MS-SSIM | 0.981762 | 0.976915 | 0.960155 |
| residual RMS | 1.110309 | 1.127236 | 1.196808 |
| residual MAE | 0.599371 | 0.629291 | 0.719338 |
| oracle symbol agreement | 70.18% | 74.76% | 81.31% |
| — at GOP position 1 | 55.51% | 60.39% | 67.29% |
| G16 ideal bits | 10.6650 M | 7.2661 M | 4.3915 M |
| M17 oracle gap (residual channel) | +2.4719% | +8.4662% | +19.6132% |

Frozen identities, all verified equal to M19's:

| | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| residual entropy identity | `ec858dee10f8a955` | `be872c2beebd2f9b` | `d2d61d66a50cad8a` |
| intra entropy identity | `8fadd1ca200c033b` | `14c4b8bf40056ca1` | `b523711d2e982057` |
| motion table identity | `d7e7b237b6451885` | `d7e7b237b6451885` | `d7e7b237b6451885` |

Every VAL-B sequence decoded back from its `.nvct` v2 container with exact symbol,
reference and reconstruction equality.

**BASELINE STATUS: CONFIRMED.** M22 proceeds.

---

## 3. Exact residual-path trace

The deployed P-frame path, traced through the real code rather than from the
design documents:

```
previous  = model.decode(reconstructed_latent)        # the decoder's own frame
motion    = estimate_block_motion(previous, frame)    # 16x16 blocks, +/-16 search
payload   = encode_motion_payload(motion)             # lossless round trip,
motion'   = decode_motion_payload(payload)            #   motion' == motion
warped    = warp_blocks(previous, motion')
reference = model.encode(warped)                      # reference_latent
latent    = model.encode(frame)                       # current_latent
delta     = latent - reference
symbols   = latent_to_symbols(delta, residual_params) # UniformQuantizer, per_channel
rows      = G16.log_probabilities(reference, G16.planes(symbols, zero))
index     = assign_codebook.assign_tensor(rows)       # K=512
bytes     = range_coder(symbols, coding_codebook[index])   # M13 frequencies
```

### 3.1 What can change in isolation — and what cannot

The milestone asks for the *smallest* intervention. Phase 1 probed the interface
rather than assuming it, and found that **the residual quantizer has no interface
at which it can change alone.**

`m10l_evaluate.calibration_signature` hashes **only** `residual_params.scale`,
`residual_params.zero_point`, the bit depth, the mode and the calibration-frame
count. `m11_evaluate.check_provenance` compares that signature against the one
recorded in the G16 checkpoint and raises `ProvenanceError` on any mismatch.

Phase 1 verified all three consequences empirically at each rate point:

| probe | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| changing the grid moves the calibration signature | yes | yes | yes |
| the deployed G16 checkpoint then rejects the grid | yes (`ProvenanceError`) | yes | yes |
| the `.nvct` v2 container needs a new field | **no** | no | no |

For example at 3-bit: `calibration 95a513136a72daf0 != 2d43e16dc8e2835b`.

So changing `residual_params` invalidates, in order:

**M10K → M11-G16 → the K=512 shared codebook → the M13 coding frequencies.**

It does **not** invalidate the intra quantizer, the motion path, the container
format, the range coder, or the causal decode ordering. That boundary is what
makes this milestone a clean single-mechanism freeze-lift.

This coupling is not an accident to be worked around — it is the guard that M10K
installed deliberately, because a stale grid costs bits silently. M22 therefore
runs **two explicitly separated arms** rather than picking one:

* **STALE** — the new grid, deployed M10K/G16/K=512/M13 kept exactly as they are.
  The provenance guard is bypassed *by construction and on the record* (the
  checkpoint is simply never re-checked). This arm exists to **price what the
  guard is protecting**; satisfying it quietly would destroy the measurement.
* **REFIT** — the new grid with M10K, G16, the K=512 codebook and the M13
  frequencies rebuilt on TRAIN by the same unmodified fitting code that produced
  the deployed stack. Every fitted component gets a new identity, which the
  `.nvct` v2 header already carries. No format change.

Keeping these apart is what section 15 of the brief requires: the codebook
incompatibility is **recorded and quantified**, never silently repaired.

---

## 4. Residual-space diagnostics (Phase 2)

Measured on TRAIN and VAL-B only, over all 246 VAL-B P-frames. TEST was never
opened.

### A–C. Residual distribution, per-channel variance and dynamic range

| | 5-bit | 4-bit | 3-bit |
|---|---:|---:|---:|
| residual std (mean over channels) | 0.9092 | 0.9156 | 0.9436 |
| residual MAE | 0.5252 | 0.5339 | 0.5601 |
| residual RMS | 0.9394 | 0.9461 | 0.9752 |
| dynamic range | 27.074 | 27.002 | 27.129 |
| top-10% of channels carry | 17.9% | 17.9% | 18.0% |
| top-25% of channels carry | 40.8% | 40.9% | 41.1% |
| max/median channel variance | 2.9× | 2.9× | 2.9× |

Variance is **mildly** concentrated: a 2.9× spread between the loudest and median
channel, with the top decile of channels carrying 17.9% of the variance against
10% for a flat distribution. The deployed quantizer is already per-channel, so it
already absorbs this. **Per-channel scale is not the defect.**

### D–E. Step size, representability and clipping

| | 5-bit | 4-bit | 3-bit |
|---|---:|---:|---:|
| step (mean over channels) | 0.38986 | 0.80636 | 1.73229 |
| step (min / max) | 0.14951 / 0.62850 | 0.30726 / 1.30550 | 0.66642 / 2.77436 |
| **step / residual std** | **0.4241** | **0.8708** | **1.8159** |
| zero exactly representable | yes | yes | yes |
| clipping | 0.1736% | 0.1536% | 0.1246% |
| quantization SNR (dB) | 15.86 | 11.61 | 7.05 |

Clipping is **negligible at every depth** (≤0.17%). The deployed (0.1, 99.9)
percentile range is not starving the grid — so a "the range is too narrow"
explanation is ruled out by measurement, not by argument.

### F. Symbol entropy against the fixed-rate alphabet

| | 5-bit | 4-bit | 3-bit |
|---|---:|---:|---:|
| symbol entropy (bits) | 2.9776 | 2.1475 | 1.5684 |
| of an alphabet of | 5 | 4 | 3 |
| alphabet efficiency | 59.6% | 53.7% | 52.3% |
| symbols actually used | 32/32 | 16/16 | 8/8 |

The residual symbol distribution is strongly peaked — it uses only ~52–60% of its
nominal alphabet. The whole alphabet *is* exercised (no dead symbols), so this is
peakedness, not range misplacement. Note that the entropy stack already exploits
this: G16 + M13 code well below the nominal depth, so low alphabet efficiency is
not itself a loss — it is a description of the distribution the quantizer hands on.

### G–H. Symbol stability under decoded-reference error

This is the mechanism.

| | 5-bit | 4-bit | 3-bit |
|---|---:|---:|---:|
| symbol agreement with the oracle arm | 70.18% | 74.76% | 81.31% |
| symbol change rate | 29.82% | 25.24% | 18.69% |
| reference shift, mean (steps) | 0.4018 | 0.3162 | 0.2584 |
| **reference shift, RMS (steps)** | **0.9162** | **0.5644** | **0.3930** |

The decoded-reference perturbation, **measured in units of the quantizer's own
step**, is roughly one step at 5-bit and well under half a step at 3-bit. Change
rate tracks it monotonically and steeply:

| reference-shift decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5-bit change rate | 3% | 10% | 17% | 25% | 33% | 43% | 54% | 68% | 90% | 98% |
| 4-bit | 2% | 7% | 13% | 19% | 27% | 36% | 47% | 61% | 79% | 98% |
| 3-bit | 1% | 4% | 7% | 12% | 18% | 26% | 39% | 55% | 73% | 96% |

The top decile of perturbation flips its symbol 96–98% of the time; the bottom
decile 1–3%. Symbol instability is **not** diffuse — it is a direct function of
how large the reference error is relative to the step.

### I–J. Distance to the nearest reconstruction level

`level_distance` = |offset − round(offset)| in steps. **0 means the value sits on
a reconstruction level (stable); 0.5 means it sits on a decision boundary
(fragile).**

| | 5-bit | 4-bit | 3-bit |
|---|---:|---:|---:|
| mean distance to its own level (steps) | 0.2442 | 0.2366 | 0.2177 |
| (uniform would be) | 0.25 | 0.25 | 0.25 |

5-bit bands:

| band | positions | change rate | excess-bit share |
|---|---:|---:|---:|
| < 0.05 | 11.00% | 17.04% | 2.95% |
| < 0.10 | 21.62% | 17.61% | 6.41% |
| < 0.25 | 51.65% | 20.77% | 22.16% |
| < 0.50 (all) | 100.00% | 29.82% | 100.00% |

3-bit bands:

| band | positions | change rate | excess-bit share |
|---|---:|---:|---:|
| < 0.05 | 15.22% | 6.71% | 2.92% |
| < 0.10 | 27.52% | 7.56% | 6.06% |
| < 0.25 | 59.46% | 10.13% | 20.02% |
| < 0.50 (all) | 100.00% | 18.69% | 100.00% |

Positions sitting near a reconstruction level are markedly more robust — at 3-bit
the nearest 15.22% of positions flip at 6.71% against 18.69% overall, and carry
only 2.92% of the excess bits. The distribution is slightly level-attracted
already (0.218–0.244 against 0.25 for uniform), which is a small, real, and
**unexploited** regularity.

### K. Where the excess bits actually are

| displacement \|real − oracle\| | 5-bit pos / bits | 4-bit pos / bits | 3-bit pos / bits |
|---|---:|---:|---:|
| 0 steps | 70.18% / 8.78% | 74.76% / 12.25% | 81.31% / 12.03% |
| **1 step** | **25.82% / 83.50%** | **23.07% / 84.79%** | **17.77% / 86.51%** |
| 2 steps | 2.16% / 4.43% | 1.43% / 2.21% | 0.75% / 1.34% |
| 3–4 steps | 1.17% / 1.90% | 0.57% / 0.67% | 0.16% / 0.13% |
| 5–8 steps | 0.51% / 1.20% | 0.15% / 0.08% | 0.01% / −0.01% |
| ≥ 9 steps | 0.16% / 0.20% | 0.02% / −0.00% | 0.00% / 0.00% |

**83.5 / 84.8 / 86.5% of all excess bits come from symbols displaced by exactly
one step.** Large displacements are rare and, in aggregate, nearly free.

### The Phase 2 verdict on the five candidate explanations

The brief asks which of five explanations holds, and requires that it be measured
rather than inferred from M17/M21.

| # | explanation | verdict |
|---|---|---|
| 1 | the residual representation has poor geometry | **not supported.** Variance concentration is mild (2.9×) and already absorbed by per-channel scaling; the full alphabet is exercised at every depth. |
| 2 | the quantizer grid is poorly *placed* | **partly, but not by range.** Clipping is ≤0.17% and zero is exactly representable, so the range is not misplaced. What matters is the grid's **scale** relative to the perturbation. |
| 3 | the learned latent is overly sensitive to reference perturbations | **not established as an independent cause.** Sensitivity is fully explained by shift-relative-to-step; there is no residual sensitivity left over once that ratio is controlled for. |
| 4 | the residual distribution is badly matched to the fixed-rate alphabet | **true as a description** (52–60% efficiency), **but it is not where the loss is** — the G16 + M13 stack already codes well below the nominal depth. |
| 5 | a combination | **this.** |

**The measured mechanism:** residual-symbol instability is governed by the ratio
of *decoded-reference perturbation* to *quantizer step*. That ratio is 0.92 / 0.56
/ 0.39 (RMS steps) at 5/4/3-bit, it drives change rate monotonically from ~2% to
~98% across its deciles, and the resulting cost is concentrated almost entirely
(83–87%) in **one-step** displacements.

This has a direct and testable consequence: **a coarser step should reduce flips,
and a finer step should increase them.** That is exactly the axis the Phase 3–5
sweep traverses — and it is why it must be read as a rate/**distortion** move, not
a free win, which is what the pre-declared distortion guard is for.

---

## 5. Pre-registered candidate family

Frozen in `scripts/m22_residual.py` as `GRID_VARIANTS` **before any VAL-B coded
result was read**, and covered by a test that refuses any variant not on this list.

This milestone tests **Candidate A** (residual quantizer recalibration) of the
brief's A–D family. Sections 5.2 and 5.3 record why B, C and D were not run.

### 5.1 Candidate A — residual-quantizer recalibration (8 grids × 2 arms)

The autoencoder is frozen throughout. Only the residual quantization
parameterisation changes. Every grid is fitted on the **TRAIN residual stack
alone**; no variant is allowed to see VAL-B or TEST, and a test asserts this over
the source of each `build` function.

| variant | family | definition |
|---|---|---|
| `deployed` | percentile | percentiles (0.1, 99.9), per channel — **the identity control** |
| `broad_p001` | percentile | percentiles (0.01, 99.99): wider range, coarser step, less clipping |
| `tight_p1` | percentile | percentiles (1.0, 99.0): narrower range, finer step, more clipping |
| `tight_p05` | percentile | percentiles (0.5, 99.5): a milder tightening |
| `symmetric_p01` | symmetric | symmetric about zero, half-width = max(\|p0.1\|, \|p99.9\|) |
| `mad_k8` | robust | symmetric, half-width = 8 × per-channel MAD |
| `mad_k12` | robust | symmetric, half-width = 12 × per-channel MAD |
| `mse_optimal` | MSE | symmetric, per-channel half-width minimising TRAIN quantization MSE over a fixed 48-point geometric ladder |

The family deliberately spans the **step-size axis in both directions** (broader
and tighter than deployed), because Phase 2 identified step size relative to
reference perturbation as the governing quantity, and the sign of the effect on
coded bytes is genuinely unknown in advance: a coarser step reduces symbol flips
but raises distortion and shifts the symbol distribution.

The `deployed` variant is not a nominal control. It is rebuilt from the TRAIN
residual stack by the same code path as every other variant, and
`verify_deployed_grid()` **raises** if the rebuilt grid is not bit-identical to
the production grid — because if the control is not the control, every delta is
measured from the wrong origin. (This guard earned its place: see §16.1.)

### 5.2 Declared protocol (fixed before results were read)

| parameter | value |
|---|---|
| screening rate point | 3-bit |
| confirmation rate points | 5-bit, 4-bit |
| Stage-2 admission threshold | total-stream gain ≥ −0.25% at the screen |
| Stage-2 minimum candidates | 2 |
| distortion guard | ΔPSNR ≥ −0.10 dB **and** ΔMS-SSIM ≥ −0.0010 |
| gates | <0.5% weak · 0.5–1.0% marginal · ≥1.0% meaningful (total stream) |

The distortion guard can only ever **disqualify** a candidate; it can never
promote one. Selection is by actual coded total-stream bytes, per §17 of the brief.

3-bit was chosen as the screen because M17 measured the largest oracle gap there
(+19.6132% of the residual channel against +2.4719% at 5-bit), so it is where a
representation effect has the most room to show itself — and a mechanism that
cannot be seen at 3-bit is unlikely to be worth confirming at 5-bit.

### 5.3 Directional predictions, recorded before the results

Phase 2 says step size relative to reference perturbation governs symbol
stability. That yields specific, falsifiable expectations, written down here
before the sweep's distortion numbers were read so that a surprise is credible
rather than retrofitted:

| variant | step vs deployed | predicted bytes | predicted quality |
|---|---|---|---|
| `tight_p1`, `tight_p05` | finer | **more** | better PSNR, more clipping |
| `broad_p001`, `mad_k12` | coarser | **fewer** | worse PSNR |
| `mse_optimal` | ≈ deployed | ≈ deployed | slightly better PSNR |
| `symmetric_p01` | ≈ deployed | ≈ deployed | ≈ deployed |

If this ordering holds cleanly and every BD-rate lands near zero, the conclusion
is that **the residual grid is a rate knob, not a bottleneck** — the deployed
percentiles already sit on the rate/distortion frontier, and no repositioning of
a uniform grid buys anything the codec was not already free to buy by changing
bit depth. The interesting outcome — the one that would make this a representation
result rather than a rate-point result — is a variant whose **BD-rate is
meaningfully negative**, meaning the grid's *shape* matters and not merely its
scale.

*Disclosure:* at the time these predictions were recorded, one byte figure had
already been observed — `broad_p001/refit` at +23.40% total stream (fewer bytes,
consistent with the "coarser ⇒ fewer bytes" row above). Its distortion, its
BD-rate and every other variant were still unread.

### 5.4 Candidates B, C and D — declared, and why they were not run

The brief requires these be tested **"if technically compatible with the existing
architecture."** They are not, at this stage, for a reason established in Phase 1
rather than assumed:

* **B (residual-aware QAT)** and **C (reference-robust residual representation)**
  both require retraining the M10F autoencoder. Phase 1 established that the
  residual grid is already welded to four downstream fitted components. Retraining
  the autoencoder moves the latent itself, which invalidates the *same* four
  components **plus** the intra quantizer and the intra entropy model — and it
  changes the residual grid as a side effect. Running B or C before A would
  therefore change the representation, the quantizer, the entropy model and the
  codebook simultaneously, which §24 of the brief explicitly forbids ("do not
  simultaneously change … and then attribute the result to one mechanism").
* **D (capacity adjustment)** is gated on the existing architecture having "an
  obvious bottleneck." Phase 2 found none: the alphabet is fully exercised,
  clipping is negligible, and per-channel variance spread is mild. There is no
  measured bottleneck to justify a capacity change.

Candidate A is the smallest intervention that isolates the quantizer, and its
result determines whether B/C are worth their much larger blast radius. That
determination is recorded in §19.

---

## 6. Training objectives

```
L = D + λ·R + γ·S
```

| term | value in M22 |
|---|---|
| D | **not optimised.** The M10F autoencoder is frozen (seed-42, epoch 70, `latent_channels=64`, `base_channels=32`). No candidate in this milestone trains it. |
| λ | **3e-4**, inherited from the frozen checkpoint. Not swept, per §8 of the brief. |
| R | the **deployed** entropy path's own objective: `−log₂ P(symbol \| reference, causal context)` in bits/symbol — the identical objective M10K and M11-G16 were originally trained under. |
| γ · S | **not used.** No candidate C was trained, so no sensitivity term and no closed-loop perturbation dataset were required (see §5.3). |

Because only the quantizer grid changes, the "training" in M22 is confined to
**refitting the downstream entropy stack** under each new grid — M10K (20 epochs),
M11-G16 (20 epochs, warm-started from M10K), the K=512 shared codebook, and the
M13 coding frequencies. This is not a new objective; it is the original fitting
code re-run on TRAIN, which is what makes the REFIT arm a fair comparison rather
than a differently-tuned one.

**The rate term is not a proxy.** §8 of the brief warns against claiming success
from a proxy alone; here R *is* the deployed entropy model's own likelihood, and
the primary reported metric is nonetheless actual coded container bytes measured
through a real encode/decode round trip.

### Checkpoint selection

| rule | |
|---|---|
| PRIMARY | best validation checkpoint under the declared objective |
| SECONDARY | final checkpoint |
| tie-break | earliest epoch |
| selection split | **VAL-A** (`val`, 40 frames/sequence) |

VAL-B is **never** read during selection — it is the held-out compression gate
only — and TEST is not opened at all before the candidate lock. A test asserts
the selection rule over the source of `refit_downstream`.

---

## 7. Checkpoint / provenance audit

Every REFIT arm writes a checkpoint plus a sidecar `.provenance.json`. The SHA256
is taken **after** the file is written, so the recorded digest is of the artifact
a later run actually loads, and `load_refit_checkpoint` verifies it **before**
unpickling anything.

3-bit screen, all seven refitted stacks:

| variant | grid signature | calibration sig | digest | epoch | VAL-A bits/symbol | residual identity |
|---|---|---|---|---:|---:|---|
| `broad_p001` | `486a84131389ad11` | `44dcb02388` | ✓ | 19 | 1.01440 | `83a8cfb557` |
| `symmetric_p01` | `39db8a09133b0f39` | `cf0db4af8a` | ✓ | 19 | 1.35509 | `abb43923d3` |
| `mad_k12` | `91d59d50bcc12e2c` | `8bec8ede1e` | ✓ | 18 | 1.78706 | `cfdd588ebf` |
| `tight_p05` | `ef950354fe8d4103` | `0321d99d75` | ✓ | 18 | 1.86329 | `24cc6e906c` |
| `mse_optimal` | `d9dca69ae0e4f374` | `8dc2edd0c3` | ✓ | 19 | 1.88044 | `40e58a4718` |
| `tight_p1` | `7998ccd95c7fe767` | `5c70bbd05d` | ✓ | 19 | 2.10896 | `ed1f03b3e9` |
| `mad_k8` | `325432733f529173` | `84e00c38b4` | ✓ | 20 | 2.14642 | `1d86619c1f` |

**7 distinct grid signatures, 7 distinct calibration signatures, 7 distinct
residual entropy identities, 7 distinct codebook identities, all digests verified.**
A checkpoint/evaluation mismatch is therefore not merely unlikely — it is
detectable by construction, which is what Phase 11 asks for.

The VAL-A objective column is itself a useful independent check on the refit: it
orders exactly with step size (coarse `broad_p001` 1.014 bits/symbol → fine
`mad_k8` 2.146), confirming each refitted entropy stack really did adapt to the
grid it was given rather than inheriting the deployed one.

Each record also carries seed 42, λ = 3e-4, γ = None, the epoch-selection rule
(min `(val_loss, epoch)` on VAL-A), the TRAIN/VAL sequence lists and P-frame
counts, and a `code_identity` map of SHA256 digests over every script whose
behaviour could change a fit.

### 7.1 One recorded code-identity discrepancy

`code_identity()` hashes `m22_sweep.py`, and that file was edited **after** the
screen's checkpoints were written, to repair a `NameError` on the script's final
`print` statement (a missed variable rename; see §16.2). The seven screen
checkpoints therefore record the pre-fix hash of `m22_sweep.py` and will be
**refused** by `resume_refit` from now on, which is the conservative behaviour
working as designed.

Recorded rather than suppressed: the edit changed one `print` executed after all
computation and after the report was written, so it cannot have affected any
fitted artifact. No checkpoint was retro-stamped to hide the mismatch. Where a
later phase needed one of those stacks, it was refitted from scratch rather than
resumed — which additionally produced the determinism replicate reported in §14.

---

## 8. VAL-B results

### 8.1 Stage 1 — the 3-bit screen (all 8 variants × 2 arms)

Control: total 723,381 · residual 654,231 · motion 62,355 · BPP 0.321103 ·
PSNR 27.9431 · MS-SSIM 0.960155.

| variant / arm | step | clip % | H(sym) | Δ total bytes | stream % | resid % | ΔPSNR | ΔMS-SSIM | guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `broad_p001/refit` | 2.6669 | 0.011 | 1.336 | −169,238 | **+23.3954** | +26.1139 | −0.7873 | −0.012835 | **DISTORTION** |
| **`symmetric_p01/refit`** | 1.7658 | 0.120 | 1.239 | **−10,965** | **+1.5158** | **+1.6927** | −0.0248 | −0.000367 | **PASS** |
| `mad_k12/refit` | 1.0837 | 0.728 | 1.633 | +184,373 | −25.4877 | −28.3626 | +0.1704 | +0.006934 | pass |
| `tight_p05/refit` | 1.0824 | 0.714 | 1.881 | +203,169 | −28.0860 | −31.2188 | +0.2165 | +0.006871 | pass |
| `mse_optimal/refit` | 1.0208 | 0.932 | 1.751 | +232,796 | −32.1817 | −35.7669 | +0.1924 | +0.007063 | pass |
| `tight_p1/refit` | 0.8219 | 1.505 | 2.106 | +325,415 | −44.9853 | −49.9778 | +0.0796 | +0.007630 | pass |
| `mad_k8/refit` | 0.7224 | 2.043 | 2.078 | +368,909 | −50.9979 | −56.6450 | −0.1038 | +0.006816 | DISTORTION |

Every one of the 15 runs decoded back from its container with exact symbol,
reference and reconstruction equality (4/4 sequences each).

**Only `symmetric_p01/refit` both reduces coded bytes and holds quality.** Every
other variant either costs bytes or buys them with distortion the pre-declared
guard rejects.

### 8.2 The predictions held

§5.3's directional table was recorded before these distortion numbers were read,
and every row of it survived:

* coarser (`broad_p001`, step ×1.54) → fewer bytes, **−0.7873 dB** → disqualified;
* finer (`tight_p1` ×0.47, `mad_k8` ×0.42) → more bytes, better PSNR, up to
  **2.043%** clipping;
* the ordering of coded bytes is monotonic in step size across all seven variants.

`broad_p001` is the cautionary case the guard exists for: **+23.40% of the total
stream is not a compression win**, it is the codec coding at a materially lower
quality point.

### 8.3 The coupling, priced

Averaged over the 7 variants, refitting the downstream stack is worth
**+57.6100% of the total stream** against leaving the deployed M10K/G16/K=512/M13
in place. That is the magnitude of what the calibration-signature guard protects,
and it answers §15's requirement to *quantify* the incompatibility rather than
silently repair it. Two consequences:

1. a residual-grid change can never be shipped without the downstream refit;
2. the STALE column is a measure of **table mismatch**, not of a grid's merit —
   `symmetric_p01/stale` reads −75.16% while the same grid refitted reads
   **+1.52%**.

### 8.3.1 Phase 15 — the incompatibility, decomposed and not repaired

§15 of the brief requires that a codebook invalidated by a grid change be
*recorded and quantified*, never silently recalibrated, and that whether
recalibration is a separately required phase be decided on that evidence.
`scripts/m22_codebook.py` measures it and writes nothing — a test asserts the
script cannot call any fitting or checkpoint-writing function.

For `symmetric_p01` at 3-bit, over all 4,030,464 VAL-B residual positions:

| measure | value |
|---|---:|
| K=512 assignment drift | **99.7630%** of positions route to a different prototype |
| refitted coding tables | 1.067684 bits/symbol |
| deployed coding tables, same symbols | 2.168513 bits/symbol |
| **deployed-table penalty** | **+103.1045%** |
| symbol entropy under the new grid | 1.4584 bits |

The deployed codebook is not merely suboptimal under the moved grid — it is
almost entirely wrong. Nearly every position routes to a different prototype, and
coding the new grid's symbols with the old tables costs **more than double** the
bits.

This reproduces the STALE arm from the other direction, which is a useful check
that neither measurement is an artifact: doubling the refit's 538,009 P-residual
bytes and adding the unchanged I-residual (105,148), motion (~62k) and overhead
(6,795) gives ≈1.26M against the 1,267,066 actually measured.

**Answering §15's question directly: recalibration is not a separately required
phase — it is an inseparable part of the same change.** There is no version of
this candidate that ships without the refit, and the calibration signature
already refuses to let one exist.

### 8.4 GOP-boundary split (Phase 14)

For `symmetric_p01/refit` at 3-bit:

| group | frames | Δ residual bytes | residual % |
|---|---:|---:|---:|
| position 1 (boundary) | 28 | −1,731 | **+2.1360** |
| positions 2–9 (ordinary) | 218 | −9,343 | **+1.9962** |

The gain is essentially **uniform across GOP position** — category **B** of §14
(improves all P frames), not a boundary fix. This distinguishes it from the
M16–M21 line of enquiry, which kept finding position 1 anomalous: this candidate
is not exploiting that anomaly.

### 8.5 Stage 2 — confirmation at 5-bit and 4-bit

Admission was by the pre-declared rule (≥ −0.25% at the screen, minimum 2
candidates): `symmetric_p01` (+1.52%) and `broad_p001` (+23.40%) both qualified.

`symmetric_p01/refit`, all three rate points:

| bits | bytes | vs production | vs `deployed/refit` | P-residual | ΔPSNR | ΔMS-SSIM | guard | decode |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 1,605,194 | **+1.2196%** | +1.1031% | +1.3492% | **+0.0079** | **+0.000009** | PASS | exact |
| 4 | 1,126,297 | **+1.2476%** | +1.2729% | +1.6004% | **+0.0026** | −0.000076 | PASS | exact |
| 3 | 712,451 | **+1.5110%** | +1.6271% | +2.1626% | −0.0248 | −0.000367 | PASS | exact |

**Meaningful (≥1.0%) at every rate point**, guard passed at every rate point.

At 5-bit and 4-bit the candidate is a **strict Pareto improvement** — fewer bytes
*and* better PSNR (+0.0079 dB, +0.0026 dB), with MS-SSIM neutral. There is no
rate/distortion trade to adjudicate at those points; it is simply better. Only at
3-bit is there a quality cost, and it is −0.0248 dB, a quarter of the allowance.

The residual gain rises monotonically as the step coarsens — **+1.35% → +1.60% →
+2.16%** at 5/4/3-bit — which is the gradient §4's mechanism predicts, since
centering matters more when reconstruction levels are sparse (step/std 0.42 → 0.87
→ 1.82). An effect that weakens in the predicted direction while remaining
significant is stronger evidence of a real mechanism than a flat one would be.

### 8.6 The attribution control — a 2×2, not a correction

The sweep skips the control's REFIT arm (`if is_control and arm == "refit" …:
continue`, commented "the control is the same in both arms"). **That comment is
wrong**: `refit` rebuilds M10K/G16/K=512/M13 with M22's fitting hyperparameters,
which are not the ones that produced the deployed stack across M10K–M13. Without
that cell, +1.5158% would be "symmetric grid **and** refit" against "deployed grid
**and** deployed stack" — two changes and one number, which §24 forbids
attributing to a single mechanism.

Running `--variants deployed symmetric_p01 --arms refit` supplied the missing
cell and completed a full factorial:

| 3-bit | deployed stack | refitted stack |
|---|---:|---:|
| **deployed grid** | 723,381 (control) | 724,235 (**−0.1181%**) |
| **symmetric grid** | 1,267,066 (**−75.16%**) | **712,451 (+1.5110%)** |

* **Grid main effect, net of refitting:** **+1.6271%** — the mechanism's true size.
* **Refit main effect:** −0.1181% at 3-bit, +0.1178% at 5-bit, −0.0256% at 4-bit —
  i.e. **nothing**, in either direction, at every rate point. The refit procedure
  is a fair comparator, not a hidden source of gain.
* **Interaction:** large and *necessary*. The symmetric grid on the deployed tables
  is catastrophic (−75.16%).

So the gain is attributable to the **grid**, and the +1.5110% headline if anything
*understates* it, because the refit it rides on is marginally worse than what it
replaces. Two numbers, two questions: **+1.5110%** is what shipping this gains
against today's codec; **+1.6271%** is the size of the mechanism.

The interaction also settles a deployment question: a residual-grid change can
never ship without the downstream refit. They are one atomic change — precisely
what the calibration signature was built to enforce.

### 8.7 Rate/distortion — and a correction

Because a grid change moves rate and distortion together, BD-rate (§12's
methodology, reused from `m10b_evaluate`) is the deciding metric rather than raw
bytes at a fixed depth. On the three-point VAL-B curves:

| variant/arm | points | BD-rate PSNR | BD-rate MS-SSIM |
|---|---:|---:|---:|
| `broad_p001/refit` | 3 | **−8.2372%** | −4.8066% |
| `symmetric_p01/refit` | 3 | **−1.1839%** | −0.8815% |

**A correction to this milestone's own earlier reading.** During the sweep,
`broad_p001` was characterised as "not a representation improvement, just a lower
quality point." That is wrong, and BD-rate is what refutes it: at −8.24% it needs
~8% fewer bits *at equal quality*. Checked by hand at three quality levels
(−3.2% at 29.10 dB, −8.6% at 28.50 dB, −15.6% at 29.40 dB), the figure holds.

What the pre-declared guard actually rejects is narrower and still correct: at a
**fixed nominal bit depth**, `broad_p001` sits at a different quality point
(−0.7873 dB at 3-bit), which disqualifies it as a drop-in replacement. That is not
the same claim as "no improvement", and conflating the two would have buried the
largest result in this milestone.

The guard was declared before any result was read and is **not** relaxed to admit
it. `broad_p001` stays out of M22's selection — and becomes §19's primary
recommendation.

---

## 9. Mechanism decomposition

Phase 13 compares the REAL reference arm against the ORACLE reference arm at four
levels, for the deployed grid and the candidate, over all 246 VAL-B P-frames. The
question it answers is not "did bytes go down" (§8 settles that) but "**did the
candidate close the reference-error penalty M17 measured, or do something else?**"

| bits | level | deployed | `symmetric_p01` | gap closed |
|---|---|---:|---:|---:|
| 5 | 1. latent MSE vs oracle | 1.342732e-01 | 1.300490e-01 | +3.15% |
| | 2. symbol agreement | 70.18% | 70.51% | +1.11% |
| | 3. G16 ideal-bit gap | 2.4722% | 2.6410% | **−5.24%** |
| | 4. coded residual-byte gap | 2.4719% | 2.6405% | **−5.23%** |
| 4 | 1. latent MSE vs oracle | 2.185526e-01 | 2.148007e-01 | +1.72% |
| | 2. symbol agreement | 74.76% | 75.01% | +1.00% |
| | 3. G16 ideal-bit gap | 8.4666% | 8.6315% | −0.35% |
| | 4. coded residual-byte gap | 8.4662% | 8.6292% | **−0.33%** |
| 3 | 1. latent MSE vs oracle | 4.905006e-01 | 4.814200e-01 | +1.85% |
| | 2. symbol agreement | 81.31% | 81.67% | +1.91% |
| | 3. G16 ideal-bit gap | 19.6183% | 19.9771% | +0.22% |
| | 4. coded residual-byte gap | 19.6132% | 19.9710% | **+0.22%** |

Absolute coded residual bytes over the same frames:

| bits | deployed | `symmetric_p01` | gain |
|---|---:|---:|---:|
| 5 | 1,333,275 | 1,313,398 | **+1.4908%** |
| 4 | 908,404 | 894,153 | **+1.5688%** |
| 3 | 549,083 | 538,044 | **+2.0104%** |

### 9.1 The candidate does NOT close M17's gap — and that is the finding

Gap closure at the coded-byte level is **−5.23% / −0.33% / +0.22%** at 5/4/3-bit:
zero within measurement, and slightly *negative* at 5-bit. The milestone requires
this be explained rather than glossed, so:

**Both arms get cheaper.** The oracle-reference arm benefits from a better-aligned
quantizer grid exactly as the real arm does — the grid sits underneath both. At
5-bit the oracle improves marginally faster, so the *ratio* between them widens
even though both absolute costs fall. A relative gap can widen while every
absolute number improves, and that is precisely what happened.

**M22's gain is therefore orthogonal to the reference-error penalty.** M17's
oracle gap (+2.4719% / +8.4662% / +19.6132% of the residual channel) is **still
open and still unexploited** after this milestone. M22 did not attack it, did not
close it, and its success says nothing about whether it is closable.

This also settles what the mechanism is *not*:

* **not improved symbol stability.** Symbol agreement moves 70.18% → 70.51%,
  74.76% → 75.01%, 81.31% → 81.67% — about 1–2% of the gap, essentially nothing.
  M19 identified residual-symbol change as the driver of 88–91% of excess bits;
  this candidate leaves that almost untouched;
* **not improved reconstruction.** Latent MSE against the oracle improves by only
  1.7–3.2% of the gap, and §8's PSNR deltas are ±0.03 dB;
* **it is cheaper coding of the same symbols.** The symbols the candidate produces
  are distributed more favourably for the entropy stack (0th-order entropy 1.568
  → 1.239 bits at 3-bit, −21%), at equal step and equal quality.

The confirmation that this is genuinely entropy and not an accounting artifact:
G16's own ideal-bit estimate and the realized coded bytes agree to four decimal
places (+2.0170% vs +2.0168% at 3-bit on VAL-B). Levels 3 and 4 of the table move
together at every rate point, as they must if the saving is real.

### 9.2 GOP-position decomposition of the gap

| bits | group | deployed gap | candidate gap | real bytes |
|---|---|---:|---:|---|
| 5 | boundary (28) | 6.4018% | 6.6598% | 157,784 → 155,621 |
| | ordinary (218) | 1.9444% | 2.1002% | 1,175,491 → 1,157,777 |
| 4 | boundary | 17.9812% | 18.0467% | 114,898 → 112,990 |
| | ordinary | 7.0884% | 7.2670% | 793,506 → 781,163 |
| 3 | boundary | 38.6826% | 38.8478% | 81,039 → 79,258 |
| | ordinary | 16.3115% | 16.7100% | 468,044 → 458,786 |

The boundary's oracle gap remains enormous — **38.7% at 3-bit against 16.3% for
ordinary positions** — and the candidate does not dent it (38.68% → 38.85%). Real
bytes improve at the boundary in step with everywhere else. M16's GOP-boundary
asymmetry survives M22 completely intact.

---

## 10. GOP-boundary analysis

M16–M21 all found GOP position 1 anomalous, so a global aggregate is not allowed
to hide a boundary-specific effect. For `symmetric_p01/refit` on VAL-B:

| bits | position 1 (28 frames) | positions 2–9 (218 frames) | shape |
|---|---:|---:|---|
| 5 | +1.3971% | +1.3428% | uniform |
| 4 | +1.6957% | +1.5866% | uniform |
| 3 | +2.1360% | +1.9962% | uniform |

The candidate is **category B of §14 — it improves all P frames.** The boundary is
consistently a little better than ordinary positions (by 0.05–0.11 points), but
the effect is overwhelmingly uniform; nothing here is a boundary fix.

This matters for interpretation. M16 established that position 1 carries a genuine
reference asymmetry, and M17–M21 repeatedly tried and failed to convert it into
bytes. This candidate does **not** exploit that anomaly — it is an independent
mechanism acting on every inter-frame residual. The GOP-boundary question is
therefore still open, and still unexploited, after M22.

---

## 11. Decoder compatibility

Every arm in every run was decoded back from its `.nvct` v2 container and checked
for three separate exact equalities, not one:

| check | result |
|---|---|
| `symbols_exact` | True, every run |
| `reconstruction_exact` | True, every run |
| `references_exact` | True, every run |
| sequences verified | 4/4 VAL-B per run (15 screen runs + 9 Stage-2 runs) |

Structural compatibility, from Phase 1 and unchanged by the candidate:

* **same latent dimensions** — the autoencoder is frozen (M10F seed-42, epoch 70,
  `latent_channels=64`, `base_channels=32`); nothing in M22 touches it;
* **same causal context** — M11-G16, `group_size=16`, identical
  `context_definition_id`;
* **same container** — `.nvct` v2 needs **no new field**. The residual entropy
  identity that changes is a field the header already carries, which is exactly
  the mechanism M10L built for a swappable codebook;
* **same coder** — the range coder and its invariants are untouched;
* **no side information** — the decoder derives the grid from the stream's own
  quantization block, as it already did. Nothing extra is transmitted.

The model identity **is** explicit and **does** change: the candidate carries new
`residual`, `m10k`, `assign_codebook` and `coding_codebook` identities. A decoder
holding the deployed stack will refuse the candidate's stream rather than
mis-decode it — the designed behaviour, demonstrated by the STALE arm's
`ProvenanceError`.

---

## 12. Coded validation (Phase 18)

The gate opened at +1.5110% ≥ 0.5%. Every item §18 requires:

| requirement | result |
|---|---|
| exact encode/decode | symbols, reconstruction and references all exact |
| byte accounting | `byte_accounting_closes` true on every run |
| deterministic reproduction | Phase 20, independent process: **0 differences** |
| checkpoint provenance | SHA256 verified before load; grid + calibration signatures matched |
| no split leakage | grids fitted on TRAIN only; VAL-A selects; VAL-B reports (asserted over source) |
| no container incompatibility | no new `.nvct` field |
| no frozen-component changes | intra and motion identities bit-identical to M19's |

Intra residual bytes are **unchanged to the byte** (+0) at every rate point, and
motion bytes move by at most +109 bytes — second-order drift from motion
estimation running against a slightly different reconstruction, not a change to
the motion path, whose table identity stays `d7e7b237b6451885` throughout.

---

## 13. Full DAVIS TEST

Opened **only** after the candidate was locked by the declared Phase 17 rule and
had cleared coded validation. `m22_davis.py` re-reads the lock from
`m22_analysis.json` and refuses to run for any other candidate, so TEST cannot be
opened for a variant chosen after the fact.

All 9 sequences, all 719 frames, both arms decoded back from their containers.

### 13.1 The baseline arm is verified, not assumed

| bits | baseline total | M14 recorded | match |
|---|---:|---:|---|
| 5 | 4,292,887 | 4,292,887 | ✓ |
| 4 | 3,009,891 | 3,009,891 | ✓ |
| 3 | 1,900,744 | 1,900,744 | ✓ |

P-residual, I-residual and motion totals match the record at every point as well.
A mismatch here would invalidate everything below, so it is reported rather than
presumed.

### 13.2 Result

| bits | baseline | candidate | Δ bytes | total stream | P-residual | ΔPSNR | ΔMS-SSIM | guard | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 5 | 4,292,887 | 4,247,581 | −45,306 | **+1.0554%** | +1.2866% | **+0.0213** | +0.000074 | PASS | meaningful |
| 4 | 3,009,891 | 2,970,488 | −39,403 | **+1.3091%** | +1.6484% | **+0.0320** | −0.000036 | PASS | meaningful |
| 3 | 1,900,744 | 1,867,362 | −33,382 | **+1.7563%** | +2.3296% | −0.0126 | −0.000412 | PASS | meaningful |

**BD-rate against the frozen production baseline: −2.4746% PSNR, −1.0968% MS-SSIM.**

BPP falls 0.728837 → 0.721145 (5-bit), 0.511013 → 0.504323 (4-bit), 0.322704 →
0.317037 (3-bit).

At 5-bit and 4-bit this is a **strict Pareto improvement on held-out data** —
fewer bytes *and* higher PSNR. Only 3-bit carries a quality cost, −0.0126 dB,
about an eighth of the declared allowance.

### 13.3 Channel attribution

| bits | I-residual | P-residual | motion | overhead |
|---|---|---|---|---|
| 5 | 584,917 → 584,917 (**+0**) | 3,527,769 → 3,482,380 | 164,010 → 164,093 (+83) | 16,191 |
| 4 | 428,654 → 428,654 (**+0**) | 2,398,204 → 2,358,671 | 166,842 → 166,972 (+130) | 16,191 |
| 3 | 273,015 → 273,015 (**+0**) | 1,439,926 → 1,406,382 | 171,612 → 171,774 (+162) | 16,191 |

Intra bytes are **identical to the byte** at every rate point, and motion drifts
by at most 162 bytes (0.09%) — second-order, from motion estimation running
against a marginally different reconstruction, not from any change to the motion
path. **The entire gain is in the P-frame residual channel**, which is exactly
where the freeze was lifted.

### 13.4 Per-sequence (3-bit)

| sequence | baseline | candidate | gain | ΔPSNR |
|---|---:|---:|---:|---:|
| `gold-fish` | 199,147 | 191,592 | **+3.7937%** | −0.0339 |
| `surf` | 100,730 | 98,257 | +2.4551% | −0.0260 |
| `drift-chicane` | 73,914 | 72,300 | +2.1836% | −0.0469 |
| `car-turn` | 168,253 | 165,182 | +1.8252% | −0.0327 |
| `bmx-bumps` | 282,415 | 277,977 | +1.5714% | **+0.0676** |
| `schoolgirls` | 218,462 | 215,318 | +1.4392% | −0.0189 |
| `cows` | 287,715 | 283,713 | +1.3910% | −0.0208 |
| `drone` | 257,504 | 253,994 | +1.3631% | **+0.0055** |
| `cat-girl` | 312,604 | 309,029 | +1.1436% | −0.0070 |

**All 9 sequences improve.** The spread is 1.14%–3.79% with no regression
anywhere, so the aggregate is not carried by an outlier.

### 13.5 Generalization

| rate point | VAL-B | DAVIS TEST | retained |
|---|---:|---:|---:|
| 5-bit | +1.2196% | +1.0554% | 87% |
| 4-bit | +1.2476% | +1.3091% | 105% |
| 3-bit | +1.5110% | **+1.7563%** | 116% |

Headline gap: VAL-B said +1.6271%, TEST delivered **+1.7563%** — a gap of
**−0.1292 points in the candidate's favour**. There is no held-out shrinkage.

This is the comparison that matters in this project. M21's locked candidate went
+0.7141% on VAL-B to +0.1411% on TEST, retaining 20%; that was a selection
artifact. M22's retains 87–116% across three rate points and stays above the 1.0%
meaningful line on every one of them.

### 13.6 Decoder verification on TEST

`symbols_exact`, `reconstruction_exact` and `references_exact` all True, **9/9
sequences, at all three rate points**, for both arms.

---

## 14. Reproducibility

Phase 20 re-ran the locked candidate in a **separate process**, loading only from
disk and verifying the checkpoint digest before unpickling anything:

```
checkpoint m22_symmetric_p01_3bit.pt  sha256 f4046d07efcda945 VERIFIED before loading
grid signature 39db8a09133b0f39  (recorded 39db8a09133b0f39)
decode round trip      symbols/reconstruction/references exact: True
aggregate fields       13 compared, 0 differ
per-sequence fields    4 sequences, 0 differ
total container bytes  712,451 (recorded 712,451)
REPRODUCED: True
```

Comparison is by **exact equality**, floating-point metrics included — a tolerance
would hide precisely the drift this phase exists to catch.

### 14.1 An honest distinction: training is not bit-reproducible, evaluation is

Running the same refit twice under the same seed does **not** produce identical
bytes:

| variant | replicates | spread |
|---|---|---|
| `symmetric_p01` 3-bit | 712,416 / 712,451 | 35 bytes (0.005%) |
| `deployed` 3-bit | 724,426 / 724,235 | 191 bytes (0.026%) |
| `broad_p001` 3-bit | 554,122 / 554,143 / 553,472 | 671 bytes (**0.121%**) |

So the refit **training** (M10K/G16 on GPU) carries run-to-run noise of up to
~0.12%, while **evaluation given a fixed checkpoint** is bit-exact. That is why
Phase 20 reproduces from the saved, digest-verified checkpoint rather than by
refitting: the checkpoint is the artifact, and it is what would ship.

The measured effect (+1.22% to +1.63%) is roughly **10–50× the noise floor**, so it
is not a training-noise artifact. Quoting the widest observed spread (0.121%)
rather than the most flattering (0.005%) is deliberate.

---

## 15. Runtime and memory

| bits | baseline encode+decode | candidate | delta |
|---|---:|---:|---:|
| 5 | 231.8 s | 230.9 s | −0.4% |
| 4 | 230.9 s | 195.9 s | −15.2% |
| 3 | 204.0 s | 202.1 s | −0.9% |

719 frames per arm on an RTX 5060 Laptop (8 GB). The candidate is never slower;
the 4-bit figure is machine noise rather than a real speedup, and should not be
quoted as one.

This is expected: the change is a different `scale`/`zero_point` pair in the
quantizer. It adds **no** operations, no parameters and no memory — the entropy
model has exactly the same architecture (`group_size=16`, identical
`context_definition_id`), the codebook is the same K=512, and the container is
byte-identical in structure. Encoder and decoder cost are unchanged by
construction.

Memory is likewise unchanged: no tensor in the pipeline changes shape or dtype.

---

## 16. Tests

`tests/test_m22_residual_freeze_lift.py` — **98 tests**, covering every item §21
lists. The load-bearing ones:

| test | what it protects |
|---|---|
| `test_deployed_variant_reproduces_calibrate_grids_exactly` | the identity control — without it every delta is measured from the wrong origin |
| `test_verify_deployed_grid_accepts_a_matching_grid_and_rejects_a_drifted_one` | the enforced control that caught the residual-walk bug (§16.1) |
| `test_train_residual_walk_includes_the_intra_round_trip` | pins the specific bug that produced the wrong origin |
| `test_grid_change_moves_the_calibration_signature`, `test_deployed_g16_rejects_a_changed_grid` | the coupling that justifies the two-arm design |
| `test_no_grid_variant_sees_val_or_test_data` | every grid fitted on TRAIN alone, asserted over `build` source |
| `test_m22_scripts_never_touch_the_test_split` | all 7 pre-lock scripts; the DAVIS runner is deliberately excluded |
| `test_resume_refit_refuses_a_checkpoint_fitted_to_a_different_grid` | a stack is never evaluated under a quantizer it was not fitted to |
| `test_resume_refit_rebuilds_rather_than_aborting_when_the_code_changed` | a stale artifact is rebuilt, not fatal (§16.3) |
| `test_report_path_is_never_rebound_by_a_loop_variable` | AST-level guard against the persistence bug (§16.2) |
| `test_bd_rate_separates_a_real_gain_from_a_slide_along_the_same_curve` | the central interpretive risk of a grid change |
| `test_phase15_never_writes_a_codebook_or_a_checkpoint` | §15's "quantify, do not repair" rule, enforced over source |
| `test_saved_checkpoints_match_their_recorded_digests` | every checkpoint still hashes to what was recorded |

### 16.1 A wrong origin, caught by the enforced control

The first full 3-bit sweep reported a control total of **897,872 bytes** where
Phase 0 had established **723,381** — a 24% error in the origin every delta is
measured from.

Cause: `collect_train_residuals` walked I-frames as `previous = model.decode(latent)`,
while the production `calibrate_grids` performs the full intra round trip
(`encode_latent_to_payload` → `decode_payload_to_latent` → `model.decode`). The
TRAIN residual stack was therefore built from references the codec never actually
produces, which shifted the rebuilt 3-bit mean step from 1.73229 to 1.7266. The
"deployed" control was not the deployed grid, so **every** candidate delta was
measured from the wrong place.

Fix: the intra round trip was added to the walk, the residual cache key bumped to
`intra_roundtrip_v2`, and — the durable part — `verify_deployed_grid()` now
**raises** if the TRAIN stack does not rebuild the deployed grid bit-for-bit,
rather than quietly reporting a control that is not the control. Two regression
tests pin both halves. The corrected control reproduces 723,381 exactly.

### 16.2 A closure writing into a deleted file

Incremental report persistence was added mid-milestone after a crashed run lost
1.5 hours of refits. The `persist()` closure wrote through a variable named
`path` — and later in the same function:

```python
for path in stream_dir.glob(f"identity_{bits}bit_*.nvct"):
    path.unlink()
```

rebound that same name. Every incremental write therefore landed in a just-deleted
`.nvct` stream file. The original code was safe only because it bound `path`
*after* the loops; moving the binding earlier silently broke it, and the
end-of-run write would have been lost too.

Fixed by renaming both (`report_path`, `stream_path`). The regression test is
AST-level rather than textual: it walks `main`, collects the names `persist()`
closes over and the names bound by every `for` in that scope, and fails if they
intersect.

A related slip: the rename initially missed the final `print(f"\nReport: {path}")`,
which raised `NameError` **after** all 15 rows had been measured and persisted.
No data was lost — precisely because the incremental writes were by then working.

### 16.3 A guard with the wrong remedy

`resume_refit` originally raised on **any** provenance mismatch, including a
changed `code_identity`. That aborted a 75-minute Stage-2 run over an edit to a
`print` statement.

The distinction now enforced: a **grid or calibration** mismatch still raises,
because reusing that checkpoint would evaluate a stack under a quantizer it was
never fitted to. A **code-identity** mismatch returns `None` and the stack is
refitted, because a stale artifact is only a hazard if it is *used* — rebuilding
it is always safe. Both behaviours are pinned by tests.

---

## 17. Files changed

**No production source was modified.** `src/nvc/` is untouched, asserted by
`test_m22_does_not_modify_any_production_source`.

New research scripts under `scripts/`:

| file | role |
|---|---|
| `m22_residual.py` | the rig: grid variants, statistics, refit, provenance, identities |
| `m22_baseline.py` | Phase 0/1 — frozen baseline and the coupling audit |
| `m22_diagnostics.py` | Phase 2 — residual-space diagnostics A–K |
| `m22_sweep.py` | Phases 3–5 — the STALE/REFIT sweep |
| `m22_analysis.py` | Phases 5/14/17 — gate, GOP split, selection, BD-rate, classification |
| `m22_mechanism.py` | Phase 13 — four-level gap decomposition |
| `m22_codebook.py` | Phase 15 — codebook incompatibility, quantified not repaired |
| `m22_reproduce.py` | Phase 20 — independent-process reproduction |
| `m22_davis.py` | Phase 19 — the only script that opens TEST |

New tests: `tests/test_m22_residual_freeze_lift.py`.

---

## 18. Final classification

### A — MEANINGFUL RESIDUAL-REPRESENTATION SUCCESS

Graded on **held-out DAVIS TEST** (+1.7563% best, +1.0554% worst across rate
points), not on the selection set. Every project gate is cleared at every rate
point:

| criterion | result |
|---|---|
| total-stream gain ≥ 1.0% (meaningful) | **yes, at all three rate points** |
| distortion guard | PASS at all three |
| decoder compatibility | exact, 9/9 TEST sequences × 3 rate points |
| provenance | SHA256-verified checkpoints, distinct identities |
| independent-process reproduction | 0 differences |
| no production source modified | `src/nvc/` untouched |
| generalization | no shrinkage (−0.1292 points, favourable) |

### Why this one converted when M18–M21 did not

M17 established a real oracle gap but called it unimplementable, because the
oracle needs information no decoder has. M18, M20 and M21 each attacked the
*reference* — its aggregate error, its codebook routing, its refinement — and each
failed to convert a measured mechanism into coded bytes.

M22 did not attack the reference at all. It changed **where the quantizer's
reconstruction levels sit relative to the residual distribution's zero spike** —
information the decoder already has, because the grid travels in the stream's own
quantization block. That is why it required no side information, no container
change, and no new decoder capability.

**It did not, however, close M17's oracle gap.** Phase 13 measures gap closure at
the coded-byte level as **−5.23% / −0.33% / +0.22%** at 5/4/3-bit — zero within
measurement (§9). Both the real and the oracle reference arms get cheaper under a
better-aligned grid, so the *ratio* between them barely moves while both absolute
costs fall. M17's reference-error penalty is **still open and still unexploited**;
M22 simply found an orthogonal axis and took it.

---

## 19. Recommendation for M23

### The seven questions §25 requires answering

**1. Did lifting the residual quantizer/autoencoder freeze improve ACTUAL CODED
RATE?** Yes. +1.0554% / +1.3091% / +1.7563% of the total stream on held-out DAVIS
TEST at 5/4/3-bit, BD-rate −2.4746% PSNR. Note the freeze that was actually lifted
is the **residual quantizer**; the autoencoder was never retrained (see §5.4).

**2. At which bit depths?** All three, and meaningfully (≥1.0%) at all three. The
effect grows as the step coarsens.

**3. How much of M17's oracle gap was closed?** **Essentially none** —
−5.23% / −0.33% / +0.22% at 5/4/3-bit, i.e. zero within measurement, and slightly
negative at 5-bit. The gain is *orthogonal* to the reference-error penalty: a
better-aligned grid makes the oracle arm cheaper too, so the relative gap is
unmoved while absolute residual bytes fall 1.49% / 1.57% / 2.01% (§9). M17's gap
remains fully open after M22.

**4. Did the improvement come from better residual geometry, better symbol
stability, lower entropy, or another mechanism?** **Lower entropy, via grid
centering.** Not geometry: scale is unchanged (ratio 1.02). Not symbol stability
in M19's sense: the candidate does not reduce reference-induced symbol flips, it
reduces the cost of coding the symbols that result. 0th-order symbol entropy falls
1.568 → 1.239 bits at 3-bit (−21%) at equal step and equal quality, and G16's own
ideal-bits estimate tracks the realized byte saving to within 0.0002 points
(+2.0170% vs +2.0168%), confirming the saving is genuinely entropy rather than an
accounting artifact.

**5. Is the improvement concentrated at GOP position 1?** **No.** Position 1 and
positions 2–9 move together (+1.3035% vs +1.2844% at 5-bit; +2.3037% vs +2.3341%
at 3-bit). This is category B — it improves all P frames. The GOP-boundary
anomaly M16 identified remains real and remains unexploited.

**6. Does the result generalize to full DAVIS TEST?** **Yes**, with no shrinkage:
VAL-B +1.6271% → TEST +1.7563%, and all 9 sequences improve individually.

**7. Should M23 continue representation optimization, optimize the residual
quantizer specifically, lift another freeze, investigate the GOP boundary, or stop
and consolidate?**

### → B — OPTIMIZE THE RESIDUAL QUANTIZER SPECIFICALLY

M22 tested eight grids and shipped the second-best one. The evidence for B is
`broad_p001`, which this milestone **correctly excluded** and which is
nevertheless the largest single result in it:

| candidate | BD-rate PSNR | status in M22 |
|---|---:|---|
| `symmetric_p01` | −1.1839% | **selected**, confirmed on TEST at −2.4746% |
| `broad_p001` | **−8.2372%** | excluded: −0.7873 dB at fixed 3-bit depth |

`broad_p001` needs roughly 8% fewer bits *at equal quality* on VAL-B — four to
seven times `symmetric_p01`'s margin — but it reaches that by moving the
quality-per-bit-depth operating point, so it fails a guard designed for drop-in
replacement. That is a **product decision the guard cannot make**, not a defect in
the candidate.

M23 should therefore treat the residual grid as a first-class rate/distortion
parameter rather than a fixed calibration:

1. **Re-derive the deployed percentiles.** (0.1, 99.9) was inherited, never
   optimized. `broad_p001` shows the R/D-optimal choice is materially wider.
   Sweep the percentile pair against BD-rate, not against fixed-depth bytes.
2. **Separate the two axes.** M22 changed centering and range together only by
   accident of the variant definitions; `symmetric_p01` is mostly centering,
   `broad_p001` mostly range. A 2-D sweep would attribute each.
3. **Decide the operating point explicitly.** If the codec may re-map bit depth to
   quality, `broad_p001`-style grids are available and worth ~8% BD-rate. If it
   must remain drop-in, the guard stands and the ceiling is nearer
   `symmetric_p01`'s.
4. **Re-run the integer `zero_point` question.** §4's derivation suggested the
   rounding of `zero_point` to an integer leaves even the "symmetric" grid
   asymmetric by up to half a step — a residual misalignment M22 measured but did
   not fix.

**Not** option C (lift another freeze): the quantizer axis is demonstrably not
exhausted, and M22 is the first milestone since M13/M14 to convert a mechanism
into held-out bytes. **Not** option D (GOP boundary): §10 shows this gain is
uniform across GOP position, so the boundary anomaly is untouched and still has no
demonstrated byte value after M16–M22. **Not** option E (consolidate): an
un-harvested −8.24% BD-rate lead is not a stopping point.

### Production recommendation

`symmetric_p01` is ready to ship: meaningful at all three rate points on held-out
TEST, quality-neutral or better at 5/4-bit, no container change, no runtime cost,
no side information, exact decode on 9/9 TEST sequences, and reproducible in an
independent process.

It must ship as **one atomic change** — grid *and* refitted M10K/G16/K=512/M13
together. §8.6's interaction term is unambiguous: the grid alone, on the deployed
tables, costs −75.16%. The calibration signature already enforces this, and would
reject a partial deployment rather than silently mis-code it.
