# Refitting the P-frame stack to the Stage 1 transform

The deployed P-frame machinery is all fitted to the 593k transform's latent and
is refused by the provenance check against a new one. This directory holds the
refit against the 64-channel GDN/residual transform with QAT
(`outputs/stage1_vimeo/colab_run4_64ch_qat/best.pt`).

Why it is worth doing: that transform is worth **−280 BD-rate points on full
video** through a fixed entropy stack
(`outputs/benchmarks/fullvideo_transform/README.md`). The refit converts a
demonstrated transform win into a deployable codec.

Order, each step warm-starting or depending on the one above:

| step | what | status |
|---|---|---|
| **M10K** | z_ref-conditioned entropy model | **done, gate passed** |
| **M10L** | refit the 512-prototype codebook | **done, gate passed** |
| **M11** | channel-autoregressive context model (G16) | **done, gate passed** |
| **M13** | codebook frequency recalibration | **done, gate passed** |
| **M14** | motion-table recalibration | **done, motion passed, intra rejected** |
| M22 | `symmetric_p01` residual grid | **blocked by ordering — see below** |
| `benchmark_parity.py` | the real full-video number | ready for the pre-M22 arm |

Nothing here overwrites the deployed artifacts: the deployed M10K lives in
`outputs/m10k_learned_entropy/` and is untouched.

## M10K — gate passed, 2026-10-24

`m10k_64ch_qat/`, 536 train and 179 validation P-frames, 12,672 parameters,
20 epochs per rate point. Held-out validation cross-entropy, bits/symbol:

| bits | M10H marginal | M10J bucketed | M10K learned | K vs J |
|---|---|---|---|---|
| 5 | 3.40547 | 3.36809 | 3.32986 | **+1.14%** |
| 4 | 2.43927 | 2.40844 | 2.37581 | **+1.35%** |
| 3 | 1.55273 | 1.52970 | 1.49572 | **+2.22%** |

Gate threshold is >0.5% over M10J on held-out validation. Passed at every rate
point.

### It transfers at essentially the same value

The question this step actually had to answer was not "does M10K work" but "does
the learned-context machinery still earn its keep on a different transform's
residuals". It does, almost identically:

| bits | K vs J, old 593k transform | K vs J, new transform |
|---|---|---|
| 5 | +1.19% | +1.14% |
| 4 | +1.44% | +1.35% |
| 3 | +1.81% | **+2.22%** |

Slightly worse at 5-bit, slightly better at 3-bit, same order throughout. So
M10L and M11 are worth refitting rather than being likely dead ends — which was
the real risk in committing to this chain.

### The new transform's residuals are HARDER per symbol

Marginal residual entropy rose at every depth:

| bits | old | new | change |
|---|---|---|---|
| 5 | 3.21763 | 3.40547 | **+5.8%** |
| 4 | 2.26931 | 2.43927 | **+7.5%** |
| 3 | 1.41047 | 1.55273 | **+10.1%** |

Each residual symbol carries more information than it used to. That is the
per-symbol counterpart of the full-video result, where the new transform spent
6–9% more bytes for +2.2 to +3.0 dB: it is not that the transform produces
cheaper residuals, it is that it produces a better reconstruction and its
residuals are correspondingly less redundant. Worth remembering when reading the
later steps — the entropy stack is working against a harder signal here, so
equal *percentage* gains are a slightly better result than they look.

## Files

- `m10k_64ch_qat/learned_entropy_{3,4,5}bit.pt` — the refitted models
- `m10k_64ch_qat/offline_gate.json` — the gate report
- `m10k_gate.log` — the run

## M10L — gate passed, K=512 selected, 2026-10-24

`m10l_64ch_qat/`, fitted on TRAIN predictions from the refitted M10K, selected on
validation, 536 train / 179 validation P-frames. The codebook sizes 16 through
512 were all swept and all passed; the pre-registered rule takes the best
held-out rate among candidates clearing the runtime bar, and that is K=512 at
every rate point — the same size the deployed codec uses.

| bits | M10K ideal | K=512 | cost vs M10K | vs M10J | table time | prototypes used |
|---|---|---|---|---|---|---|
| 5 | 3.32987 | 3.33076 | **+0.03%** | +1.11% | **−94.0%** | 501 / 512 |
| 4 | 2.37582 | 2.37693 | **+0.05%** | +1.31% | **−88.2%** | 503 / 512 |
| 3 | 1.49574 | 1.49635 | **+0.04%** | +2.18% | **−79.3%** | 506 / 512 |

`l1` was swept alongside `code_length` as the assignment metric and lost, so
`code_length` is selected — again matching the deployed choice, and chosen on
4-bit validation before any test data was read.

### It behaves exactly as it does on the deployed transform

On the old transform the codebook retained M10K's gain to within +0.02% while
removing 70–88% of residual-coding time. Here it retains it to within
+0.03–0.05% and removes 79–94% of table time. The approximation is as tight on
this transform's distributions as on the old one, so nothing about the wider,
harder residuals defeats the prototype representation.

Over 500 of the 512 prototypes are used at every rate point, so K=512 is not
over-provisioned — the distributions genuinely occupy the codebook.

Artifacts: `codebooks/codebook_{3,4,5}bit_K512_code_length.json` are the ones
M11 and the deployed path consume.

## M11 — gate passed, G=16 selected, 2026-10-24

Two steps: `m11_offline_gate.py` establishes that causal residual context carries
information M10K's z_ref did not already provide, then `m11_train.py` trains the
networks and arithmetic-codes every VAL-B P-frame to check the gain survives in
real bytes.

### Offline gate

Chosen context `neighbourhood_x_channel` — **the same one the deployed codec
chose** — net of the random control, on VAL-B:

| bits | net gain over M10L | random control |
|---|---|---|
| 5 | +1.774% | +0.001% |
| 4 | +4.971% | +0.013% |
| 3 | +8.032% | +0.000% |

Controls at ~0.00% are the validity check: the gain is causal signal, not
estimator bias. Threshold is 0.5% at every point.

### Trained models, real coded bytes

Both transforms select **G=16** under the pre-declared rule (largest group size
keeping ≥50% of the G=1 VAL-A context gain), so the deployed four-step decode
schedule carries over unchanged. Round-trip exact at every rate point, 13,536
parameters, identical to the deployed model.

| bits | old transform | new transform | change | decode ms/P |
|---|---|---|---|---|
| 5 | +6.408% | **+5.346%** | −1.06 | 6.73 |
| 4 | +7.120% | **+5.953%** | −1.17 | 7.71 |
| 3 | +6.856% | **+5.315%** | −1.54 | 6.40 |

(coded bytes against M10L, through the codebook path)

### The context is worth less here, and that is the expected direction

M11 earns about 1.1–1.5 points less on the new transform, and the offline gate
said the same thing more strongly (2–3 points less). Both point the same way: a
better transform decorrelates more, so there is less cross-channel and spatial
redundancy left for a causal context model to collect. The transform has taken
some of what M11 used to.

This matters for the projection in
`outputs/benchmarks/fullvideo_transform/README.md`, which estimated the learned
stack would be worth about 77.8 BD-rate points on the new transform because that
is what it is worth on the old one. M11 is the largest single entropy
contributor and it is down by roughly a fifth, so **that estimate is optimistic**
and should be replaced by the `benchmark_parity.py` measurement rather than
quoted.

Artifacts: `m11_64ch_qat/m11_training.json`, plus the G16 entropy models and
their K=512 codebooks.

## M13 — gate passed, 2026-10-25

Recalibrates the K=512 codebook's coding frequencies against TRAIN's true symbol
histogram, with assignments held fixed, so symbols and reconstructions are
unchanged and only coding cost falls. TRAIN fits counts, VAL-A selects the
smoothing strength, VAL-B reports.

| bits | H_old | H_new (integer table) | gain | verdict |
|---|---|---|---|---|
| 5 | 3.07555 | 3.04604 | +0.960% | marginal |
| 4 | 2.19005 | 2.13296 | **+2.607%** | meaningful |
| 3 | 1.39555 | 1.33688 | **+4.204%** | meaningful |

Selected smoothing strength 256 at both lower rate points. The deployed M13 was
worth +1.4–4.0% of real coded bytes, so this is the same order — with 5-bit
landing just under the 1% "meaningful" bar rather than above it, which is the one
place this refit is slightly weaker than the original.

## M14 — motion passed, intra rejected, 2026-10-25

Recalibrates the motion and intra symbol tables on broader TRAIN coverage (576
frames across all 72 TRAIN sequences, against the deployed calibration's ~400
from the first few sequences in manifest order).

| bits | intra gain | verdict | motion gain | verdict |
|---|---|---|---|---|
| 5 | +0.2528% | weak | **+13.011%** | meaningful |
| 4 | +0.2864% | weak | **+12.536%** | meaningful |
| 3 | +0.2497% | weak | **+11.609%** | meaningful |

The intra arm is correctly rejected as weak at every rate point — exactly as it
was on the deployed transform, so that negative result reproduces. The motion
arm is **better** here than the deployed +9.2% on its channel. Motion is a small
share of the stream (3.6–7.8% for this transform, per
`outputs/benchmarks/fullvideo_transform/README.md`), so expect a few tenths of a
percent of total bytes, as it was deployed.

## M22 cannot go last — an ordering mistake, found before it cost anything

I sequenced this refit in milestone order: M10K, M10L, M11, M13, M14, M22. That
is the wrong order, and `scripts/m22_residual.py` says why in its own docstring:

> the residual quantizer has no interface at which it can be changed in
> isolation. Changing it invalidates, in order, the M10K entropy model, the
> M11-G16 model trained from it, the K=512 assignment codebook fitted to G16's
> predictions, and the M13 coding frequencies fitted to that codebook's
> assignments.

`m10l_evaluate.calibration_signature` hashes the residual quantizer's scale and
zero_point, and `m11_evaluate.check_provenance` refuses any M11 checkpoint whose
digest does not match. So M22's `symmetric_p01` grid is not a sixth recalibration
to append — it is a different grid that invalidates the four steps above it.

### What that means for what is already done

It is not wasted. `benchmark_parity.py` ships two arms, and they map cleanly:

- **`nvc_deployed`** — the stack on `calibrate_grids`' plain percentile residual
  grid. **This is exactly what the refit above has rebuilt**, and it is complete
  and internally consistent.
- **`nvc_m22`** — the same stack on the `symmetric_p01` grid, with M10K, M10L,
  M11 and M13 refitted against it. The paper's Table I quotes this arm.

So the refit so far is a complete `nvc_deployed` arm for the new transform, and
`benchmark_parity.py` can measure it now. The M22 arm needs the residual grid
changed first and then M10K → M10L → M11 → M13 re-run against it, which is
roughly the same three hours again for the ~1.06–1.76% of bytes (BD-rate −2.47%)
that M22 was worth.

M14 is unaffected either way: M22 explicitly freezes the motion table, so the
motion recalibration is independent of the residual grid.
