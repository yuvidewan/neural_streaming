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
| M10L | refit the 512-prototype codebook | not started |
| M11 | channel-autoregressive context model (G16) | not started |
| M13 / M14 / M22 | codebook, motion-table and residual-grid recalibrations | not started |
| `benchmark_parity.py` | the real full-video number | not started |

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
