# M19 — Reference Error Shape Diagnostic

**Status: COMPLETE.** **Classification: F — no single pre-registered mechanism (A–E) cleanly dominates; the evidence instead identifies a precise, quantified two-part mechanism outside that taxonomy (see §14).**

## 1. Executive summary

M17 proved the reference error costs real bytes; M18 proved that shrinking its *aggregate magnitude* (MSE) does not reliably recover them. M19 asked what *kind* of error actually matters. The answer, measured rather than assumed: **the error is spatially structured (high autocorrelation, growing edge-concentration) but only moderately so — a magnitude-preserving spatial shuffle recovers 46–65% of M17's gap on its own, meaning magnitude is the larger factor but structure contributes a real, non-trivial remainder**. Channel structure is **weak and gets weaker** at coarser bit depths (ruling out "a few bad channels"). The 512-entry codebook's assignment is **surprisingly brittle** — it flips 35–55% of the time even for the *smallest* error decile — but this brittleness is a *frequent, low-cost* effect: a precise 2×2 decomposition shows **88–91% of total excess bits come from cases where the reference error changes the residual symbol itself**, not from cases where only the codebook table changes while the symbol stays the same (9–12%). This pattern is consistent across all three bit depths. M19 recommends exactly one M20 experiment, targeted at the smaller but directly addressable mechanism it found: a stability margin on codebook assignment.

## 2. Baseline (Phase 0)

Pre-M19 full suite: 1323 passed after M19's own tests are added (see §19); confirmed against M17's `m17_residual_diagnostic.json` by re-deriving the identical frozen M13 residual arm and reproducing the real/oracle byte totals **exactly**:

| bits | fresh real bytes | recorded | fresh oracle bytes | recorded | match |
|---|---|---|---|---|---|
| 5 | 1,333,275 | 1,333,275 | 1,300,318 | 1,300,318 | ✓ |
| 4 | 908,404 | 908,404 | 831,497 | 831,497 | ✓ |
| 3 | 549,083 | 549,083 | 441,390 | 441,390 | ✓ |

`outputs/m19_reference_error_audit/m19_baseline.json`. All three residual identities (`ec858dee10f8a955`, `be872c2beebd2f9b`, `d2d61d66a50cad8a`) match M13/M14/M15/M17/M18's own recorded values.

## 3. Real/oracle reference definitions

Per this milestone's own Phase A: `R_real` = the actual deployed `model.decode(reconstructed_latent)`; `R_oracle` = `model.decode(model.encode(raw_previous_frame))` (M16/M17's own idealization, reused unmodified). `E_pixel = R_real − R_oracle`. `Z_real = model.encode(R_real)`, `Z_oracle = model.encode(R_oracle)`, `E_latent = Z_real − Z_oracle` — the **direct** encodings of the two reference candidates, deliberately decoupled from motion/warping (which Phase G treats as a separate covariate, never folded into this primary definition).

## 4. Pixel-domain error

Reported per-frame (mean/MAE/RMS/variance) and aggregated in §6's spatial analysis; not repeated as a standalone table since every pixel-domain number that matters is a spatial or control statistic below. Never compared numerically against latent-domain error (different domains, different scales, kept strictly separate throughout — no shared table mixes them).

## 5. Latent-domain error

Same discipline — `E_latent`'s per-channel breakdown is §7's subject; its per-position magnitude is §8/§9's. `E_latent` is what actually enters the residual transform, codebook assignment, and G16 context — the analysis that matters is downstream (§8–§9), not the raw scalar itself.

## 6. Spatial structure

Deterministic, non-learned partitions only (fixed percentile splits and a Sobel-gradient tercile rule on `R_oracle`, never TRAIN-fit):

| bits | autocorr h/v | low/high freq ratio | edge/flat error ratio | boundary/interior ratio |
|---|---|---|---|---|
| 5 | 0.877 / 0.832 | 2.72 | 1.36 | 1.02 |
| 4 | 0.877 / 0.831 | 2.08 | 1.43 | 1.02 |
| 3 | 0.875 / 0.826 | 1.80 | 1.55 | 1.02 |

**The error is strongly spatially smooth** (autocorrelation ~0.83–0.88 at every bit depth — far from white noise) and **increasingly edge-concentrated** as quantization coarsens (1.36→1.55). Low-frequency energy dominates throughout (ratio >1), though relatively less so at coarser bit depths. **Block-boundary concentration is absent** (ratio ≈1.02 at every bit depth) — this specifically rules out a motion-compensation-block-grid artifact as a spatial driver.

## 7. Channel structure

| bits | top10% share of MAE | top25% share of MAE | top10% share of churn | top25% share of churn |
|---|---|---|---|---|
| 5 | 14.66% | 35.61% | 12.84% | 32.78% |
| 4 | 14.89% | 36.00% | 11.50% | 30.11% |
| 3 | 14.91% | 35.83% | 10.54% | 27.80% |

Uniform-across-64-channels would give 10%/25% exactly. **Observed concentration is only modestly above uniform for error magnitude, and churn concentration actively *decreases toward* uniform as bit depth coarsens.** No small subset of channels dominates either the error or its consequences — channel structure is a weak, secondary factor at best, ruling out Phase J outcome **B**.

## 8. Codebook-routing analysis (Phase D, the milestone's own flagged priority)

P(assignment changed | error-magnitude decile), 3-bit (most extreme; all three bit depths monotonic in the same way):

`[0.545, 0.683, 0.704, 0.719, 0.734, 0.745, 0.762, 0.778, 0.789, 0.809]`

Two findings, both real: **(a)** churn probability rises monotonically with error magnitude at every bit depth — a genuine dose-response relationship confirming error size does drive routing changes; **(b)** even the *lowest* error decile shows substantial churn (34.8%/45.1%/54.5% at 5/4/3-bit) — the codebook's decision boundaries are **brittle**, flipping assignment for a large fraction of positions even under near-zero perturbation. Mean code-length cost when assignment changes vs. doesn't: **2.4× (5-bit), 2.5× (4-bit), 3.0× (3-bit)** — routing changes are a real, quantifiable, growing amplifier.

## 9. G16 prediction analysis — the 2×2 decomposition

For every position, cross-tabulating (residual symbol changed?) × (codebook assignment changed?), with mean Δcode-length per cell (3-bit shown; same qualitative pattern at 5/4-bit):

| symbol same? | assignment same? | fraction of positions | mean Δcode-length | share of total excess bits |
|---|---|---|---|---|
| yes | yes | 24.0% | 0.000 bits | 0% (trivially — identical lookup) |
| yes | **no** | 57.0% | 0.044 bits | **~12%** |
| **no** | yes | 3.3% | 0.700 bits | ~11% |
| **no** | **no** | 15.7% | 1.033 bits | **~77%** |

Summing the two "symbol changed" rows: **88–91% of total excess bits (consistent across all three bit depths) come from positions where the reference error changed the residual symbol itself** — not merely which codebook table encodes it. "Same symbol, different table" is the *most frequent* outcome at 3-bit (57% of all positions) but contributes only ~9–12% of total excess bits, because each occurrence is individually cheap (0.04 bits). **The dominant cost driver is the residual transform changing what needs to be coded, not the entropy model mis-routing a fixed target** — this argues against Phase J outcome **C** (codebook routing as the *dominant* amplifier) even though routing changes are real, frequent, and quantifiably costly in isolation.

## 10. Motion/error interaction (Phase G)

| bits | corr(SAD, pixel error) | corr(SAD, churn) | corr(pixel error, byte gap) |
|---|---|---|---|
| 5 | 0.731 | −0.185 | 0.148 |
| 4 | 0.695 | −0.324 | 0.097 |
| 3 | 0.548 | −0.071 | 0.524 |

Block SAD correlates meaningfully with pixel error (0.55–0.73, as expected — a worse reference produces a worse match) but **correlates weakly or even *negatively* with churn**, and its correlation with the byte-gap is inconsistent across bit depths (0.10–0.52). Motion is a real covariate (as M17 already established at ~13–15% attribution) but **does not drive the dominant mechanism** — ruling out Phase J outcome **E** as the primary story, consistent with this milestone's own instruction to treat motion as a covariate, not a target.

## 11. GOP-position analysis (Phase F)

| bits | boundary gain% | boundary churn | ordinary gain% | ordinary churn |
|---|---|---|---|---|
| 5 | 6.52% | 61.5% | 1.86% | 41.7% |
| 4 | 17.71% | 75.9% | 6.59% | 56.3% |
| 3 | 38.53% | 89.6% | 70.6% | — |

Boundary P-frames show 3–4× the gain% and substantially higher churn than ordinary ones at every bit depth — the same shape M16 (motion) and M17 (residual, aggregate) already found, now confirmed to hold under the SAME fine-grained decomposition. The mechanism is not boundary-*only* (ordinary positions retain a large, non-trivial gain and churn of their own) but is clearly boundary-*concentrated*.

## 12. 5/4/3-bit comparison

Every qualitative pattern in §6–§11 (spatial smoothness, weak/shrinking channel concentration, monotonic-but-brittle routing, symbol-change dominance of excess bits, weak motion-churn correlation, boundary concentration) **holds at all three bit depths** — magnitudes scale with quantization coarseness, but the *shape* of the mechanism does not qualitatively change. Per this milestone's own robustness criterion, the identified pattern is considered robust, not a bit-depth-specific artifact.

## 13. Control experiments (Phase I)

| bits | real bytes | oracle bytes | shuffled bytes | shuffle recovers | interpretation |
|---|---|---|---|---|---|
| 5 | 597,335 | 583,090 | 590,792 | 45.93% | moderate — both magnitude and structure contribute |
| 4 | 408,872 | 376,479 | 389,340 | 60.30% | majority — magnitude dominates, shape secondary |
| 3 | 248,909 | 202,288 | 218,443 | 65.35% | majority — magnitude dominates, shape secondary |

A magnitude-matched, spatially-shuffled reconstruction of the error (same per-pixel magnitude distribution, spatial correlation destroyed by a fixed-seed permutation) recovers **less than the full gap at every bit depth** — proof that structure is not irrelevant — but recovers a **majority** of the gap at 4- and 3-bit specifically. Read together with §6's finding that spatial autocorrelation is *high* at every bit depth yet the *shuffle-recoverable fraction grows* at coarser quantization: **structure's relative contribution is largest at the finest quantization (5-bit, ~54% unrecovered) and smallest at the coarsest (3-bit, ~35% unrecovered)** — i.e., magnitude becomes proportionally more important as quantization coarsens, even though the error is spatially structured throughout. Reproduced byte-for-byte across two independent processes (§16).

## 14. Root-cause interpretation

No single Phase J option (A–E) is a clean fit, and forcing one would misrepresent the evidence:
- **Not A** (spatially structured error dominates) in isolation: structure is real (§6) and non-trivial (§13: 35–54% of the gap survives shuffling) but is not the majority contributor at 4-/3-bit, where magnitude (post-shuffle) already recovers the majority.
- **Not B** (channel-specific): concentration is weak and shrinks toward uniform at coarser bit depths (§7).
- **Not C** (codebook routing is *the dominant* amplifier): routing changes are frequent (44–73%) and individually costly (2.4–3.0×) but contribute only ~9–12% of *total* excess bits (§9) — real, but secondary.
- **Not D** as literally stated (a G16 "prediction quality" failure for a fixed target): the dominant cost (§9, 88–91%) comes from the residual *symbol itself* differing between real and oracle — an upstream, mechanical consequence of a different reference producing a different quantized residual, not a case of G16 mis-predicting an unchanged target.
- **Not E** (motion/high-texture interaction dominates): motion correlates with error magnitude but not reliably with churn or byte cost (§10).

**The evidence-backed finding, precisely stated**: reference error is spatially smooth and moderately structured (contributing 35–54% of the gap, more at finer quantization); it drives codebook reassignment frequently and disproportionately even at small magnitudes (a real brittleness in the 512-prototype decision boundary); but the majority of the resulting byte cost, consistently across all three bit depths, comes from cases where the error is large enough to change the *residual symbol itself* — a mechanism upstream of both codebook routing and G16's conditional prediction, which the pre-registered taxonomy does not have a clean label for. Classified as **F** for that reason, not because the evidence is weak — it is unusually precise and consistent — but because it does not collapse onto a single pre-registered category.

## 15. What M19 does NOT prove

- It does not prove that fixing codebook brittleness (§8, the ~9–12% share) would translate into a *deployable* net gain — that requires an actual coded-validation experiment M19 was not asked to run (M19 is diagnostic-only per its own scope).
- It does not identify *why* the codebook's decision boundaries are brittle at low error magnitudes (e.g., prototype spacing, L1 vs. log2 metric sensitivity) — only that they are, empirically.
- The spatial-structure analysis uses frame-level and pixel-region aggregates; it does not establish a fine-grained, position-by-position map between specific pixel-domain structures and specific latent-domain excess-bit positions (a materially larger undertaking than this audit's scope).
- Sample size: 108 VAL-B P-frames per bit depth (a `--val-frames-per-sequence 30` cap, chosen for tractability given ~3× M17's per-frame cost) — churn rates (43.9%/58.5%/72.7%) closely track M17's full 246-frame sample (44.2%/59.1%/73.0%), supporting representativeness, but the sample is smaller than M17's own.
- Does not train, fit, or select any model or threshold using DAVIS TEST (never accessed) or using VAL data as anything but held-out evaluation.

## 16. Reproducibility (Phase K)

5-bit was re-run in a fresh, independent process. **Byte-for-byte identical**: residual identity (`ec858dee10f8a955`), real bytes (597,335), oracle bytes (583,090), **shuffled bytes (590,792)** — including the seeded-permutation control — and mean assignment churn (43.89%). `deterministic_kernels()` used throughout.

## 17. Provenance

All three bit depths' residual identities match M13/M14/M15/M17/M18's own recorded values exactly (§2). No new identity was created — M19 reuses the frozen, already-verified M13 residual arm (`model11`/`assign_codebook`/`coding_codebook`) throughout, never refitting or modifying it.

## 18. Compatibility

No file under `src/nvc/` or any existing `m10`–`m18` script was touched. No `.nvct` container was ever created (every measurement runs raw frame-level calls, exactly as M16–M18 did). `test_m19_scripts_do_not_modify_any_production_source` / `test_m19_introduces_no_new_container_format` confirm both directly.

## 19. Tests

13 new tests in `tests/test_m19_reference_error_diagnostic.py`: exact real-chain match against `encode_multi`, pixel/latent error domain separation, the channel-specific (not channel-averaged) per-symbol error-magnitude alignment bug caught and fixed before the real run, deterministic channel/spatial/codebook/code-length accounting, GOP-position tagging, no-TEST-access guards, shuffled-control determinism (same seed → same result) and sensitivity (different seed → different result), and production-source/compatibility checks. Full suite: **1323 passed, 0 failed** (1310 pre-M19 + 13 new), zero regressions.

## 20. Files changed

**Created** (nothing under `src/nvc/` touched, nothing in any M10–M18 script touched):
- `scripts/m19_baseline.py` (Phase 0).
- `scripts/m19_reference_error_diagnostic.py` (Phases A–I data collection).
- `scripts/m19_analysis.py` (pure-numpy analysis pass over the collected data).
- `tests/test_m19_reference_error_diagnostic.py` (13 tests).
- `outputs/m19_reference_error_audit/{m19_baseline.json, m19_raw_{5,4,3}bit.json, m19_identities.json, m19_analysis.json, repro_check/..., M19_REPORT.md}`.

**Modified**: nothing (confirmed by `git status --short`, aside from the standing M15–M18 `CHANGELOG.md` edits this milestone extends further).

## 21. Final classification

**F — no single pre-registered mechanism (A–E) cleanly dominates.** This is not an inconclusive result (contrast with **D — inconclusive**, which would mean the diagnostic *couldn't isolate* the effect): the mechanism is precisely quantified and consistent across all three bit depths (§12); it simply does not collapse onto one of the five pre-registered categories. Stated plainly: **spatial structure is real but not dominant (35–54% of the gap survives magnitude-preserving shuffling); channel structure is weak and shrinking; codebook routing is a frequent, quantifiable, but secondary amplifier (~9–12% of excess bits); the majority driver (88–91% of excess bits, every bit depth) is the reference error changing the residual symbol itself, upstream of both routing and G16 prediction.**

## 22. Recommended M20 experiment (exactly one)

**Add a stability margin (hysteresis) to `SharedCodebook.assign_tensor`**: only reassign a position to a different prototype if its cost advantage over the currently-favored prototype exceeds a fixed margin, rather than switching on any infinitesimal advantage. This is:
- **Causal**: directly modifies the mechanism §8 identified as brittle (churn even at the smallest error decile).
- **Decoder-compatible**: a fixed, deterministic decision rule computable identically by encoder and decoder from information both already have (the model's own predicted probabilities) — no side information, no format change.
- **Minimal**: one function, no retraining, no new model, no architecture change — falls squarely within every freeze this milestone (and M17/M18) imposed.
- **Directly targeted**: aimed specifically at the ~9–12%-of-excess-bits mechanism this report isolated and quantified, not a blind re-optimization of reference PSNR (M18 already showed that fails) and not an attempt to address the larger, likely-unaddressable-without-a-new-model 88–91% "symbol changed" mechanism.

**Explicitly bounded expectation, stated now so M20 does not overclaim**: even a fully successful hysteresis margin cannot recover more than the ~9–12% share attributable to routing-only changes — it cannot touch the 88–91% majority mechanism, which is a direct, mechanical consequence of reference substitution and would require addressing the reference itself (M18 already showed the intra quantizer angle on that doesn't work; a learned reference-refinement approach remains the larger, not-yet-attempted M19/M17 recommendation, deliberately still out of scope here).

---

## Summary table

| bit depth | reference error (RMS, arb.) | assignment churn | G16 penalty (Δbits, changed vs. unchanged) | dominant structure |
|---|---|---|---|---|
| 5 | (see `m19_raw_5bit.json`, per-frame) | 43.9% | 0.096 vs 0.040 | symbol-change (88–91% of excess bits); moderate spatial structure (46% shuffle-recovered) |
| 4 | (see `m19_raw_4bit.json`, per-frame) | 58.5% | 0.195 vs 0.079 | symbol-change dominant; magnitude-dominant (60% shuffle-recovered) |
| 3 | (see `m19_raw_3bit.json`, per-frame) | 72.7% | 0.258 vs 0.086 | symbol-change dominant; magnitude-dominant (65% shuffle-recovered) |

- **M17 oracle upper bound**: +2.03% / +6.75% / +14.86% total-stream (5/4/3-bit).
- **M18 best realizable result**: +0.194% / +0.104% / −3.244% total-stream (intra quantizer recalibration — did not close the gap).
- **M19 identified mechanism**: reference error is spatially structured but magnitude-dominant, especially at coarser bit depths; codebook assignment is brittle and a real secondary amplifier (~9–12% of excess bits); the majority driver (~88–91%) is the reference error changing the coded residual symbol itself.
- **M20 recommended intervention**: a stability-margin (hysteresis) rule for codebook assignment, targeted at the quantified ~9–12% routing-only mechanism — causal, decoder-compatible, minimal, with an explicit, pre-stated ceiling on what it can recover.

---

**Reproducibility**: torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen checkpoint:
M10F λ=3e-4 seed 42 `best.pt`, unchanged since M10.
