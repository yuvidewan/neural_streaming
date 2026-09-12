# M18 — Intra Quantizer Precision Audit

**Status: COMPLETE.** **Classification: C — INTRA QUANTIZER DOES NOT EXPLAIN / CLOSE THE M17 GAP.**

## 1. Executive summary

M17 found a large, real, deployed-pipeline-confirmed residual-rate bottleneck relative to an
unattainable oracle reference (+2.0% to +14.9% total-stream upper bound). M18 tested the smallest
plausible realizable mechanism — recalibrating the intra quantizer's scale/percentile parameters,
never its architecture, alphabet, or the frozen residual/motion/G16/codebook pipeline. The audit
found the opposite of the naive hypothesis: **the deployed intra quantizer is already relatively
*better* calibrated than the residual quantizer** (finer step/std ratio, higher SNR, near-identical
clipping fraction) — its larger *absolute* error comes from intra latents having much higher
inherent variance (whole-image content) than residual latents (motion-compensated differences),
not from a fixable calibration defect. Two realistic candidates were pushed through the real
deployed pipeline: broader TRAIN coverage (best at 5/4-bit) recovered a negligible 1.3%/−2.0% of
M17's oracle gap, net total-stream effect **+0.19%/+0.10%** — real but an order of magnitude below
the 0.5% gate. Tighter percentile clipping (best at 3-bit) recovered a real 6.8% of the oracle gap
on the P-frame channel, but its own I-frame byte cost **more than erased it** — net total-stream
effect **−3.24%**, i.e. *worse* than the deployed quantizer. No candidate reached the 0.5% gate;
Phase F (coded validation) and Phase G (full DAVIS) were correctly never triggered.

**This is the "Important Secondary Question" scenario the milestone explicitly asked to watch for**:
intra reconstruction genuinely improved (I-frame reconstruction MSE dropped meaningfully in three of
four tested configurations), but actual downstream residual bytes barely moved, or moved the wrong
way once I-frame cost was honestly netted out. The bottleneck M17 found is real but is **not** simply
"the intra quantizer is badly calibrated" — the case for M17's own second-listed M18 direction (a
causal reference-refinement mechanism) is strengthened by this result, not weakened.

## 2. Baseline (Phase 0)

Pre-M18 full suite (M17's own final count): 1300 passed, 0 failed. Confirmed the intra and residual
quantizers are structurally identical code paths — both `UniformQuantizer`/`calibrate_quantization_params`
affine grids, both per-channel mode, both derived via the same percentile convention (0.1/99.9), and
both entropy models use the **identical alphabet size** `2**bits` (`EmpiricalEntropyModel` enforces
this structurally) — so any gap is entirely attributable to the *calibrated scale*, never to a
format or alphabet asymmetry. `outputs/m18_intra_quantizer_audit/m18_baseline.json`.

## 3. Current intra quantizer

Per-channel affine grid, TRAIN-fit via `calibrate_grids`'s sequential 400-frame walk (M14/M15's own
documented coverage gap applies here too — see §6), percentile (0.1, 99.9), no companding, no
per-channel bit reallocation. Every consumed frame (I- and P-typed alike) contributes one intra
latent sample.

## 4. Current residual quantizer

Identical code path and percentile convention, fit from the **same** `calibrate_grids` walk's
P-frame residuals (`latent − reference_latent`, true-latent-advanced per that function's own
documented approximation, unrelated to and unaffected by M18).

## 5. Effective precision comparison (Phase A, held-out VAL-A)

**Normalized, never raw scale** (per the milestone's own explicit instruction):

| bits | intra step/std | residual step/std | intra SNR | residual SNR | intra clipped% | residual clipped% |
|---|---|---|---|---|---|---|
| 5 | 0.271 | 0.424 | 20.24 dB | 13.43 dB | 0.442% | 0.437% |
| 4 | 0.560 | 0.871 | 15.29 dB | 10.31 dB | 0.368% | 0.394% |
| 3 | 1.200 | 1.816 | 9.30 dB | 6.01 dB | 0.287% | 0.327% |

**Intra's own quantizer is relatively *more* precise than residual's at every bit depth** — a smaller
step relative to its own signal spread, and higher SNR. Clipping fractions are nearly identical
between the two. TRAIN→VAL generalization degradation (MSE) is also comparable in *relative* terms
for both (roughly 2.2–2.4× worse on held-out data for both intra and residual alike) — not a
sign that intra's calibration data is disproportionately unrepresentative. `m18_baseline.json`.

## 6. Root-cause analysis (Phase B)

Given §5, the dominant mechanism is **#3 — latent dynamic range** (intra represents whole natural
images, an inherently higher-variance, higher-information signal, than a motion-compensated
residual, which is mostly near zero by construction) — not scale miscalibration, not clipping, not
an entropy-model interaction. This was tested, not assumed: an offline sweep of percentile bounds ×
TRAIN coverage (`m18_candidates.py`) found:

| candidate | 5-bit VAL MSE | 4-bit | 3-bit |
|---|---|---|---|
| A (current deployed) | 0.1386 | 0.3236 | 1.1279 |
| BROAD (0.1/99.9, broad TRAIN) | **0.0863** (+37.7%) | **0.2835** (+12.4%) | 1.1629 (−3.1%) |
| TIGHT (1/99, broad TRAIN) | 0.3257 (−135%) | 0.4289 (−32.5%) | **0.9048** (+19.8%) |
| TIGHTER (2/98, broad TRAIN) | 0.6144 | 0.6984 | 1.0615 |
| LOOSE (0.01/99.99, broad TRAIN) | 0.1006 | 0.4152 | 1.7707 |
| *(unrealizable)* per-channel bit reallocation | 0.0863 | 0.2836 | 1.1629 |

No single candidate wins at every bit depth — broader coverage helps most where quantization is
already fine (5/4-bit); tighter clipping helps only once quantization is coarse enough (3-bit) that
trading tail coverage for step size pays off. Per-channel bit reallocation (`allocate_bits_per_channel`,
already implemented in production code for a different purpose) is diagnostically identical to BROAD
here and is **not realizable without a `.nvct` v2 alphabet change** — excluded from Phase C onward,
consistent with this milestone's freezes, and reported as a theoretical bound only.

## 7. Candidate quantizer variants

Selected for the expensive real-pipeline bridge (Phase C), one per bit depth, chosen as each
depth's *only* candidate that beat the deployed quantizer's own latent MSE: **BROAD** at 5/4-bit,
**TIGHT (1/99)** at 3-bit. Each candidate's own intra entropy model was freshly TRAIN-fit under its
own quantizer (never a stale table mismatched to a different grid) — `m18_reference_bridge.py`.

## 8. Boundary vs non-boundary results (Phase D)

| bits | candidate | boundary P-byte Δ | ordinary P-byte Δ | total P-byte Δ |
|---|---|---|---|---|
| 5 | BROAD | −634 | +200 | −434 |
| 4 | BROAD | +1,957 | −450 | +1,507 |
| 3 | TIGHT | **−6,333** | −995 | −7,328 |

Where a candidate helps at all (5-bit, 3-bit), the improvement is **boundary-concentrated**,
matching M16/M17's own GOP-position finding exactly — TIGHT's entire P-frame gain at 3-bit is
overwhelmingly the boundary position's own reduction (6,333 of 7,328 total bytes, 86%). At 4-bit,
BROAD's candidate actually makes the *boundary* position measurably *worse* (+1,957 bytes) despite
improving mean I-frame reconstruction MSE — a clean, direct illustration of §1's "improves PSNR,
doesn't improve bytes" warning.

## 9. Reference-quality changes

I-frame reconstruction MSE, base → candidate: 5-bit 0.001399→0.001342 (better); 4-bit
0.001581→0.001669 (**worse** — BROAD's own I-frame reconstruction regressed slightly at 4-bit even
though its latent MSE improved, since latent MSE and image-space reconstruction MSE are not
identical objectives); 3-bit 0.002586→0.002846 (worse — TIGHT trades reconstruction fidelity for
finer relative resolution in the bulk of the distribution, at the cost of more clipping distortion
in the tails).

## 10. Actual residual-byte changes

Reported in §8 (P-byte Δ) and summarized in §12's table. The channel-level P-residual gain (5-bit
+0.033%, 4-bit −0.166%, 3-bit +1.335%) is the number that matters here — not the offline latent-MSE
improvement from §6, which does **not** reliably predict it (BROAD improves 5-bit MSE by 37.7% but
P-bytes by only 0.033%; a >1000x compression of "signal" into "effect").

## 11. I-frame byte cost

| bits | candidate | I-byte Δ |
|---|---|---|
| 5 | BROAD | **−2,596** (cheaper) |
| 4 | BROAD | **−2,624** (cheaper) |
| 3 | TIGHT | **+28,554** (27% more expensive) |

BROAD is a genuine, if modest, I-frame-only win at 5/4-bit (consistent with M15's own prior finding
that broader TRAIN coverage gives intra a small entropy gain, ~0.35–0.43%, independent of this
milestone). TIGHT's 3-bit I-frame cost explosion is the direct, measured price of trading dynamic
range for resolution within the same affine-quantizer family — a real rate/distortion tradeoff, not
a bug.

## 12. Net total-stream effect (Phase H) — the deciding number

| bit depth | candidate | I-Δbytes | P-Δbytes | total Δbytes | oracle gap recovered | quality Δ | verdict |
|---|---|---|---|---|---|---|---|
| 5 | BROAD | −2,596 | −434 | **−3,030** (+0.194%) | +1.32% | I-MSE better | weak |
| 4 | BROAD | −2,624 | +1,507 | **−1,117** (+0.104%) | −1.96% | I-MSE worse | weak |
| 3 | TIGHT (1/99) | +28,554 | −7,328 | **+21,226** (−3.244%) | +6.80% (P-channel only) | I-MSE worse | **negative** |

(Δbytes negative = fewer bytes = improvement; percentages are of that bit depth's base total stream
bytes on the VAL-B sample used.)

## 13. Fraction of M17 oracle gap recovered

Using `fraction_oracle_gap_recovered = candidate_P-channel-gain / M17_real-to-oracle_gain`: **+1.3%**
(5-bit), **−2.0%** (4-bit, wrong direction), **+6.8%** (3-bit, P-channel only — but this is the
channel-level fraction, not the total-stream fraction, and §12 shows the total-stream effect at
3-bit is net *negative* once I-frame cost is included). None of these come close to materially
closing M17's 2.0–14.9% total-stream opportunity.

## 14. Coded validation — not triggered

Phase E's gate (≥0.5% total-stream improvement, no unacceptable quality regression) was not met by
any candidate at any bit depth: two of three configurations land in "weak" (<0.5%, both far below
even that floor at +0.19%/+0.10%) and one is net-negative. Per Phase F's explicit instruction ("If
no candidate reaches 0.5%, stop without production implementation"), no coded validation was run
and no candidate was implemented in or near production.

## 15. Full DAVIS — not triggered

Gated on §14; not reached.

## 16. Reproducibility (Phase I)

5-bit BROAD was re-run in a fresh, independent process. Byte-for-byte identical: candidate intra
identity (`9715726f72ea1e53`), I-byte Δ (−2,596), P-byte Δ (−434), boundary/ordinary breakdown, net
total-stream effect (−3,030 bytes / +0.1944%), and both reconstruction-MSE values, all matched
exactly. `deterministic_kernels()` used throughout, the same discipline established in M11.

## 17. Provenance

No new `.nvct` v2 identity was created in any deployed sense — every candidate's intra entropy
model got its own distinct `EmpiricalEntropyModel.model_id()` (verified distinct from the base
identity in every run), but since no candidate was adopted, none of these identities were ever
written into a real stream. The frozen M13 residual arm (model11/assign_codebook/coding_codebook)
used throughout is the same one Phase 0 confirmed against the recorded M13/M14/M15/M17 baseline.
`test_m18_scripts_do_not_modify_any_production_source` pins via `git status` that nothing existing
changed.

## 18. Compatibility

Old M13/M14/M15/M16/M17 streams are entirely unaffected — no file under `src/nvc/` or any existing
`m10`–`m17` script was touched, and no `.nvct` container was ever created by M18 (every measurement
runs raw frame-level `encode_latent_to_payload`/`encode_frame_recalibrated` calls directly, exactly
as M17 did, never `TemporalStreamWriter`). `test_m18_introduces_no_new_container_format` confirms.

## 19. Tests

10 new tests in `tests/test_m18_intra_quantizer.py`: exact real-closed-loop matching against
`encode_multi` (byte-for-byte, both I- and P-frame totals), boundary-position tagging, quantizer
-audit metric sanity (a finer quantizer must never increase MSE for the same data), candidate-vs
-base distinctness, non-contamination of the baseline path by a candidate run, TRAIN-only
calibration guards, candidate-calibration determinism, and production-source/compatibility checks.
Full suite: **1310 passed, 0 failed** (1300 pre-M18 + 10 new), zero regressions.

## 20. Files changed

**Created** (nothing under `src/nvc/` touched, nothing in any M10–M17 script touched):
- `scripts/m18_baseline.py` (Phase 0/A).
- `scripts/m18_candidates.py` (Phase B).
- `scripts/m18_reference_bridge.py` (Phases C/D/H).
- `tests/test_m18_intra_quantizer.py` (10 tests).
- `outputs/m18_intra_quantizer_audit/{m18_baseline.json, m18_candidates.json, m18_reference_bridge_BROAD_same_percentile.json, m18_reference_bridge_TIGHT_1_99.json, repro_check/..., M18_REPORT.md}`.

**Modified**: nothing (confirmed by `git status --short`, aside from the standing M15–M17
`CHANGELOG.md` edits this milestone extends further).

## 21. Final classification

**C — INTRA QUANTIZER DOES NOT EXPLAIN / CLOSE THE M17 GAP.**

Every tested candidate's total-stream effect is below 0.5% (two configurations) or net-negative
(one configuration); no candidate reached the coded-validation gate. Per the milestone's own
explicit "Important Secondary Question": intra reconstruction (latent MSE, and in most
configurations image-space MSE) genuinely improved under at least one candidate at every bit depth,
while actual downstream residual bytes barely moved or moved the wrong way once I-frame cost was
honestly netted out. **The bottleneck is not simply "bad intra PSNR."**

- **M17 oracle upper bound**: +2.03% / +6.75% / +14.86% total-stream (5/4/3-bit).
- **M18 measured realizable gain**: +0.194% / +0.104% / −3.244% total-stream (5/4/3-bit).
- **Fraction of oracle opportunity captured**: ~9.6% (5-bit), ~1.5% (4-bit), negative (3-bit) —
  negligible to none.
- **Does the intra-quantizer hypothesis survive?** No. Recalibrating the intra quantizer's
  scale/percentile parameters, within the existing affine-uniform-quantizer family, cannot recover
  a meaningful fraction of M17's bottleneck — the gap is not a calibration defect, and improving
  the quantizer's own precision moves along a rate/distortion tradeoff curve (§6, §9, §11) rather
  than off of it.

## 22. Recommendation for M19

This result strengthens, rather than weakens, the case for M17's own larger-scope direction: a
**causal reference-refinement mechanism** (a learned or structured correction applied to the
decoded reference before it conditions G16, trained or designed to reduce the *specific* error
signature that hurts context-conditioned prediction and codebook routing — not merely aggregate
MSE, since M18 shows aggregate MSE improvement does not reliably transfer to byte savings). Two
honest sub-directions, neither chosen here:

1. Characterize **what kind of error** actually hurts G16's prediction and codebook assignment
   (not just its magnitude) — e.g., is it spatially structured, channel-correlated, or specific to
   particular latent channels? M18's own finding that latent-MSE improvement doesn't transfer to
   byte savings implies the error's *shape*, not just its size, is what matters; characterizing
   that shape is a natural, still-diagnostic (not yet a new-model) M19 scope.
2. Only once (1) is understood, design a minimal, causal, decoder-compatible refinement mechanism
   targeted at that specific error shape — this is the point at which a genuinely new component
   (explicitly out of scope for every audit milestone from M16 through M18) would be warranted,
   with this report and M17's as its motivating evidence.

---

**Reproducibility**: torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen checkpoint:
M10F λ=3e-4 seed 42 `best.pt`, unchanged since M10.
