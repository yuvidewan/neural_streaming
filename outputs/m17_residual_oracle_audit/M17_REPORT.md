# M17 — Residual Oracle Audit Through the Deployed M11-G16 + M13 Pipeline

**Status: COMPLETE.** **Classification: A — RESIDUAL ORACLE GAP IS A MEANINGFUL TOTAL-STREAM BOTTLENECK (confirmed under the real deployed pipeline, not a proxy).**

## 1. Executive summary

M16's simplified per-channel proxy estimated a residual-side oracle gain of +1.3% to +12.7% (channel-level) and flagged it explicitly as *not decision-grade* — a lead, not a finding. M17 reproduces the same real-vs-oracle comparison through the **actual deployed** M11-G16 causal-context model, the real 512-entry codebook assignment, M13's recalibrated frequencies, and the real arithmetic coder — never a proxy. The result is **larger than the proxy suggested, not smaller**: +2.47% / +8.47% / +19.61% channel-level gain at 5/4/3-bit, translating to a **+2.03% / +6.75% / +14.86% total-stream upper bound** — comfortably above the 1% "meaningful" line at every tested bit depth. The effect decomposes as ~85% reference-pixel quality, ~15% motion-vector re-selection, and is **boundary-dominated** (sharp at GOP position 1, substantial-but-flatter everywhere else) — the same qualitative shape M16 found for motion, now confirmed to matter far more on the channel that actually dominates the byte budget. Every oracle-variant byte count was corroborated by direct entropy-decoder round-trip (243/243 payloads matched exactly) and cross-process reproducibility (byte-identical totals across two independent Python processes). **No production change was made or is recommended by M17 itself** — the oracle reference is fundamentally unavailable to any real decoder without transmitting the previous frame twice, and every mechanism that could plausibly capture part of this gap (a new reference-refinement model, a different quantizer) is explicitly outside this milestone's freezes. M17's job was to determine whether the bottleneck is real; it is, decisively, and the specific mechanism is now precisely characterized for whoever takes on M18.

## 2. Baseline (Phase 0)

Pre-M17 full suite (M16's own final count): **1289 passed, 0 failed**. Re-derived the exact deployed M13 residual arm (model11 + assign_codebook + M13-recalibrated coding_codebook) fresh at all three bit depths and compared against `outputs/m13_recalibration/m13_davis_benchmark.json`'s own recorded provenance:

| bits | fresh identity | recorded identity | match |
|---|---|---|---|
| 5 | `ec858dee10f8a955` | `ec858dee10f8a955` | ✓ |
| 4 | `be872c2beebd2f9b` | `be872c2beebd2f9b` | ✓ |
| 3 | `d2d61d66a50cad8a` | `d2d61d66a50cad8a` | ✓ |

Byte-for-byte identical, confirming zero drift since M13 first shipped this arm. `outputs/m17_residual_oracle_audit/m17_baseline.json`.

## 3. Exact real-reference flow

Traced directly from `m13_recalibration.encode_frame_recalibrated` (called unmodified throughout M17, never reimplemented):

1. **Real residual**: `delta = latent − reference_latent`, where `reference_latent = model.encode(warp_blocks(previous_reconstruction, motion, block_size))` — the real, causal, motion-compensated reference.
2. **G16 causal context**: `model11.planes(target_symbols, zero)` — channel-group causal planes (`group_size=16`); only channel-groups *before* the current one are visible, using the *true* target symbols (known at encode time), zero-padded for the first group.
3. **G16 prediction**: `model11.log_probabilities(reference_latent, planes)` — per-position predicted distribution, conditioned on **both** the reference latent and the causal context.
4. **Codebook assignment**: `assign_codebook.assign_tensor(rows)` where `rows` are the step-3 probabilities — argmin distance to the *original, deployed* 512 prototypes (never the recalibrated ones — M13's own invariant, reused unmodified).
5. **M13 recalibrated frequencies**: `coding_codebook.cumulative[table_index]` — a separate, TRAIN-fit/VAL-A-smoothed `SharedCodebook`, used only for the coder's cumulative table.
6. **Arithmetic coding**: `nvc.compression.range_coder.encode_symbols(...)` → real payload bytes.

Because assignment (step 4) depends on `rows`, which depends on **both** the reference latent (step 3) **and** the true symbols being coded (step 2, causally) — changing the reference changes assignment, which changes which frequency row codes each symbol. This is the concrete mechanism the rest of this report measures. Full trace: `m17_reference_flow.json` (folded into `m17_baseline.json`'s `pipeline_trace` section).

## 4. Oracle construction

Identical to M16's own idealization, reused rather than redefined: `oracle_previous = model.decode(model.encode(raw_previous_frame))` — the autoencoder round-trip of the **true, uncoded** previous frame, recomputed fresh from ground truth every step (never chained, never accumulating). Three variants, matching Phase C's minimum requirement exactly:

- **A_real** — real reference + real motion vectors. This *is* what the deployed coder produces; its bytes were verified to equal `encode_multi`'s own actual P-frame residual byte totals exactly (`test_diagnose_residual_oracle_total_bytes_match_encode_multi_total`).
- **B_oracle_ref_real_motion** — oracle reference pixels, but motion vectors are still A's real ones (unchanged). Isolates the reference-*pixel* effect from any motion-vector effect.
- **C_full_oracle** — oracle reference **and** motion re-estimated against it. The full, unattainable upper bound; explicitly labeled as a separate experiment since it changes motion vectors too, per the milestone's own instruction.

B and C are read-only side channels; only A ever advances the real chain (verified: two independent calls to the diagnostic reproduce byte-identical A totals — `test_oracle_variants_never_affect_the_real_chain`).

## 5. Real vs. oracle residual statistics (Phase B), VAL-B, all P-frames

| bits | mean bytes A | mean bytes B | mean bytes C | fraction symbols changed A→C | fraction assignments changed A→C |
|---|---|---|---|---|---|
| 5 | 5419.8 | 5303.5 | 5285.8 | 29.8% | 44.2% |
| 4 | 3692.7 | 3427.5 | 3380.1 | 25.2% | 59.1% |
| 3 | 2232.0 | 1861.5 | 1794.3 | 18.7% | 73.0% |

**Codebook-assignment churn is large and grows as quantization coarsens** — at 3-bit, nearly three-quarters of all positions route to a *different* one of the 512 prototypes depending on reference quality alone. This is the direct mechanism behind the entropy cost difference: a "wrong" prototype gives a poorly-matched frequency table for whatever symbol actually occurs.

## 6. Actual deployed entropy-model comparison — the primary metric

This is not a marginal-entropy or per-channel-proxy estimate; every number above and below is `len(payload)` from the real, unmodified `encode_frame_recalibrated`, or its own reported ideal bits. Mean ideal bits/frame (A vs. C): 5-bit 43,354→42,282; 4-bit 29,537→27,036; 3-bit 17,852→14,349 — tracking the byte totals closely, confirming the arithmetic coder realizes close to the entropy model's own ideal cost in both scenarios (no coder-overhead artifact inflating the apparent gain).

## 7. Decomposition of the gap (Phase C)

| bits | A→C total (full oracle) | A→B (reference-pixel effect) | B→C (motion-vector effect, on top of oracle ref) |
|---|---|---|---|
| 5 | +2.4719% | **+2.1462%** (87%) | +0.3328% (13%) |
| 4 | +8.4662% | **+7.1808%** (85%) | +1.3848% (15%) |
| 3 | +19.6132% | **+16.6030%** (85%) | +3.6096% (15%) |

**Reference *pixel* quality is the dominant mechanism (~85% of the effect) at every bit depth**, not motion-vector re-selection. This is a clean, consistent split — the same relative proportion holds regardless of bit depth, which is itself evidence the decomposition is measuring a real, stable mechanism rather than noise. It also explains why M16 found motion's *own* channel unaffected at the total-stream level (motion is a small byte share) while this milestone finds residual — the channel actually carrying ~76–82% of the bytes — much more exposed to the *same* underlying reference-quality effect, through a different pathway (context-conditioned prediction and codebook routing, not motion search).

## 8. GOP-position analysis (Phase D)

Bytes gap (A vs. C) by position within the GOP (`gop_size=10`):

| position | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| **1 (boundary)** | **6.40%** | **17.98%** | **38.68%** |
| 2 | 1.80% | 6.80% | 17.69% |
| 3 | 1.94% | 7.42% | 18.16% |
| 4 | 1.81% | 6.59% | 15.95% |
| 5 | 2.56% | 8.50% | 18.17% |
| 6 | 1.85% | 6.93% | 15.22% |
| 7 | 1.95% | 7.35% | 16.11% |
| 8 | 1.85% | 6.52% | 13.91% |
| 9 | 1.83% | 6.72% | 15.31% |

**Classification: B — boundary-dominated**, not boundary-only (positions 2–9 still show a large, itself-meaningful gap on their own — 6.5–8.5% at 4-bit is far from negligible), not diffuse (position 1 is consistently 2–3× every other position), and not monotonically increasing (positions 2–9 stay roughly flat, no ramp toward position 9). This is the *same qualitative shape* M16 found for motion — reinforcing that intra quantization is simply coarser than residual quantization at matched nominal bit budgets, and that asymmetry propagates through both the motion and residual coding pathways identically in shape, very differently in consequence.

## 9. 5/4/3-bit results — summary table

| bits | channel gain (A→C) | total-stream upper bound | verdict |
|---|---|---|---|
| 5 | +2.4719% | +2.0314% | meaningful |
| 4 | +8.4662% | +6.7459% | meaningful |
| 3 | +19.6132% | +14.8590% | meaningful |

## 10. Channel-level gains

Reported in §7 and §9. Symbol-histogram divergence (symmetric KL, A vs. C, secondary diagnostic): 0.0010 / 0.0033 / 0.0073 bits at 5/4/3-bit — small in absolute terms (the *marginal* symbol distribution barely shifts) yet the *coded bytes* differ substantially, because the mechanism is context/assignment routing (§5, §7), not a marginal-distribution shift a histogram-divergence metric would fully capture. This is exactly why the milestone insisted on the real deployed pipeline rather than a marginal or per-channel proxy.

## 11. Total-stream upper bounds

Using the actual M13/M14-deployed P-frame residual byte share (75.8–82.2% of total stream bytes, from `m14_davis_benchmark.json`'s own recorded byte accounting):

| bits | residual channel gain | P-residual byte share | **total-stream upper bound** |
|---|---|---|---|
| 5 | +2.4719% | 82.18% | **+2.0314%** |
| 4 | +8.4662% | 79.68% | **+6.7459%** |
| 3 | +19.6132% | 75.76% | **+14.8590%** |

All three clear the 1.0% "meaningful" threshold by a wide margin — 3-bit's upper bound alone is larger than M13's *entire* deployed recalibration gain (+4.04% at 3-bit, M13's own headline number).

## 12. Coded validation gate (Phase F) — triggered

Phase E cleared 0.5% at every rate point, so Phase F ran. **What it can and cannot mean is stated explicitly, per the milestone's own emphasis**: the oracle reference cannot exist in any real decoder (it requires the raw previous frame, which is never transmitted). Phase F therefore does not — and could not — produce a "deployable oracle stream." What it verifies is narrower and load-bearing: that every reported oracle-variant byte count is a **real, correctly-decodable entropy-coder output**, not a computation artifact.

Result: **243/243 payloads** (81 per bit depth: A/B/C × 27 P-frames on a held-out sequence) decoded back to their exact source symbols via the unmodified `decode_frame_recalibrated`, given the same reference a hypothetical oracle-aware decoder would need. Sample-level gains on this smaller single-sequence check (+1.91% / +6.16% / +15.85%) closely track the full VAL-B numbers (§9), confirming the effect is not a single-sequence artifact. `m17_coded_validation_gate.json`.

**This corroborates that the bytes are genuine. It does not mean the gain is implementable.** No `.nvct` v2 container was created at any point in M17 — every measurement operates on raw frame payloads via `encode_frame_recalibrated`/`decode_frame_recalibrated` directly, exactly as M13/M14/M15's own coded-validation scripts do at the frame level before assembling a stream.

## 13. Reproducibility (Phase H)

The 5-bit diagnostic was re-run in a fresh, independent process (`--output-dir .../repro_check`). Byte-for-byte identical: residual-arm identity (`ec858dee10f8a955`), total bytes (A=1,333,275, B=1,304,660, C=1,300,318), and channel gain (+2.4719%) all matched exactly. `deterministic_kernels()` is used throughout, the same discipline established in M11 and reused unmodified in every milestone since.

## 14. Provenance

No new `.nvct` v2 identity was created (§12). The model11/assign_codebook/coding_codebook objects used throughout are the *same* ones whose identities were confirmed against the recorded M13 baseline in Phase 0 (§2) — never a new or modified identity. `test_m17_scripts_do_not_modify_any_production_source` pins directly (via `git status`) that no existing tracked file changed.

## 15. Compatibility

Old M13/M14/M15/M16 streams are entirely unaffected — no production file under `src/nvc/` or any existing `m10`–`m16` script was touched. `test_m17_introduces_no_new_container_format` confirms no new format version or container-writing code was introduced.

## 16. Tests

11 new tests in `tests/test_m17_residual_diagnostic.py`, covering exact real-reference-chain matching (byte-for-byte against `encode_multi`), oracle-path isolation, no-TEST-access, full determinism, the A/B/C decomposition's structural validity, GOP-position tagging, oracle-variant round-trip decoding, and production-source/compatibility guards. Full suite: **1300 passed, 0 failed** (1289 pre-M17 baseline + 11 new), zero regressions.

## 17. Files changed

**Created** (nothing under `src/nvc/` touched, nothing in any M10–M16 script touched):
- `scripts/m17_baseline.py` (Phase 0/A).
- `scripts/m17_residual_diagnostic.py` (Phases B/C/D/E — `diagnose_residual_oracle`, `_summarize`, `_histogram_divergence`).
- `scripts/m17_coded_validation_gate.py` (Phase F).
- `tests/test_m17_residual_diagnostic.py` (11 tests).
- `outputs/m17_residual_oracle_audit/{m17_baseline.json, m17_residual_diagnostic.json, m17_coded_validation_gate.json, repro_check/m17_residual_diagnostic.json, M17_REPORT.md}`.

**Modified**: nothing (confirmed by `git status --short`, aside from the standing M15/M16 `CHANGELOG.md` edits this milestone extends further, §19).

## 18. Final classification

**A — RESIDUAL ORACLE GAP IS A MEANINGFUL TOTAL-STREAM BOTTLENECK.**

Both of the milestone's own requirements for "A" are met, not merely the channel-level one: the comparison ran through the actual deployed entropy pipeline (§3–§6), not a proxy, and the total-stream upper bound is **≥1% at every tested bit depth** (§11), corroborated by a real round-trip decode check (§12) and cross-process reproducibility (§13) — not selected merely because the channel-level number looked large.

**The distinction the milestone asked to keep explicit, stated one more time plainly**: "residual entropy improves under a perfect, unattainable reference" is now a confirmed, precisely-quantified fact. "The complete compressed video stream gets materially smaller" is **not** something M17 has shown how to achieve — the oracle is defined to be inaccessible to any real decoder, and every mechanism that could plausibly close even part of this gap (a reference-refinement model, a redesigned quantizer) is outside this milestone's freezes by explicit instruction. M17 answers *whether a bottleneck exists*, decisively yes; it does not answer *how to capture it*, and was not asked to.

## 19. Recommendation for M18

The natural next step is the one M17 was explicitly not allowed to attempt: design and evaluate a **realizable** mechanism to close some fraction of this gap without violating causality (the decoder must never need information it wasn't sent). Concretely, in order of increasing scope:

1. **Smallest, most surgical**: since §8 shows the effect is boundary-dominated, investigate whether a **better intra quantizer specifically** (matching residual quantization's effective precision at the same nominal bit budget, not a general redesign) closes a meaningful fraction of the boundary-position gap alone — this is a quantizer-calibration question, not a new-model question, and might not even require retraining.
2. **Larger scope**: a decoder-side (or joint encoder/decoder) reference-refinement step — some processing applied to `reference_latent` before it conditions the G16 model, trained to reduce the effective quantization-error signature. This is a new model by the letter of M17's freezes and would need its own dedicated milestone, explicitly scoped and load-bearing on this report's numbers as its motivating evidence.
3. Either direction should re-confirm the oracle upper bound is still what this report says it is on the *specific* candidate mechanism's own held-out data before investing in a full implementation — the discipline this entire M14→M17 chain has followed throughout.

---

## Summary

- **Baseline tests**: 1289 passed (pre-M17, M16's own final count)
- **Final tests**: 1300 passed, 0 failed — zero regressions
- **Residual channel gain**: +2.47% (5-bit) / +8.47% (4-bit) / +19.61% (3-bit)
- **Total-stream upper bound**: +2.03% (5-bit) / +6.75% (4-bit) / +14.86% (3-bit)
- **Coded validation**: triggered (Phase E cleared 0.5% at every rate point) — 243/243 payloads round-trip verified; corroborates the bytes are genuine, does NOT mean the oracle is implementable
- **Final classification**: **A — meaningful total-stream bottleneck**, confirmed under the real deployed pipeline
- **Recommended M18 direction**: investigate a realizable, causal mechanism to close part of the gap — starting with the boundary-position-specific intra-quantizer angle (smallest scope) before considering a dedicated reference-refinement model (largest scope, its own milestone)

**Reproducibility**: torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen checkpoint: M10F λ=3e-4 seed 42 `best.pt`, unchanged since M10.
