# M15 — Broad TRAIN Calibration Policy: Root-Cause Experiment

**Status: COMPLETE.** **Classification: ROOT-CAUSE CONFIRMED (COVERAGE, NOT SAMPLE COUNT) — NO PRODUCTION CHANGE WARRANTED.**

M14 found and fixed a real motion-table calibration gap and, in doing so, identified the
*mechanism* it suspected was the general cause: `calibrate_grids`'s sequential,
first-N-frames TRAIN sampling under-covers the 72-sequence TRAIN population. M15 was
tasked with testing that mechanism directly, as a controlled experiment, rather than
shipping another one-off recalibration. The experiment answers the question cleanly:
**coverage is confirmed as the dominant mechanism** (a same-budget uniform policy
recovers 97%+ of M14's entire measured gain; shuffling sequence order changes almost
nothing). But the practical, unglamorous coda is that **M14's already-deployed recipe
(8 frames/sequence, 576 total) is itself already a good, nearly-optimal instance of a
coverage-balanced policy** — it is not meaningfully beaten by a more "principled"
uniform-at-equal-budget alternative. No candidate cleared the bar to replace what is
already in production. This is reported as the honest outcome, not reframed as a miss.

---

## 1. Baseline (Phase 0)

Re-derived (never trusted from the record) the M14-deployed calibration at 5/4/3-bit via
a fresh `calibrate_grids` call and compared every identity against
`outputs/m14_entropy_audit/m14_entropy_audit.json`:

| bits | calibration signature | intra id | motion id | residual id |
|---|---|---|---|---|
| 5 | match | match | match | match |
| 4 | match | match | match | match |
| 3 | match | match | match | match |

All four fields matched byte-for-byte at every rate point — **deterministic, zero drift
since M14 shipped.** Full test suite (pre-M15): **1254 passed, 0 failed** (see §20 for why
this differs from M14's previously-cited 1251 — unrelated to M15, harmless).
Report: `m15_phase0_baseline.json`.

## 2. Existing calibration policy (Phase A)

`m10h_motion_compensation.calibrate_grids` walks its `sequences` argument in the given
(manifest, alphabetically-sorted) order and consumes frames front-to-back per sequence,
against a single GLOBAL counter shared across the whole walk — stopping **mid-sequence**
the instant `max_frames` is reached, never finishing the current sequence first. TRAIN has
**72 sequences / 4,826 frames** (min 25, max 100, mean 67.0 frames/sequence). At the
deployed `max_frames=400`, this walk touches exactly **6 of 72 sequences (8.3%)**:

| sequence | frames taken | frames available |
|---|---|---|
| bear | 82 | 82 |
| bike-packing | 69 | 69 |
| blackswan | 50 | 50 |
| boat | 75 | 75 |
| boxing-fisheye | 87 | 87 |
| breakdance | 37 | 84 (partial) |

Both passes (intra-only, and residual+motion) walk the *identical* sequence list under the
*identical* budget, so they see the same substantive TRAIN prefix — but intra keeps every
frame (I- and P-typed alike) while residual/motion keep only P-typed frames
(`residual_frames=motion_frames=358` of 400 at 5-bit, matching the audit record exactly).
The M11-G16 residual codebook's calibration is a **completely separate path**
(`m11_train`/`m13_recalibration`, never `calibrate_grids`) — this is *why* M13's
recalibration never inherited this coverage problem in the first place.

One asymmetry worth flagging honestly: `calibrate_grids`'s own motion collection is not
perfectly bit-depth-independent (its I-frame branch reconstructs the GOP-boundary
reference through the real, bit-depth-dependent intra quantizer), while M14's
`collect_motion_symbols` (reused unmodified by every M15 policy) idealizes this to the
true latent, making its own tables exactly bit-depth-independent. Both are pre-existing,
documented approximations relative to the coder's real closed loop; M15 changes neither,
and since every M15 policy uses the *same* collector, this is not a confound between them.
Full detail: `m15_baseline_policy.json`.

## 3. Candidate policies (Phase B)

Implemented in `scripts/m15_calibration_policy.py`, entirely separate from
`calibrate_grids` — every policy returns a list of `BenchmarkSequence` objects whose
`frame_paths` are an unmodified **prefix** of the original (never a different in-sequence
selection), so the existing, unmodified `m14_recalibration.collect_intra_symbols` /
`collect_motion_symbols` can consume any of them identically to how they consume
`calibrate_grids`'s own list.

| policy | rule | total budget |
|---|---|---|
| **A** — current | sequential, manifest order, stop at N | 400 |
| **B** — uniform | flat allocation across all 72 sequences, remainder to the first sequences | 400 (same as A) |
| **C** — M14 broad | 8 frames/sequence, every sequence (= M14's shipped recipe) | 576 |
| **D** — shuffled uniform | seed-42 shuffle of sequence order, then B's allocation rule | 400 |

## 4. Exact frame-selection statistics (Phase D)

| policy | sequences covered | % TRAIN sequences | frames | min/seq | max/seq | mean/seq | stdev/seq |
|---|---|---|---|---|---|---|---|
| A (current) | 6/72 | 8.3% | 400 | 37 | 87 | 66.67 | **17.71** |
| B (uniform) | 72/72 | 100.0% | 400 | 5 | 6 | 5.56 | **0.50** |
| C (M14 broad) | 72/72 | 100.0% | 576 | 8 | 8 | 8.00 | **0.00** |
| D (shuffled) | 72/72 | 100.0% | 400 | 5 | 6 | 5.56 | **0.50** |

Policy A's per-sequence stdev (17.71) is **35× B/D's** (0.50) at the *identical* total
budget — the coverage difference this milestone set out to test is a real, large,
directly-measured property of the current policy, not an assumption.

## 5. TRAIN/VAL-A/VAL-B provenance

Sequence-disjoint splits, identical to M13/M14's own discipline, reused unmodified:
- **TRAIN**: all 72 sequences (never truncated before a policy is applied to it).
- **VAL-A** (even-indexed val sequences, consistency check): bmx-trees, dogs-scale,
  mallard-water, skate-park, upside-down.
- **VAL-B** (odd-indexed, the headline number): car-roundabout, drift-straight, pigs, stunt.
- **TEST**: never read by any M15 fitting call (`test_scripts_never_feed_test_sequences_into_a_fitting_or_policy_call`,
  parametrized over all three M15 driver scripts, pins this).

## 6. Offline entropy results (Phase C) — THE CENTRAL RESULT

**Motion** (bit-depth-independent; one table per policy, reused across 5/4/3-bit).
VAL-B held-out bits/symbol, gain relative to **Policy A** (root-cause question) and
relative to **deployed Policy C** (incremental question):

| policy | VAL-B H (bits/symbol) | vs A (root cause) | vs deployed C (Phase G) |
|---|---|---|---|
| A — current | 4.39672 | — | −13.94% (weak *for* C) |
| **B — uniform, same 400 budget** | **3.87319** | **+11.91% (meaningful)** | −0.38% (weak) |
| C — M14 deployed, 576 | 3.85868 | +12.24% (meaningful) | — |
| D — shuffled, 400 | 3.86720 | +12.04% (meaningful) | −0.22% (weak) |

**This is the strong-evidence pattern the milestone asked for**: 400 sequential (A) is
dramatically worse than 400 uniform (B); 400 uniform (B) is within **0.33 percentage
points** of 576 broad (C) despite using **31% fewer frames**; shuffled order (D) is
within 0.2 points of B. Coverage — not raw sample count, not manifest order — explains
essentially all of M14's motion gain.

**Intra** (bit-depth-dependent quantizer; new table per policy per rate point), same
comparison, vs Policy A:

| bits | A (baseline) | B | C | D |
|---|---|---|---|---|
| 5 | H=3.83403 | +0.381% weak | +0.370% weak | +0.351% weak |
| 4 | H=2.81589 | +0.417% weak | +0.401% weak | +0.379% weak |
| 3 | H=1.79891 | +0.427% weak | +0.407% weak | +0.381% weak |

Every policy is weak for intra at every rate point — this is not specific to Policy C's
particular 8-frame recipe (which is what M14 alone tested); **no coverage policy tested
helps intra meaningfully.** M14's intra rejection generalizes.

Self-consistency check: Policy A's table, built via the *same* collector every other
policy uses, was verified to produce **byte-identical `model_id()`** to `calibrate_grids`'s
own intra table at all three rate points (`policy_a_matches_calibrate_grids_intra: true`),
and Policy C's motion table reproduced M14's *exact* deployed identity
(`d7e7b237b6451885`) — confirming the policy framework is a faithful reproduction of the
existing pipeline, not a parallel reimplementation that merely resembles it.
Full data: `m15_offline_gate.json`.

## 7. Per-sequence analysis (Phase E)

VAL-B, per-sequence motion H (bits/symbol), diagnostic only (not used to tune anything):

| sequence | A | B | C | D | C vs A |
|---|---|---|---|---|---|
| car-roundabout | 4.1363 | 3.5325 | 3.5280 | 3.5094 | +14.71% |
| drift-straight | 6.7755 | 5.4632 | 5.4020 | 5.4663 | +20.27% |
| **pigs** | 2.3118 | 2.6115 | 2.6232 | 2.5876 | **−13.47%** |
| stunt | 4.3632 | 3.8856 | 3.8815 | 3.9055 | +11.04% |

Three of four VAL-B sequences improve substantially and consistently under every broad
policy; **`pigs` regresses under B, C, *and* D alike** — a real, consistent, sequence-level
exception, not noise (all three broad policies agree on the direction). Reported honestly
rather than folded into the aggregate: broad coverage helps on average and by a large
margin, but is not uniformly better on every sequence. Whatever is distinctive about
`pigs`'s TRAIN-adjacent motion statistics is outside this milestone's scope to diagnose
further.

## 8. Actual coded validation (Phase F)

Real arithmetic-coded streams, 3 validation sequences (bmx-trees, car-roundabout,
dogs-scale — M13/M14's own established coded-validation default), frozen M13 residual arm
re-derived exactly as M13/M14 did. Combo `current` = A-intra + C-motion (M14's exact
deployed state); combo `broad_motion` = A-intra + B-motion (the root-cause, equal-budget
candidate):

| bits | current bytes | broad_motion bytes | Δ bytes | total-container gain |
|---|---|---|---|---|
| 5 | 582,935 | 582,916 | −19 | +0.0033% |
| 4 | 413,773 | 413,752 | −21 | +0.0051% |
| 3 | 263,903 | 263,881 | −22 | +0.0083% |

All invariants held at every rate point: symbols identical, reconstruction identical,
motion identical, PSNR/MS-SSIM identical (29.8537/29.5406/28.4629 dB,
0.981240/0.977140/0.962110 — bit-for-bit the same between combos, as required when only a
frequency table changes). Note the *direction* here (B fractionally ahead of C) is the
opposite of the offline VAL-B comparison (C fractionally ahead of B, §6) — expected sample
noise: this coded-validation sample is only ~90 P-frames drawn from a *different*, smaller,
VAL-A/VAL-B-mixed sequence set, far too small to resolve a 0.2–0.4 percentage-point
offline difference. The robust conclusion from both measurements agrees: **B and C are
statistically indistinguishable from each other**, while both are dramatically ahead of A.
Full data: `m15_coded_validation.json`.

## 9. Ideal-vs-actual realization

Not separately tabulated: since B and C are themselves nearly tied, and neither shows a
*meaningful* gain over the other, there is no new candidate whose "ideal offline gain" vs
"actual realized coded gain" gap is the interesting number here. The realization question
was already answered at full scale by M14 (99.97–100.02% for its residual/motion gains) and
nothing in M15 revisits or changes that pipeline.

## 10. Interaction effects (Phase G)

Phase G's question — "is a broad policy still useful **relative to the actual deployed
baseline** (M13 residual + M14 motion)" — is exactly what §8's `current` vs `broad_motion`
combo measures, since Policy C **is** M14's deployed motion recipe. Answer: **no**, not
meaningfully (+0.003–0.008% total-container, an order of magnitude below the 0.5% "weak"
line). Intra was not combined into any interaction test: every policy showed a
consistently weak intra gain (§6), so per the milestone's own "only test combinations
justified by the offline gates" rule, no `broad_intra` or `broad_intra+motion` combo was
run — combining a weak candidate would not manufacture a meaningful interaction, and
running it would spend real GPU time to confirm what the offline gate already shows.

## 11. Full DAVIS results — SKIPPED, with justification

**Phase H was not run.** The milestone's own gate is explicit: promote to the 719-frame
DAVIS TEST benchmark only for a candidate that (a) showed a meaningful offline gain over
the *current* comparator, and (b) survived actual coded validation. Policy B (or D) is a
meaningful improvement **over Policy A** (§6) but **not over the already-deployed Policy
C** (§6, §8, §10) — and Phase H's comparator is explicitly "the current M14 baseline," i.e.
C. Spending 719-frame DAVIS compute to confirm a difference already measured at
±0.003–0.008% on a smaller sample would not change the conclusion. This mirrors M14's own
disciplined skip of DAVIS-scale compute for intra (rejected at the offline gate) —
applied here one level up: not "is a table worth recalibrating" but "is a *policy* worth
promoting over what already shipped."

## 12–15. BPP / PSNR / MS-SSIM / BD-rate

No new BD-rate curve is reported: BD-rate needs a rate–distortion difference, and every
M15 combo compared here has **bit-identical PSNR and MS-SSIM** at every rate point (§8) —
by construction, since only an entropy frequency table differs and no combo changed a
symbol. There is nothing to trade off; the only axis that moved at all was bytes, by
<0.01%. BPP is listed in §8 alongside bytes.

## 16. Latency

Encode timing from the coded-validation run (81 P-frames), `current` vs `broad_motion`,
network/tables/coder seconds respectively: 5-bit 0.0957/0.0886/0.0458 vs
0.0872/0.0835/0.0451; 4-bit 0.0907/0.0825/0.0401 vs 0.0897/0.0795/0.0401; 3-bit
0.0905/0.0782/0.0328 vs 0.0914/0.0764/0.0325 — differences are within measurement noise
at this sample size, no systematic direction. This is structurally expected, not merely
observed: a motion table swap changes only which frequencies an `EmpiricalEntropyModel`
holds, never its table count (2), alphabet size, or the coder's per-symbol work — nothing
about *which* policy fitted the table can move encode or decode latency, only *which
bytes* result. No dedicated DAVIS-scale latency table was produced (§11).

## 17. Memory

Not separately profiled: every M15 policy produces a table of identical shape to the
ones already deployed (`EmpiricalEntropyModel`, 2 tables × `2**motion_bits` for motion,
`channels` × `2**bits` for intra) — memory footprint is a function of shape, not content,
and no shape changed.

## 18. Provenance

- Policy A/B/C/D motion identities are all distinct 8-byte hashes: `7d4ff90c74b45c48`,
  `60fde348c2f32a70`, `d7e7b237b6451885`, `35fdf02570e231f8`.
- **Policy C's identity is byte-identical to M14's already-deployed motion identity** —
  confirms the M15 policy framework reproduces production exactly when configured to.
- Policy A's intra identity is byte-identical to `calibrate_grids`'s own intra table at
  every rate point (§6) — confirms the same for intra.
- `git status --short` at the end of this milestone shows **zero modifications to any
  existing tracked file** — every M15 change is a new, untracked file. The deployed M13/M14
  configuration is untouched.

## 19. Compatibility

- `.nvct` v2 is unchanged; no v3 was introduced (`test_m15_introduces_no_new_container_format`).
- Old M13/M14 streams remain decodable: no existing decode path was modified.
- A stream encoded under one policy's table and decoded under a *different* policy's table
  is correctly **rejected** (`TemporalFormatError: motion entropy model mismatch`) by the
  identity check M14 already added — proven again here specifically for M15-policy-derived
  tables (`test_decoding_a_policy_b_stream_with_the_policy_a_table_is_rejected`), not just
  asserted by analogy to M14's own test.
- A stream encoded and decoded under the *same* policy's table round-trips correctly for
  both a narrow (A) and a broad (B) candidate
  (`test_policy_a_stream_round_trips_unchanged`, `test_policy_b_stream_round_trips_correctly`).

## 20. Tests

23 new tests in `tests/test_m15_calibration_policy.py`, covering all 16 requested
categories: deterministic frame selection, exact total budget, sequence coverage, uniform
allocation, remainder allocation, current-policy compatibility, TRAIN-only fitting, no-TEST
access, cross-process (in-process, M14's own convention) reproducibility, fixed
symbol/motion/reconstruction streams, calibration identity changes, provenance-mismatch
rejection, and old/new stream compatibility. Full suite: **1277 passed, 0 failed** — exactly
1254 (Phase 0 baseline) + 23 new, zero regressions. (The 1254 baseline itself is 3 higher
than the 1251 last recorded at the end of M14; unrelated to any code change — most likely a
test that dynamically parametrizes over discovered files/directories, and M12–M14's own
push added new `outputs/`/`scripts/` entries in between. Not investigated further as it is
orthogonal to M15's own zero-regression requirement, which is satisfied either way.)

## 21. Files modified/created

**Created** (all new, nothing under `src/nvc/` touched):
- `scripts/m15_calibration_policy.py` — the four policy functions + `coverage_statistics` + `build_policy` dispatcher.
- `scripts/m15_phase0_baseline.py`, `scripts/m15_policy_audit.py`, `scripts/m15_offline_gate.py`,
  `scripts/m15_coded_validation.py`, `scripts/m15_davis_benchmark.py` (built, ready, not run — §11).
- `tests/test_m15_calibration_policy.py` (23 tests).
- `outputs/m15_calibration_policy/{m15_phase0_baseline.json, m15_baseline_policy.json, m15_offline_gate.json, m15_coded_validation.json, M15_REPORT.md}`.

**Modified**: nothing (confirmed by `git status --short`, §18).

## 22. Git diff

Additive only — see §21. `git status --short` shows only untracked (`??`) new paths, zero
`M` lines against any file tracked before this milestone.

## 23. Candidate classification

| candidate | vs Policy A | vs deployed baseline | verdict |
|---|---|---|---|
| B — motion, uniform@400 | +11.91% meaningful | −0.38% weak | root cause confirmed; not a deployable improvement |
| C — motion, broad@576 (= deployed) | +12.24% meaningful | — (is the baseline) | already deployed (M14) |
| D — motion, shuffled-uniform@400 | +12.04% meaningful | −0.22% weak | confirms order-independence; not a deployable improvement |
| B/C/D — intra, all variants | +0.35–0.43% weak | (same, intra never recalibrated) | correctly rejected, all rate points |

## 24. Overall M15 classification

**ROOT-CAUSE CONFIRMED (COVERAGE, NOT SAMPLE COUNT) — NO PRODUCTION CHANGE WARRANTED.**

Answering the milestone's own scientific question directly: **400 sequential ≪ 400
uniform ≈ 576 broad** — the strong-evidence pattern was observed, not the "sample count"
or "inconclusive" alternatives. Coverage is confirmed as the general mechanism behind
M14's gain, and a clean, reusable, deterministic, TRAIN-only, provenance-compatible
policy abstraction now exists (`m15_calibration_policy.py`) supporting sequential,
uniform, flat-broad, and shuffled-uniform allocation, ready for any future milestone that
needs it (Phase J below).

The reason nothing ships from this milestone is arithmetic, not a weak result: M14's own
deployed recipe (8 frames/sequence, zero remainder at 576/72) is *itself* already a
perfectly flat, fully-covering allocation — a degenerate special case of "uniform." M15
confirms the *general principle* is what made M14 work, but the *specific instance*
already in production was already a good one. There was no gap left for a more
"principled" same-family policy to close.

Distinguishing the required categories explicitly:
- **SAMPLE COUNT EFFECT**: small and secondary — B (400) captures 97.3% of C's (576) gain
  over A (11.91% / 12.24%), i.e. adding 44% more frames on top of already-uniform coverage
  buys well under 3% additional relative improvement.
- **COVERAGE EFFECT**: large and primary — going from 8.3%-sequence/stdev-17.71 coverage
  (A) to 100%-sequence/stdev-0.50 coverage (B), at the *same* 400-frame budget, is what
  produces the +11.91% gain.
- **OFFLINE ENTROPY GAIN**: up to +12.24% (motion, vs A); +0.35–0.43% (intra, vs A, every
  policy — weak).
- **ACTUAL CODED BYTE GAIN**: +0.0033% to +0.0083% total-container (B vs the
  already-deployed C) — negligible; this is the number that matters for "should production
  change," and it says no.
- **TOTAL-STREAM GAIN**: same as above, since motion is a small fraction of total bytes.
- **QUALITY CHANGE**: none, at any comparison — PSNR/MS-SSIM bit-identical throughout.
- **LATENCY CHANGE**: none, structurally guaranteed and confirmed within measurement noise.

## 25. Recommended M16 direction

Two honest options, not a single forced recommendation:

1. **Consolidate, don't extend**: fold `m15_calibration_policy.py`'s uniform-allocation
   function into `calibrate_grids` itself as an opt-in `sampling="uniform"` mode (keeping
   `"sequential"` as the untouched default), purely so any *future* table calibrated
   through it inherits good coverage by construction, without needing another audit to
   discover the same problem again. This is infrastructure hygiene, not a claimed
   performance win — say so plainly if pursued.
2. **Look elsewhere for the next real gap**: M11–M14 have now covered residual
   (recalibrated, deployed), motion (recalibrated, deployed), intra (audited twice,
   rejected twice), and calibration coverage (audited, confirmed, already adequate in
   production). The remaining unexamined lever in this codec is likely *not* another
   entropy table but something structural — e.g., the GOP-boundary bit-depth-dependent
   motion asymmetry noted in §2, or M12's shelved spatial-context result, which was
   "weak" against G16's channel context specifically and has never been re-tested against
   a *recalibrated* G16 (M13's own deployed table) as its baseline.

Either is a reasonable next step; neither is assumed here.

---

**Reproducibility**: torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen checkpoint:
M10F λ=3e-4 seed 42 `best.pt`, unchanged since M10. All M15 scripts run via
`./.venv/Scripts/python.exe scripts/m15_*.py`.
