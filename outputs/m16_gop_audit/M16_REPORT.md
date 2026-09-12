# M16 — GOP-Boundary Reference / Motion-Calibration Asymmetry Audit

**Status: COMPLETE.** **Classification: REFERENCE DISCREPANCY CONFIRMED AND GOP-BOUNDARY-CONCENTRATED, BUT NOT A TOTAL-STREAM BOTTLENECK FOR MOTION (Outcome C) — NO PRODUCTION CHANGE.**

M14/M15 documented a bit-depth asymmetry in the *calibration-time helper functions* (`calibrate_grids`, `collect_motion_symbols`) concentrated at GOP boundaries. M16 asks whether the **real deployed coder** has a matching, materially-sized inefficiency. It does — the boundary effect is real, sharply concentrated at exactly one position per GOP, and grows fast with coarser quantization — but motion is too small a share of the total stream for even its **theoretical, unimplementable perfect-reference upper bound** to cross 0.5% of total bytes at any tested bit depth. A separate, much larger apparent gap on the **residual** channel was surfaced by the same diagnostic, but only under a simplified proxy model, not the real deployed entropy coder — flagged as an M17 candidate, not acted on here.

---

## 1. Baseline (Phase 0)

Full test suite before any M16 change: **1277 passed** (M15's own final count), 0 failed. `git status --short` showed only M15's own pending, previously-reported changes (all new files, `CHANGELOG.md` modified) — nothing further to re-verify beyond what M15 already established, since M16 makes zero changes to any M13/M14/M15 identity, calibration, or table. (M16's own new tests bring the suite to 1289; see §21.)

## 2. Exact reference-flow trace (Phase A)

Traced `scripts/m13_closed_loop.encode_multi`/`decode_sequence` directly — the **actual deployed coder**, not the calibration-time helper functions M14/M15 examined:

| step | what it is | bit-depth dependent? |
|---|---|---|
| true latent | `model.encode(frame)` | no |
| quantized latent | integer symbols inside the entropy-coded payload | yes (bits) |
| decoded latent (`reconstructed_latent`) | `decode_payload_to_latent(...)` (I) or `reference_latent + symbols_to_latent(...)` (P) | yes |
| reconstruction (`previous`) | `model.decode(reconstructed_latent)` — **pixel space, this is the actual motion-estimation reference** | yes |
| re-encoded latent (`reference_latent`) | `model.encode(warp(previous, motion))` — fresh every P-frame | yes (via `previous`) |

Measured I-frame quantization error (mean \|decoded − true latent\|, pixel-space mean \|reconstruction − frame\|), real checkpoint:

| bits | latent-space error | pixel-space error |
|---|---|---|
| 5 | 0.1786 | 0.0326 |
| 4 | 0.3710 | 0.0350 |
| 3 | 0.7829 | 0.0437 |

**Important correction made during this audit, stated plainly rather than glossed over**: Phase A's own initial reading of the source code concluded that GOP-boundary-specific bit dependence is *only* a property of `calibrate_grids`/`collect_motion_symbols` (which really do have an explicit boundary/non-boundary code-path asymmetry — see those functions' own P-frame branches), and that the real coder is "uniformly" bit-dependent since every frame's reference is a real quantized reconstruction with no special-case branch for boundary frames. That much is still true of the *code structure*. But Phase B/C's actual measurements (§5) show the real coder nonetheless produces a **sharply boundary-concentrated effect empirically**, for a different, mechanistic reason: intra quantization is measurably coarser (relative to signal) than residual quantization at the same nominal bit budget, so the one reference transition that goes through intra quantization (I → first P) takes a much bigger one-time quality hit than any transition that goes through residual quantization (P → next P). No code branch causes this; the quantizers' relative precision does. Both things are true at once: the *code* has no boundary special-case, and the *data* shows the boundary position behaving very differently anyway.

Full trace: `m16_reference_flow.json`.

## 3. GOP-boundary behavior

At the deployed `gop_size=10`: I-frames at 0, 10, 20, …; boundary P-frames (immediately following an I-frame) at 1, 11, 21, …; ordinary P-frames everywhere else. Confirmed both by direct inspection of `gop_frame_types` and by a dedicated test (`test_gop_boundary_positions_match_gop_frame_types`).

## 4. Bit-depth dependence

VAL-B (144 P-frames, 4 sequences), real deployed reference vs. oracle (true-latent) reference:

| bits | mean SAD real | mean SAD oracle | gap | fraction motion vectors changed |
|---|---|---|---|---|
| 5 | 26.835 | 26.435 | +1.51% | 11.31% |
| 4 | 27.613 | 26.435 | +4.46% | 18.41% |
| 3 | 30.032 | 26.435 | +13.61% | 30.48% |

Monotonic and substantial: at 3-bit, **nearly a third of all motion vectors differ** from what a perfect-reference encoder would choose, purely because of accumulated reference quantization error. The oracle values are *identical* across all three bit depths (26.435, 26.435, 26.435) — confirms the oracle construction is genuinely bit-depth-independent by design, an internal correctness check as much as a result.

## 5. Motion-vector differences — the per-GOP-position picture (Phases B/C)

This is the central empirical finding, and it does **not** match a simple "error compounds monotonically across the GOP" story. Per-position mean SAD (VAL-B, 16 P-frames per position per bit depth):

| GOP position | 5-bit real / oracle (gap %) | 4-bit (gap %) | 3-bit (gap %) |
|---|---|---|---|
| **1 (boundary)** | 26.99 / 25.65 (**+5.2%**) | 28.72 / 25.65 (**+11.9%**) | 34.52 / 25.65 (**+34.6%**) |
| 2 | 28.65 / 28.42 (+0.8%) | 29.28 / 28.42 (+3.0%) | 31.54 / 28.42 (+11.0%) |
| 3 | 25.45 / 25.06 (+1.6%) | 26.12 / 25.06 (+4.3%) | 28.25 / 25.06 (+12.7%) |
| 4 | 25.76 / 25.53 (+0.9%) | 26.44 / 25.53 (+3.6%) | 28.47 / 25.53 (+11.5%) |
| 5 | 24.93 / 24.67 (+1.0%) | 25.60 / 24.67 (+3.7%) | 27.60 / 24.67 (+11.9%) |
| 6 | 28.57 / 28.35 (+0.8%) | 29.23 / 28.35 (+3.1%) | 30.96 / 28.35 (+9.2%) |
| 7 | 26.63 / 26.27 (+1.4%) | 27.28 / 26.27 (+3.8%) | 29.21 / 26.27 (+11.2%) |
| 8 | 27.73 / 27.47 (+1.0%) | 28.39 / 27.47 (+3.4%) | 30.35 / 27.47 (+10.5%) |
| 9 | 26.80 / 26.50 (+1.1%) | 27.47 / 26.50 (+3.6%) | 29.40 / 26.50 (+10.9%) |

**Position 1 is a sharp, isolated spike, not the start of a ramp.** At every bit depth it is roughly **3–4× the relative gap of every other position**, and positions 2–9 stay comparatively flat across the rest of the GOP rather than climbing further — there is no evidence of the "accumulating error, worst right before the next I-frame" pattern a naive propagation model would predict. The autoencoder's repeated encode/decode cycling apparently does not let ordinary-P-frame error accumulate the way intra error hits once, hard, at the boundary.

## 6. SAD differences

Reported inline in §4/§5 (mean and per-position); medians tracked the same pattern (available in the JSON, not separately tabulated since the mean already tells the same story with no distortion from outliers here).

## 7. Residual differences

Mean residual energy (\|delta\|, real vs. oracle) is reported per-frame in the raw diagnostic rows; the entropy-relevant summary is §8's bits/symbol comparison, which is the number that actually maps to bytes.

## 8. Offline entropy impact (Phase D)

TRAIN-fit (M15's own broad policy C, 576 frames — chosen deliberately over `calibrate_grids`'s narrow sequential sample, applying M15's own finding rather than repeating the mistake it just diagnosed), VAL-B held-out:

| bits | motion H real | motion H oracle | **motion channel gain** | residual H real (proxy) | residual H oracle (proxy) | **residual channel gain (proxy)** |
|---|---|---|---|---|---|---|
| 5 | 3.86479 | 3.85868 | **+0.158% (weak)** | 3.16224 | 3.12138 | +1.292% |
| 4 | 3.89925 | 3.85868 | **+1.040% (meaningful)** | 2.24552 | 2.14391 | +4.525% |
| 3 | 4.01841 | 3.85868 | **+3.975% (meaningful)** | 1.41844 | 1.23842 | +12.692% |

Motion is measured with the **same entropy-model class actually deployed** (`EmpiricalEntropyModel`, 2 tables) — this comparison is rigorous. **Residual is a diagnostic proxy only**: a simple per-channel empirical model, matching `calibrate_grids`'s own (already-dead-weight, per M14) convention — **not** the real deployed M11-G16 channel-autoregressive model with M13's recalibrated codebook. G16's context conditioning already captures some of the same structure a naive per-channel model misses, so this number almost certainly **overestimates** what would survive under the real entropy coder. Building the real comparison would mean re-deriving the entire M13 recalibration pipeline twice (real-reference symbols and oracle-reference symbols) — a materially larger undertaking than this audit's scope, and explicitly not attempted here; see §25.

## 9. Oracle upper bound (Phase E) — channel vs. total-stream, not conflated

Using each bit depth's actual deployed byte shares (M14's own DAVIS numbers: motion ≈ 3.8–9.0% of total bytes; P-frame residual ≈ 75.8–82.2%):

| bits | motion channel gain | **motion → total-stream** | residual channel gain (proxy) | **residual → total-stream (proxy, likely overestimate)** |
|---|---|---|---|---|
| 5 | +0.158% | **+0.006% (weak)** | +1.292% | +1.062% |
| 4 | +1.040% | **+0.058% (weak)** | +4.525% | +3.606% |
| 3 | +3.975% | **+0.359% (weak)** | +12.692% | +9.615% |

**Motion's oracle upper bound never exceeds 0.36% of total stream bytes, even at the theoretical best case (3-bit) and even granting the oracle 100% realization** (never achievable in practice, since it requires a reference no real decoder can produce). This is decisively **weak** by the milestone's own thresholds. Per Phase E's explicit instruction — **the oracle is not meaningful for motion; this milestone stops here for the motion question.** No Phase F/G/H was run for a motion-reference correction.

The residual proxy numbers are large enough to reach "meaningful" even at 5-bit, but per §8's caveat this is not decision-grade evidence — it is a lead, not a finding, and is not acted on in M16 (§25).

## 10. Implementation decision

**No implementation was built.** Per the milestone's own explicit instruction ("If the oracle is not meaningful, STOP. Do not build a solution to a bottleneck that does not exist"), motion's oracle upper bound (§9) does not justify Phase F. Two honest, related observations for the record, neither acted on:

- The effect is **concentrated at exactly one GOP position**, not diffuse — so *if* a future milestone found this worth revisiting (e.g., after a residual-side investigation independently justified touching the reference chain), the targeted fix would be narrow (improve or refine only the boundary transition) rather than a general reference-quality change.
- No candidate in Phase F's own example list (decoded/reconstructed reference, re-encode consistently, calibration-matches-deployed, boundary-only change) was evaluated, since none was warranted.

## 11. Actual coded validation

Not run — gated on §10; no candidate reached Phase F.

## 12. Full DAVIS results

Not run — gated on §11 (Phase H requires Phase G first, which requires Phase F).

## 13–17. BPP / PSNR / MS-SSIM / BD-rate / per-sequence results

Not applicable — no coded stream was produced or compared; every measurement in this report is either an offline reference/entropy diagnostic (§4–§9) or a direct source-code trace (§2–§3). Quality and rate figures require an actual coded stream, which this milestone correctly did not produce given §9's outcome.

## 18. Latency

Not separately measured. The diagnostic itself is roughly 2× the cost of a single real encode pass (it also runs a full oracle-side motion search and an extra encode/decode per P-frame) - this cost belongs to the diagnostic tooling, not to anything that would ship. No deployed latency changes, since nothing deploys.

## 19. Provenance

Unaffected. No `.nvct` v2 identity field, calibration signature, or entropy-table identity changes as a result of M16 — the diagnostic never feeds its oracle computation back into anything that gets encoded (verified directly: `real_previous`/`reconstructed_latent` in `diagnose_sequence` are computed by the identical formula `encode_multi` uses, and are the *only* state that advances the loop; the oracle is a read-only side channel). `test_diagnose_sequence_real_chain_matches_encode_multi_exactly` proves the real chain byte-for-byte, not just by construction argument.

## 20. Compatibility

No stream semantics changed. Old M13/M14/M15 streams are unaffected (nothing in `src/nvc/` or any existing script was modified — `git status --short` shows only new M16 files and the standing M15 `CHANGELOG.md` edit). `test_m16_scripts_do_not_modify_any_production_source` pins this directly rather than asserting it.

## 21. Tests

12 new tests in `tests/test_m16_reference_diagnostic.py`, covering Phase J categories 1–7 (GOP-boundary identification, reference-flow correctness — including the load-bearing exact-match-with-`encode_multi` test, bit-depth-specific behavior, oracle/reference comparison, no-TEST access, determinism, existing-stream/production-source compatibility). Categories 8–9 (new-stream round trip, provenance mismatch) are explicitly not applicable — no implementation was reached (§10). Full suite: **1289 passed, 0 failed** (1277 M15 baseline + 12 new), zero regressions.

## 22. Files modified/created

**Created** (nothing under `src/nvc/` touched, nothing in any M10–M15 script touched):
- `scripts/m16_reference_audit.py` (Phase A trace).
- `scripts/m16_reference_diagnostic.py` (Phases B/C/D/E — `diagnose_sequence`, `_block_sad`, `_summarize`).
- `tests/test_m16_reference_diagnostic.py` (12 tests).
- `outputs/m16_gop_audit/{m16_reference_flow.json, m16_reference_diagnostic.json, M16_REPORT.md}`.

**Modified**: nothing (confirmed by `git status --short`, aside from the pre-existing M15 `CHANGELOG.md` edit this milestone did not touch further).

## 23. Git diff

Additive only — see §22.

## 24. Final classification

**REFERENCE DISCREPANCY CONFIRMED AND GOP-BOUNDARY-CONCENTRATED — NOT A TOTAL-STREAM BOTTLENECK FOR MOTION (Outcome C).**

Distinguishing the required categories explicitly:
- **REFERENCE DISCREPANCY**: real, measured directly (not assumed) — real vs. oracle SAD gap grows from +1.5% (5-bit) to +13.6% (3-bit) over all P-frames, and is sharply concentrated at GOP position 1 (+5.2% to +34.6%) versus a comparatively flat +0.8–12.7% at every other position.
- **MOTION-VECTOR EFFECT**: real and non-trivial — up to 30.5% of motion vectors differ from the oracle's choice at 3-bit, over all P-frames; boundary-position-only fractions run even higher (up to 45.3% at 3-bit position 1).
- **RESIDUAL EFFECT**: measured only via a diagnostic proxy, not the deployed entropy model; larger apparent effect (up to +12.7% channel-level at 3-bit) but explicitly not decision-grade.
- **OFFLINE ENTROPY EFFECT**: motion channel gain reaches "meaningful" (+1.04% to +3.98%) at 4- and 3-bit under the real entropy-model class.
- **ORACLE UPPER BOUND**: motion's translates to at most **+0.36% of total stream bytes** (3-bit, best case) — weak throughout. This is the number that actually gates the decision.
- **ACTUAL CODED GAIN**: not measured — no candidate reached coded validation.
- **QUALITY CHANGE**: none — no candidate was implemented.
- **LATENCY CHANGE**: none — no candidate was implemented.

The milestone's own audit-first discipline worked exactly as designed: a real, measurable, mechanistically-understood discrepancy was found and precisely characterized, and was **rejected as a bottleneck** once translated to the metric that actually matters (total-stream bytes), rather than being chased on the strength of an eye-catching channel-level or motion-vector-level number.

## 25. Recommended M17 direction

Two candidates, not a single forced pick:

1. **The residual proxy finding, done rigorously.** §8/§9's residual numbers are the more promising lead by far (potentially "meaningful" even at 5-bit), but were deliberately not trusted here because they come from a model cruder than what is actually deployed. An M17 would need to reproduce the real vs. oracle reference comparison through the *actual* M11-G16 + M13 recalibrated-codebook pipeline (prototype assignment, context conditioning, the works) rather than a simple per-channel proxy, before any bytes-level claim could be trusted. This is a bigger undertaking than M16 (it's closer in shape to re-running M13's own pipeline twice) but has a real chance of landing above the "meaningful" line where M16's motion result did not.
2. **The boundary-position-specific mechanism itself**, independent of whether §1 is pursued: since the effect is a sharp spike at exactly one GOP position rather than a diffuse drift, a future investigation into *why* intra quantization is so much coarser than residual quantization at matched nominal bit budgets (a quantizer-design question, not a reference-handling one) could be worth its own narrow audit — separate from, and prerequisite to, deciding whether a boundary-only correction is worth building at all.

Either is a reasonable next step; neither is assumed here, and this report does not recommend one over the other.

---

**Reproducibility**: torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen checkpoint:
M10F λ=3e-4 seed 42 `best.pt`, unchanged since M10. Diagnostic run via
`./.venv/Scripts/python.exe scripts/m16_reference_diagnostic.py`.
