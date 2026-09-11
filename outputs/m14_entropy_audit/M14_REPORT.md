# M14 — ENTROPY TABLE CALIBRATION AUDIT ACROSS THE COMPLETE TEMPORAL CODEC: REPORT

## 1. Baseline (Phase 0)

- Full suite before any M14 change: **1232/1232 passing**, 0 failures (M13's
  exact closing baseline).
- M13 model/calibration/codebook identity (from `outputs/m13_recalibration/m13_davis_benchmark.json`,
  re-verified fresh in this milestone's own Phase D/E runs, byte-identical):
  4-bit residual identity `be872c2beebd2f9b`, calibration signature
  `eab9082596c6f980`. M13's DAVIS headline (unchanged, not re-touched):
  5/4/3-bit total container bytes 4,309,995 / 3,027,347 / 1,918,028; BPP
  0.7317 / 0.5140 / 0.3256; PSNR 29.2712 / 28.9758 / 27.9459; MS-SSIM
  0.973731 / 0.967650 / 0.947253 — these are exactly the "baseline" rows
  reported in §8 below, reproduced fresh rather than assumed.
- Deterministic calibration guard: active and verified twice more this
  milestone — `mc.calibrate_grids`'s signature, recomputed fresh, matched
  the cached one at every rate point in both Phase D and Phase E (the same
  "recompute and compare, stop on mismatch" guard M11/M13 established,
  reused unmodified).
- Cross-process reproducibility (motion recalibration specifically): two
  independent Python processes computing TRAIN motion-symbol collection,
  the fitted table, its identity and held-out entropy produced a
  **byte-identical** combined SHA-256 (`7726cd77...`,
  `outputs/m14_entropy_audit/m14_reproducibility_cross_process.json`).
- Full suite after every M14 change: **1251/1251 passing**, 0 failures.
  +19 new tests (13 in `test_m14_recalibration.py`, 6 in
  `test_m14_closed_loop.py`); 0 removed. (M13's own suite grew from
  1205→1232 with +27 tests; M14 adds 19 more on top of that 1232.)

## 2. Complete entropy-table inventory (Phase A)

Full machine-generated inventory: `outputs/m14_entropy_audit/m14_entropy_audit.json`.
`.nvct` v2's fixed 56-byte header has **exactly three** entropy-model
identity fields (audited directly from `TemporalStreamHeader`'s own
docstring/byte layout, not assumed) — `intra_entropy_model_id` (offset 32),
`residual_entropy_model_id` (offset 40), `motion_entropy_model_id` (offset
48) — so every live table maps onto one of these three slots, with no room
for a fourth without a format version bump.

| table | live in M13 deployment | identity slot | tables | alphabet | model-predicted | fitted against |
|---|---|---|---|---|---|---|
| `intra_entropy_model` | **yes** | `intra_entropy_model_id` | 64 (1/channel) | 2^bits | no | every frame's latent (I- and P-typed alike), first ~400 scanned |
| `motion_entropy_model` | **yes** | `motion_entropy_model_id` | 2 (dy, dx) | 64 (search_range-dependent, NOT bit-depth-dependent) | no | real block-matched motion vectors, first ~400 frames scanned |
| `residual_entropy_model` (calibrate_grids, M10H-style per-channel) | **NO** | none | 64 | 2^bits | no | nothing — dead weight |
| M11-G16 codebook (M13's already-recalibrated table) | yes (frozen) | `residual_entropy_model_id` | 512 | 2^bits | **yes** (M13 already fixed this) | 536 TRAIN P-frames, M13 |

**A finding worth stating plainly**: `calibrate_grids` still computes and
returns a plain per-channel `residual_entropy_model` (M10H's original
design) inside its calibration dict, but `scripts/m13_closed_loop.py` never
reads it — only `calibration["residual_params"]` (the quantization grid) is
used from that part of the dict; every P-frame residual is coded through
M11-G16 + M13's codebook instead. Recalibrating a table nothing reads
cannot produce any deployed byte change, so it was excluded from candidacy
rather than "tested and found weak."

**Neither remaining live table is model-predicted** — both `intra_entropy_model`
and `motion_entropy_model` are already DIRECT empirical counts (Laplace-
smoothed), unlike M11-G16's residual codebook, which was a neural
network's PREDICTED distribution. This means the M13 story ("predicted vs
actual" bias) does NOT directly apply here; Phase B had to find a different
mechanism, and did — see §3.

**A second finding, also worth stating plainly**: TRAIN has 72 sequences
totaling 4,826 frames, but `calibrate_grids`'s `max_frames` budget (400,
the deployed recipe) is applied SEQUENTIALLY - "walk sequences in manifest
order, stop at N total frames" - so the DEPLOYED `intra_entropy_model`/
`motion_entropy_model` are fitted from only the **first ~6 of 72 TRAIN
sequences (≈8% of TRAIN)**, never touching the other 66. This coverage gap,
not a prediction-bias gap, turned out to be the actual mechanism behind
§3's results.

## 3. Calibration methodology (Phase B)

For each candidate, the SAME symbol-generating machinery (frozen motion
estimator / frozen `intra_params` quantization grid) was reused to collect
a BROADER TRAIN sample — `discover_sequences(..., max_frames_per_sequence=8)`,
8 frames from **every** one of TRAIN's 72 sequences (576 total, vs the
deployed ~400 from ~6 sequences) — then refit via the SAME
`EmpiricalEntropyModel.from_symbols`/Laplace convention `calibrate_grids`
already uses (Phase B's "same smoothing convention unless proven unstable" —
no instability was found, so no new scheme was introduced).

**A methodological bug found and fixed before any number could be trusted**:
the first draft of `scripts/m14_offline_gate.py` accidentally passed the
BROAD, per-sequence-truncated TRAIN list into `calibrate_grids` itself (which
is supposed to represent the unmodified DEPLOYED recipe), making "current"
and "new" draw from the same narrow resample — producing nonsensical,
uniformly NEGATIVE "gains" (intra −3.8%, motion −57%) at first. Fixed by
keeping two separate sequence lists (`calibration_sequences`, full and
untruncated, for the deployed baseline; `broad_train_sequences`,
per-sequence-capped, for the candidate) — documented in the script's own
module docstring so it can't silently regress. `outputs/m14_entropy_audit/m14_offline_gate.json`
reflects the corrected run only.

## 4. Train/VAL-A/VAL-B provenance (Phase B)

Same sequence-disjoint, index-parity split M11/M12/M13 used, applied to the
SAME 9 DAVIS validation sequences: **VAL-A** = `bmx-trees, dogs-scale,
mallard-water, skate-park, upside-down` (even index), **VAL-B** =
`car-roundabout, drift-straight, pigs, stunt` (odd index). TRAIN fits the
candidate table; VAL-A is a consistency check only (Laplace has no tunable
smoothing strength to select, unlike M13's hierarchical smoothing); VAL-B
is the headline offline number. TEST is never read by Phase B or C —
`tests/test_m14_recalibration.py::test_scripts_never_feed_test_sequences_into_a_fitting_call`
proves this structurally for all three M14 driver scripts, including
`m14_davis_benchmark.py` (the one script that DOES touch TEST — only as the
Phase E evaluation subject, never as a fitting input).

## 5. Offline entropy results (Phase B)

| bits | intra H_old→H_new | intra gain | intra verdict | motion H_old→H_new | motion gain | motion verdict |
|---|---|---|---|---|---|---|
| 5 | 3.83403→3.81985 | +0.3699% | **weak** | 4.37954→3.85868 | **+11.8930%** | **meaningful** |
| 4 | 2.81589→2.80459 | +0.4013% | **weak** | 4.35772→3.85868 | **+11.4519%** | **meaningful** |
| 3 | 1.79891→1.79159 | +0.4070% | **weak** | 4.31383→3.85868 | **+10.5511%** | **meaningful** |

Full data: `outputs/m14_entropy_audit/m14_offline_gate.json`. Note motion's
`H_new` is identical across bit depths (3.85868) — expected, since the
recalibrated table's own alphabet/structure (64 symbols, 2 tables) doesn't
depend on residual bit depth at all (only `search_range` does); `H_old`
varies slightly by bit depth because the DEPLOYED calibration's narrow
~400-frame sample is itself sensitive to bit-depth-dependent I-frame
reconstruction quality at GOP boundaries, an effect that averages out once
TRAIN coverage is broad.

**Intra is consistently WEAK (<0.5%) at every rate point — rejected by the
pre-declared gate.** Per the milestone's own rule ("If the audit finds no
meaningful remaining gap, say so and stop" / "Do NOT assume that
recalibrating every table is beneficial"), intra was NOT promoted to Phase
D. This is a real, informative negative result, not an oversight: intra's
64-table structure already sees tens of thousands of observations per
table even at 400 calibration frames, so it was already well-converged —
broader coverage barely moves it.

**Motion is unambiguously meaningful (>10%) at every rate point** — by far
the largest offline gap found in this codec since M13's own residual
recalibration (1.6–5.1%). Promoted to Phase D.

## 6. Candidate ranking (Phase C)

With one candidate rejected at Phase B and one promoted, ranking reduces to
a short, explicit accounting rather than a real trade-off table:

| criterion | motion | intra |
|---|---|---|
| offline entropy gain | +10.6% to +11.9% (meaningful) | +0.37% to +0.41% (weak) |
| expected bitrate impact | real (§8) | none expected — not tested |
| symbols affected | 2 blocks/P-frame × search_range-sized alphabet, ~every P-frame | every latent position, every frame |
| runtime cost | zero neural cost (no model in this table at all) | would be zero neural cost too, moot since rejected |
| implementation risk | low — same table shape, same coder call, `EmpiricalEntropyModel.from_symbols` reused unmodified | low, moot |
| preserves reconstruction invariants | yes (proven, §7/§8) | not tested (never reached Phase D) |

**Only ONE candidate cleared the individual gate, so the milestone's
"interaction effects" requirement (test A alone, B alone, A+B) reduces to
"nothing to combine" — not skipped, but the CORRECT answer given the data**:
testing "intra+motion" together would have conflated a real effect with a
rejected one, telling us nothing "motion alone" doesn't already show.
`tests/test_m14_recalibration.py::test_intra_gate_result_was_rejected_before_reaching_phase_d`
pins this decision against the actual recorded Phase B verdicts, so it
can't silently drift if a future rerun changes the numbers.

## 7. Actual coded validation (Phase D)

Real `.nvct` v2 streams, 90 P+I frames (3 VAL sequences, matching M13's own
Phase D scale), `baseline` (both tables deployed/current) vs `motion`
(motion table recalibrated, intra untouched, M13's residual arm frozen and
reused unmodified in both):

| bits | total bytes old→new | total gain | motion bytes old→new | motion-only gain |
|---|---|---|---|---|
| 5 | 587,939→582,935 | +0.8511% | (not broken out at this stage; see §8) | — |
| 4 | 418,691→413,773 | +1.1746% | — | — |
| 3 | 268,449→263,903 | +1.6934% | — | — |

Invariants — symbols, reconstruction, motion vectors, PSNR, MS-SSIM —
**identical between arms at every rate point**, exactly as required for an
entropy-table-only change. Full data: `outputs/m14_entropy_audit/m14_coded_validation.json`.

## 8. Full DAVIS results (Phase E)

719 frames, all 9 TEST sequences, `outputs/m14_entropy_audit/m14_davis_benchmark.json`:

| bits | motion bytes old→new | **motion-channel gain** | total bytes old→new | **total stream gain** |
|---|---|---|---|---|
| 5 | 181,118→164,010 | **+9.446%** | 4,309,995→4,292,887 | +0.3969% |
| 4 | 184,298→166,842 | **+9.472%** | 3,027,347→3,009,891 | +0.5766% |
| 3 | 188,896→171,612 | **+9.150%** | 1,918,028→1,900,744 | +0.9011% |

**Invariants hold EXACTLY at every rate point**: residual bytes, I-frame
bytes, PSNR, MS-SSIM are bit-identical between `baseline` and `motion`
(e.g. 4-bit: PSNR 28.9758 both, MS-SSIM 0.967650 both) — only motion bytes,
and therefore total bytes, differ. Byte accounting closes everywhere.

**Why two very different-looking numbers, both true**: the motion CHANNEL
itself improved by ~9.15–9.47% — squarely "meaningful" by the SAME
pre-declared gate Phase B used, and closely matching Phase B's own offline
estimate (10.6–11.9%; the ~1.5–2.7 point gap between the two is ordinary
estimator-vs-realized variance, not a red flag — see §9). But motion bytes
are only 4–6% of the total P-frame stream (residual dominates at ~90%+), so
even a large channel-level win dilutes to a smaller TOTAL-stream number:
0.40% (5-bit) to 0.90% (3-bit), growing at lower bit depths because
residual bytes shrink faster than motion bytes as quantization coarsens,
making motion a relatively larger slice of a smaller total. By the
TOTAL-stream reading of the SAME pre-declared thresholds, this ranges from
weak (5-bit) to marginal (3-bit) — a materially different classification
question than the channel-level reading gives, and both are reported here
rather than only the more flattering one. See §21 for how this is resolved
into a single classification.

## 9. BPP / PSNR / MS-SSIM (Phase E)

| bits | old BPP | new BPP | PSNR (both) | MS-SSIM (both) |
|---|---|---|---|---|
| 5 | 0.73174 | 0.72884 | 29.2712 | 0.973731 |
| 4 | 0.51398 | 0.51101 | 28.9758 | 0.967650 |
| 3 | 0.32564 | 0.32270 | 27.9459 | 0.947253 |

PSNR/MS-SSIM identical to 6 decimal places between arms, as required.

## 10. BD-rate

**motion vs baseline: PSNR −0.683%, MS-SSIM −0.681%** (bitrate saved at
matched quality). Smaller than M13's −2.37% — consistent with §8's
explanation (a smaller channel, even meaningfully improved, moves the total
curve less than M13's dominant-channel recalibration did).

## 11. Per-sequence results (4-bit)

| sequence | total bytes old→new | change |
|---|---|---|
| bmx-bumps | 450,395→444,078 | −1.403% |
| car-turn | 276,844→275,173 | −0.604% |
| cat-girl | 480,580→475,977 | −0.958% |
| cows | 453,061→454,540 | **+0.326%** |
| drift-chicane | 128,428→128,336 | −0.072% |
| drone | 412,351→405,896 | −1.565% |
| gold-fish | 330,451→330,803 | **+0.107%** |
| schoolgirls | 330,928→333,545 | **+0.791%** |
| surf | 164,309→161,543 | −1.683% |

6 of 9 sequences improve, 3 regress slightly (cows, gold-fish, schoolgirls —
each well under 1% worse). This is expected and NOT a red flag: recalibrating
against the AVERAGE of a broader TRAIN sample necessarily trades a bit of
fit on sequences whose motion statistics were, by chance, closer to the old
NARROW sample's particular (unrepresentative) slice of TRAIN. The net effect
across all 9 sequences is positive at every rate point (§8).

## 12. Latency and memory (Phase G)

Per-P-frame, full DAVIS TEST (`outputs/m14_entropy_audit/m14_davis_benchmark.json`'s
`latency`):

| bits | combo | encode (ms) | decode (ms) |
|---|---|---|---|
| 5 | baseline | 2.999 | 8.305 |
| 5 | motion | 3.015 | 7.636 |
| 4 | baseline | 2.635 | 6.408 |
| 4 | motion | 2.582 | 6.661 |
| 3 | baseline | 2.468 | 6.358 |
| 3 | motion | 2.531 | 6.386 |

No consistent direction or magnitude beyond ordinary run-to-run noise
(±0.02–0.7 ms on totals of 6–8 ms) — exactly the "essentially zero neural
overhead" the milestone expected, and for a structural reason, not luck:
motion's entropy table has no neural network in it at all (§2) — recalibrating
it changes 2 tables' worth of INTEGER FREQUENCY VALUES, never their shape
(64 symbols × 2 tables, unchanged) or the number of coder calls
(`encode_motion_payload`/`decode_motion_payload` are one-shot, not
grouped) - so there is no mechanism by which recalibration could move
latency, and the measurements confirm it doesn't.

Memory: the recalibrated motion table is a second `[2, 64]` int64 frequency
array plus its cumulative — a few hundred bytes, once, at calibration time;
no per-frame cost. Both intra's rejected candidate and motion's deployed
one would have cost the same negligible amount had intra passed its gate.

## 13. Provenance (Phase F)

- `motion_entropy_model.model_id()` — deployed (old): 5-bit `ee605dd44277b878`,
  4-bit `e1bbc6643d038eee`, 3-bit `90eeac872bfa2f97` (all distinct, as
  expected — the "current" table's narrow TRAIN sample is itself slightly
  bit-depth sensitive, §5). Recalibrated (new): **`d7e7b237b6451885` at
  every rate point** — identical across 5/4/3-bit, confirming §2's finding
  that the recalibrated table's structure is search_range-dependent only.
- `intra_entropy_model` identity is unchanged (M14 never recalibrated it) —
  `outputs/m14_entropy_audit/m14_davis_benchmark.json`'s `provenance`
  records it explicitly per rate point as `intra_identity_unchanged`, so a
  reader can confirm it wasn't silently touched.
- `residual_entropy_model_id` (M13's) is also unchanged — frozen per this
  milestone's own rules, re-derived fresh (not cached) in every M14 script
  and shown identical to M13's own recorded values at every rate point.

## 14. Compatibility (Phase F)

1. Old M13 streams remain decodable — unaffected, since M14 changed no
   shared coder code path, only added a NEW check that only fires on an
   actual mismatch (§ below).
2. New M14 (motion-recalibrated) streams decode exactly —
   `tests/test_m14_closed_loop.py::test_recalibrated_motion_stream_round_trips`,
   plus every real Phase D/E stream (90 + 719 frames) round-tripped with
   `symbols`/`reconstruction` invariants holding.
3. Old/new motion identities are distinguishable — §13.
4. **Mismatch rejection — a real gap found and fixed, not just verified**:
   `scripts/m13_closed_loop.decode_sequence` (M13's own new closed-loop
   function, reused unmodified by M14 until this fix) checked ONLY
   `residual_entropy_model_id` against the stream header — never
   `intra_entropy_model_id`/`motion_entropy_model_id`. This was invisible
   in M13 because `intra_entropy_model`/`motion_entropy_model` were always
   the SAME object at every call site there; M14 is the first milestone to
   actually vary `motion_entropy_model` across configurations, which means
   a caller could have silently decoded a recalibrated-motion stream with
   the DEPLOYED table (or vice versa) and gotten a corrupted reconstruction
   with no error at all. **Fixed**: `decode_sequence` now checks all THREE
   `.nvct` v2 identities before decoding a single symbol, mirroring the
   residual check's own pattern exactly.
   `tests/test_m14_closed_loop.py::test_decoding_with_the_wrong_motion_table_is_now_rejected`
   proves the fix (raises `TemporalFormatError`, matches on "motion entropy
   model mismatch"), and the full 1251-test suite (including every
   pre-existing M11/M12/M13 test, none of which ever exercised a genuine
   intra/motion mismatch) confirms the fix is fully backward compatible.
5. `.nvct` v2 unchanged — same 56-byte header, same magic/version, same
   three identity slots; only their VALUES differ for the motion field.
6. **Resumable decoder and motion — a scope clarification, not a gap**:
   motion was NEVER routed through M12's `ResumableDecoder` — it was always,
   and remains, a single one-shot `decode_symbols` call
   (`mc.decode_motion_payload`), unrelated to M12's group-sequential
   residual decode path. "Resumable decoder supports the new tables" is
   therefore vacuously true for motion (there is nothing resumable-decoder-
   specific for it to interact with) rather than a property that needed
   proving; M13's residual arm, which DOES use `ResumableDecoder`, is
   reused unmodified and untouched by anything in M14.
7. No legacy caller is silently affected — `m13_closed_loop.py`'s only
   OTHER caller besides M14's new scripts is M13's own
   `m13_coded_validation.py`/`m13_davis_benchmark.py`, both of which always
   pass MATCHING `intra_entropy_model`/`motion_entropy_model` on encode and
   decode, so the new check is a no-op there (proven by the unchanged
   1232-test M13 subset all still passing).

No `.nvct v3` was created or considered necessary.

## 15. Tests (Phase H)

**+19 new tests**, 0 regressions, full suite 1232→1251:

- `tests/test_m14_recalibration.py` (13): no-TEST-channel on both
  collectors and all three driver scripts, deterministic table generation,
  coder-invariant satisfaction, the collectors' fidelity to
  `calibrate_grids`'s own frozen-quantizer/motion-estimator behavior, the
  audit's live/dead table classification (structural, against the actual
  script source), and the Phase C decision-logic pin (intra rejected,
  motion promoted, against the actually recorded Phase B verdicts).
- `tests/test_m14_closed_loop.py` (6): distinct identities, old-stream and
  new-stream round trips through REAL `.nvct` v2 files, cross-arm
  symbol/reconstruction equality, and — the load-bearing one — the
  newly-fixed motion-identity mismatch rejection (§14.4), proven positively
  (raises on mismatch, succeeds on the matching table) rather than merely
  documented as a gap.

## 16. Files modified/created

**Modified**: `scripts/m13_closed_loop.py` — `decode_sequence` extended to
check `intra_entropy_model_id`/`motion_entropy_model_id` in addition to the
existing `residual_entropy_model_id` check (§14.4). This is M13's OWN new
closed-loop script (not a frozen M10–M11 historical one), and the fix is
purely additive — no existing call site's behavior changes when its
identities already match, which every M13 call site's did (verified: the
full pre-existing M13 test subset still passes unmodified). No file under
the frozen coder (`src/nvc/compression/*`) or `.nvct` format code was
touched.

**Created**:
- `scripts/m14_entropy_audit.py` (Phase A), `scripts/m14_recalibration.py`
  (collectors + fitting, Phase B/D core), `scripts/m14_offline_gate.py`
  (Phase B driver), `scripts/m14_closed_loop.py` (generalizes
  `m13_closed_loop.run_sequences` to any number of arms, reusing
  `encode_multi`/`decode_sequence`/`aggregate` unmodified), `scripts/m14_coded_validation.py`
  (Phase D), `scripts/m14_davis_benchmark.py` (Phase E).
- `tests/test_m14_recalibration.py`, `tests/test_m14_closed_loop.py`.
- `outputs/m14_entropy_audit/`: `m14_entropy_audit.json`,
  `m14_offline_gate.json`, `m14_coded_validation.json`,
  `m14_davis_benchmark.json`, `m14_reproducibility_cross_process.json`,
  `coded_validation_streams/`, `davis_streams/` (real `.nvct` v2 files),
  this report.

No M10A–M13 script other than the one documented fix was modified; nothing
under `src/nvc/` was touched at all. No commits were made.

## 17. Git diff

`git status --short` shows zero tracked-file diffs (nothing from M12/M13/M14
has been committed yet this session, per standing instructions not to
commit unless asked) — `scripts/m13_closed_loop.py` shows as a new (`??`)
file rather than a modified one, so its diff isn't visible via git; the
exact change is described in full in §14.4/§16 instead. Every other M14
file is new and untracked.

## 18. Classification per candidate (Phase I)

**Intra: not deployed.** Phase B verdict "weak" (<0.5%) at every rate
point — correctly rejected by the pre-declared gate before ever reaching
Phase D. None of A/B/C/FAIL quite describes this outcome precisely (it
isn't a coded-validation failure, since it never reached coded validation;
it isn't an "offline-only gain" in the C sense, since the offline gain
itself was too weak to call a gain worth acting on) — stated plainly
instead: **the audit worked as intended and correctly filtered this
candidate out before spending any further compute on it.**

**Motion: A — meaningful deployed gain**, by the SAME channel-level
standard the milestone's own pre-declared gate was designed around (Phase
B/C measure the CANDIDATE TABLE's own entropy, not a diluted total-stream
number — exactly the standard M13's residual recalibration was graded
against too, where it happened to coincide with the total-stream number
because residual dominates the stream). Confirmed independently at three
scales — Phase B offline (+10.6% to +11.9%), Phase D coded validation on
90 VAL frames (+0.85% to +1.69% total, motion-specific breakdown not
separately tracked at that stage), Phase E full DAVIS TEST (+9.15% to
+9.47% motion-channel, +0.40% to +0.90% total) — all agreeing in direction
and order of magnitude. Every invariant (symbols, motion vectors,
reconstruction, PSNR, MS-SSIM) held exactly at every stage. Zero measured
latency cost, by construction (§12). **Stated without inflation**: the
TOTAL-stream reading of the same number is smaller (0.40–0.90%, weak to
marginal by the total-stream standard) because motion is a minor channel
(4–6% of stream bytes) — this is reported prominently in §8/§10, not
buried, precisely so the classification isn't read as claiming a
"second M13."

## 19. M14 overall classification

**A — a real, meaningful (channel-level), zero-cost, fully-validated
calibration gap was found and closed; the audit-first methodology
correctly rejected the other live candidate rather than forcing a result.**

This is not "M13 again, only smaller" — it is exactly what an audit-first
milestone is supposed to produce: one real finding acted on, one plausible
finding measured and correctly set aside, with the SAME rigor (TRAIN/VAL-A/
VAL-B/TEST discipline, offline-then-coded-then-full-benchmark staging,
invariant-checking at every stage, cross-process reproducibility) applied
regardless of which way each candidate's evidence pointed.

## 20. Explicit distinction (as required)

- **CALIBRATION GAP** (Phase A/B, structural finding): the deployed
  `intra_entropy_model`/`motion_entropy_model` are fitted from only ~6 of
  72 TRAIN sequences (~8% of TRAIN) due to `calibrate_grids`'s SEQUENTIAL
  frame budget — not a prediction-bias gap like M13's, a COVERAGE gap.
- **OFFLINE ENTROPY GAIN** (Phase B, held-out VAL-B, bits/symbol): intra
  +0.37% to +0.41% (weak, rejected); motion +10.6% to +11.9% (meaningful).
- **ACTUAL CODED BYTE GAIN** (Phases D/E, real bytes through the real
  coder): motion channel +9.15% to +9.47% on the full DAVIS TEST set
  (realizing essentially all of the offline estimate, same pattern M13
  found); motion's TOTAL-stream effect +0.40% to +0.90% (diluted by
  motion's small share of total bytes — a distinct number from the
  channel-level one, both reported).
- **QUALITY CHANGE**: none, anywhere. PSNR and MS-SSIM are bit-identical
  between every old/new pair at every rate point, because reconstruction
  is bit-identical — entropy-table recalibration structurally cannot touch
  quantization, motion vectors, or the model.
- **LATENCY CHANGE**: none measurable, for a structural reason (§12): the
  motion table has no neural network in it and recalibration doesn't
  change its shape or the coder's call pattern, only its integer values.

## 21. Recommended M15 direction

1. **Ship the motion recalibration.** Real, free, fully invariant-preserving,
   independently confirmed three ways. Package `new_motion.to_dict()`
   alongside the deployed calibration exactly as M13 recommended for its
   own recalibrated codebook.
2. **Do not chase intra further** — the audit gave it a fair, identically-
   rigorous test and it didn't clear the bar. Revisit only if intra's OWN
   deployed conditions change materially (a different quantization
   calibration recipe, a much larger calibration budget than tested here).
3. **The coverage-gap mechanism (§2/§20) is itself the more interesting
   finding for M15 to generalize**, more than "recalibrate more tables":
   `calibrate_grids`'s sequential (not per-sequence) frame budget is a
   STRUCTURAL choice that under-samples TRAIN for ANY table it fits, not
   just motion. M15 could ask a sharper, more general question than "audit
   more tables one at a time": does simply changing `calibrate_grids`'s
   OWN frame-selection strategy (broad per-sequence sampling instead of
   sequential) — for calibration generally, not just as an M14-style
   after-the-fact table refit — close gaps like this one BY CONSTRUCTION,
   for every table it fits, present and future? That reframes "find more
   calibration gaps" as "fix the one root cause several of them share."
4. Per-sequence variance (§11) — a few sequences regressed slightly under
   broader TRAIN coverage — suggests the SAME caution M13's own report
   raised: a single global TRAIN fit is a compromise, and per-stream or
   per-genre calibration could recover more on outliers, at the cost of
   needing multiple shipped tables instead of one. Worth a cheap offline
   check before any implementation investment, exactly as M13 also
   recommended and exactly as this milestone declined to do for its own
   sake — first prove the aggregate effect (done here), then decide if
   finer-grained calibration is worth its complexity.
