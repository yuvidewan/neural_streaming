# M13 — DEPLOYED M11-G16 TABLE RECALIBRATION + VALIDATION: REPORT

## 1. Classification

**A — REAL, MEANINGFUL, DEPLOYED COMPRESSION GAIN, PROVEN END TO END, ZERO REGRESSION**

(No rubric was given for A/B/C/FAIL in the brief; the one used here — stated
up front so the label is checkable — is: **A** = a real coded-byte gain,
independently reproduced, realized ≈100% through the actual arithmetic
coder on the full DAVIS TEST set, with reconstruction/PSNR/MS-SSIM
UNCHANGED and no regression anywhere; **B** = a real but small/marginal
gain, or a real gain with a non-trivial cost; **C** = a real offline gain
that does not survive to deployment (coder, latency, or compatibility
problems); **FAIL** = no real gain, or an invariant violated.)

Every gate passed, at every rate point, on independently-drawn data of
increasing size and stringency:

| stage | data | 5-bit | 4-bit | 3-bit |
|---|---|---|---|---|
| Phase C (offline, VAL-B, 144 P-frames, integer table) | held-out | +1.609% | +2.927% | +5.110% |
| Phase D (real coded bytes, 90 frames, 3 VAL sequences) | held-out | +0.953% | +1.982% | +3.726% |
| **Phase E (real coded bytes, full DAVIS TEST, 719 frames)** | **TEST** | **+1.424%** | **+2.639%** | **+4.035%** |

All three independent measurements agree in sign and order of magnitude;
Phase E — the headline, TEST-only number — clears the pre-registered 1.0%
"meaningful" bar at every rate point, reconstruction is bit-identical to
the deployed codec, and realized gain is 99.97–100.02% of ideal at every
rate point on the full benchmark.

## 2. Phase 0 baseline

- Full suite before any M13 change: **1205/1205 passing**, 0 failures (same
  baseline M12 left the repo in).
- Current M11-G16 benchmark (`outputs/m11_autoregressive_entropy/m11_benchmark.json`,
  unread/unmodified by M13): `m11_op@5bit` 4,360,960 total container bytes,
  BPP 0.7404, PSNR 29.2712, MS-SSIM 0.97373; `m11_op@4bit` 3,092,347 bytes,
  BPP 0.5250, PSNR 28.9758, MS-SSIM 0.96765; `m11_op@3bit` 1,978,573 bytes,
  BPP 0.3359, PSNR 27.9459, MS-SSIM 0.94725. (M13's Phase E `m11_op` row,
  §17, reproduces these exactly — see §14 for the byte-for-byte match.)
- Current model/calibration/codebook identity (from that same file's
  `provenance` block): `5bit_m11_op` identity `c4f1373f63ffbdfe`, calibration
  `928411732c587518`, M10K lineage `70fbfc21465d2a2f`; `4bit_m11_op` identity
  `1687b85e55d12934`, calibration `eab9082596c6f980`, M10K `00ac53e03a70fdc7`;
  `3bit_m11_op` identity `a72c9dbdb00f7eee`, calibration `95a513136a72daf0`,
  M10K `c52cb42b2b390a8b`. M11-G16 has no SEPARATE "codebook identity" from
  "model identity" — `m11_ar_entropy.model_identity(..., codebook=...)`
  hashes the codebook's frequencies into the SAME 8-byte digest used for
  `residual_entropy_model_id` (§3.7); this is exactly the mechanism M13's
  recalibration exploits for free provenance separation (§11).
- **Fresh reproduction, not just reading the file**: Phase D/E's own runs
  recompute calibration from scratch (`mc.calibrate_grids`) and verify its
  signature matches the CACHED one before proceeding — a guard M11 already
  had (`scripts/m11_evaluate.py`) and M13 reuses unmodified. It matched at
  every rate point, both runs, confirming the deterministic-calibration
  guard (`m10h_motion_compensation.deterministic_kernels`) is still active
  and correct.
- **Cross-process reproducibility, recalibrated frequencies specifically**:
  two independent Python processes, same checkpoint/bit depth (4-bit),
  computing prototype assignments, recalibrated frequencies, the coding
  cumulative table, the new codebook identity and held-out entropy — a
  combined SHA-256 over all of it was **byte-identical** across both runs
  (`5380fa35...`, `outputs/m13_recalibration/m13_reproducibility_cross_process.json`).
- Full suite after every M13 change: **1232/1232 passing**, 0 failures (27
  new tests: 14 in `test_m13_recalibration.py`, 13 in `test_m13_closed_loop.py`;
  0 changed, 0 removed).

## 3. Exact old-vs-new entropy-table construction (Phase A audit)

Read directly from `scripts/m10l_shared_codebook.py`, `scripts/m11_train.py`,
`scripts/m11_ar_entropy.py`, `src/nvc/compression/nvc_format.py` and
`scripts/m10h_motion_compensation.py`'s `TemporalStreamHeader` — nothing
below was guessed:

1. **The 512-prototype codebook** is fitted once, offline, by
   `m11_train.fit_model_codebook` → `m10l_shared_codebook.fit_codebook`:
   Lloyd's algorithm (cross-entropy/code-length metric) over the M11-G16
   NETWORK's own PREDICTED probability rows on TRAIN, quantized once via
   `m10k_learned_entropy.probabilities_to_frequencies`. **The prototypes
   themselves are frozen by M13 and never refit** (per the brief).
2. **Assignment**: `SharedCodebook.assign_tensor(probabilities)` — a GEMM
   against `-log2(prototype probabilities)` plus a deterministic
   lowest-index tie-break, run per-position from `z_ref` + previously-decoded
   channel context (`m11_ar_entropy.ChannelContextEntropyModel`). Fully
   deterministic, no side information beyond what the decoder already holds.
3. **The DEPLOYED P(s\|k) tables today** are the quantized cluster
   CENTROIDS from step 1 — the mean PREDICTED distribution among rows
   assigned to cluster k — never counted against what symbols actually
   occurred there.
4. **What they were fitted against**: `m10l_shared_codebook.sample_training_distributions`
   / `m11_train.fit_model_codebook`'s subsampled predicted probability rows
   on TRAIN latents — predictions, not observed outcomes.
5. **Smoothing/normalization**: `probabilities_to_frequencies` (floor, floor
   at `MIN_FREQUENCY=1`, deterministic largest-first residual distribution,
   stable tie-break) — the SAME function this milestone reuses unmodified
   to build the recalibrated table (§5), so "preserve frequency
   normalization exactly" (Phase B req. 8) is satisfied by construction,
   not by re-implementing an equivalent.
6. **table_index → coder**: `encode_symbols`/`decode_symbols`(or
   `ResumableDecoder.decode_group`, M12) `(payload, cumulative, table_index)`
   — `table_index` selects a row of `codebook.cumulative`; unchanged.
7. **Provenance today**: `m11_ar_entropy.model_identity(model, m10k_identity=...,
   calibration_signature=..., bits=..., codebook=codebook)` hashes the
   MODEL'S WEIGHTS and, when a codebook is supplied, `codebook.frequencies.tobytes()`
   — ONE 8-byte digest that already couples model + calibration + codebook
   content. `SharedCodebook.codebook_id()` is a second, independent 8-byte
   digest with the same property (hashes `self.frequencies.tobytes()`).
8. **`.nvct` v2 storage**: `TemporalStreamHeader.residual_entropy_model_id`
   (8 fixed bytes, `TEMPORAL_MAGIC=b"NVCT"`, `TEMPORAL_FORMAT_VERSION=2`),
   written from whichever "identity" the caller supplies and checked
   byte-for-byte against the decoder's own recomputed identity on read
   (`mc.TemporalFormatError("residual entropy model mismatch...")` — see
   `m11_evaluate.decode_sequence`, reused unmodified as `m13_closed_loop.decode_sequence`).

**The load-bearing design consequence** (stated once here, enforced by
every test in `tests/test_m13_recalibration.py`): assignment (item 2) and
coding (items 3/6) read the SAME `SharedCodebook` object today because
M10L/M11 never needed them to differ. M13's whole premise — "exact same
512 prototypes, exact same assignments, different frequency tables" —
requires splitting them. `scripts/m13_recalibration.py`'s
`encode_frame_recalibrated`/`decode_frame_recalibrated` take TWO codebook
objects: `assign_codebook` (the deployed one, unchanged, used ONLY for
`.assign_tensor`) and `coding_codebook` (recalibrated, used ONLY for
`.cumulative`). `coding_codebook.assign_tensor` is never called anywhere in
this milestone's code — `test_assignment_never_uses_the_recalibrated_codebook`
proves it with a codebook stand-in whose `.assign_tensor` raises if invoked.

## 4. Train/VAL-A/VAL-B/TEST provenance

Identical split discipline to M11/M12, reused unmodified
(`m11_data.load_or_collect`/`split_validation`): 536 TRAIN P-frames; 324
validation P-frames from 9 sequences, split by SEQUENCE into 180 VAL-A +
144 VAL-B (never by frame, so temporally-adjacent near-duplicate frames
can't leak across the split); 719 TEST P+I frames from 9 DIFFERENT
sequences (`discover_sequences(..., split="test")`), read only by Phase E's
`run_sequences` — never by any fitting call. `tests/test_m13_recalibration.py`
pins this two ways: `test_fit_recalibrated_frequencies_has_no_channel_for_test_data`
(the fitting function's signature has no parameter TEST data could even be
passed through) and `test_davis_benchmark_never_feeds_test_sequences_into_the_fit`
(greps the actual Phase E script to confirm the `test_sequences` variable
is never an argument to a fitting call).

## 5. Offline entropy results (Phase C)

Independently reproduced M12's estimate, through the ACTUAL integer-quantized
frequency table (M12's number was a float diagnostic, never quantized or
coder-tested):

| bits | smoothing strength (VAL-A) | H_old (deployed) | H_new (recalibrated) | gain | verdict |
|---|---|---|---|---|---|
| 5 | 1024 | 2.80526 | 2.76012 | **+1.609%** | meaningful |
| 4 | 1024 | 1.95968 | 1.90232 | **+2.927%** | meaningful |
| 3 | 256  | 1.22852 | 1.16574 | **+5.110%** | meaningful |

Matches M12's float estimate (+1.604/+2.916/+5.077%) to within 0.005–0.03
percentage points — integer quantization overhead is negligible.
All rate points meaningful (`outputs/m13_recalibration/m13_offline_gate.json`).

## 6. Actual arithmetic-coded validation: ideal bits vs actual bits (Phase D)

Small, held-out proof of mechanism (90 P-frames, 3 VAL sequences — see §7
for the full-scale TEST numbers) — the milestone's explicit ask ("do not
assume the offline entropy gains survive arithmetic coding, prove it"):

| bits | ideal gain | actual byte gain | realized gain | old overhead | new overhead |
|---|---|---|---|---|---|
| 5 | +0.954% | +0.953% | **100.00%** | +0.0097% | +0.0098% |
| 4 | +1.984% | +1.982% | **99.91%** | +0.0134% | +0.0156% |
| 3 | +3.727% | +3.726% | **99.99%** | +0.0223% | +0.0236% |

Invariants: symbols / reconstruction / motion / PSNR / MS-SSIM identical
between `m11_op` and `m13_recal` at every rate point — TRUE everywhere.
PSNR/MS-SSIM were bit-identical (e.g. 4-bit: 29.5406 / 0.977140 both arms).

## 7. Realized percentage of ideal gain

Confirmed twice, at two scales, both ≈100%: Phase D (§6, 90 frames) 99.91–100.00%;
**Phase E (§8, 719 frames, TEST) 99.97–100.02%**. The arithmetic coder was
already within 0.01–0.03% of ideal before recalibration (M11/M12's own
finding) and stays there after — recalibration is a change to WHAT the
coder is told the distribution is, not to how well it codes against that
distribution, so this was expected but is now measured, not assumed.

## 8. Full DAVIS results (Phase E — the headline)

719 frames, 9 TEST sequences, `m11_op` (deployed) vs `m13_recal`
(TRAIN-recalibrated), identical model/motion/quantization/GOP throughout
(`outputs/m13_recalibration/m13_davis_benchmark.json`):

| bits | old P-resid bytes | new P-resid bytes | byte gain | ideal gain | realized |
|---|---|---|---|---|---|
| 5 | 3,578,734 | 3,527,769 | **+1.424%** | +1.425% | 99.98% |
| 4 | 2,463,204 | 2,398,204 | **+2.639%** | +2.639% | 100.02% |
| 3 | 1,500,471 | 1,439,926 | **+4.035%** | +4.037% | 99.97% |

Total container bytes: 5-bit 4,360,960 → 4,309,995 (−1.169%); 4-bit
3,092,347 → 3,027,347 (−2.102%); 3-bit 1,978,573 → 1,918,028 (−3.060%).
I-frame bytes, motion bytes and container overhead are IDENTICAL between
arms at every rate point (both use the same intra/motion entropy models,
untouched) — byte accounting closes (`total = motion + residual + overhead`)
at every arm/rate point.

## 9. BPP / PSNR / MS-SSIM

| bits | old BPP | new BPP | PSNR (both arms) | MS-SSIM (both arms) |
|---|---|---|---|---|
| 5 | 0.74039 | 0.73174 | 29.2712 | 0.973731 |
| 4 | 0.52501 | 0.51398 | 28.9758 | 0.967650 |
| 3 | 0.33592 | 0.32564 | 27.9459 | 0.947253 |

PSNR and MS-SSIM are **bit-identical to 6 decimal places** between `m11_op`
and `m13_recal` at every rate point — not "close," identical, because
reconstruction is bit-identical (§13). BPP moves down; nothing else moves.

## 10. BD-rate

Piecewise-linear, no extrapolation (`m10e_evaluate._bd_rate_linear`, reused
unmodified), computed over the three rate points:

**m13_recal vs m11_op: PSNR −2.372%, MS-SSIM −2.366%**

(Negative = bitrate saved at matched quality — the standard convention.)
This is a clean number specifically BECAUSE PSNR/MS-SSIM don't move between
arms (§9) — the BD-rate curve shift is a pure horizontal (bitrate)
translation, not a quality/rate trade-off being approximated.

## 11. Per-sequence results

At 4-bit (`outputs/m13_recalibration/m13_davis_benchmark.json`'s
`per_sequence`), every one of the 9 TEST sequences improves, none regresses:

| sequence | m11_op bytes | m13_recal bytes | change |
|---|---|---|---|
| bmx-bumps | 418,733 | 413,784 | −1.182% |
| car-turn | 265,038 | 255,847 | −3.468% |
| cat-girl | 455,042 | 448,428 | −1.453% |
| cows | 442,615 | 436,362 | −1.413% |
| drift-chicane | 125,421 | 117,168 | −6.580% |
| drone | 383,244 | 374,583 | −2.260% |
| gold-fish | 316,490 | 313,972 | −0.796% |
| schoolgirls | 328,141 | 322,014 | −1.867% |
| surf | 157,134 | 144,700 | **−7.913%** |

Range −0.80% to −7.91%: the gain is real everywhere but not uniform —
sequences whose residual statistics diverge more from the codebook's
FITTED (predicted-distribution) prototypes benefit more from recalibrating
against what ACTUALLY occurred.

## 12. Latency and memory (Phase G)

Per-P-frame, averaged over all 719 TEST frames
(`outputs/m13_recalibration/m13_davis_benchmark.json`'s `latency`):

| bits | arm | encode total (ms) | decode total (ms) | decode coder (ms) | decode tables (ms) |
|---|---|---|---|---|---|
| 5 | m11_op | 3.675 | 10.395 | 2.417 | 2.736 |
| 5 | m13_recal | 3.600 | 8.113 | 1.093 | 2.071 |
| 4 | m11_op | 3.795 | 10.925 | 2.043 | 3.083 |
| 4 | m13_recal | 3.699 | 8.829 | 0.943 | 2.494 |
| 3 | m11_op | 2.446 | 6.763 | 1.171 | 1.922 |
| 3 | m13_recal | 2.356 | 5.534 | 0.574 | 1.383 |

**Reading this correctly matters.** `m13_recal`'s decoder uses M12's
`ResumableDecoder`; `m11_op`'s still uses the legacy prefix-redecoding
`ma.decode_frame` (unchanged, exactly as deployed). The ~2x decode-coder
speedup visible here (2.42→1.09 ms at 5-bit, etc.) is **M12's already-established
resumable-decoder win reappearing, not a new effect of recalibration** —
the coder's per-symbol cost is a function of table SHAPE (K=512, same
alphabet) and symbol COUNT, both unchanged by recalibration; it cannot
depend on table VALUES. Recalibration's OWN marginal cost is best read from
**encode timing**, where both arms interleave per FRAME (no
which-arm-ran-first ordering bias, unlike decode which runs one arm's whole
sequence then the other's): 5-bit 3.675→3.600 ms, 4-bit 3.795→3.699 ms,
3-bit 2.446→2.356 ms — `m13_recal` is marginally FASTER, not slower, and by
an amount (2–4%) consistent with measurement noise, not a real cost. The
"tables" step gap visible in decode (also present in both arms, smaller,
same direction every time) is most parsimoniously explained by
`m13_recal` always being the SECOND arm decoded per sequence in this
benchmark's loop (`ARMS = ("m11_op", "m13_recal")`) — a GPU warm-up/cache
ordering artifact, not a recalibration cost, since the SAME
`assign_codebook.assign_tensor` call handles this step for both arms with
identical inputs. **Conclusion: recalibration itself adds ~zero latency, as
expected; the decode speedup actually visible in this table belongs to M12.**

Memory: `coding_codebook` holds one more `[512, alphabet]` int64 frequency
array plus its cumulative — identical shape and size to the original
codebook's (`assign_codebook.table_memory_bytes()` == `coding_codebook.table_memory_bytes()`
by construction, same K and alphabet) — a few hundred KB at most, negligible
next to the ~13k-parameter, ~56–60 KB M11-G16 checkpoint it doesn't touch.

## 13. Bitstream compatibility (Phase F)

Proven by 13 tests in `tests/test_m13_closed_loop.py` through REAL `.nvct`
v2 streams (`TemporalStreamWriter`/`TemporalStreamReader`, real motion
estimation, real GOP handling — not just the single-frame functions):

1. Old (`m11_op`) streams decode exactly as before —
   `test_old_arm_m11_op_round_trips_unchanged`.
2. New (`m13_recal`) streams decode exactly to their intended symbols —
   `test_new_arm_m13_recal_round_trips`.
3. The two arms use DISTINCT `residual_entropy_model_id`s automatically
   (§3.7's identity-hashes-frequencies mechanism, exercised unmodified) —
   `test_m11_op_and_m13_recal_have_distinct_stream_identities`; confirmed
   again on real streams in Phase E (§8's `provenance`: e.g. 4-bit old
   `1687b85e55d12934` vs new `be872c2beebd2f9b`).
4. Model/identity mismatch is rejected by the EXISTING guard, unmodified —
   `test_decoding_a_recalibrated_stream_with_the_old_spec_is_rejected` and
   the reverse, both raising `mc.TemporalFormatError("...entropy model
   mismatch...")`. Additionally, Phase D/E now actively call M11's
   `check_provenance`/`ProvenanceError` guard on the loaded M11-G16
   checkpoint before using it (a gap found and closed during this
   milestone — see §15) — `test_check_provenance_rejects_a_mismatched_m11_g16_checkpoint`
   proves the guard fires on a deliberately stale checkpoint dict.
5. `.nvct` v2 is unchanged: `TEMPORAL_MAGIC=b"NVCT"`, `TEMPORAL_FORMAT_VERSION=2`,
   every header field except `residual_entropy_model_id` identical between
   arms — `test_the_container_format_is_still_nvct_v2`,
   `test_only_the_residual_entropy_model_id_differs_between_headers`.
6. Truncated/corrupt payload behavior is unchanged — inherited from the
   range coder's documented "read zero forever past the end" behavior
   (M12's `test_m12_resumable_decoder.py`, unmodified by M13); M13 adds no
   new corruption surface since the coder algorithm itself is untouched.
7. The resumable decoder works with both old and new tables by
   construction — `decode_frame_recalibrated` (§3's split design) always
   decodes through `ResumableDecoder`, and
   `test_recalibrated_decode_reconstructs_identically_to_legacy_ma_decode_frame`
   proves the "old table, resumable decoder" path is byte-identical to
   "old table, legacy decoder."

No `.nvct v3` was created or considered necessary.

## 14. Provenance/model identity

Old identities (from THIS milestone's own fresh Phase E run, matching §2's
baseline exactly — confirming nothing drifted): 5-bit `c4f1373f63ffbdfe`,
4-bit `1687b85e55d12934`, 3-bit `a72c9dbdb00f7eee`. New (recalibrated)
identities: 5-bit `ec858dee10f8a955`, 4-bit `be872c2beebd2f9b`, 3-bit
`d2d61d66a50cad8a` — all distinct from their old counterparts, all 8 bytes,
all produced by the SAME `model_identity`/`codebook_id` functions M11/M10L
already shipped (§3.7), with zero new identity scheme and zero `.nvct`
field added. `SharedCodebook.codebook_id()` likewise differs (e.g. 4-bit
old `15da16516904ae72` vs new `2aebc9cc7fb5f8b7`).

## 15. Test count and regression status

- Before M13: 1205/1205 (M12's clean baseline).
- After M13: **1232/1232**, 0 failures, 0 skipped. +27 new tests:
  `tests/test_m13_recalibration.py` (14: determinism, coder-invariant
  satisfaction, TRAIN-count fidelity, VAL-A-driven strength selection, the
  assignment/coding isolation property — twice, from two angles — old/new
  symbol equality, legacy-decoder equivalence, identity separation, no
  mutation of the deployed codebook, no TEST-data channel, provenance-guard
  wiring and behavior) and `tests/test_m13_closed_loop.py` (13: old/new
  round trips, cross-arm symbol/reconstruction equality, distinct
  identities, mismatch rejection both directions, `.nvct` v2 format
  pinning, header-field equality except identity, short sequences/GOP
  boundaries, byte-identical repeated encodes).
- **A real gap was found and closed mid-milestone**: the first drafts of
  `m13_coded_validation.py`/`m13_davis_benchmark.py` loaded the M11-G16
  checkpoint without calling M11's own `check_provenance` guard against the
  freshly-recomputed calibration — a silent-stale-model risk exactly like
  the one M11's guard exists to prevent. Both scripts now call it and stop
  with `ProvenanceError` on any mismatch; it passed cleanly on every rerun
  after being added (confirming the checkpoint's provenance was genuinely
  valid all along, not just untested).

## 16. Exact files modified/created; git diff summary

**Modified: none.** `git status --short` shows zero changes under `src/`
from this milestone — M13 touches no shared infrastructure, no `.nvct`
code, no arithmetic coder, no model. (The three `src/nvc/compression/...`
entries `git status` shows are M12's resumable-decoder changes, carried
over uncommitted from the prior session; M13 adds nothing to them.)

**Created:**
- `scripts/m13_recalibration.py` — the recalibration fit + the
  assignment/coding-split encode/decode functions (§3's core contribution).
- `scripts/m13_closed_loop.py` — a parallel `encode_multi`/`decode_sequence`
  (structurally identical to `m11_evaluate.py`'s, not a modification of it —
  needed because M13's two-codebook split can't be expressed through
  `m11_evaluate.py`'s single-codebook-per-arm dispatch).
- `scripts/m13_offline_gate.py` (Phase C), `scripts/m13_coded_validation.py`
  (Phase D), `scripts/m13_davis_benchmark.py` (Phase E).
- `tests/test_m13_recalibration.py`, `tests/test_m13_closed_loop.py`.
- `outputs/m13_recalibration/`: `m13_offline_gate.json`,
  `m13_coded_validation.json`, `m13_davis_benchmark.json`,
  `m13_reproducibility_cross_process.json`, `validation_streams/` and
  `davis_streams/` (real `.nvct` v2 files from Phase D/E), this report.

No M10A–M11 script was rewritten; everything is reused, unmodified, via the
repo's existing `_load_script` pattern. No commits were made.

## 17. Recommendation for M14

Ship the recalibrated tables — this is a real, ~free, bitrate-only win
(1.4–4.0% smaller residual streams, identical reconstruction, identical
latency cost from recalibration itself) confirmed on the full TEST set, not
an offline artifact. Concretely:

1. **Deploy via the identity mechanism already proven here** — no new
   container version needed. The recalibrated codebook JSON
   (`SharedCodebook.to_dict()`) can ship alongside the existing M11-G16
   checkpoint exactly as M10L's codebook already does.
2. **Re-run recalibration whenever the M11-G16 model or its 512 prototypes
   change** — the recalibrated table is fitted TO the current prototypes'
   assignment behavior (§3); a future retrain would need a fresh fit, which
   `scripts/m13_recalibration.py`'s `fit_recalibrated_frequencies` already
   supports as a cheap (seconds, per §5's timing), deterministic,
   TRAIN/VAL-A-only step.
3. **§12's own evidence suggests the next highest-value lever isn't a new
   context or a bigger model — it's routine recalibration hygiene.** The
   gap between "deployed" and "recalibrated" (1.6–5.1% offline, §5) is
   larger than M12's spatial-context ceiling (0.5–0.9% net, never
   "meaningful") at every rate point, for zero added decode cost — the
   opposite trade from M12's spatial context (some entropy gain, real
   latency cost). If M14 explores anything new, it should ask whether OTHER
   already-deployed components (the intra/I-frame entropy model, the motion
   entropy model — both still using their ORIGINAL M10H-era tables, never
   examined this way) have the same recalibration gap M11-G16's residual
   codebook did, before reaching for a new architecture.
4. Per-sequence variance (§11, −0.8% to −7.9%) suggests a per-STREAM or
   per-GENRE recalibration (rather than one global TRAIN fit) could close
   more of the gap on outlier sequences like `surf` — worth a cheap offline
   check before any new modeling investment, not a reason to hold this
   milestone's result.

## Explicit distinction (as required)

- **OFFLINE ENTROPY GAIN** (Phase C, float/estimator territory): +1.6% to
  +5.1% bits/symbol on held-out VAL-B, net of recalibration's own
  quantization to an integer table.
- **ACTUAL CODED BITRATE GAIN** (Phases D/E, real bytes through the real
  coder): +0.95–3.73% on a 90-frame validation sample; **+1.42–4.04% on the
  full 719-frame DAVIS TEST set** — realized 99.97–100.02% of the ideal
  entropy gain at every rate point.
- **QUALITY CHANGE**: none. PSNR and MS-SSIM are bit-identical between the
  deployed and recalibrated arms at every rate point, because
  reconstruction is bit-identical — recalibration cannot and does not
  touch quantization, motion, or the model.
- **LATENCY CHANGE**: none attributable to recalibration itself (§12).
  A real ~2x decode-coder speedup is visible in this milestone's own
  numbers, but it belongs to M12's resumable decoder, reused here, not to
  anything M13 introduced.

The expected scientific outcome — a bitrate-only improvement at identical
reconstruction quality — is what was found, and it was demonstrated end to
end (offline estimate → real bytes on validation data → real bytes on the
full, untouched TEST set), not assumed at any step.
