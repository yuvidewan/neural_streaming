# Plan: the second audit (fixed 2026-11-23, before any check runs)

Roadmap item 8 (`docs/ROADMAP_2026-11-17.md`): "Audit was one pass. A second audit of the
entropy coder, the container and the docs." After item 7 (MCL-JCV) the user also asked
whether there is data leakage. So this audit adds a fourth part: **leakage and test-set
reuse**.

This is an **audit, not a codec change**. Nothing is trained, and no setting is tuned. The
audit has no pass/fail as a whole; it closes when every check below has a written result.
Results go to `outputs/p12_audit2/README.md` as one table: check id, what was done, result
(ok / finding / known limit), severity, action.

## Severity (fixed now)

- **Material**: one of these holds:
  - the finding moves any headline BD-rate quoted in README.md by **1 point or more**;
  - it breaks a "real bytes" or "decode-exact" claim;
  - it shows a setting was **picked on a test set** (DAVIS TEST, UVG or MCL-JCV).

  A material finding is fixed, and every affected number is re-measured and re-quoted in
  this milestone, the old number shown next to the new one. A fix that would *improve* our
  numbers gets the same scrutiny as one that hurts them.
- **Minor**: real, but under those limits. Fixed if cheap, otherwise listed.
- **Known limit**: not a bug, but a reader should know it (for example, a GPU-only decoder).
  It is listed in the README limits.

## A. Entropy coder and bitstream (`scripts/p4_gate3.py`, `src/nvc/compression/`)

The current decode-exact check decodes in the **same process**, with the same Python
objects, right after encoding, and then deletes the stream. These checks make it stricter:

- **A1. Fresh-process decode.** Keep a sample of streams:
  - 2 DAVIS TEST videos × 5 λ;
  - 1 UVG video × λ 1e-2 and 1e-4.

  A separate script decodes each one in a **new Python process** from only the `.nvct` file
  and the model files on disk. Pass condition: a SHA-256 of the decoded frames equals the
  hash the encoder wrote, for every stream.
- **A2. Nothing outside the stream.** List every input the decoder reads. Each one must be
  either in the stream or a declared model file bound by the stream's identity hash.
  - Test: decode a stream with the **wrong λ's models**, and with a wrong I-coder or P-coder.
    Each must raise a model-mismatch error and must not return frames.
  - The identity check compares only 8 bytes; report how many models it really covers.
- **A3. Byte accounting.** For the sampled streams, the file size must equal header +
  framing + the sum of the per-frame motion and residual payloads, counted from the file
  itself. Any header field that wide I-frames do not use (for example, the old
  `intra_params`) is counted and reported as overhead we pay but don't need.
- **A4. Range coder edge cases.** All of these must round-trip bit for bit, or raise a clear
  error:
  - symbols at exactly ±`SYMBOL_RADIUS`, and one beyond (must raise);
  - scales at `SCALE_FLOOR` and at the top of the scale table;
  - an all-zero latent;
  - a one-symbol stream;
  - an empty x payload.

  Then a randomized test: 1,000 random arrays through the C coder. If the commented-out
  Python reference is reinstated for the test only, both must give identical bytes.
- **A5. Damaged streams (report only).** Truncate and bit-flip 20 sampled streams. The
  decoder must raise or return frames; it must not hang or crash the process. Silent
  garbage is listed as a known limit; this codec has no error resilience.
- **A6. Cross-device decode (report only).** Decode the 10 DAVIS streams from A1 on the CPU.
  Floating-point nets are expected to differ, so a failure is a **known limit**: decode needs
  the same GPU and kernels. It is stated in the README, not fixed here.

## B. Measurement fairness (container, anchors, BD-rate)

- **B1. Colour-conversion floor of the anchor pipe.** `ffmpeg_round_trip` feeds RGB to
  ffmpeg, which converts to YUV with its default matrix and range, and converts back for
  scoring. That loss is charged to x264/x265 in our RGB metric, so it could **favour us**.
  - Measure: x264 lossless (`-qp 0`) through the same pipe, 4:2:0 and 4:4:4, on DAVIS TEST,
    UVG and MCL-JCV, giving PSNR and MS-SSIM ceilings. Record which matrix and range ffmpeg
    used.
  - **Material** if the 4:4:4 lossless PSNR is below **50 dB** on any set (our points reach
    about 41 dB). In that case the 4:4:4 anchor is re-run with an explicit BT.709 full-range
    conversion, and every vs-x264 number is re-quoted.
  - The 4:2:0 ceiling is reported; its chroma loss is the known reason 4:4:4 is the stricter
    anchor.
- **B2. BD-rate implementation.** Recompute every README headline BD-rate with an
  independent implementation: the standard Bjøntegaard cubic fit of log-rate against
  quality, over the overlap only. Compare with our piecewise-linear `bd_rate`. The
  difference is reported per number. It is **not** a finding unless the two disagree on the
  **sign** of a headline, or by 1 point or more on one quoted as a win.
- **B3. Anchor settings.**
  - ffprobe one anchor stream per arm: the GOP must really be 30, with no scene-cut I-frames,
    and low-delay must really have no B-frames.
  - Check preset `medium`.
  - Report the fixed `FRAMERATE = 30` against the real fps (24/25/30 on MCL-JCV, 50/120 on
    UVG). Re-encode one UVG and one MCL-JCV video at its real fps at 3 CRFs and report the
    byte difference. It is **material** only if a 5-point BD-rate moves by 1 point or more.
- **B4. Same frames, same metric.** For one video per set, hash the uint8 frames our codec
  is scored against and the frames the anchor is scored against; the hashes must be equal.
  Confirm that both are scored by `bp.score_frames`.

## C. Leakage and test-set reuse

- **C1. Training data against test videos (needs a download; see below).**
  - Perceptual-hash every frame of DAVIS TEST, UVG (first 96 frames) and MCL-JCV (first 96
    frames).
  - Re-fetch each Vimeo chunk used in training, **one at a time**. Hash its frames, compare,
    and **delete the chunk** before fetching the next (the delete-after-use rule).
  - A pair closer than a fixed Hamming distance (**≤ 6 of 64 bits**, set now) is a candidate.
    Every candidate is checked by eye from saved thumbnails.
  - A confirmed overlap with a test video is **material**: that video is dropped from every
    affected headline, and the headline is re-quoted with and without it.
- **C2. Test-reuse ledger.** From the plan docs and git history since the A2 gate
  (2026-11-16), list every decision and the split that informed it. Each one gets one of
  three classes:
  - *chosen on validation* (VAL-A or VAL-B): fine;
  - *direction from test*: a test result made us work on something, but no number was
    picked on test. This is the slow kind of reuse; it is listed, not fixed;
  - *picked on test*: a number (λ, steps, threshold, model) was chosen by looking at a test
    result. This is **material**: re-pick it on VAL-B and re-quote the result.

  The ledger also states plainly that MCL-JCV has informed no decision.
- **C3. Split integrity.** The DAVIS VAL-A, VAL-B and TEST video lists must be disjoint, and
  each must match what its plan says. UVG and MCL-JCV must never appear in a training or
  validation manifest.

- **C2 amendment (2026-11-23, written after the ledger was drafted and before any re-pick
  run):** the ledger found one *picked on test* entry. The A2 gate 3 (2026-11-16) ran two
  arms on DAVIS TEST: A2 P-coder + old I-frames, +40.4% vs x264; and + wide I-frames, +39.9%.
  The wide arm became the base of everything after it, so the choice was made on TEST.
  - **Re-pick on VAL-B:** both arms, as in gate 3 (GOP 30, the 3 original lambdas, untuned,
    the rig's motion coding, real bytes, decode-exact), coded on the 4 VAL-B videos with the
    current code.
  - **Rule:** BD-rate PSNR (piecewise-linear) of wide against old. Wide is confirmed if it is
    at or below +0.5%; old wins if wide is above +0.5%. The 0.5-point tie band is the same
    one `PLAN_encode_tuning.md` used; within it, wide stays because the later low-rate and
    MS-SSIM models exist only with wide I-frames.
  - **If old wins:** material. Every headline at the 3 original lambdas is re-quoted with old
    I-frames, and it is stated that the later models (λ 3e-3 and 1e-2, and the MS-SSIM set)
    have no old-I version.

## D. Docs

- **D1. Headline numbers re-derived.** A script reads every headline in the README.md
  "Latest" block, and each `outputs/p*/README.md` verdict since p4, and recomputes it from
  that milestone's JSON. Any mismatch over 0.1 point is listed and fixed in the docs.
- **D2. Stale claims.** The rest of README.md still describes the M22 codec. Either mark it
  plainly as history or move it to a history file; no number changes.
  - The paper figures and text (`docs/paper*`, `docs/research_paper/`) are **only
    compared**, not edited. They are the user's.
- **D3. Commands run.** Every command line in the p4–p11 READMEs runs at least as `--help`
  or a 1-video smoke test with the current code. Broken ones are fixed or marked.

## Order and time (estimates)

| Part | Needs | Time |
|---|---|---|
| D1, D3, C2, C3, B2, B3 ffprobe | CPU | about 3–4 h |
| A2–A5 | CPU (small GPU) | about 2 h |
| A1, A6 (re-encode the sample, keep the streams) | GPU | about 1.5 h |
| B1 lossless floor, 3 sets × 2 formats | CPU | about 1.5 h |
| B3 real-fps re-encodes | CPU | about 30 min |
| C1 Vimeo re-fetch and hashing | **download**, CPU | about 3 h |
| Write-up | | about 1 h |
| **Total** | | **about 1 day of machine time** |

Heavy jobs run one at a time, chained in one detached script.

**C1 needs a new download:** the Vimeo training chunks again, one at a time, each deleted
after hashing (about 2–3 h of download). It runs only after the user approves it. If not
approved, C1 is reported as "not done", and the rest of the audit is unaffected.

## What this audit does not do

No model is trained or changed, and no λ, step count or threshold is tuned. MCL-JCV stays
report-only. If a material finding changes a headline, the new number is reported as it
comes out. The roadmap's goal (item 1) is still judged only by a real-bytes, decode-exact
run against the stricter anchor.
