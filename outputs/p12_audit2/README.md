# Second audit (roadmap item 8) — 2026-11-23

Plan, with every rule fixed before its check ran: [`docs/PLAN_audit2.md`](../../docs/PLAN_audit2.md)
(two C2 amendments, both committed before their runs: c68c161, af0b7ee). All numbers:
[`results.json`](results.json), built by [`report.py`](report.py) from the per-check JSONs.
No model was trained or changed, and no setting was tuned.

**In short:** the bitstream, the byte counts, the data splits and every README headline hold
up. Both decisions that had been made on a test set were re-made on VAL-B and **came out the
same** (wide I-frames; I-frame every 30). No test frame was found in the Vimeo training data: of 641,907 Vimeo frames, 21 pairs fell under the hash threshold, and all 21 are different scenes on inspection. Two findings are formally *material*
under the plan's rules, and **both make our quoted numbers stricter than they need to be, not
looser**: x264's fixed 30 fps input on UVG, and 1 KB per stream of header we write but don't use.
The BD-rate method matters for the numbers that sit near a tie.

## Findings

| Check | What was done | Result | Severity |
|---|---|---|---|
| A1 fresh-process decode | 12 kept streams (DAVIS bmx-bumps, car-turn × 5 λ; UVG Beauty × 1e-2, 1e-4) decoded in a new process from the file and model files only | **12/12 bit-identical** to the encoder | ok |
| A2 wrong models | the bmx-bumps 1e-3 stream with the 3e-4 coders, the P-coder only, the I-coder only, the 3e-3 coders, and the 3e-4 rig | **all 5 raise** a model-mismatch error; the control decodes | ok |
| A3 byte accounting | header + quantization blocks + 9-byte frame records + payloads, against the file size | **adds up exactly** on all 12; per-frame records match the payloads | ok |
| A3 unused header | wide I-frames don't use the two old quantization blocks, 1,024 B per stream | up to **2.6%** of the smallest stream (DAVIS car-turn, λ 1e-2). Dropping them would move DAVIS by **−0.9 points** vs x264 4:2:0 and **−1.1** vs 4:4:4; UVG and MCL-JCV by −0.05 | material by the 1-point rule, **in our disfavour**; not fixed here (a container change); listed as a free gain |
| A4 range coder | edge cases (±255, smallest/largest scale, all-zero, one symbol, empty input) and 1,000 random arrays, C coder vs the retired Python reference | **all round-trip; 1000/1000 byte-identical** to the reference; out-of-range symbols and empty input raise | ok |
| A5 damaged streams | 10 truncations, 10 single bit flips | truncations: **10/10 raise**; bit flips: **10/10 decode to wrong frames silently**; no hang, no crash | known limit: no error detection in the container |
| A6 CPU decode | the 10 DAVIS streams decoded on the CPU | **0/10 identical** | known limit: decode needs the same GPU and kernels |
| B1 anchor pipe ceiling | x264 lossless through the same RGB → YUV → RGB pipe | 4:4:4: **51.9–52.5 dB** (lowest video), 4:2:0: 37.4 (DAVIS) / 41.7 (UVG) / 37.8 (MCL-JCV) dB lowest | 4:4:4 ok (bar 50 dB); 4:2:0 ceiling is why 4:2:0 flatters us at high quality (already stated) |
| B2 BD-rate method | the 23 headlines with a cubic Bjøntegaard fit and PCHIP, against our piecewise-linear | 5 differ by ≥ 1 point or in sign (table below); **4 of 5 move in our favour** | finding, minor: tie claims depend on the method |
| B3 anchor structure | ffprobe of every arm | low-delay arms: **I at 0/30/60, no B-frames** as designed; default arms: one I-frame and B-frames (encoder default, by design) | ok |
| B3 frame rate | the anchors are fed at 30 fps; re-encoded at the real fps | UVG Beauty (120 fps): x264 needs **+1.6% (4:2:0) / +1.1% (4:4:4) more** bits at real fps; MCL-JCV videoSRC05 (25 fps): +0.4 / +0.2 | UVG: material by the 1-point rule, **in our disfavour** (the 30 fps anchor is the stronger one); MCL-JCV ok |
| B4 same frames | anchor and codec reports: same videos, same pixel counts; loader vs PNG on disk | **identical** on DAVIS, UVG, MCL-JCV | ok |
| C1 Vimeo overlap | pHash of 4,271 test frames against every frame of Vimeo chunks 1–10, ≤ 6 of 64 bits | 641,907 Vimeo frames hashed (all 10 chunks). **21 candidates**, all at exactly 6 bits, from 3 MCL-JCV videos (videoSRC06: 1, SRC28: 4, SRC08: 16). Checked by eye: **all 21 are different scenes** (grey sky vs a cliff face; lit windows vs a bride; a stage couple vs skydivers), matched only on overall brightness layout. Nearest frame per video: DAVIS 8–12 bits, UVG 8–14, MCL-JCV 6–14 | ok: **no overlap**; no video dropped |
| C2 ledger | every decision since the A2 gate, and the GOP choice before it | 2 *picked on test*, both re-made on VAL-B (below); the rest chosen on VAL-B or *direction from test* | ok after re-pick |
| C3 splits | DAVIS TRAIN / VAL-A / VAL-B / TEST lists, folders, manifests, near-duplicates | **disjoint**; no foreign videos; UVG/MCL-JCV in no manifest; **no DAVIS TEST frame within 6 bits** of any TRAIN/VAL frame (closest 12) | ok |
| D1 headlines | all 23 README "Latest" numbers rebuilt from the raw point files with merges written anew | **23/23 match** (also the ranges, the B-frame +0.1..+10.1%, and 6/30, 11/30) | ok |
| D2 stale text | README's M22 sections | marked as history at both entry points; no number changed. The paper (`docs/paper/`) was only compared: its current-codec numbers match `p8_motion`, and it already discloses the GOP choice | ok |
| D3 commands | 6 milestone `report.py` reruns, `p4_gate3 --verdict`, 9 `--help`, 229 unit tests | **every results.json regenerates byte-identical**; all run; the new `--sequence-major` anchor loop had **no test** → added one (25/25 pass) | minor, fixed |

## C1: the Vimeo check in detail

The 21 candidates are only 3 distinct scene pairs, repeated over neighbouring frames:

| Test video | Vimeo chunk | Candidates | What the two frames show |
|---|---|---|---|
| MCL-JCV videoSRC06 | 3 | 1 | an almost empty grey sky vs a sunlit cliff with a bird |
| MCL-JCV videoSRC28 | 7 | 4 | glowing windows on a dark building vs a bride walking on grass |
| MCL-JCV videoSRC08 | 9 | 16 | a stage couple on a dark background vs skydivers against the sky |

A 64-bit pHash keeps only the coarse light/dark layout, so low-detail frames with a similar layout can
land within 6 bits. All 21 side by side: [`c1_candidates_sheet.jpg`](c1_candidates_sheet.jpg).
The Vimeo hashes are kept (`c1_vimeo_hashes_chunk*.npy`, 5 MB), so the check can be re-run at any
threshold without downloading again. No DAVIS TEST or UVG frame came within 8 bits of any Vimeo frame.

## C2: the two decisions made on test data, re-made on VAL-B

| Decision | Made on TEST | VAL-B re-pick (rule fixed beforehand) | Result |
|---|---|---|---|
| Wide vs old I-frames (A2 gate 3, 2026-11-16) | wide, +39.9% vs +40.4% | wide vs old: **−0.59%** (bar ≤ +0.5%) | **wide confirmed** |
| I-frame spacing (P-frame pilot, 2026-11-14) | 30, of 10/20/30 | vs GOP 30: GOP 20 **+1.27%**, GOP 10 **+8.05%** (bar: either below −0.5%) | **GOP 30 confirmed** |

Every stream in these runs decodes bit for bit. So no headline changes. Spacings above 30 were
never tried.

The rest of the ledger, all chosen on VAL-B or only *direction from test* (a test result made us
work on something, but no number was picked on it): A1 vs A2 (VAL-B); encode-tuning steps/lr
(VAL-B); motion λ_m (VAL-B, motivated by UVG's motion share); B-frame settings (VAL-B); the
low-rate models (motivated by cows/schoolgirls on TEST; λ values fixed in the plan); the MS-SSIM
set (motivated by TEST's MS-SSIM; k from CompressAI). MCL-JCV informed nothing.

## B2: where the BD-rate method matters

| Headline | Quoted (piecewise-linear) | Cubic | PCHIP |
|---|---|---|---|
| MS-SSIM set vs MSE set, DAVIS (MS-SSIM) | −22.4% | −20.7% | −22.8% |
| MS-SSIM set vs x264 4:2:0, UVG (MS-SSIM) | +0.4% | −2.8% | −2.3% |
| MCL-JCV MSE set vs x264 4:2:0 (PSNR) | −1.4% | −2.5% | −1.4% |
| MCL-JCV MSE set vs x264 4:2:0 low-delay (PSNR) | −12.3% | −13.9% | −12.4% |
| MCL-JCV MS-SSIM set vs x264 4:2:0 (MS-SSIM) | −0.4% | −2.4% | −2.8% |

The quoted method stays (fixed since M10A). It is the least favourable to us in 4 of these 5.
The DAVIS MS-SSIM win (−20.7% cubic) still clears its −10% bar. Any "tie" claim within about
±3% should be read as method-dependent. All 23 rows: [`d1_b2_headlines.json`](d1_b2_headlines.json).

## The two material findings, and why nothing is re-quoted yet

> **Update 2026-11-25:** the user asked to fix both, and both are now fixed and re-quoted in
> [`outputs/p13_audit_fixes/README.md`](../p13_audit_fixes/README.md). The header fix moved DAVIS by
> about a point as expected. The frame-rate fix needed extra CRFs (0–6) at 120 fps to cover our quality
> range; with them UVG vs x264 4:2:0 moved −18.1 → −20.2%.

Both are real, both are over the plan's 1-point limit, and both mean the codec is **better**
than quoted:
1. **UVG anchors at 30 fps.** x264's CRF rate control depends on the frame rate. Fed at the
   real 120 fps, x264 needs 1.1–1.6% more bits for the same PSNR on Beauty, so our 30 fps
   anchor is the harder one. Re-running all UVG anchors at 120 fps (about 2.5 h) would improve
   every UVG number by about a point.
2. **1 KB of unused header per stream.** Removing it is a container change plus re-coding. It
   would improve DAVIS by about a point and the 1080p sets by 0.05.

The plan says a material finding is re-measured and re-quoted, old next to new. These two would
only flatter the codec, and the stricter numbers are the honest default, so they are **left as
quoted** pending the user's decision.

## Notes on how the audit ran

- The GPU checks (A1–A6, B1, B3) started at 00:49 while C1 was still downloading. The queue
  script waited for "C1 exit" and matched the line of C1's first, crashed attempt. That broke the
  one-job-at-a-time rule. Results are unaffected (GPU and network work), but the timing fields in
  `a_*.json` may be slightly slow.
- C1 re-downloaded each Vimeo chunk (about 9 GB), hashed it inside the zip, and deleted it before
  the next one. At most one chunk was ever on disk.
- Scripts: `bitstream.py` (A1–A3, A5, A6), `a4_range_coder.py`, `fairness.py` (B1, B3, B4),
  `headlines.py` (D1, B2), `c1_leakage.py`, `c3_splits.py`, `run_*.sh`. Kept streams and scratch
  files live in `C:/audit2_scratch` and are not committed.
