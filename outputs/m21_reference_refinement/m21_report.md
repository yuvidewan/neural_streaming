# M21 — Causal Reference Refinement

**Status: COMPLETE.** **Classification: C — NO CODED-RATE IMPROVEMENT** (on held-out DAVIS TEST; VAL-B reached B and did not replicate — §7, §10, §12).

## 1. Executive summary

M17 proved the decoded reference costs real bytes; M18 proved shrinking its aggregate error does not recover them; M19 showed the dominant mechanism is the reference error changing the residual *symbol*; M20 closed the routing branch. M21 tested the one remaining direct mechanism: **can a small, causal, decoder-available refinement of the previous reconstruction reduce residual-symbol error and buy coded bytes?**

18 pre-registered candidates — 12 pixel-domain and 5 latent-domain transforms plus the identity control — were run through the real deployed closed loop (`.nvct` v2, real motion coding, real M11-G16 + K=512 + M13 residual coding, real arithmetic coder) on all 246 VAL-B P-frames at 5/4/3-bit, and every one was decoded back from its container and checked. **All 18 are decoder-compatible** — 54 sequence-level round trips with symbols, reconstructions *and* refined reference latents identical every time, zero side information, no container change.

**The rate answer is a qualified negative.** A Phase 2 oracle diagnostic run *before* any candidate was built delivered the milestone's most important finding: at 5-bit and 4-bit, **not one candidate improves any of the four metrics** — not pixel MSE against the oracle, not latent MSE, not symbol agreement, not coded bytes. The deployed reference is a strict local optimum under this family. At 3-bit a coherent subset (the three mildest pixel smoothers) improves **all four together**, which is precisely *not* the M18 failure mode, and that survived into the closed loop: `px_median3_a50` (a half-strength 3×3 median) reached **+0.7141% of the total stream on VAL-B at 3-bit with PSNR and MS-SSIM both up** — a pure win, clearing the 0.5% gate.

**It did not replicate on TEST.** Gated in by VAL-B and locked by the declared Phase 7 rule before DAVIS was opened, the candidate delivered **+0.1411% on the full 719-frame DAVIS TEST at 3-bit** — a fifth of the VAL-B estimate and well below the 0.5% production line — while costing −0.8228% at 5-bit and −0.6341% at 4-bit. Applied at every rate point its BD-rate is +0.059% PSNR (slightly worse) / −0.181% MS-SSIM (slightly better): a wash. The gap is sequence variance, not a bug: **6 of 9 TEST sequences improve**, but `schoolgirls` alone costs +7,971 bytes and flips the aggregate from marginal to weak. VAL-B's four sequences happened to contain no such case.

Two findings survive as real and replicated. First, the mechanism is **not** picture quality: at 3-bit the winner makes pixel MSE against the oracle *worse* (−2.7%) while improving motion SAD (+12.0% of the way to the oracle), residual RMS (+23.9%) and coded bytes (+5.0%) — it is a better *prediction* target, not a better picture, which is exactly the distinction M18 said to insist on. Second, the gain is **GOP-boundary-located**: on TEST at 3-bit the entire effect is at position 1 (+2.4954% of its residual bytes) while positions 2–9 get slightly *worse* (−0.2177%) — independently replicating the boundary spike M16/M17/M19 all found, and pinning the refinement's benefit to it.

**No production change is made or recommended.** `src/nvc/` is untouched, `.nvct` v2 is unchanged, and the identity arm reproduces M14's recorded DAVIS totals **exactly at all three rate points**. Per the milestone's own closing constraint: simple causal refinements have now been tested and they do not close the reference bottleneck. The evidence points to lifting the residual-quantizer/autoencoder freeze (§18).

## 2. Frozen baseline (Phase 0)

Pre-M21 full suite: **1379 passed, 0 failed** (`m21_pre_tests.log`), matching M20's recorded count. Working tree clean at `a7891dd7` apart from M21's own new files (checked in-script, not asserted).

The deployed arm — M13-recalibrated residual tables **plus M14's recalibrated motion table** — was rebuilt from scratch and run through M21's own closed loop on VAL-B:

| bits | residual id | intra id | motion id | P-residual | M17 recorded | match |
|---|---|---|---|---|---|---|
| 5 | `ec858dee10f8a955` | `8fadd1ca200c033b` | `d7e7b237b6451885` | 1,333,275 | 1,333,275 | ✓ |
| 4 | `be872c2beebd2f9b` | `14c4b8bf40056ca1` | `d7e7b237b6451885` | 908,404 | 908,404 | ✓ |
| 3 | `d2d61d66a50cad8a` | `b523711d2e982057` | `d7e7b237b6451885` | 549,083 | 549,083 | ✓ |

M17/M19/M20 produced those totals with a frame-level loop that has **no container and no motion coding**; M21's produces them with the full `.nvct` v2 closed loop. Agreement is therefore a cross-check of two independently written pipelines, not a tautology. Full VAL-B accounting:

| bits | total | P-residual | I-residual | motion | overhead | BPP | PSNR | MS-SSIM | symbols |
|---|---|---|---|---|---|---|---|---|---|
| 5 | 1,625,012 | 1,333,275 (82.05%) | 225,575 | 59,367 | 6,795 | 0.721330 | 29.4393 | 0.981762 | 4,030,464 |
| 4 | 1,140,526 | 908,404 (79.65%) | 165,243 | 60,084 | 6,795 | 0.506270 | 29.1027 | 0.976915 | 4,030,464 |
| 3 | 723,381 | 549,083 (75.91%) | 105,148 | 62,355 | 6,795 | 0.321103 | 27.9431 | 0.960155 | 4,030,464 |

Byte accounting closes at every rate point; all four sequences decode-verified exact (symbols, reconstruction and reference latents). Residual bytes by GOP position, 3-bit, showing the boundary spike this milestone later acts on:

| position | 0 (I) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| frames | 29 | 28 | 28 | 28 | 28 | 27 | 27 | 27 | 27 | 26 |
| bytes/frame | 3,626 | **2,894** | 2,290 | 2,110 | 2,155 | 1,972 | 2,149 | 2,116 | 2,212 | 2,168 |

Position 1 costs **26% more than position 2** — M16/M17/M19's spike, reproduced in the real container. `m21_baseline.json`.

`BASELINE STATUS: CONFIRMED`. A regression test (`test_frozen_baseline_identities_still_match_m19`, `test_recorded_val_b_baseline_matches_m17`) pins all of it.

## 3. Exact reference-path trace (Phase 1)

Measured on a real P-frame, not assumed from earlier reports (`m21_reference_path.json`):

| stage | shape | dtype | device | range | availability |
|---|---|---|---|---|---|
| `previous` | [1,3,256,256] | float32 | cuda | [0.0138, 0.9888] | **both sides** — decoded previous reconstruction (Sigmoid output) |
| `current_frame` | [1,3,256,256] | float32 | cuda | [0.0000, 1.0000] | **encoder only** |
| `current_latent` | [1,64,16,16] | float32 | cuda | [−27.60, 34.86] | **encoder only** |
| `motion` | [2,16,16] | int64 | cuda | [−16, 14] | encoder only *before* coding |
| `decoded_motion` | [2,16,16] | int64 | cpu | [−16, 14] | **both sides** |
| `warped` | [1,3,256,256] | float32 | cuda | [0.0138, 0.9888] | **both sides** |
| `reference_latent` | [1,64,16,16] | float32 | cuda | [−27.24, 34.66] | **both sides** |
| `delta` | [1,64,16,16] | float32 | cuda | [−9.95, 18.89] | **encoder only** |
| symbols | 16,384 | int64 | — | [0, 31] | alphabet 2^bits |
| G16 rows | [16384, 32] | float32 | cuda | — | 492 of 512 prototypes used |

**The motion payload round trip is lossless** (`motion_roundtrip_lossless: true`) — the encoder warps with exactly the vectors the decoder reads, which is what makes a pixel-domain refinement upstream of warping safe.

**Narrowest valid insertion points.** Exactly two tensors are held identically by both sides before the current frame's symbols exist:

- **PIXEL — `previous`.** Upstream of both motion estimation and warping. Refining it also re-aims the motion search, which is legitimate: the decoder never estimates motion, it reads coded vectors, and needs the refined `previous` only to warp.
- **LATENT — `reference_latent`.** Downstream of motion, so motion vectors are untouched (pinned by `test_latent_candidates_leave_the_motion_field_untouched`).

Forbidden inputs, enforced by signature: current source frame, current target latent, current residual, future frames, pre-coding motion vectors.

## 4. Oracle diagnostic (Phase 2) — the milestone's most informative result

Before building anything, every candidate was measured open-loop against M16/M17's oracle reference on all 246 VAL-B P-frames, keeping the four quantities M18 proved do not move together strictly separate.

**Does any candidate move the reference *toward* the oracle?**

| bits | oracle gap (residual channel) | lower pixel MSE | lower latent MSE | higher symbol agreement | fewer coded bytes |
|---|---|---|---|---|---|
| 5 | +2.472% | **NONE** | **NONE** | **NONE** | **NONE** |
| 4 | +8.466% | **NONE** | **NONE** | **NONE** | **NONE** |
| 3 | +19.613% | `px_box3_a25`, `px_binom5_a25`, `px_ae_reproject_a25` | + `px_median3_a50` | `px_median3_a50`, `px_box3_a25`, `px_binom5_a25` | same three |

**At 5- and 4-bit the deployed reference is a strict local optimum**: of 17 transforms × 4 metrics × 2 rate points, 136 comparisons, not one is an improvement. At 3-bit — where the residual quantizer is coarsest and the reference therefore carries the most quantization noise — three mild smoothers improve **all four metrics simultaneously**. That coherence is the signal: it is the opposite of M18's pattern, where MSE moved without bytes following.

Two candidates deserve specific mention because they were the theoretically motivated ones and both failed:

- **`px_ae_reproject`** (`model.decode(model.encode(R))`, the only candidate using the model's own notion of a valid reconstruction) nearly *doubles* pixel error against the oracle at 5-bit (7.51e-5 → 1.36e-4) and costs 5.2% of the residual channel. The autoencoder is lossy, so a round trip of an already-good reconstruction adds a second dose of its own loss rather than projecting toward the oracle.
- **Latent-domain smoothing is catastrophic.** `lat_box3_a50` costs **+39.2% / +49.6% / +57.8%** of the residual channel at 5/4/3-bit. Latent channels are not spatially smooth in a way that tolerates blurring; pixel-domain and latent-domain refinement are not interchangeable.

`m21_oracle.json`.

## 5. Pre-registered candidate definitions (Phase 3)

Frozen in `m21_refinement.CANDIDATES` before any VAL-B coded result was inspected; `m21_sweep.py` may only run what is listed there (`test_candidate_family_is_pre_registered_and_well_formed`). Every definition is closed-form, parameter-free and deterministic.

**Family A — pixel domain, applied to `previous`.** `clamp01` throughout, because `estimate_block_motion` documents its inputs as [0,1].

| name | definition |
|---|---|
| `px_median3` | `clamp01(median_3x3(R))` |
| `px_median3_a50` | `clamp01(R + 0.50·(median_3x3(R) − R))` |
| `px_box3_a25` / `a50` / `a100` | `clamp01(R + α·(box_3x3(R) − R))`, α ∈ {0.25, 0.50, 1.00} |
| `px_binom5_a25` / `a50` | `clamp01(R + α·(binomial_5x5(R) − R))`, α ∈ {0.25, 0.50} |
| `px_unsharp3_b25` / `b50` | `clamp01(R + β·(R − binomial_3x3(R)))`, β ∈ {0.25, 0.50} |
| `px_ae_reproject` | `model.decode(model.encode(R))` |
| `px_ae_reproject_a25` / `a50` | `R + α·(model.decode(model.encode(R)) − R)` |

**Family B — latent domain, applied to `reference_latent`.** `lat_box3_a25`, `lat_box3_a50`, `lat_unsharp3_b25`, `lat_ae_reproject` (`model.encode(model.decode(Z))`), `lat_ae_reproject_a50`.

The unsharp variants exist so the sweep brackets the identity from **both** sides rather than only testing low-pass — without them a null result could not distinguish "smoothing does not help" from "the identity is optimal".

## 6. Decoder-compatibility proof (Phase 4)

Encoder and decoder are implemented separately (`encode_sequence_refined` / `decode_sequence_refined`) and apply the same refinement at the same point from state each side already has: `previous`, which the decoder reconstructed, and the motion field it just decoded. **No side information is transmitted and the `.nvct` v2 header is read, never extended** — all three entropy identities are still checked before a symbol is decoded.

Verified, not argued:

| check | result |
|---|---|
| symbols round-trip exactly | ✓ every candidate, every sequence, every rate point |
| reconstruction identical | ✓ |
| **refined reference latents identical** | ✓ — the quantity that would diverge if the rule were not reproducible |
| motion vectors identical | ✓ (decoder reads coded vectors) |
| VAL-B sequence-level round trips | 18 candidates × 4 sequences at 4-bit + 4 × 4 at 5-bit + 4 × 4 at 3-bit = **104** |
| DAVIS sequence-level round trips | 2 arms × 9 sequences × 3 rate points = **54** |
| container header fields identical across candidates | ✓ (`test_no_side_information_enters_the_container`) |

**No candidate was decoder-incompatible**, so none was discarded on those grounds. A negative control (`test_a_decoder_using_the_wrong_refinement_diverges`) confirms the refinement is load-bearing: decoding a refined stream with the identity rule produces different reference latents, so the equality result above is not vacuous.

Additionally, the identity candidate reproduces `m13_closed_loop.encode_multi`'s deployed arm **byte-for-byte, including the container file itself** (`test_identity_candidate_reproduces_the_deployed_closed_loop`) — the origin every M21 delta is measured from.

## 7. VAL-B results (Phase 5)

Protocol declared in `m21_refinement.py` before any result was read: screen all 18 at 4-bit; confirm at 5- and 3-bit every candidate with a Stage-1 total-stream effect ≥ −0.25%, with a floor of the 3 best-ranked so a bit-depth-dependent candidate cannot be screened out by one rate point. The threshold is a fixed number, never "keep the best".

**Stage 1 — 4-bit, all 18** (positive % = fewer bytes):

| candidate | Δ total | stream % | Δ resid | Δ motion | ΔPSNR | ΔMS-SSIM | dec |
|---|---|---|---|---|---|---|---|
| identity | 0 | +0.0000 | 0 | 0 | 0 | 0 | ✓ |
| `px_box3_a25` | +1,622 | −0.1422 | +1,601 | +21 | +0.0071 | +0.000074 | ✓ |
| `px_median3_a50` | +2,231 | −0.1956 | +2,251 | −20 | +0.0076 | +0.000072 | ✓ |
| `px_binom5_a25` | +4,523 | −0.3966 | +4,468 | +55 | +0.0102 | +0.000078 | ✓ |
| `px_unsharp3_b25` | +10,819 | −0.9486 | +10,785 | +34 | −0.0032 | −0.000044 | ✓ |
| `px_ae_reproject_a25` | +10,983 | −0.9630 | +10,899 | +84 | +0.0003 | −0.000029 | ✓ |
| `px_ae_reproject_a50` | +24,025 | −2.1065 | +23,795 | +230 | −0.0020 | −0.000061 | ✓ |
| `px_unsharp3_b50` | +29,727 | −2.6064 | +29,650 | +77 | −0.0084 | −0.000114 | ✓ |
| `px_median3` | +30,325 | −2.6589 | +30,201 | +124 | +0.0105 | +0.000091 | ✓ |
| `px_binom5_a50` | +34,765 | −3.0482 | +34,723 | +42 | +0.0164 | +0.000099 | ✓ |
| `lat_ae_reproject` | +40,556 | −3.5559 | +40,511 | +45 | −0.0052 | −0.000148 | ✓ |
| `px_ae_reproject` | +54,413 | −4.7709 | +53,771 | +642 | −0.0057 | −0.000154 | ✓ |
| `px_box3_a50` | +20,354 | −1.7846 | +20,308 | +46 | +0.0126 | +0.000091 | ✓ |
| `lat_ae_reproject_a50` | +14,116 | −1.2377 | +14,118 | −2 | −0.0027 | −0.000050 | ✓ |
| `px_box3_a100` | +93,916 | −8.2344 | +93,658 | +258 | +0.0181 | +0.000080 | ✓ |
| `lat_unsharp3_b25` | +112,690 | −9.8805 | +112,593 | +97 | −0.0634 | −0.000603 | ✓ |
| `lat_box3_a25` | +159,766 | −14.0081 | +159,710 | +56 | +0.0318 | +0.000020 | ✓ |
| `lat_box3_a50` | +452,546 | −39.6787 | +452,521 | +25 | −0.0129 | −0.000322 | ✓ |

Every candidate costs bytes at 4-bit. Admitted to Stage 2: `px_median3_a50`, `px_box3_a25` (threshold) + `px_binom5_a25` (floor) — exactly the three the oracle diagnostic had flagged as 3-bit improvers, so the loose threshold did the job it was set loose for.

**Stage 2 — 5-bit and 3-bit:**

| candidate | 5-bit stream % | 4-bit stream % | **3-bit stream %** | best verdict |
|---|---|---|---|---|
| `px_median3_a50` | −0.5004 | −0.1956 | **+0.7141** | **marginal** |
| `px_box3_a25` | −0.3070 | −0.1422 | +0.4005 | weak |
| `px_binom5_a25` | −0.5863 | −0.3966 | +0.3868 | weak |

**`px_median3_a50` at 3-bit is a pure win on VAL-B**: −5,166 bytes (+0.7141%), BPP 0.321103 → 0.318810, **PSNR +0.0175 dB and MS-SSIM +0.000423** — fewer bytes *and* better quality, so no rate/distortion flag applies. I-frame bytes are unchanged (the refinement touches P-frames only); motion moves +12 bytes. All four VAL-B sequences improve (car-roundabout +0.419%, drift-straight +0.241%, pigs +2.215%, stunt +0.366%), all four gain PSNR. Encode cost +1.3% (103.6s → 104.9s).

Gate: **+0.7141% ≥ 0.5% → coded validation and full DAVIS both gated ON.**

## 8. Mechanism decomposition (Phase 6)

For the locked candidate at 3-bit, open-loop, expressed as the fraction of the identity→oracle distance closed:

| quantity | identity | `px_median3_a50` | oracle | fraction of the gap closed |
|---|---|---|---|---|
| pixel MSE vs oracle | 4.836e-4 | 4.965e-4 | 0 | **−2.67% (worse)** |
| latent MSE vs oracle | 0.490501 | 0.476613 | 0 | +2.83% |
| motion SAD | 0.036659 | 0.036068 | 0.031730 | **+11.99%** |
| residual mean abs | 0.719338 | 0.703077 | 0.587106 | +12.30% |
| residual RMS | 1.19681 | 1.17522 | 1.10637 | **+23.87%** |
| symbol agreement with oracle | 81.31% | 81.56% | 100% | +1.33% |
| G16 ideal bits | 4.392e6 | 4.348e6 | 3.530e6 | +4.99% |
| **actual coded bytes** | 549,083 | 543,698 | 441,390 | **+5.00%** |

**The gain does not come from picture quality — pixel MSE against the oracle gets *worse*.** It comes from the reference being a better *prediction target*: motion matches better, the residual is smaller in magnitude, and the symbol distribution shifts toward the oracle's. This is exactly the distinction M18 insisted on after finding that aggregate MSE improvement does not transfer to bytes; M21 finds the converse — a byte gain with a pixel-MSE *loss*. Any future work that optimizes reference refinement against reconstruction error is optimizing the wrong objective.

Per the milestone's six sub-questions: (1) reference pixel quality — *worse*, not the source; (2) reference latent — improved, +2.8%; (3) motion — improved, +12.0%, and motion bytes are essentially unchanged (+12 on VAL-B, −14 on TEST), so this is better matching, not cheaper vectors; (4) residual symbol distribution — improved, +1.3% agreement and +23.9% RMS, the dominant channel; (5) G16 entropy model — +5.0% of ideal bits, tracking the coded bytes almost exactly, so the entropy model is faithfully pricing the better residual rather than being separately improved; (6) codebook assignment — not separately isolated, and M20 already bounded any routing-only component at 9–12% of the excess.

## 9. GOP-position analysis (Phase 6)

| set | bits | boundary (position 1) | ordinary (positions 2–9) |
|---|---|---|---|
| VAL-B | 3 | −1,903 bytes (**+2.3483%**) | −3,275 bytes (+0.6997%) |
| **DAVIS TEST** | 3 | −5,337 bytes (**+2.4954%**) | +2,669 bytes (**−0.2177%**) |
| DAVIS TEST | 4 | −1,326 bytes (+0.4354%) | +20,494 bytes (−0.9789%) |
| DAVIS TEST | 5 | +2,703 bytes (−0.6476%) | +32,597 bytes (−1.0480%) |

**On TEST at 3-bit the entire gain is at the GOP boundary, and ordinary positions get slightly worse.** The boundary effect is remarkably stable across the two sets (+2.35% VAL-B, +2.50% TEST) while the ordinary-position effect is not (+0.70% VAL-B, −0.22% TEST) — which is precisely where the generalization gap comes from. Position 1's reference is the I-frame reconstruction, whose quantization-noise profile differs from a P-frame reconstruction's, and mild median filtering suits it specifically.

This independently replicates the boundary spike M16 (motion), M17 (residual) and M19 (error shape) all found, and localizes a real mechanism to it — the most durable positive finding in M21.

## 10. Gate decision

| stage | best result | threshold | outcome |
|---|---|---|---|
| Phase 5 VAL-B | **+0.7141%** (3-bit, `px_median3_a50`) | ≥ 0.5% | **PASS → marginal** |
| Phase 7 candidate lock | `px_median3_a50`, by the declared rule, before TEST was opened | — | locked |
| Phase 8 coded validation | exact round trips, byte accounting closes, identities unchanged | — | **PASS** |
| Phase 9 DAVIS TEST | **+0.1411%** (3-bit) | ≥ 0.5% | **FAIL → weak** |

The candidate was selected on VAL-B, so grading on VAL-B would be grading on the selection set. **The classification is taken from the held-out DAVIS TEST result**, and both numbers are reported.

## 11. Coded validation (Phase 8)

Triggered and passed. M21's closed loop *is* the coded validation — there is no separate proxy path:

- exact encode/decode on every candidate at every rate point (§6);
- deterministic reproduction (§13);
- byte accounting closes at every rate point (`motion + residual + overhead == container`);
- `.nvct` v2 unchanged, all header fields identical across candidates;
- all five frozen entropy identities unchanged (§14);
- **no production source modified** — `src/nvc/` and every M10–M20 script untouched.

## 12. Full DAVIS TEST (Phase 9)

9 sequences, 719 frames. The only TEST access in M21, after the candidate was locked — `m21_davis.py` refuses to run otherwise (it re-reads `m21_analysis.json` and checks both the locked name and the gate).

**The identity arm reproduces M14's recorded production totals exactly at all three rate points** (`identity_matches_m14_recorded: true` ×3) — the direct comparison against the frozen M13/M14/M20 baseline:

| bits | arm | total | P-resid | I-resid | motion | BPP | PSNR | MS-SSIM |
|---|---|---|---|---|---|---|---|---|
| 5 | identity | 4,292,887 | 3,527,769 | 584,917 | 164,010 | 0.728837 | 29.2712 | 0.973731 |
| 5 | candidate | 4,328,210 | 3,563,069 | 584,917 | 164,033 | 0.734834 | 29.2725 | 0.973756 |
| 5 | **Δ** | **+35,323 (−0.8228%)** | +35,300 | 0 | +23 | +0.005997 | +0.0014 | +0.000025 |
| 4 | identity | 3,009,891 | 2,398,204 | 428,654 | 166,842 | 0.511013 | 28.9758 | 0.967650 |
| 4 | candidate | 3,028,977 | 2,417,372 | 428,654 | 166,760 | 0.514253 | 28.9777 | 0.967732 |
| 4 | **Δ** | **+19,086 (−0.6341%)** | +19,168 | 0 | −82 | +0.003240 | +0.0019 | +0.000082 |
| 3 | identity | 1,900,744 | 1,439,926 | 273,015 | 171,612 | 0.322704 | 27.9459 | 0.947253 |
| 3 | candidate | 1,898,062 | 1,437,258 | 273,015 | 171,598 | 0.322249 | 27.9584 | 0.947707 |
| 3 | **Δ** | **−2,682 (+0.1411%)** | −2,668 | 0 | −14 | −0.000455 | **+0.0125** | **+0.000454** |

**BD-rate with the candidate applied at every rate point: PSNR +0.059% (worse), MS-SSIM −0.181% (better)** — a wash, and not a basis for deployment.

**Per-sequence, 3-bit** — where the generalization gap lives:

| sequence | identity | candidate | Δ | stream % | ΔPSNR |
|---|---|---|---|---|---|
| surf | 100,730 | 98,767 | −1,963 | **+1.949** | +0.0259 |
| gold-fish | 199,147 | 196,162 | −2,985 | **+1.499** | −0.0015 |
| drone | 257,504 | 253,769 | −3,735 | **+1.450** | +0.0218 |
| cat-girl | 312,604 | 310,772 | −1,832 | +0.586 | +0.0159 |
| bmx-bumps | 282,415 | 281,325 | −1,090 | +0.386 | +0.0013 |
| car-turn | 168,253 | 168,186 | −67 | +0.040 | +0.0185 |
| drift-chicane | 73,914 | 74,025 | +111 | −0.150 | +0.0147 |
| cows | 287,715 | 288,623 | +908 | −0.316 | +0.0071 |
| **schoolgirls** | 218,462 | 226,433 | **+7,971** | **−3.649** | +0.0092 |

**6 of 9 sequences improve, three by ~1.5–1.9%** — the effect is real. But `schoolgirls` alone costs +7,971 bytes, more than the other eight sequences' +10,653 of savings combined minus the aggregate. Excluding it the total would be ≈ +0.63%; including it, +0.1411%. VAL-B's four sequences contained no comparable case, which is the whole explanation for the +0.573-point generalization gap — a sampling accident, not a methodological error, and an honest reminder that a 4-sequence offline gate is thin for an effect of this size.

Runtime overhead is negligible: +1.9% / −1.4% / −0.3% encode at 5/4/3-bit (a 3×3 median on a 256×256 frame, once per P-frame). Memory overhead is one extra frame-sized tensor.

## 13. Reproducibility (Phase 10)

The 3-bit VAL-B sweep — identity plus all three Stage-2 candidates — was re-run in a **fresh, independent Python process**. Every recorded quantity is identical: container bytes, residual bytes (I and P), motion bytes, BPP, PSNR, MS-SSIM and ideal bits (`all_identical: true`). `deterministic_kernels()` throughout; the motion table is cached but validated by identity, never by existence. A miniature independent-process check also runs in the test suite.

## 14. Provenance

All five frozen identities are unchanged at every rate point and match M13/M14/M19/M20's recorded values: residual (`ec858dee10f8a955` / `be872c2beebd2f9b` / `d2d61d66a50cad8a`), intra (`8fadd1ca200c033b` / `14c4b8bf40056ca1` / `b523711d2e982057`), motion (`d7e7b237b6451885` at all three), plus the M10L assignment codebook and M13 coding codebook ids. The M14 motion table was refitted from its own broad-TRAIN allocation and **accepted only because its identity matched** `d7e7b237b6451885`.

No `.nvct` container change, no new field, no new format version. Every stream M21 wrote is a normal v2 stream, and all were deleted after verification.

## 15. Tests

**60 new tests** in `tests/test_m21_reference_refinement.py`:

| area | coverage |
|---|---|
| refinement definitions | pre-registration well-formed; identity is the identity; shape/dtype guard; unknown candidate refused |
| determinism | every non-AE candidate, pixel and latent, repeat-identical and range-preserving |
| kernel correctness | `median3` against an explicit replicate-padded window; blends interpolate identity↔smoother; unsharp is the exact sign-flipped counterpart |
| causality | `Refinement.__call__` signature pinned; call sites may not mention frame/delta/symbols; encode and decode take the same inputs; assignment independent of the current group's symbols |
| frozen baseline | identity reproduces `m13_closed_loop` **byte-for-byte including the container file**; byte accounting closes; recorded VAL-B totals pinned; M19 identities pinned |
| decoder equality | 7 candidates × chained frames: symbols, reconstructions **and reference latents** identical; negative control proves the refinement is load-bearing |
| no side information | container header fields identical across candidates; all three entropy identities still checked |
| refinement is real | a non-trivial candidate provably changes the reference; latent candidates leave motion untouched |
| protocol & gates | declared screen/confirm/threshold/floor constants; project thresholds; selection rule uses coded bytes not quality; tie-break prefers cheaper; quality-bought byte savings flagged |
| no TEST fitting | five scripts scanned; VAL-B and TRAIN selection pinned |
| provenance | no production source modified; no new container format; deployed coder and container reused |
| reproducibility | fresh-interpreter subprocess comparison |

**Full suite: 1439 passed, 0 failed** (1379 pre-M21 + 60 new). **Zero regressions.**

## 16. Files changed

**Created** — nothing under `src/nvc/` touched, nothing in any M10–M20 script touched:

- `scripts/m21_refinement.py` — the candidate family, the refined encoder/decoder pair, the frozen rig, the declared protocol.
- `scripts/m21_baseline.py` — Phases 0 and 1.
- `scripts/m21_oracle.py` — Phase 2.
- `scripts/m21_sweep.py` — Phases 3–7.
- `scripts/m21_analysis.py` — Phases 6/7 and the gate decision.
- `scripts/m21_davis.py` — Phase 9.
- `tests/test_m21_reference_refinement.py` — 60 tests.
- `outputs/m21_reference_refinement/` — `m21_baseline.json`, `m21_reference_path.json`, `m21_oracle.json`, `m21_sweep.json`, `m21_sweep_repro.json`, `m21_analysis.json`, `m21_davis.json`, `m21_deployed_motion_table.json`, this report and the run logs.

## 17. Final classification

**C — NO CODED-RATE IMPROVEMENT.**

Not D: every candidate is decoder-compatible, proven by 158 sequence-level round trips with reference latents, symbols and reconstructions all exact and zero side information. Not E: the evidence is unusually clean — 136 open-loop comparisons at 5/4-bit unanimously negative, a coherent 3-bit crossover confirmed in the closed loop, byte-exact reproducibility, and a TEST run whose identity arm reproduces production exactly. Not A or B: the held-out DAVIS TEST result is **+0.1411%** at its single best rate point and negative at the other two, against a 0.5% line; deployed at every rate point the BD-rate is a wash.

Stated plainly: **VAL-B reached B (+0.7141%) and it did not replicate on TEST (+0.1411%).** Reporting B would mean grading on the set the candidate was selected on.

## 18. Recommendation for M22

**The next milestone should lift the residual-quantizer/autoencoder freeze (option 2).**

M21 was the last cheap structural experiment available under the current freezes, and the reason to stop iterating on reference heuristics is now evidential rather than aesthetic:

1. **The deployed reference is a local optimum.** At 5- and 4-bit, 136 open-loop comparisons across 17 transforms and 4 metrics produced *zero* improvements. Not "small improvements" — zero. No local, parameter-free operator moves this reference toward the oracle at the rate points that carry most of the bitrate.
2. **Where refinement does help, it helps for a reason that is nearly exhausted.** The 3-bit gain is quantization-noise removal, it recovers only 5.0% of M17's oracle gap, and on TEST it is confined to GOP position 1. Chasing the remaining 95% with better heuristics is not a promising use of a milestone.
3. **The four milestones now agree on where the problem is.** M18: aggregate reference error is not the lever. M19: the dominant cost is the reference error changing the residual *symbol*. M20: routing is not exploitable. M21: the reference cannot be improved by causal post-processing. Every one of those points at the thing none of them was allowed to touch — **the residual quantizer and the autoencoder that define what a symbol is**.

M22 should therefore be scoped as an explicit, pre-registered freeze-lift on the residual quantizer / autoencoder, with the same discipline these milestones have used: a declared candidate family, a held-out gate, coded-byte primacy, and decoder compatibility proven rather than argued. M21's §8 supplies its objective: optimize the reference as a **prediction target** (residual magnitude and symbol agreement), never as a **picture** — the winner here improved bytes while making pixel MSE worse, and any training loss built on reconstruction error would have rejected it.

Two smaller things worth carrying forward, neither worth a milestone on its own:

- **The GOP-boundary effect is now three-times confirmed and mechanically actionable.** M21 found a real, replicated +2.5% residual saving at position 1 on TEST at 3-bit. A boundary-specific treatment (rather than a global one) is a cheap, well-localized option if M22 wants a low-risk secondary target.
- **The offline gate is too thin.** A 4-sequence VAL-B over-estimated this effect five-fold because it contained no `schoolgirls`-like case. Widening VAL-B, or reporting per-sequence spread alongside the aggregate, would have caught that before TEST was opened — and costs almost nothing.

---

**Explicit answer to the milestone's closing question — what should M22 do?**

> **2 — lift the residual-quantizer/autoencoder freeze.** Not (1): causal reference refinement has now been tested across 18 pre-registered candidates at three rate points and is a local optimum at the rate points that matter. Not (3): M17–M21 have collectively eliminated motion, intra quantization, codebook routing and reference post-processing as the bottleneck, and all four point at the same remaining structure. Not (4): the bottleneck M17 measured is still real and still large (+2.0/+6.7/+14.9% total-stream), so consolidating now would be stopping with the main result unaddressed, not finished.
