# M20 — Codebook-Assignment Hysteresis

**Status: COMPLETE.** **Classification: C — HYSTERESIS DOES NOT IMPROVE CODED RATE.**

## 1. Executive summary

M19 found the 512-entry codebook's assignment is brittle — it flips for 35–55% of positions even in the *smallest* reference-error decile — and that 9–12% of the reference-error excess bits come from positions where the residual symbol is unchanged and only the entropy table moves. M19 recommended one experiment: a stability margin on the assignment. M20 ran it, rigorously, and the answer is a clean negative.

**Hysteresis is fully decoder-compatible and costs bytes at every margin, every state, and every bit depth.** 24 candidate configurations (3 previous-assignment definitions × 8 non-zero margins) were evaluated against the deployed rule on all 246 VAL-B P-frames at 5/4/3-bit. The best of the 72 (state, margin, bit-depth) results is **+0.0007% of the total stream at 4-bit — eight bytes out of 908,404** — statistically indistinguishable from zero, and 700× below the project's 0.5% "weak" line. Every other configuration is worse, monotonically so: at the largest margin tested the cost reaches −2.4% / −4.0% / −6.8% of the total stream at 5/4/3-bit.

The mechanism of the failure is measured, not guessed. Holding a position on its previous assignment makes the actual code length **worse 47–74% of the time** — the argmin is a weak but genuinely positive predictor, and overriding it is a slightly-losing bet taken tens of millions of times. And the milestone's explicitly-warned-about failure mode was observed directly: **14 configurations reduce routing churn while increasing coded bytes**, most starkly `channel_group @ 0.500` at 5-bit, which cuts churn by 7.8 points and routing-only cases by 19.4% while costing 5.5% of the total stream. Per the milestone's own rule these are classified counterproductive, not as churn wins.

Two reference points (clearly separated from the candidate sweep) bound the result. A genuinely decoder-available alternative rule — routing by cross-entropy against M13's *recalibrated coding* tables instead of the original prototypes — is also worse (−1.39% / −1.93% / −3.37%), which retrospectively validates M13's design decision to keep assignment on the original prototypes. And the non-causal per-symbol oracle's spectacular-looking +57–58% is shown to be **vacuous**: it needs 1.7–3.0 bits/position of assignment entropy to save 0.8–1.8 bits/position, so it costs more side information than it saves at every bit depth.

**No production change is made or recommended.** `src/nvc/` is untouched, no `.nvct` stream was written, and all 117 existing M13–M15 streams are proven unaffected. Per the milestone's own framing: this is a useful negative — it establishes that the codebook brittleness M19 observed is real and measurable but **not exploitable as a bitrate optimization under a causally decoder-compatible stability rule**.

## 2. Baseline (Phase 0)

Pre-M20 full suite: **1323 passed, 0 failed** (`m20_pre_tests.log`), matching M19's recorded count exactly. Working tree clean at `a1565ae1` apart from M20's own new, untracked files.

The frozen M11-G16 + M13 residual arm was rebuilt from scratch and checked against M17's recorded byte totals and M19's recorded identities:

| bits | residual identity | assign codebook | coding codebook | real bytes | recorded | oracle bytes | recorded |
|---|---|---|---|---|---|---|---|
| 5 | `ec858dee10f8a955` | `c4bb904fe6f88a5f` | `474d54d3673ade9f` | 1,333,275 | 1,333,275 ✓ | 1,300,318 | 1,300,318 ✓ |
| 4 | `be872c2beebd2f9b` | `624682536c203a54` | `2d319551c6d8bec1` | 908,404 | 908,404 ✓ | 831,497 | 831,497 ✓ |
| 3 | `d2d61d66a50cad8a` | `666c05b46e355317` | `50cd2a7277d8fa97` | 549,083 | 549,083 ✓ | 441,390 | 441,390 ✓ |

All nine identities match M13/M14/M15/M17/M18/M19's own recorded values. `BASELINE STATUS: CONFIRMED`.

M19's routing statistics were **recomputed from scratch** (not read out of M19's JSON) over all 246 VAL-B P-frames — if the ~9–12% routing-only share had not reproduced, M20 would have had no premise:

| bits | churn | symbol-change rate | **routing-only share of excess bits** | symbol-change share |
|---|---|---|---|---|
| 5 | 44.18% | 29.82% | **8.78%** | 91.22% |
| 4 | 59.13% | 25.24% | **12.25%** | 87.75% |
| 3 | 73.03% | 18.69% | **12.03%** | 87.97% |

The full 2×2 reproduces M19's §9 table essentially exactly (M19's 3-bit: 24.0 / 57.0 / 3.3 / 15.7% of positions with shares 0 / 12 / 11 / 77%; M20's 3-bit: 23.79 / 57.52 / 3.19 / 15.50% with shares 0 / 12.03 / 10.66 / 77.31%). `m20_baseline.json`.

## 3. Assignment-rule trace (Phase A)

Every property below was **probed on the real frozen rig**, not read off the source. `m20_assignment_trace.json`.

| property | finding |
|---|---|
| candidate prototypes | all K = 512, no pruning or masking; 492 / 501 / 508 distinct prototypes actually used on the probe frame at 5/4/3-bit |
| cost metric | `cost(i,k) = Σ_s p_i(s)·−log₂ q_k(s)` — cross-entropy, i.e. expected code length **in bits**. Argmin is identical to the KL argmin (they differ by `H(p)`, constant in k) |
| probabilities → code lengths | `q_k` = `frequencies / TOTAL_FREQUENCY` — the **integer coder tables**, not the pre-rounding Lloyd centroids. So the cost scores the table the arithmetic coder will really use |
| tie-breaking | lowest prototype index. Probed on an all-K tie and an exact 2-way tie between indices 7 and 300: numpy and torch reducers both return `[0, 7]` |
| depends on the **current target symbol**? | **No.** Flipping the symbol at the last decoding group's centre moved **zero** assignments |
| depends on causal context? | **Yes, causally.** Flipping a group-0 symbol left every group-0 assignment unchanged and moved later groups — exactly the `G·floor(c/G)` structure |
| encoder state | reference latent; the full symbol plane (only its causal part can reach any assignment); the frozen prototype frequencies |
| decoder state before coding position i | the identical reference latent; every symbol in earlier decoding groups; the identical frozen frequencies |
| prototype index reconstructible at decoder? | **Demonstrated**, not claimed — see §4 |

**Assignment-cost scale** (runner-up advantage = 2nd-best minus best cost), the number that makes M19's brittleness concrete:

| bits | mean | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|
| 5 | 0.00637 | 0.00165 | 0.00385 | 0.00811 | 0.01465 | 0.03683 |
| 4 | 0.00504 | 0.00140 | 0.00328 | 0.00650 | 0.01135 | 0.02821 |
| 3 | 0.00413 | 0.00106 | 0.00252 | 0.00521 | 0.00954 | 0.02527 |

**The median position prefers its chosen prototype over the runner-up by 0.003 bits.** That is M19's brittleness in numerical form: with 512 prototypes packed this tightly, the argmin boundary is genuinely knife-edge, and a small perturbation of `p` flips it. It also explains, in advance, why a margin cannot help: the *decision* is marginal, but so is the *consequence*.

**Numerical soundness.** A float64 recompute of the cost matrix disagrees with the deployed float32 GEMM at **1 position out of 16,384** at 5-bit (0 at 4- and 3-bit), and at that one position the float64 runner-up advantage is **1×10⁻⁹ bits** — an exact tie, not a precision defect. This is *not* an encoder/decoder mismatch: both sides run the identical float32 kernel on the identical input (§4 proves they agree exactly). It is reported because it quantifies how tight the decision boundary is.

## 4. Decoder-compatibility proof (Phase B)

**The proposed state, explicitly.** For each position `i`:

```
previous_assignment  prev(i)        one of three pre-declared definitions (below)
current_candidate    k*(i)   = argmin_k cost(i, k)            (the deployed rule)
candidate_cost               = cost(i, k*(i))
previous_cost                = cost(i, prev(i))
margin                                                         (bits, swept)
advantage(i)                 = previous_cost − candidate_cost   ( ≥ 0 always )

a(i) = prev(i)  if prev(i) is defined and advantage(i) <  margin      [STRICT <]
       k*(i)    otherwise
```

| state | prev(i) | undefined at |
|---|---|---|
| `temporal` | the **final** assignment at the same flat position in the previous P-frame of the same GOP | GOP position 1 |
| `channel_group` | the **final** assignment at the same (h,w) in channel c−G (the previous G16 decoding group) | group 0 |
| `raster` | the **final** assignment at flat index i−1 in the coder's own C-major order | i = 0 |

The strict `<` makes **margin = 0 an exact identity control by construction** — `advantage` is never negative, so no position can ever be held.

**Why every quantity is available on both sides.** `cost(i,·)` depends only on `rows = G16(reference, context_planes(symbols))`, whose planes for channel c use only channels before `G·floor(c/G)` (§3, probed). And choosing a different *table* does not change `rows`: the coding is lossless, so the decoder recovers identical symbols, identical context planes, identical `rows`. **Hysteresis cannot feed back into its own inputs.** All three `prev(i)` definitions are then things the decoder has already produced — the previous frame's assignment, the previous group's assignment, or the previous position's, and within a group every cost is known before *any* symbol of that group is decoded.

**Executable proof, not an argument.** `decode_frame_hysteresis` rebuilds the assignment group-by-group from the payload alone, with **zero side information** and no `.nvct` change:

- **Phase B probe:** 9 (state, margin) combinations × 3 bit depths × 2 chained frames = 54 encode/decode pairs. `decoder_compatible = True` for every one — assignments identical, baseline assignments identical, symbols round-tripped exactly.
- **In the sweep:** 200 further round trips per bit depth (600 total), covering every candidate configuration. **All exact.**
- **Test:** `test_encoder_and_decoder_agree_on_every_assignment` (9 parametrisations), plus `test_no_side_information_is_written_into_the_payload` and signature checks that neither side can take state the other lacks.

**No candidate was invalid.** All three states are decoder-compatible; nothing had to be stopped for needing unavailable information. (The `oracle_table` reference point in §9 *is* non-causal — which is precisely why it is labelled a bound and never a candidate.)

## 5. Margin sweep (Phase C)

The grid and the states are frozen in `m20_hysteresis.py` — declared in the module, never in a caller that could pick them after seeing a number (`test_the_margin_is_declared_before_any_held_out_result_is_read`). Fitting used VAL-B only; **DAVIS TEST was never accessed.**

The brief's grid is `0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10`. Cost units *are* bits of expected code length, so those values are directly interpretable — but Phase C asks the sweep to reach the 90th percentile of the positive cost advantage, so a 10-P-frame pilot measured that distribution first and **0.25 and 0.50 were appended before any full VAL-B result was read.** Measured on all 246 frames, the advantage the margin is actually compared against:

| bits | state | mean | p25 | p50 | p75 | p90 | exactly 0 |
|---|---|---|---|---|---|---|---|
| 5 | temporal | 0.0933 | 0.0028 | 0.0215 | 0.0825 | 0.2281 | 18.9% |
| 5 | channel_group | 0.2856 | 0.0000 | 0.1035 | 0.3903 | 0.7060 | 26.7% |
| 5 | raster | 0.0987 | 0.0052 | 0.0251 | 0.0871 | 0.2416 | 14.9% |
| 4 | temporal | 0.0823 | 0.0036 | 0.0202 | 0.0731 | 0.1981 | 16.7% |
| 4 | channel_group | 0.4175 | 0.0000 | 0.0738 | 0.8193 | 1.2075 | 26.8% |
| 4 | raster | 0.0910 | 0.0074 | 0.0284 | 0.0845 | 0.2216 | 11.5% |
| 3 | temporal | 0.0729 | 0.0028 | 0.0168 | 0.0660 | 0.1811 | 15.6% |
| 3 | channel_group | 0.9095 | 0.0000 | 0.0948 | 1.9187 | 2.5704 | 26.1% |
| 3 | raster | 0.0891 | 0.0064 | 0.0279 | 0.0931 | 0.2261 | 10.0% |

The final grid `0 → 0.5` therefore spans ≈p10 through well past p90 for `temporal` and `raster`, and 0 through ≈p75 for `channel_group` — the required coverage, on the correct scale, in the correct units.

## 6. Routing churn changes (Phase D / F)

Full curves are in §7 and `m20_analysis.json`. Churn against the oracle arm, at the extremes:

| bits | margin 0 | temporal @0.50 | channel_group @0.50 | raster @0.50 |
|---|---|---|---|---|
| 5 | 44.18% | 61.48% (+17.3) | 36.42% (**−7.8**) | 64.84% (+20.7) |
| 4 | 59.13% | 76.23% (+17.1) | 54.21% (**−4.9**) | 74.57% (+15.4) |
| 3 | 73.03% | 89.53% (+16.5) | 68.56% (**−4.5**) | 82.86% (+9.8) |

Two distinct behaviours, both informative. `temporal` and `raster` **increase** churn: once each arm is sticky, the real and oracle chains lock onto *different* histories and drift further apart than the memoryless rule ever let them. `channel_group` genuinely **reduces** churn, by up to 7.8 points — and costs 5.5% of the total stream doing it. Churn reduction is reported here because the milestone asks for it; it is explicitly **not** the success criterion, and §8 shows why that rule was the right one.

## 7. Actual coded-byte changes (Phase D, primary metric)

VAL-B residual payload bytes from the real arithmetic coder, Δ against margin = 0. Positive Δ = **more bytes = worse**.

**5-bit** (baseline 1,333,275 bytes)

| margin | temporal | channel_group | raster |
|---|---|---|---|
| 0.001 | +3 | +1 | +4 |
| 0.005 | +158 | +47 | +147 |
| 0.010 | +459 | +216 | +546 |
| 0.020 | +1,422 | +685 | +1,706 |
| 0.050 | +4,860 | +2,569 | +6,084 |
| 0.100 | +10,621 | +6,606 | +13,764 |
| 0.250 | +24,081 | +27,423 | +37,856 |
| 0.500 | +38,863 | +89,515 | +78,696 |

**4-bit** (baseline 908,404 bytes)

| margin | temporal | channel_group | raster |
|---|---|---|---|
| 0.001 | +28 | **−8** | +8 |
| 0.005 | +285 | +43 | +176 |
| 0.010 | +877 | +235 | +700 |
| 0.020 | +2,247 | +806 | +2,272 |
| 0.050 | +6,708 | +3,021 | +7,496 |
| 0.100 | +15,180 | +6,837 | +16,443 |
| 0.250 | +31,951 | +15,388 | +41,541 |
| 0.500 | +45,202 | +30,167 | +83,124 |

**3-bit** (baseline 549,083 bytes)

| margin | temporal | channel_group | raster |
|---|---|---|---|
| 0.001 | +17 | **−3** | **−1** |
| 0.005 | +445 | +24 | +214 |
| 0.010 | +1,255 | +154 | +820 |
| 0.020 | +3,071 | +580 | +2,427 |
| 0.050 | +9,129 | +2,284 | +8,517 |
| 0.100 | +19,324 | +5,266 | +19,595 |
| 0.250 | +37,835 | +10,722 | +55,025 |
| 0.500 | +49,415 | +15,239 | +111,817 |

Three negative numbers exist in 72 cells: **−8, −3 and −1 bytes**, all at the smallest margin, all at or below one part in 10⁵. They are noise, and the sweep's shape classifier treats them as such (§10).

**Every configuration's margin = 0 point is byte-identical to the deployed `m13.encode_frame_recalibrated`** — verified on 8 frames per bit depth inside the sweep, and pinned by `test_margin_zero_reproduces_the_deployed_assignment_exactly`. Every Δ above is therefore a delta from production, not from a re-implementation.

## 8. 2×2 decomposition (Phase E)

The six questions Phase E asks, answered from the data:

**1–2. Do "same symbol, assignment changed" cases disappear?** Mostly the opposite. The routing-only cell's population, Δ vs margin = 0:

| bits / state | @0.010 | @0.050 | @0.100 | @0.250 | @0.500 |
|---|---|---|---|---|---|
| 5 temporal | +8.13% | +33.25% | +43.48% | +49.47% | +49.51% |
| 5 channel_group | +1.75% | +2.99% | +0.94% | **−4.40%** | **−19.41%** |
| 5 raster | +6.11% | +25.09% | +33.58% | +43.48% | +53.27% |
| 4 channel_group | +0.81% | +0.83% | **−1.55%** | **−6.55%** | **−8.74%** |
| 3 channel_group | +0.52% | **−0.70%** | **−2.78%** | **−5.42%** | **−6.18%** |

Only `channel_group`, at large margins, actually shrinks the routing-only population — by up to 19.4%. Everything else *enlarges* it, for the same reason churn rises: two sticky chains diverge.

**3. How many held positions get a worse assignment?** This is the decisive number.

| bits | margin 0.001 | 0.010 | 0.050 | 0.100 | 0.250 | 0.500 |
|---|---|---|---|---|---|---|
| 5 temporal | 51.00% | 52.58% | 55.50% | 57.24% | 59.03% | 59.89% |
| 4 temporal | 51.04% | 54.84% | 58.95% | 62.17% | 65.13% | 65.63% |
| 3 temporal | 52.54% | 59.58% | 61.81% | 65.47% | 73.05% | 73.68% |
| 3 channel_group | 47.04% | 49.66% | 54.47% | 56.47% | 57.29% | 57.24% |

Holding a position makes its actual code length **worse more often than better**, at essentially every setting, and increasingly so as the margin grows and as quantization coarsens. The argmin under a noisy `p` is a *weak* predictor — barely above a coin flip at the tightest margins — but it is a **positive** one, and hysteresis is a bet against it. The bet is placed tens of millions of times and it loses on aggregate every time.

**4. Does symbol-change behaviour remain unchanged?** Yes, exactly. The symbol-change rate against the oracle is **bit-identical across all 27 configurations** at every bit depth (29.8191% / 25.2395% / 18.6916%). Hysteresis re-routes which table codes an unchanged symbol; it cannot change the symbol.

**5. Does G16 context remain unchanged?** Yes, and necessarily so: the context planes are a function of the decoded symbols alone, and (4) shows those are invariant. This is also what makes the rule decoder-compatible in the first place (§4).

**6. Does actual code length improve or worsen?** It worsens. Mean code length per symbol, 5-bit: 2.64610 bits at margin 0, rising to 2.66719 (`temporal @0.100`), 2.67344 (`raster @0.100`), 2.82378 (`channel_group @0.500`).

**Verdict on the Phase E criterion.** The milestone's desired behaviour is *routing-only churn decreases AND actual coded bits decrease*. That never happens. What does happen, in **14 configurations**, is the explicitly-flagged failure mode: churn decreases while bits increase. Per the milestone's own instruction these are classified **counterproductive** — most clearly `channel_group @0.500` at 5-bit (churn −7.8 points, routing-only cases −19.4%, **coded bytes +6.7%**).

## 9. Reference points (NOT M20 candidates)

Two bounds were measured alongside the sweep and are reported separately so a null hysteresis result can be read as "not exploitable *this way*" rather than "not exploitable at all".

| rule | causal? | 5-bit | 4-bit | 3-bit |
|---|---|---|---|---|
| `coding_metric` — argmin cross-entropy against M13's **recalibrated** coding tables | **yes**, decoder-available | −1.3890% stream | −1.9290% | −3.3652% |
| `oracle_table` — the table minimising the true symbol's actual code length | **no** (reads the symbol) | +57.3799% stream | +51.1672% | +58.2031% |

**`coding_metric` is a real, deployable rule and it is worse.** The deployed assignment scores the *original* prototypes while the coder spends bits on M13's *recalibrated* ones — a mismatch M13 introduced by design. It is tempting to call that a bug. It is not: routing by the recalibrated tables re-routes 77–92% of positions, collapses the assignment entropy from ~8.6 bits to 7.5–7.9, and *loses* 1.4–3.4% of the stream. M13's per-prototype counts were fitted under the original partition; routing by the recalibrated tables breaks the correspondence between the k the counts were fitted for and the k that gets used. **M13's one-way assign/coding split is retrospectively validated as the better choice, not an oversight.**

**`oracle_table`'s +57% is vacuous, and quantifiably so.** It collapses to 17 / 11 / 7 distinct tables (of 512), with assignment entropy 3.02 / 2.27 / 1.72 bits per position — because the table index has become a *channel for the symbol itself*. Compare what it saves against what it would have to transmit:

| bits | code length saved | assignment entropy needed | net |
|---|---|---|---|
| 5 | 2.646 − 0.798 = **1.848** bits | **3.016** bits | **−1.17 bits** |
| 4 | 1.803 − 0.645 = **1.158** bits | **2.274** bits | **−1.12 bits** |
| 3 | 1.090 − 0.252 = **0.838** bits | **1.718** bits | **−0.88 bits** |

It costs more side information than it saves at every bit depth. It is included because it makes the general point precisely: *any* "perfect routing" bound that reads the symbol it routes is measuring a side channel, not an opportunity.

## 10. Rate-response curve (Phase F)

Monotonicity was not assumed; the shape was classified per (bit depth, state) from the measured curve:

| bits | temporal | channel_group | raster |
|---|---|---|---|
| 5 | immediately harmful | immediately harmful | immediately harmful |
| 4 | immediately harmful | immediately harmful | immediately harmful |
| 3 | immediately harmful | immediately harmful | immediately harmful |

"Immediately harmful" here means: no margin produces a gain above 0.01% of the total stream — fifty times below the project's "weak" line, so the guard cannot be masking anything the gate would have cared about — and the best non-zero point is +0.0000% to +0.0007%. **There is no finite optimum, no flat region beyond the noise floor, and no bit-depth-dependent reversal.** The response is monotonically *harmful* in the margin, at every state and every bit depth. It steepens with coarser quantization (`raster @0.500`: −4.85% at 5-bit → −7.29% at 4-bit → −15.43% at 3-bit), which is consistent with §8's finding that "held made it worse" also rises with coarseness.

## 11. GOP position (Phase G)

For the best candidate — `channel_group @ 0.001`, the only configuration with a non-negative mean across bit depths:

| bits | group | frames | baseline bytes | candidate bytes | Δ | routing-only Δ | held |
|---|---|---|---|---|---|---|---|
| 5 | boundary (pos 1) | 28 | 157,784 | 157,783 | −1 | −0.00% | 0.373% |
| 5 | ordinary (2–9) | 218 | 1,175,491 | 1,175,493 | +2 | +0.06% | 0.368% |
| 4 | boundary (pos 1) | 28 | 114,898 | 114,899 | +1 | −0.01% | 0.359% |
| 4 | ordinary (2–9) | 218 | 793,506 | 793,497 | −9 | +0.04% | 0.425% |
| 3 | boundary (pos 1) | 28 | 81,039 | 81,041 | +2 | +0.01% | 0.302% |
| 3 | ordinary (2–9) | 218 | 468,044 | 468,039 | −5 | +0.00% | 0.470% |

**Fraction of M19's routing-only opportunity recovered: −0.03% / +0.08% / +0.02%** — i.e. none, at either GOP position. The effect is too small to be boundary-concentrated or general; it is simply absent. Worth noting structurally: `temporal` hysteresis is *inactive by construction* at GOP position 1 (no previous P-frame), so it could never have addressed the boundary spike M16/M17/M19 all found — a limitation of the mechanism, independent of this null result.

## 12. 5/4/3-bit results

Every qualitative finding holds at all three bit depths: decoder compatibility, monotonic harm, "held made it worse" above 50%, invariant symbols and context, and the counterproductive churn-down/bits-up cases. Magnitudes scale with coarseness (the harm is ~2–3× larger at 3-bit than 5-bit at the same margin), but **no bit depth shows a different shape**. Per the project's robustness convention the result is considered robust rather than a bit-depth artifact.

## 13. Total-stream impact (Phase H)

Converted using the deployed arm's own P-frame residual byte share from `m14_davis_benchmark.json` (82.18% / 79.68% / 75.76% at 5/4/3-bit) — the same conversion M17 used.

| bits | best candidate | Δ bytes | residual % | **total-stream %** | verdict |
|---|---|---|---|---|---|
| 5 | `channel_group @ 0.001` | +1 | −0.0001% | **−0.0001%** | weak |
| 4 | `channel_group @ 0.001` | −8 | +0.0009% | **+0.0007%** | weak |
| 3 | `channel_group @ 0.001` | −3 | +0.0005% | **+0.0004%** | weak |

Best total-stream gain anywhere in the sweep: **+0.0007%**, against a 0.5% gate. `CODED VALIDATION GATED: False`. `FULL DAVIS GATED: False`.

## 14. Coded validation (Phase I) — NOT triggered

Gate not reached (§13), so no candidate was implemented, and correctly so. What *was* verified regardless, because it validates the measurement itself rather than a candidate:

- 600 decoder round trips across every configuration and bit depth — **all exact** (symbols and assignments).
- 24 margin = 0 frames checked byte-for-byte against the deployed `m13.encode_frame_recalibrated` — **all identical**, payload and ideal bits.
- Symbols, codebook indices, entropy code lengths, payload bytes and the three entropy identities all accounted for in `m20_sweep.json` / `m20_baseline.json`.
- Reconstruction is untouched by construction (§8 item 4), so PSNR/MS-SSIM could not have moved.

## 15. Full DAVIS (Phase J) — NOT triggered

Gate not reached. **DAVIS TEST was never accessed by any M20 script** — pinned by `test_m20_scripts_never_touch_the_test_split` across all six scripts and `test_val_b_selection_never_returns_test_sequences`.

## 16. Reproducibility (Phase K)

The 5-bit sweep — all 27 configurations — was re-run in a **fresh, independent Python process**. Every recorded quantity is identical: residual bytes, oracle bytes, ideal bits, held positions, held-made-worse counts, routing churn, mean code length, total excess bits and the Δ against margin 0. `all_identical: true`.

This is a stronger check than it looks: the repro run used a *different* round-trip verification schedule (`--roundtrip-every 0` vs `41`), so the byte-identical result also shows the verification passes do not perturb the measurement. `deterministic_kernels()` throughout. A miniature independent-process check also runs inside the test suite (`test_hysteresis_is_reproducible_in_an_independent_process`).

## 17. Provenance (Phase L)

No candidate adopted, so the obligation is to prove existing streams are unaffected. All **117** `.nvct` streams from M13/M14/M15 were re-parsed with the unmodified `TemporalStreamReader`:

- container format version: **[2]** for every stream — no new version introduced;
- 3 distinct intra identities, 6 residual, 5 motion — **every one attributable to a pre-M20 milestone's own recorded JSON** (M10H/M10I/M11/M13/M14/M15), found by scanning those JSONs rather than against a hardcoded list, with M20's own output directory excluded so it cannot vouch for itself;
- every stream still parses and yields its declared frame count;
- **`.nvct` streams written by M20: 0.**

`EXISTING STREAMS UNAFFECTED: True`. `m20_provenance.json` also records each file's SHA-256 and length.

## 18. Compatibility

No file under `src/nvc/` and no existing `m10`–`m19` script was modified — `test_m20_does_not_modify_any_production_source` checks `git status` directly. No new container format (`test_m20_introduces_no_new_container_format`, all six scripts). The coder is reused, never reimplemented (`test_m20_reuses_the_deployed_coder_rather_than_a_reimplementation`). All global freezes observed: λ = 3e-4, M10F seed-42 checkpoint, M11-G16, M10L K=512 prototypes, M13 frequencies, M14 motion entropy, M15 calibration policy, both quantizers, motion estimator, GOP = 10, the range coder, `.nvct` v2, the evaluation convention, deterministic kernels.

## 19. Tests

**56 new tests** in `tests/test_m20_codebook_hysteresis.py`, covering all twelve Phase M items:

| # | requirement | tests |
|---|---|---|
| 1 | baseline assignment reproduction | cost matrix reproduces `assign_tensor` exactly; cost is cross-entropy against the integer coder tables |
| 2 | margin = 0 exact equivalence | argmin for every state even with a differing previous; **byte-identical to `m13.encode_frame_recalibrated`**; plus a guard that a positive margin is not silently inert |
| 3 | encoder/decoder assignment equivalence | 9 parametrisations (3 states × 3 margins) over chained frames; unknown-state rejection |
| 4 | hysteresis state determinism | repeat-identical per state; inputs not mutated |
| 5 | no unavailable decoder information | `apply_hysteresis` signature pinned (no symbol/target arg); assignment invariant to the current group's symbols; encode/decode signature symmetry; no side channel in the payload |
| 6 | margin sweep determinism | declared grid fixed/ordered/unique; configuration list deterministic and shares margin 0; held-count monotone in margin |
| 7 | routing-only decomposition | 2×2 partitions every position and sums to the total excess; held positions classified better/worse/equal; GOP bucketing |
| 8 | actual byte accounting | `code_with_assignment` matches the deployed coder; ideal bits track the chosen tables with the per-symbol oracle as the floor; diagnostics labelled non-candidates |
| 9 | no TEST fitting | six scripts scanned; VAL-B selection pinned to `split="val"` and a fixed stride; margin declared in the module |
| 10 | provenance | no entropy identity changes when the rule runs; recorded identities still match M19 |
| 11 | compatibility | no production source modified; no new container; deployed coder reused |
| 12 | independent-process reproducibility | fresh-interpreter subprocess comparison across every state × margin |

Also covered: response-shape classification without assuming monotonicity (4 shapes), gate thresholds matching the project-wide ones, and that `verdict()` cannot even see churn.

**Full suite: 1379 passed, 0 failed** (1323 pre-M20 + 56 new). **Zero regressions.**

## 20. Files changed

**Created** — nothing under `src/nvc/` touched, nothing in any M10–M19 script touched:

- `scripts/m20_hysteresis.py` — the rule, the encoder/decoder pair, the frozen rig, the declared sweep.
- `scripts/m20_baseline.py` — Phase 0.
- `scripts/m20_assignment_trace.py` — Phases A and B.
- `scripts/m20_sweep.py` — Phases C–G data collection.
- `scripts/m20_analysis.py` — Phases F/G/H and the decision, pure analysis over recorded JSON.
- `scripts/m20_provenance.py` — Phase L.
- `tests/test_m20_codebook_hysteresis.py` — 56 tests.
- `outputs/m20_codebook_hysteresis/` — `m20_baseline.json`, `m20_assignment_trace.json`, `m20_sweep.json`, `m20_sweep_repro.json`, `m20_analysis.json`, `m20_provenance.json`, this report, and the run logs.

## 21. Final classification

**C — HYSTERESIS DOES NOT IMPROVE CODED RATE.**

Not D: hysteresis *is* decoder-compatible, proven by 654 exact encode/decode round trips with zero side information. Not B or A: the best total-stream result is +0.0007%, against a 0.5% floor for "marginal". The classification is driven entirely by coded bytes — churn reduction was measured, reported, and explicitly refused as a success criterion, which mattered here: 14 configurations reduce churn while costing bytes.

## 22. Recommendation for M21

**Do not pursue codebook routing further.** M20 closes three doors at once: the stability margin (this milestone), the recalibrated-metric rule (§9, also decoder-available, also worse), and the whole class of symbol-reading "oracle routing" bounds (§9, shown to cost more side information than they save). Combined with M19's own bound — routing-only is at most 9–12% of the excess bits — the expected value of further work on the 512-prototype assignment is close to zero.

**Go after the 88–91%.** M17 established the reference bottleneck is real (+2.0 / +6.7 / +14.9% total-stream upper bound); M18 showed shrinking the error's *aggregate magnitude* does not transfer to bytes; M19 showed the dominant cost is the reference error changing the *residual symbol itself*; M20 now shows the remaining 9–12% is not exploitable by re-routing. The mechanism that has never been tested is the one all four milestones point at: **something that changes the residual symbol distribution rather than its entropy coding.** Within the current freezes the concrete candidates are (a) a reference-refinement step applied identically at encoder and decoder — the only thing that could move symbols rather than tables, and the direction M18's negative result explicitly pointed toward; or (b) accepting that the remaining gap requires lifting a freeze (the residual quantizer or the autoencoder), and scoping M21 as an explicit, pre-registered freeze-lift rather than another surgical optimization.

One narrower observation worth recording, from §3: the median position prefers its prototype over the runner-up by **0.003 bits** with all 512 prototypes in use. That is an unusually tight packing, and it suggests K = 512 may be past the point of useful resolution for this predictor — a cheap, well-scoped question (does K = 128 or 256 cost anything?) that bears on latency and table memory even though M20's evidence says it will not move the rate.

---

```
margin | 5-bit Δbytes | 4-bit Δbytes | 3-bit Δbytes | churn reduction | total-stream effect | verdict
-------+--------------+--------------+--------------+-----------------+---------------------+--------
                                    state = temporal
 0.001 |           +3 |          +28 |          +17 |          +0.08% |            -0.0017% | weak
 0.005 |         +158 |         +285 |         +445 |          -1.02% |            -0.0320% | weak
 0.010 |         +459 |         +877 |       +1,255 |          -3.16% |            -0.0928% | weak
 0.020 |       +1,422 |       +2,247 |       +3,071 |          -6.53% |            -0.2361% | weak
 0.050 |       +4,860 |       +6,708 |       +9,129 |         -11.34% |            -0.7158% | weak
 0.100 |      +10,621 |      +15,180 |      +19,324 |         -14.67% |            -1.5507% | weak
 0.250 |      +24,081 |      +31,951 |      +37,835 |         -16.91% |            -3.1689% | weak
 0.500 |      +38,863 |      +45,202 |      +49,415 |         -16.97% |            -4.3926% | weak
                                  state = channel_group
 0.001 |           +1 |           -8 |           -3 |          -0.01% |            +0.0004% | weak
 0.005 |          +47 |          +43 |          +24 |          -0.20% |            -0.0033% | weak
 0.010 |         +216 |         +235 |         +154 |          -0.46% |            -0.0184% | weak
 0.020 |         +685 |         +806 |         +580 |          -0.71% |            -0.0643% | weak
 0.050 |       +2,569 |       +3,021 |       +2,284 |          -0.38% |            -0.2461% | weak
 0.100 |       +6,606 |       +6,837 |       +5,266 |          +0.79% |            -0.5778% | weak
 0.250 |      +27,423 |      +15,388 |      +10,722 |          +3.07% |            -1.5064% | weak
 0.500 |      +89,515 |      +30,167 |      +15,239 |          +5.72% |            -3.4219% | weak
                                     state = raster
 0.001 |           +4 |           +8 |           -1 |          +0.05% |            -0.0003% | weak
 0.005 |         +147 |         +176 |         +214 |          -0.59% |            -0.0180% | weak
 0.010 |         +546 |         +700 |         +820 |          -2.05% |            -0.0694% | weak
 0.020 |       +1,706 |       +2,272 |       +2,427 |          -4.60% |            -0.2131% | weak
 0.050 |       +6,084 |       +7,496 |       +8,517 |          -8.37% |            -0.7358% | weak
 0.100 |      +13,764 |      +16,443 |      +19,595 |         -11.19% |            -1.6647% | weak
 0.250 |      +37,856 |      +41,541 |      +55,025 |         -14.34% |            -4.5229% | weak
 0.500 |      +78,696 |      +83,124 |     +111,817 |         -15.31% |            -9.1895% | weak
-------+--------------+--------------+--------------+-----------------+---------------------+--------
     0 |           +0 |           +0 |           +0 |          +0.00% |            +0.0000% | identity control
```

Positive Δbytes = more bytes = worse. "Churn reduction" is averaged over the three bit depths; negative means churn *increased*. "Total-stream effect" is averaged over the three bit depths; the verdict uses the best of the three.

**Explicitly:**

- **M19 routing-only opportunity** — 8.78% / 12.25% / 12.03% of reference-error excess bits at 5/4/3-bit (23,140 / 75,362 / 103,645 bits = 2,893 / 9,420 / 12,956 bytes on VAL-B).
- **M20 maximum possible target** — +0.1783% / +0.8263% / +1.7875% of the total stream at 5/4/3-bit. This is the ceiling if the routing-only excess were eliminated *completely*, which would itself require the oracle's assignment — information no real decoder has.
- **M20 measured gain** — **−0.0001% / +0.0007% / +0.0004%** of the total stream. Best across the entire sweep: **+0.0007%** (4-bit, `channel_group @ 0.001`, −8 bytes of 908,404).
- **M20 fraction of routing-only opportunity recovered** — **−0.03% / +0.08% / +0.02%**. Effectively zero at every bit depth.
- **Is hysteresis worth keeping?** **No.** It is decoder-compatible and correctly implemented, and it does not reduce coded bytes at any margin, in any state, at any bit depth. It is not adopted, not shipped, and not recommended. The codebook brittleness M19 observed is real — the median prototype decision is won by 0.003 bits — but it is **observable without being exploitable** under the current entropy objective.

*A successful M20 would only ever have addressed the 9–12% routing-only component. It does not, and in no reading does M20 solve the reference bottleneck: the dominant 88–91% mechanism, where reference error changes the residual symbol itself, is entirely untouched by this milestone.*
