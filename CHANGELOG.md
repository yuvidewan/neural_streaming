# Changelog

This file is the running record of fixes and changes made to this codec,
newest entry first. It exists so anyone (Aditya included) can catch up on
what changed and why without reconstructing it from commit messages.

**Maintenance rule:** when we ship something new, add a new dated entry at
the top. If a later fix changes or supersedes an earlier entry's numbers or
conclusions, that later entry says so explicitly and links back — we don't
silently rewrite history here, but we do keep this file accurate as the
single current picture, not just an append-only log of what we believed at
the time.

---

## 2026-09-07 — M10E: final λ selection with two seeds — improvement durable, λ NOT resolved

**Source:** M10D left the converged optimum unbounded below (its best arm, λ = 6.0e-4, was the
smallest λ it tested) and noted that with a ~1.4-point noise floor and ~2-point differences, one run
per λ was marginal. M10E answers both: it extends the sweep downward **and** runs two independent
seeds per λ.
**Tests:** 738 passing (full suite), zero regressions. 18 new tests.
**Scope:** no architecture, quantizer, entropy-coder or `.nvc` change; no `src/nvc/` change at all;
`train_autoencoder.py`, `m10c_convergence.py` and `m10a_pilot.py` left byte-identical.

### Design

Five λ × two seeds (42, 43) = **10 runs**, 18,120 steps each (30 epochs × 604 batches — the M10C/M10D
budget, so results are directly comparable), all from the M8-QAT checkpoint `90d51157…`, identical
batch/LRs/QAT/scale-tracking. **Only λ and seed vary.**

| arm | λ | role |
|---|---|---|
| CTRL | 0 | matched control / noise floor |
| LOWER | 3.0e-4 | below M10D's best |
| MID_LOW | 4.5e-4 | between 3e-4 and M10D's best |
| CURRENT_BEST | 6.0e-4 | M10D's winner, replicated |
| UPPER_REFERENCE | 7.5e-4 | local shape above the best |

A fail-closed **fairness preflight** (18 boolean checks) ran before training and is recorded in
`training_summary.json`: checkpoint hash, two seeds, five design λ, exactly one control, every
selected λ carrying every seed, distinct output dirs, no pre-existing checkpoints, identical
deterministic rate-estimator init, scale tracking on everywhere, `resume-model-only` semantics,
shared batch/LRs/QAT/epochs/manifest. All passed.

BD-rate is computed **paired per seed** — each run against the control of *its own* seed — so
seed-level effects in the control cancel.

### Noise floor, re-measured independently

CTRL s42 vs CTRL s43 are identical but for the seed, so their gap **is** run-to-run noise:

| bits | s42 BPP | s43 BPP | ΔBPP | ΔPSNR |
|---|---|---|---|---|
| 8 | 1.8444 | 1.8451 | +0.04% | +0.054 |
| 6 | 1.3420 | 1.3428 | +0.05% | +0.056 |
| 4 | 0.8310 | 0.8317 | +0.08% | +0.023 |

**CTRL-vs-CTRL BD-rate = −1.62%**, i.e. a **1.62-point floor**. M10D's independent estimate from a
different pair of runs was 1.41 points — two independent measurements agreeing to ~0.2 points is
good evidence the floor is real and stable.

### Actual `.nvc` at convergence (DAVIS test, 719 frames, fresh per-run calibration)

All 30 calibrations (10 models × 3 depths) passed the 2% guard at **0.120–0.195%** clipping,
train-split only, 400 frames, each fitted to its own checkpoint.

| λ | 8-bit BPP / PSNR | 6-bit | 4-bit |
|---|---|---|---|
| CTRL (mean) | 1.8447 / 30.026 | 1.3424 / 29.879 | 0.8313 / 28.032 |
| LOWER 3.0e-4 | 1.7691 / 30.110 | 1.2665 / 29.998 | 0.7594 / 28.660 |
| MID_LOW 4.5e-4 | 1.7593 / 30.092 | 1.2568 / 29.976 | 0.7501 / 28.606 |
| CURRENT_BEST 6.0e-4 | 1.7535 / 29.984 | 1.2510 / 29.865 | 0.7452 / 28.555 |
| UPPER_REFERENCE 7.5e-4 | 1.7489 / 29.735 | 1.2465 / 29.626 | 0.7416 / 28.410 |

Every rate-aware λ **strictly dominates CTRL at 4-bit** (lower BPP, higher PSNR, higher MS-SSIM).

### Paired BD-rate: rate loss wins big, λ choice does not separate

| λ | s42 | s43 | mean | spread | ×floor |
|---|---|---|---|---|---|
| **LOWER 3.0e-4** | −17.18% | −16.39% | **−16.78%** | 0.79 | 10.4× |
| **MID_LOW 4.5e-4** | −16.85% | −16.21% | **−16.53%** | 0.64 | 10.2× |
| CURRENT_BEST 6.0e-4 | −14.81% | −12.24% | −13.52% | 2.57 | 8.3× |
| UPPER_REFERENCE 7.5e-4 | −15.46% | −2.96% | −9.21% | 12.49 | 5.7× |

Pairwise, against the 1.62-point floor:

| comparison | gap | verdict |
|---|---|---|
| LOWER vs MID_LOW | 0.25 pts (0.2×) | **tied within noise** |
| MID_LOW vs CURRENT_BEST | 3.01 pts (1.9×) | marginal |
| LOWER vs CURRENT_BEST | 3.26 pts (2.0×) | distinguishable, barely |
| LOWER vs UPPER_REFERENCE | 7.57 pts (4.7×) | distinguishable |

So: **λ = 3e-4 and 4.5e-4 are the top two and are indistinguishable from each other**, and their edge
over M10D's winner (6e-4) is ~3 points — about 2× the floor. Real, but modest.

### Three caveats that stop this from being a clean lock

1. **One run's evaluation is contaminated.** UPPER_REFERENCE@s43 is the only run of the ten with a
   meaningful best-vs-final gap (**4.08%**; the other nine are 0.00–1.23%). Its final epoch 70 dipped
   ~0.3 dB (PSNR 29.88 → 29.55 in one epoch), and the fixed "evaluate the final 18,120-step snapshot"
   convention locked that dip in. That single epoch produces its −2.96% BD-rate and the 12.49-point
   spread. **λ = 7.5e-4 is not as bad as its mean suggests** — its clean seed gives −15.46%. The
   convention was not changed after the fact; it is the same one M10D used, and changing it
   mid-experiment would have been exactly the silent design change the brief forbids.
2. **MS-SSIM ranks the λ almost in reverse.** CURRENT_BEST −19.57%, MID_LOW −19.46%,
   UPPER_REFERENCE −19.43%, LOWER −19.06% — total spread **0.51 points**. The PSNR-based preference
   for the low end is *not* corroborated by the perceptual metric, which sees all four as equivalent.
3. **On the clean seed alone, the whole λ range spans 2.37 points (1.5× floor).** Seed 42:
   −17.18 / −16.85 / −14.81 / −15.46 across 3e-4 → 7.5e-4, and not monotonic. The λ-dependence inside
   [3e-4, 7.5e-4] is weak.

### The optimum is still unbounded below

LOWER (3e-4) is again the smallest λ tested and again the best on PSNR — the same open flank M10D
had. The mitigating evidence is that the curve is **flattening**: 7.5e-4 → 6e-4 → 4.5e-4 → 3e-4 gives
−9.21 → −13.52 → −16.53 → −16.78, so the last step buys only 0.25 points against 3.01 for the one
before it. That is consistent with sitting near the bottom of a broad basin rather than still
descending — but it is inference from shape, not a measured minimum.

### Proxy vs actual — two questions, kept apart

**A. Does proxy R order actual `.nvc` bitrate?** **Yes, at all three depths.** Seed-averaged cross-λ
ordering is perfectly monotonic at 8/6/4-bit (Spearman +1.000 at 8 and 6-bit). The analysis flags
`rank_agreement=False` at 4-bit (Spearman +0.976 over the 8 rate-aware runs), but the single
inversion is **between the two seeds of the same λ** (LOWER s42 0.7594 vs s43 0.7593) — a **0.0053%**
difference, ~15× smaller than the 0.08% control-to-control BPP noise at that depth. It is a
noise-level tie inside one λ, not a cross-λ ordering failure.

**B. Does the λ minimising proxy R also minimise BD-rate?** **No.** Lowest proxy R is
UPPER_REFERENCE (7.5e-4); best BD-rate is LOWER (3e-4). The proxy estimates **rate**, not the
rate/quality tradeoff, and must never be used alone to choose λ. This reproduces M10D's finding.

### M10D cross-check — strong reproducibility

λ = 6.0e-4 seed 42, M10D vs M10E, independently trained and independently calibrated:

| bits | M10D BPP | M10E BPP | ΔBPP | ΔPSNR |
|---|---|---|---|---|
| 8 | 1.7530 | 1.7534 | +0.02% | +0.012 |
| 6 | 1.2506 | 1.2509 | +0.02% | −0.049 |
| 4 | 0.7447 | 0.7451 | +0.06% | −0.003 |

Bitrate reproduces to **0.06%** and PSNR to **0.05 dB** across milestones. The pipeline is stable; the
noise lives in quality, not rate.

### Pareto

All four rate-aware λ are non-dominated at all three depths; CTRL is dominated everywhere. As noted
in M10D, a λ sweep necessarily spreads points along a curve, so this is close to vacuous — BD-rate is
the discriminating metric.

### Verdict: DURABLE IMPROVEMENT BUT λ NOT YET RESOLVED

What is settled, at 10 runs and two seeds:

- Rate-aware training beats the matched control by **−9% to −17% BD-rate**, 6–10× the measured floor,
  on both seeds, on the deployed `.nvc` path. Not in doubt.
- The converged optimum is **at or below 4.5e-4**, i.e. below M10D's 6.0e-4 and well below the
  9.0757e-4 inherited from M9's 500-step pilot.

What is **not** settled:

- A single λ. 3e-4 and 4.5e-4 tie on PSNR (0.25 pts apart), MS-SSIM prefers neither, and the minimum
  is still on the boundary of the tested range.

**Engineering recommendation (a judgement call, not a measured minimum): λ = 4.5e-4.** It is tied with
the best PSNR result, has the **smallest seed spread of any arm (0.64)**, is second-best on MS-SSIM,
and — unlike 3e-4 — sits in the *interior* of the tested range, so it is robust to the unresolved
question of what happens below 3e-4. Locking 3e-4 would place the operating point on an untested edge.

### If this is to be resolved

One experiment closes it: λ ∈ {1.0e-4, 2.0e-4} at 18,120 steps, two seeds, same control. If BD-rate
degrades below 3e-4, the basin is bracketed on both sides and 3e-4–4.5e-4 can be locked on evidence.
Evaluating **`best.pt` rather than the final snapshot** (or averaging the last few epochs) would also
remove the UPPER_REFERENCE-style single-epoch contamination — but that is a convention change that
must be applied uniformly and re-run, not retrofitted.

### Reproducibility

seeds 42/43, torch 2.13.0+cu130, RTX 5060 Laptop GPU, ~11.4 min/run, ~2 h total for the grid.
Per-snapshot SHA256 in `snapshots.csv`. Snapshots retained at 604/1,812/4,832/10,268/18,120 steps for
all 10 runs. Best-checkpoint selection used only current-objective history in every run (40 stale
pure-MSE records ignored each time). All commands require the project venv
(`./.venv/Scripts/python.exe`).

---

## 2026-09-06 — M10D: converged λ refinement — the optimum moves DOWN (DURABLE IMPROVEMENT)

**Source:** the open question after M10C — λ = 9.0757e-04 was inherited from a 500-step D/R
measurement and had never been refined at a converged budget.
**Tests:** 720 passing (full suite), zero regressions. 16 new tests.
**Scope:** five arms at the established full budget. No architecture, quantizer, entropy-coder or
`.nvc` change; no `src/nvc/` change at all; `train_autoencoder.py`, `m10c_convergence.py` and
`m10a_pilot.py` all left byte-identical.

### Design

Five arms, 18,120 steps each (30 epochs × 604 batches — the M10C budget, so results are directly
comparable), all from the M8-QAT checkpoint `90d51157…`, identical seed/batch/LRs/QAT/scale
tracking, **only λ differs**. Concentrated around the M10C operating point rather than sweeping
downward again — M10B settled the low direction at pilot scale.

| arm | λ |
|---|---|
| CTRL | 0 |
| LOW | 6.0e-4 |
| CENTER | 9.0757e-4 (the M10C point, replicated) |
| HIGH | 1.35e-3 |
| VERY_HIGH | 1.8e-3 |

A programmatic **fairness preflight** ran before training and is recorded in
`training_summary.json`: checkpoint hash, distinct λ, distinct output dirs, deterministic and
identical rate-estimator initialisation (loc=0, log_scale=0), scale tracking on for every arm,
`resume-model-only` semantics, shared seed/batch/LRs/QAT/epochs. It refuses to train on failure.

### The nondeterminism noise floor, finally measured

CENTER is configuration-identical to M10C-L, so the gap between them **is** run-to-run noise:

| bits | M10C-L BPP | CENTER BPP | ΔBPP | ΔPSNR |
|---|---|---|---|---|
| 8 | 1.7447 | 1.7448 | +0.01% | +0.026 |
| 6 | 1.2423 | 1.2425 | +0.01% | +0.026 |
| 4 | 0.7371 | 0.7373 | +0.02% | +0.019 |

Bitrate reproduces to **0.02%** — far tighter than expected. But PSNR moves +0.02–0.03 dB, and
because BD-rate integrates rate against quality, that small quality shift reads as **−1.41% BD-rate**
between two identical configurations. **That 1.41 points is the noise floor for every BD-rate
comparison in this project**, and it is the number previous milestones lacked.

### Actual `.nvc` at convergence (DAVIS test, 719 frames, fresh per-arm calibration)

| arm | λ | 8-bit BPP / PSNR | 6-bit | 4-bit |
|---|---|---|---|---|
| CTRL | 0 | 1.8445 / 30.106 | 1.3422 / 29.956 | 0.8312 / 28.068 |
| **LOW** | 6.0e-4 | **1.7530 / 29.998** | **1.2506 / 29.929** | **0.7447 / 28.561** |
| CENTER | 9.0757e-4 | 1.7448 / 29.931 | 1.2425 / 29.805 | 0.7373 / 28.543 |
| HIGH | 1.35e-3 | 1.7403 / 29.857 | 1.2380 / 29.785 | 0.7337 / 28.449 |
| VERY_HIGH | 1.8e-3 | 1.7366 / 29.757 | 1.2345 / 29.654 | 0.7307 / 28.282 |

All 15 calibrations passed the 2% guard at 0.120–0.195%, train-split only, 400 frames, seed 42,
each fitted to its own checkpoint (provenance recorded per row in `calibration_report.json`).

All four rate-aware arms **strictly dominate CTRL at 4-bit** (lower BPP, higher PSNR, higher
MS-SSIM). At 8/6-bit they trade a little PSNR for 5–8% bitrate, which BD-rate resolves in their
favour.

### BD-rate: λ = 6.0e-4 is the best point tested

| arm | λ | vs CTRL (PSNR) | vs CTRL (MS-SSIM) | vs M10C-L | ×noise floor |
|---|---|---|---|---|---|
| **LOW** | 6.0e-4 | **−12.96%** | −19.31% | **−4.26%** | 3.0× |
| CENTER | 9.0757e-4 | −10.99% | −20.07% | −1.41% | 1.0× (= noise) |
| HIGH | 1.35e-3 | −10.97% | −19.21% | +1.20% | 0.9× |
| VERY_HIGH | 1.8e-3 | −7.50% | −15.59% | +7.86% | 5.6× worse |

LOW beats the M10C operating point by −4.26% BD-rate, three times the measured noise floor. Netting
off the replicate's own −1.41% offset, the attributable gain is about **−2.9 points** — real, but
modest, and it should be described that way rather than as a −4.3% headline.

Above the centre the curve degrades monotonically: HIGH is indistinguishable from CENTER (+1.20%,
inside noise) and VERY_HIGH is clearly worse (+7.86%).

### The optimum has moved down with training budget

This is the substantive finding. M10B tested λ = 3e-4 at **500 steps** and found it worse than
9.0757e-4. M10D tests λ = 6.0e-4 at **18,120 steps** and finds it better. Those are not
contradictory — they are different budgets — but together they say the converged optimum sits
**below** the pilot-derived one. The λ inherited from M9's 500-step D/R balance was slightly too
high for a converged model, and the useful region is broad and shallow on the low side
(LOW → CENTER → HIGH spans only ~2 BD-rate points) while falling away sharply above it.

### Proxy vs actual at convergence

Proxy ranking `VERY_HIGH < HIGH < CENTER < LOW` matched actual `.nvc` BPP ordering **exactly at all
three bit depths** (Spearman +1.000; Pearson +0.990…+0.994). The proxy remains directionally
reliable for λ selection at convergence — but note it orders arms by *bitrate*, not by RD quality:
it ranks VERY_HIGH cheapest, and VERY_HIGH is the worst model by BD-rate. The proxy predicts rate,
not the rate/quality tradeoff, and must not be used alone to pick λ. (n=4; ordering is the usable
signal, correlations indicative.)

### Pareto frontier

At 8-bit all six models (including CTRL and M10C-L) are non-dominated; at 6-bit five; at 4-bit five
of six. A λ sweep necessarily spreads points along a curve, so "extends the frontier" is nearly
vacuous here — BD-rate is the discriminating metric, not raw dominance counting.

### Verdict: DURABLE IMPROVEMENT

λ = 6.0e-4 improves the deployed frontier over M10C-L by −4.26% BD-rate (−2.9 points net of the
replicate offset), at the full budget, with fresh per-arm calibration, at 3× the measured noise
floor. The improvement is modest and the region is broad — this is a refinement, not a step change,
and M10C-L was already close to optimal.

### Reproducibility

seed 42, torch 2.13.0+cu130, RTX 5060 Laptop GPU, ~23 min/arm. Per-snapshot SHA256 in
`snapshots.csv`. Snapshots retained at 604/1,812/4,832/10,268/18,120 steps for every arm.
Best-checkpoint selection used only current-objective history in all five arms (40 stale pure-MSE
records ignored each time).

### Next experiment (M10E) — proposed, not run

**Bracket the converged optimum from below.** LOW is the smallest λ tested at full budget and is the
best, so the optimum is again unbounded on one side — the same situation M10A left, and the reason
M10B was needed. Test λ ∈ {3.0e-4, 4.5e-4} at 18,120 steps against the same control. M10B's
pilot-scale rejection of 3e-4 does not carry over, because M10D has just shown the optimum shifts
with budget.

Design note for whoever runs it: with a 1.41-point BD-rate noise floor and differences of ~2 points
in this region, a single run per λ is marginal. Two seeds per arm would make the comparison
conclusive, at double the compute.

---

## 2026-09-06 — M10C: rate-aware advantage is DURABLE and grows with training budget

**Source:** the open question left by M10A and M10B — every result so far was a 500-step pilot, and
M9 Section 9F showed pilot-scale RD orderings can reverse by convergence.
**Tests:** 704 passing (full suite), zero regressions. 12 new tests.
**Scope:** two arms at the established full-training budget. No architecture, quantizer,
entropy-coder or `.nvc` change; no `src/nvc/` change at all; no new rate mechanism; no λ sweep.

### The question

M10A measured −6.59% BD-rate for λ = 9.0757e-04 against a λ=0 control; M10B established the optimum
is bracketed there. But both were 500-step pilots, so the number was not yet a durable result.
M10C trains λ=0 and λ=9.0757e-04 to the M9-final budget — 30 epochs × 604 batches = **18,120
optimizer steps** — retaining checkpoints so the advantage can be read as a function of budget.

Both arms received an identical budget, seed, data order, transforms, LRs, QAT config and
evaluation. Every comparison below is **paired by step count**: a snapshot is only ever compared
against the other arm's snapshot at the same number of steps.

### The answer: it survives, and it roughly triples

BD-rate vs the λ=0 control at the same budget (piecewise-linear, the conservative methodology):

| steps | BD-rate (PSNR) | BD-rate (MS-SSIM) |
|---|---|---|
| 604 | −8.11% | −11.55% |
| 1,812 | −15.77% | −18.46% |
| 4,832 | −14.12% | −22.08% |
| 10,268 | −16.48% | −21.05% |
| **18,120** | **−17.10%** | **−20.20%** |

The advantage appears immediately, grows steeply to ~1,800 steps, then plateaus in a −14% … −17%
band and is at its largest at the final budget. It does **not** decay, converge to the control, or
reverse — the M9 pilot-to-final failure mode does not occur here.

### Actual `.nvc` at convergence (18,120 steps, DAVIS test, 719 frames)

| bits | CTRL BPP / PSNR / MS-SSIM | M10C-L BPP / PSNR / MS-SSIM | ΔBPP | ΔPSNR | ΔMS-SSIM |
|---|---|---|---|---|---|
| 8 | 1.8446 / 29.861 / 0.9742 | 1.7447 / 29.905 / 0.9753 | **−5.42%** | **+0.044** | **+0.0011** |
| 6 | 1.3422 / 29.723 / 0.9720 | 1.2423 / 29.779 / 0.9738 | **−7.44%** | **+0.057** | **+0.0018** |
| 4 | 0.8311 / 27.942 / 0.9385 | 0.7371 / 28.524 / 0.9529 | **−11.31%** | **+0.582** | **+0.0144** |

**Strictly dominant at every bit depth**: lower bitrate *and* higher PSNR *and* higher MS-SSIM,
simultaneously, three times over — no interpolation or single-operating-point argument required.
Compression ratio rises from 28.88× to 32.56× at 4-bit.

All 30 calibrations (10 snapshots × 3 depths) passed the 2% guard at 0.116–0.195%, train-split only,
400 frames, seed 42, per-channel 0.1/99.9. No calibration was reused between snapshots or arms.

### Pareto frontier

Over all 10 snapshots and all three bit depths, the non-dominated set is **entirely rate-aware**:

- 8-bit: `M10C-L@18120`
- 6-bit: `M10C-L@10268`, `M10C-L@18120`
- 4-bit: `M10C-L@4832`, `M10C-L@10268`, `M10C-L@18120`

No control snapshot appears on the frontier at any bit depth.

### Scale tracking is doing real work, not the M9 exploit

| steps | latent abs-mean | latent range | tracked bin width | fitted density scale | proxy R | actual BPP (4-bit) |
|---|---|---|---|---|---|---|
| 604 | 4.4100 | 81.1 | 3.1030 | 4.333 | 0.7254 | 0.8203 |
| 1,812 | 3.3631 | 63.4 | 2.2054 | 2.726 | 0.6858 | 0.7919 |
| 4,832 | 2.5264 | 47.1 | 1.3461 | 1.557 | 0.6593 | 0.7682 |
| 10,268 | 1.9296 | 41.6 | 0.9577 | 1.104 | 0.6476 | 0.7516 |
| 18,120 | 1.8872 | 51.3 | 0.8661 | 1.027 | 0.6339 | 0.7371 |

The latent shrinks 2.3× over training and the tracked bin width follows it down 3.6× — but unlike
M9, **actual bitrate falls with it** (0.8203 → 0.7371, −10.1%). Under M9's frozen bin width the
same shrinkage bought ~0% real bitrate. Proxy R moves only 0.7254 → 0.6339 (1.14×) across the whole
run, so the proxy is no longer paying out for scale; the real gain is coming from the latent
becoming genuinely cheaper to code.

The control's own latent grows slightly (6.68 → 7.64) with no bitrate benefit, which is what
"no rate pressure" looks like.

### Proxy-vs-actual alignment holds through convergence

Rank agreement between proxy R and actual `.nvc` BPP across the five budgets: **True at 8-, 6- and
4-bit.** This is a different question from M10A/M10B — there the ranking was across λ at one
budget; here it is across budgets at one λ — and the proxy passes both.

### Head-to-head against M9's frozen bin width

Same λ, same 18,120-step budget, each measured against its own λ=0 control:

| | BD-rate vs own control |
|---|---|
| M9-L (frozen bin width) | −11.76% |
| **M10C-L (scale-tracked)** | **−17.10%** |

So scale tracking did not merely make the proxy honest — it produced a materially better trained
model at the same λ and the same budget.

### Verdict: DURABLE IMPROVEMENT

M10A's −6.59% is not a pilot artefact. At convergence it is −17.10%, with strict dominance on all
three metrics at all three bit depths, a fully rate-aware Pareto frontier, healthy calibration, and
proxy/actual alignment intact.

### Reproducibility

seed 42, torch 2.13.0+cu130, NVIDIA GeForce RTX 5060 Laptop GPU. Per-snapshot SHA256 recorded in
`snapshots.csv` and `training_summary.json`. The documented cuDNN nondeterminism still applies —
reruns will differ slightly and were not repeated for identical hashes, as no correctness issue was
found. Best-checkpoint selection used only current-objective history in both arms (40 stale
pure-MSE records from M8 ignored each time). CTRL 22.4 min, M10C-L 23.7 min.

### Next milestone — proposed, not run

The rate objective is now settled: it works, at a known λ, and the gain is durable. Two candidates,
neither started:

1. **Re-derive λ at the converged operating point.** Every λ decision so far was made from
   500-step evidence, and the M10C latent statistics at 18,120 steps differ substantially from the
   pilot's. The measured D/R balance has moved, so the optimum may have moved with it — this is the
   cheap, in-scope check, and it reuses the M10B harness exactly.
2. **Take the M10C-L model to a full deployment evaluation** (H.264/H.265 comparison, the operating
   points the roadmap actually cares about), since it is now the best model the project has.

Option 1 is the smaller and more informative next step.

---

## 2026-09-06 — M10B: low-λ sweep — hypothesis refuted, optimum bracketed (FAIL on hypothesis)

**Source:** the open question left by M10A (entry below): its useful RD region appeared to lie at or
below its smallest λ, which was untested from underneath.
**Tests:** 692 passing (full suite), zero regressions. 12 new tests.
**Scope:** controlled 500-step diagnostic only. No architecture, quantizer, entropy-coder or `.nvc`
change; no `src/nvc/` change at all; no long training run.

### The question

M10A's BD-rate versus its λ=0 control was −6.59% at λ=9.0757e-04, −1.61% at 2.87e-03 and +42.08% at
9.0757e-03. The gain therefore peaked at the *smallest* λ tested, leaving the optimum bracketed on
only one side. M10B swept below it — 3e-4, 1e-4, 3e-5 — to find out whether the frontier kept
improving.

### It does not. The gain decays monotonically to zero as λ → 0.

BD-rate versus the λ=0 control, every λ now tested (piecewise-linear, the conservative methodology
M10A settled on):

| λ | BD-rate vs CTRL | vs M10A-L |
|---|---|---|
| 3.0000e-05 | −1.01% | +6.19% |
| 1.0000e-04 | −1.27% | +6.06% |
| 3.0000e-04 | −3.78% | +3.44% |
| **9.0757e-04** | **−6.59%** | — (reference) |
| 2.8700e-03 | −1.61% | +1.61% |
| 9.0757e-03 | +42.08% | — |

**The optimum is now bracketed on both sides**, which it was not after M10A. The curve is
single-peaked at λ ≈ 9.08e-04, and the useful range is roughly 3e-4 … 2.9e-3. Every M10B arm is
worse than M10A-L; none is worth carrying forward.

This is the expected shape in hindsight — λ → 0 must converge to the control by construction — but
it was not measured, and "the optimum is below the smallest point tested" was a live hypothesis
that the evidence now refutes.

### Actual `.nvc` results (DAVIS test, 719 frames, fresh per-model calibration)

| model | λ | 8-bit BPP / PSNR | 6-bit BPP / PSNR | 4-bit BPP / PSNR |
|---|---|---|---|---|
| CTRL | 0 | 1.8592 / 29.839 | 1.3571 / 29.710 | 0.8456 / 27.922 |
| M10A-L | 9.0757e-04 | 1.8369 / 29.771 | 1.3343 / 29.677 | **0.8236 / 28.398** |
| M10B-1 | 3.0000e-04 | 1.8507 / 29.846 | 1.3483 / 29.716 | 0.8374 / 28.156 |
| M10B-2 | 1.0000e-04 | 1.8562 / 29.843 | 1.3538 / 29.693 | 0.8428 / 28.030 |
| M10B-3 | 3.0000e-05 | 1.8583 / 29.845 | 1.3561 / 29.714 | 0.8450 / 27.989 |

BPP versus the control shrinks smoothly with λ: −2.60% → −0.96% → −0.32% → −0.06% at 4-bit. All 9
new calibrations passed the 2% guard (0.116–0.193%), train-split only, 400 frames, seed 42.

**Pareto frontier:** M10A-L is non-dominated at all three bit depths. M10B-1 is also non-dominated
at 8- and 6-bit, but only as a marginally-higher-rate/marginally-higher-quality point that is worse
by BD-rate — it does not extend the frontier usefully. M10B-2 and M10B-3 are dominated everywhere.

### Proxy versus actual — stronger than M10A

| bit depth | rank agreement | Spearman | Pearson | actual BPP spread |
|---|---|---|---|---|
| 8-bit | **True** | +1.000 | +1.000 | 1.17% |
| 6-bit | **True** | +1.000 | +1.000 | 1.63% |
| 4-bit | **True** | +1.000 | +1.000 | 2.60% |

Proxy ranking `M10A-L < M10B-1 < M10B-2 < M10B-3` matched actual `.nvc` BPP ordering exactly at all
three depths. This is a **harder** test than M10A's: these four models span only 1.17–2.60% in
actual BPP, and the proxy still ordered them correctly every time. M10A's central claim — that the
scale-tracked proxy predicts deployed bitrate ordering — survives extension to the low-λ regime and
is strengthened by it. (n=4; correlations indicative, rank agreement is the evidence. The λ=0
control is excluded from the ranking: `0.0 * rate` has exactly zero gradient, so its estimator is
unfitted by construction and its proxy R is not a meaningful quantity.)

### Training behaviour

All five arms completed exactly 500 steps. No NaN/Inf, no latent collapse or explosion. `best.pt`
written for every arm, in every case selected using only the current objective's history (40 stale
pure-MSE records from M8 ignored each time).

Latent scale and tracked bin width move together and converge smoothly toward the control as λ falls
— which is the scale-tracking mechanism behaving exactly as designed:

| arm | λ | latent abs-mean | latent range | tracked bin width | fitted density scale |
|---|---|---|---|---|---|
| CTRL | 0 | 7.8031 | 137.86 | 4.1230 | 1.0000 (unfitted) |
| M10A-L | 9.0757e-04 | 5.2630 | 97.34 | 3.2101 | 4.3725 |
| M10B-1 | 3.0000e-04 | 6.8127 | 123.61 | 3.8023 | 5.3851 |
| M10B-2 | 1.0000e-04 | 7.4329 | 132.86 | 4.0086 | 5.7668 |
| M10B-3 | 3.0000e-05 | 7.6854 | 136.36 | 4.0878 | 5.9111 |

Rate-estimator gradient norm scales with λ as expected (1.48e-05 → 6.57e-07 across the sweep) and is
exactly 0 for the control.

### Verdict: FAIL (on the hypothesis, not on the system)

Classified against M10B's own decision gate, whose FAIL clause is "lower λ provides no useful RD
improvement". That is precisely what was measured: no M10B arm improves on M10A-L at any bit depth
by BD-rate. Nothing destabilised, no alignment broke, and no new exploit appeared — training was
clean and proxy/actual alignment was perfect. This is a well-executed experiment with a negative
result, and the negative result is worth more than another sweep would have been: it closes off the
low-λ direction and brackets the optimum.

### Next experiment — proposed, not run

**Do not sweep λ further.** The optimum is bracketed and further refinement between 3e-4 and 2.9e-3
would chase differences smaller than the run-to-run nondeterminism documented in M10A (~1.4% on val
D). The genuinely open question is different, and it is the one every result so far shares:
**everything is a 500-step pilot.** M10A-L's −6.59% BD-rate has never been tested at a training
budget where the model actually converges, and a pilot-scale advantage is not guaranteed to survive
convergence — M9 saw exactly that kind of reordering between pilot and final runs. The next
experiment should therefore hold λ = 9.0757e-04 fixed and vary the *budget*, against the same λ=0
control at the same budget. That needs an explicit authorisation for long training.

### Reproducibility

Same cuDNN nondeterminism noted in M10A: repeat runs differ slightly. The λ=0 control and M10A-L
were **retained from M10A** rather than retrained, so their checkpoints, calibrations and benchmark
rows are the exact artifacts M10A measured — re-running them would have introduced a second
nondeterministic copy of the same experiment into the comparison.

---

## 2026-09-06 — M10A: scale-tracked rate proxy validated in real training (PASS)

**Source:** the 9F.5 latent-shrinkage exploit in [MILESTONE_9_PLAN.md](MILESTONE_9_PLAN.md), and the
`update_bin_width()` fix shipped in the 2026-09-05 entry below, which explicitly deferred the
question this entry answers.
**Tests:** 680 passing (full suite), zero regressions. 15 new tests. (The entry below reports 662;
the suite was at **665** when M10A began — 3 tests were added between that entry being written and
this one, unrelated to M10A.)
**Scope:** controlled diagnostic only. No architecture, quantizer, entropy-coder or `.nvc` change;
no hyperprior, autoregressive context, temporal prediction or motion compensation; no long final
training run.

### What M9 discovered

M9 Section 9F.5 measured, on the real codec, that the deployed quantizer recalibrates its grid to
each model's own latent range. So a model that shrinks its latent uniformly pays almost nothing on
real `.nvc` bits — the grid shrinks with it. The proxy, scoring against a bin width frozen at
construction, read that same shrinkage as a large rate reduction. Gradient descent found the
exploit: M9-H shrank its latent 2.5x versus M9-L for a **3.7x lower proxy R** and **+0.7% actual
BPP** — slightly worse, not better. Proxy-vs-actual rank agreement was **False at all three bit
depths**.

### What scale tracking was intended to fix

`RateEstimator.update_bin_width()` EMA-tracks the observed latent dynamic range so the proxy's bin
width follows the latent, as the deployed calibrator's does. The entry below measured the mechanism
statically (+1.4592 → +0.5667 bpp reward, a 61% reduction) and stated plainly that real retraining
was still required. This is that retraining.

### Phase 3 — the static diagnostic, corrected and extended

Reproduced on the real M8-QAT checkpoint over 20 DAVIS batches, reward for shrinking to 0.25z:

| Configuration | Reward for pure shrinkage |
|---|---|
| A. frozen bin width (M9 behaviour) | **+1.8406 bpp** |
| B. tracked bin width, density frozen at init | **+1.3067 bpp** (−29.0%) |
| C. tracked bin width **and** adapted density | **+0.0499 bpp** (−97.3%) |

**Tracking the bin width alone is necessary but not sufficient**, and that is the substantive
correction to the earlier 61% figure. Laplace is a location–scale family: scaling the latent, the
bin width, `loc` and `scale` together by the same factor leaves every bin probability — and hence
the rate — *exactly* unchanged (proved to 1e-6 in
`test_rate_is_exactly_invariant_when_bin_width_and_density_both_scale`). Configuration C is what
real training reaches, because `--rate-lr` gives `loc`/`log_scale` their own optimizer group
precisely so they follow the latent. Configuration B — bin width tracked, density stuck at
initialization — is the one that leaves a large residual, and it is not the training regime.

For reference the deployed calibrator's own bin width scaled 3.6837 → 0.9209 across the same
sweep (4.0x for a 4x shrink); the tracked proxy managed 3.3202 → 0.8984 (3.7x).

### Experimental setup

Four independent 500-step arms (5 epochs × 100 batches), all from the **same** M8-QAT checkpoint
with `--resume-model-only`, SHA256 verified `90d51157356953db…` before training and refused
otherwise. Same lambdas as M9's final runs, deliberately — the question is whether the *same* rate
pressure now behaves differently.

| | Value |
|---|---|
| Arms | λ=0 control, 9.0757e-04 (L), 2.8700e-03 (M), 9.0757e-03 (H) |
| Model LR / rate LR | 1e-4 / 1e-2 |
| Scale tracking | **enabled**, EMA momentum 0.99 |
| Seed / batch / QAT | 42 / 8 / 4-bit per_channel |

The λ=0 control separates ordinary DAVIS fine-tuning from the rate term. It landed within **0.1% BPP**
of M8-QAT at every bit depth (0.8456 vs 0.8447 at 4-bit), which is the expected result for a
500-step run and validates that the pilot pipeline reproduces the baseline.

### Observed proxy behaviour

| arm | val D | proxy R | PSNR | latent abs-mean | tracked bin width |
|---|---|---|---|---|---|
| CTRL | 1.2938e-03 | 2.2409\* | 29.58 | 7.8031 | 4.1230 |
| M10A-L | 1.3158e-03 | 0.7345 | 29.50 | 5.2630 | 3.2101 |
| M10A-M | 1.4162e-03 | 0.6822 | 29.14 | 3.0101 | 2.1381 |
| M10A-H | 1.7229e-03 | 0.6190 | 28.19 | 1.6773 | 1.3208 |

\* the control's estimator receives exactly zero gradient at λ=0 and stays at initialization, so its
R is the unfitted-prior value and is not comparable to the others.

The latent still shrinks strongly with λ (**4.65x** spread across arms), but the proxy no longer
pays for it: proxy R spread across the λ>0 arms fell from M9's **3.73x** to **1.19x**, against an
actual-BPP spread of 1.055x. The bin width tracked the latent (3.12x spread) as designed.

### Actual `.nvc` results (DAVIS test split, fresh per-model calibration)

All 12 calibrations passed the 2% guard (0.116–0.194% clipping).

| model | bits | BPP | PSNR | MS-SSIM | vs control BPP | vs control PSNR |
|---|---|---|---|---|---|---|
| M10A-CTRL | 8 / 6 / 4 | 1.8592 / 1.3571 / 0.8456 | 29.839 / 29.710 / 27.922 | 0.9732 / 0.9713 / 0.9373 | — | — |
| M10A-L | 8 / 6 / 4 | 1.8369 / 1.3343 / 0.8236 | 29.771 / 29.677 / 28.398 | 0.9738 / 0.9723 / 0.9481 | −1.20 / −1.68 / −2.60% | −0.067 / −0.033 / **+0.476** |
| M10A-M | 8 / 6 / 4 | 1.8136 / 1.3109 / 0.8010 | 29.313 / 29.250 / 28.299 | 0.9715 / 0.9704 / 0.9516 | −2.45 / −3.40 / −5.27% | −0.526 / −0.461 / +0.378 |
| M10A-H | 8 / 6 / 4 | 1.7924 / 1.2897 / 0.7809 | 28.202 / 28.151 / 27.409 | 0.9624 / 0.9613 / 0.9427 | −3.59 / −4.97 / **−7.65%** | −1.637 / −1.559 / −0.512 |

BD-rate versus the control (piecewise-linear, negative better): **M10A-L −6.59%**, M10A-M −1.61%,
M10A-H +42.08%.

### Proxy versus actual — the headline

| | M9 (frozen bin width) | M10A (scale-tracked) |
|---|---|---|
| Rank agreement, 8-bit | **False** | **True** |
| Rank agreement, 6-bit | **False** | **True** |
| Rank agreement, 4-bit | **False** | **True** |
| Spearman (proxy vs BPP) | — (ordering inverted) | **+1.000** at all three depths |
| Pearson | — | +0.814 / +0.816 / +0.818 |
| Actual BPP response to λ | ~0%, non-monotone (H worse than L) | monotone, up to −7.65% at 4-bit |
| Proxy R spread across λ | 3.73x | 1.19x |

Proxy R ranking `H < M < L < CTRL` matches the actual-BPP ranking at 8, 6 and 4 bits exactly.
**n=4, so the correlations are indicative only** — the rank agreement is the primary evidence, and
it is unanimous across three independent bit depths.

**Absolute BPP is NOT comparable between M9 and M10A**: M9's models are 30-epoch (18,120-step) final
runs, M10A's are 500-step pilots. Only the proxy/actual *alignment* is being compared.

### Verdict: PASS

- The exploit is substantially reduced: latent magnitude still varies 4.65x while proxy R varies
  1.19x, and the static diagnostic shows a 97.3% reduction in shrinkage reward for the configuration
  training actually reaches.
- Proxy R is materially better aligned with actual BPP: rank agreement went from False at every bit
  depth to True at every bit depth.
- λ>0 now produces a real RD tradeoff: actual bitrate responds monotonically to λ (up to −7.65%) and
  costs distortion, where in M9 it responded ~0% and only cost distortion.

**Caveat, stated plainly:** the tradeoff being *real* is not the same as it being *favourable* at
every λ. Only M10A-L improves on the control by BD-rate (−6.59%); M10A-H buys −7.65% bitrate for
−0.512 dB at 4-bit, which is +42% BD-rate — a bad trade. The useful λ range is narrow and sits at or
below the measured balance point, the same conclusion M9 reached, now for the right reason.

### Stability

No NaN/Inf in any arm. No latent collapse or explosion (abs-mean 7.80 → 1.68, abs-max bounded). No
pathological bin width. `best.pt` written for all four arms, in every case selected using only the
current objective's validation history — 40 stale pure-MSE records from M8 correctly ignored (the
M9F.1 guarantee, re-verified per arm).

### Note on reproducibility

Re-running the pilot produced different checkpoint hashes: cuDNN's autotuned kernels are
non-deterministic by default, so 500 steps accumulate small differences (val D moved ~1.4% for one
arm). Calibration and benchmarking were re-run against the final checkpoints so every number here
matches the artifacts on disk. Anyone reproducing this should expect small variation, or force
deterministic kernels.

### Next experiment (M10B) — proposed, not run

Sweep λ **below** M10A-L (e.g. 3e-4, 1e-4, 3e-5) at the same 500 steps, with scale tracking on. The
whole useful RD range now sits at or under L, and it is unsampled: L is the smallest λ tested and is
already the best. The open question is whether a smaller λ gives a better BD-rate than −6.59%, or
whether the gain peaks and turns over. This is the cheapest experiment that could move the
milestone, and it needs no new mechanism.

---

## 2026-09-05 — Calibration crash fix, stream header overhead, RD-proxy exploit fix, per-channel bit allocation, companding

**Commit:** [`380c4b3`](https://github.com/yuvidewan/neural_streaming/commit/380c4b3a1df1fc1499ee640f4e32acedac0d4fd0)
**Source:** [OPTIMIZATION_ANALYSIS.md](OPTIMIZATION_ANALYSIS.md) audit (items B1, Q1, Q2, Q3) plus the
9F.5 rate-distortion-proxy bug documented in [MILESTONE_9_PLAN.md](MILESTONE_9_PLAN.md).
**Tests:** 662 passing (full suite), zero regressions. 91 new tests added across this change.

Five fixes, all additive — every new field on `QuantizationParams` / `RateEstimator` defaults to
off, and every calibration file or checkpoint already on disk loads unchanged.

### 1. Fixed: `torch.quantile` crash above 16.7M elements (B1)

**The bug:** `torch.quantile` hard-refuses any input above 2²⁴ = 16,777,216 elements
(`RuntimeError: quantile() input tensor is too large`). `calibrate_quantization_params`
(`src/nvc/compression/calibration.py`) called it once per channel on the full calibration set.
With the default 64×16×16 latent this crashes at:

| Mode | Crashes above |
|---|---|
| `global` | 1,024 calibration frames |
| `per_channel` (default) | 65,536 calibration frames |

Not firing today only because `scripts/calibrate_quantizer.py` defaults to 400 frames — a latent
crash waiting for anyone who raises `--max-batches`.

**The fix:** `_quantile_safe()` — below a 10M-element limit it is byte-for-byte identical to
`torch.quantile`; above it, a deterministic random subsample of exactly that many values is used
instead of crashing.

**Result:** no crash above the limit; a subsampled 1st/99th percentile estimate on a 12,800-value
synthetic population agreed with the full-population estimate within 5% relative error across two
independent draws. Below the limit, output is unchanged (bit-exact, verified by test).

### 2. Fixed: per-frame header overhead (Q1)

**The bug (really: waste):** the `.nvc` header re-sends all per-channel `scale`/`zero_point`
values on *every single frame* — 512 of the header's 549 bytes, identical from frame to frame in
a sequence since calibration is fixed. On a ~13KB frame payload that's roughly 4% of every frame
spent re-transmitting data the decoder already has after frame 1.

**The fix:** an additive `.nvcs` stream container — `NVCStreamHeader` / `NVCStreamWriter` /
`NVCStreamReader` (`src/nvc/compression/nvc_format.py`) plus `encode_frame_payload` /
`decode_frame_payload` / `build_stream_header` (`src/nvc/compression/codec.py`). One 33-byte fixed
header + the parameter block once per stream, then a 4-byte length prefix per frame. The original
single-frame `.nvc` format (`NVCHeader`/`NVCWriter`/`NVCReader`/`encode_frame`/`decode_frame`) is
completely unchanged and still works exactly as before — both formats live side by side.

**Result:** stream output is bit-exact against the existing per-frame path (same payload bytes,
same decoded pixels — verified by test) and measurably smaller for any sequence of more than one
frame. **Not yet wired into `benchmark_rd.py`'s actual encoding path** — the primitive is done and
tested; plugging it into the benchmark harness is the natural next step, not done here.

### 3. Fixed: the RD-proxy could be cheated by shrinking the latent uniformly (9F.5)

**The bug:** `RateEstimator`'s differentiable proxy rate loss used a *frozen* bin width fixed at
calibration time. During QAT, the model could reduce the proxy's reported rate simply by shrinking
its latent's dynamic range uniformly — with a frozen bin width, a smaller latent maps to fewer
occupied bins in the proxy's eyes, but the *actual* rate paid by the real quantizer + entropy coder
does not improve, because real calibration would just re-derive a smaller `scale` for the smaller
range. The proxy and the real coder disagreed, and gradient descent found the exploit.

**The fix:** `RateEstimator.update_bin_width()` — an EMA (momentum 0.99 default) that tracks the
*actual* observed dynamic range during training and updates the proxy's bin width to match, so
shrinking the latent no longer fools the proxy. Wired into `train_one_epoch_with_rate` (called
after every optimizer step); deliberately **not** called during validation, so validation always
measures against the calibration-time bin width. CLI: `--rate-track-scale` / `--rate-scale-momentum`
in `scripts/train_autoencoder.py`.

**Result, measured on a real M8 checkpoint:** artificially shrinking the latent's dynamic range and
checking the reported proxy-rate reward for doing so —

| | Reward for pure latent shrinkage |
|---|---|
| Frozen bin width (before) | **+1.4592 bpp** |
| Tracked bin width (after) | **+0.5667 bpp** |

A 61% reduction in the exploit's reward. **What this does *not* establish:** this proves the
mechanism closes, not that M9-M/M9-H (the two milestone configurations that previously showed no
real improvement) will now train better — that needs an actual GPU retraining run to confirm, which
was out of scope here.

### 4. Added: per-channel bit allocation (Q3) — mechanism works, real-data payoff didn't show up

**The idea:** spend more quantization levels on channels carrying more information, fewer on
channels that barely vary (the same axis JPEG's quantization matrix exploits). Half the
infrastructure (per-channel calibration and statistics) already existed — this is "wire it
through," not "build it."

**What was built:**
- `QuantizationParams.bits_per_channel` + a new `effective_q_max` property
  (`src/nvc/compression/quantization.py`) — `bits` stays the entropy table's alphabet depth
  everywhere it already was (so `.nvc` headers, the entropy model, and the range coder need **zero**
  changes); `bits_per_channel` only ever *narrows* an individual channel's clamp bound below that
  shared depth. Unused high symbol values are handled for free by the entropy model's existing
  Laplace smoothing.
- `calibrate_quantization_params(bits_per_channel=...)` — gives a restricted channel a
  proportionally *coarser scale* over the same calibrated range, not just a harsher clamp on an
  unchanged fine step (`src/nvc/compression/calibration.py`).
- `allocate_bits_per_channel()` — the classical water-filling formula from calibration-latent
  variance: `bits_c = avg + 0.5·log2(var_c / geomean(var))`, integer-rounded and rebalanced to hit
  the exact requested average bit budget.
- CLI: `--allocate-bits-per-channel` / `--allocate-average-bits` /
  `--min-bits-per-channel` / `--max-bits-per-channel` in `scripts/calibrate_quantizer.py`.

**Real-data validation (`outputs/checkpoints/vimeo_qat_noise_best.pt`, 400 calibration frames,
64 channels, per-channel variance ratio 26× across channels):** allocating an average of 6
bits/channel (range came out 5–7 bits, inside an 8-bit table) against a **plain uniform 6-bit
baseline at the same average bit budget** —

| | MSE | Mean per-channel empirical entropy |
|---|---|---|
| Uniform 6-bit (baseline) | 0.05635 | 5.216 bits/symbol |
| Water-filled, avg 6-bit | 0.05591 (0.8% better) | 5.249 bits/symbol (0.6% *worse*) |

Essentially a wash, both directions within noise. **Likely reason:** the existing per-channel
percentile-range calibration already adapts each channel's *scale* to its own spread — a
high-variance channel's step is already coarser and a low-variance channel's already finer before
any bit reallocation happens, so water-filling on top of that captures little marginal signal the
scale adaptation hadn't already captured.

**Bottom line:** the mechanism is implemented correctly and verified end to end (a channel given
fewer allocated bits genuinely produces measurably fewer distinct symbols and a coarser step, not
merely a clamp) — but on this codec's actual latent statistics, it is not the free win the
classical formula promises. Anyone using this needs to measure per-checkpoint, not assume a win.

### 5. Added: non-uniform quantization via companding (Q2) — small real win, gamma-sensitive

**The idea:** the latent is documented (`calibration.py`'s own docstring) as "sharply peaked with
long tails" — equal-width bins waste most of their resolution on nearly-empty tail regions. A
power-law companding transform `y = sign(x)·|x|^γ` (mu-law-style, chosen over an iterative
Lloyd-Max fit for simplicity) compresses large magnitudes and expands small ones before the
existing uniform grid, buying finer steps near zero where the density actually is.

**What was built:** `QuantizationParams.companding_gamma` — calibration companies the latent before
computing percentiles; `UniformQuantizer` companies before quantizing and expands (`x =
sign(y)·|y|^(1/γ)`) after dequantizing. `scale`/`zero_point` are computed *in the companded domain*
and stored exactly as before, so — same as bit allocation — **zero changes** to `nvc_format.py`,
`entropy_model.py`, or the range coder; only the calibration file gains one extra recorded number.
CLI: `--companding-gamma` in `scripts/calibrate_quantizer.py`.

**Real-data validation (same checkpoint, 32 calibration frames, 8-bit per-channel), latent MAE/MSE
at a *fixed* bit depth:**

| γ | MAE | MSE |
|---|---|---|
| 1.0 (no companding) | 0.0433 | 0.02316 |
| 0.7 | **0.0387 (−11%)** | **0.02285 (−1.3%)** |
| 0.5 | 0.0413 (roughly a wash) | 0.02343 (slightly worse) |
| 0.3 | 0.0532 (**worse than uncompanded**) | 0.02597 (worse) |

**Bottom line:** real, but modest, and **not monotonic in γ** — mild companding (γ≈0.7) helps,
aggressive companding hurts. This latent is peaked, but not peaked enough to reward the strong
companding its qualitative description might suggest. Needs a small per-checkpoint gamma sweep
before use, not a fixed default.

### Also fixed along the way

While adding tests for per-channel bit allocation, a real bug surfaced: `torch.clamp` in this
project's PyTorch version (2.13.0) refuses a call mixing a scalar `min` with a tensor `max` — which
is exactly what `UniformQuantizer.quantize()` did the moment `effective_q_max` became a per-channel
tensor. Fixed by branching on `effective_q_max`'s type (`torch.clamp` for the scalar case,
`torch.minimum` for the tensor case). Caught by the new tests before this ever shipped.

### Files changed

`MILESTONE_9_PLAN.md` · `OPTIMIZATION_ANALYSIS.md` · `scripts/calibrate_quantizer.py` ·
`scripts/train_autoencoder.py` · `src/nvc/compression/__init__.py` ·
`src/nvc/compression/calibration.py` · `src/nvc/compression/codec.py` ·
`src/nvc/compression/nvc_format.py` · `src/nvc/compression/quantization.py` ·
`src/nvc/training/rate_estimator.py` · `src/nvc/training/trainer.py` · `src/nvc/utils/config.py` ·
`tests/test_entropy_coding.py` · `tests/test_rate_estimator.py`
