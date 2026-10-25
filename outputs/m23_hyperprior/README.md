# M23 Phase A - the hyperprior gate: PASSES on the residual, FAILS on intra

2026-10-25. `scripts/m23_offline_gate.py`, VAL-B (held out), TEST never opened,
side channel charged at its real coded cost, every arm net of a whole-split
permutation control.

**Classification: SUCCESS on the residual channel (79.6-84.2% of the stream),
NEGATIVE on intra (11.9-12.2%).** The mechanism is real; its size as a drop-in
is modest, and its actual value is what it unlocks.

## Result

| bits | channel | arm | baseline | net b/sym | side | gain | control | NET | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 5 | intra | `hyper_only` | 3.94833 | 4.03063 | 0.33% | −2.0843% | −1.5871% | **−2.0843%** | weak |
| 5 | residual | `hyper_only` | 3.26082 | 3.24038 | 0.51% | +0.6266% | | | marginal |
| 5 | residual | **`hyper_plus_zref`** | 3.26082 | 3.20951 | 0.53% | +1.5734% | −8.5301% | **+1.5734%** | **meaningful** |
| 4 | intra | `hyper_only` | 2.91941 | 3.02387 | 0.44% | −3.5778% | −2.1225% | **−3.5778%** | weak |
| 4 | residual | `hyper_only` | 2.33491 | 2.33225 | 0.63% | +0.1139% | | | weak |
| 4 | residual | **`hyper_plus_zref`** | 2.33491 | 2.27749 | 0.66% | +2.4588% | −7.4870% | **+2.4588%** | **meaningful** |
| 3 | intra | `hyper_only` | 1.87539 | 1.93545 | 0.52% | −3.2029% | −2.4049% | **−3.2029%** | weak |
| 3 | residual | `hyper_only` | 1.47709 | 1.51995 | 0.78% | −2.9019% | | | weak |
| 3 | residual | **`hyper_plus_zref`** | 1.47709 | 1.44518 | 0.71% | +2.1599% | −6.4860% | **+2.1599%** | **meaningful** |

Bars are the project's pre-registered ones: <0.5% weak, 0.5-1.0% marginal, >1.0%
meaningful.

## The three things that make this a measurement

**The side channel is charged, and it is genuinely cheap.** 18,432 hyper values
against 2,359,296 symbols - 0.78% of the positions - costing 0.0103-0.0170
bits/symbol, which is **0.53-0.71% of the net**. Nothing was clamped at any rate
point, so the learned scale fits its alphabet and the measurement is not
distorted by the bound.

**The control confirms content specificity rather than discounting it.** Every
residual control LOST 6.49-8.53%: a hyper-latent from a different frame does not
merely fail to help, it actively misleads a predictor trained to use it. That is
the strongest form this control can take. Under M11's rule a negative control is
not added back, so the reported net equals the measured gain.

**`hyper_only` collapsing is the informative part.** +0.63% / +0.11% / −2.90% as
the quantizer coarsens: transmitted side info cannot carry the channel alone, and
gets worse the less there is to describe. But `hyper_plus_zref` gains
+1.57-2.46% at the same time. **The side channel is complementary to `z_ref`,
not a substitute for it** - which is the useful finding, because `z_ref` is free.

## Intra fails, and says where intra's headroom actually is

The hyperprior loses to the deployed static per-channel histogram at every rate
point (−2.08 / −3.58 / −3.20%), and its control loses too, so this is not a
control artefact - the arm is simply worse.

The reason is worth keeping: **the static histogram is already close to the
marginal optimum**, and a small transmitted scale map cannot pay for itself
against it. What the intra path lacks is not a better marginal, it is *any*
conditioning - compare the baselines, 3.948 bits/symbol for intra against 3.261
for the residual with `z_ref`. So the intra lead is a causal/autoregressive
context on the latent itself, which costs no side information at all, and NOT a
transmitted hyperprior. That redirect is this half of the gate's real output.

## Read the size of this correctly

**The baseline is M10K's family, not the deployed stack.** The shipped P-residual
path adds M11-G16's causal context and M13 on top of M10K, worth a further
+5.3-6.0% of coded bytes on this transform. The +1.57-2.46% here is a gain over
`z_ref` conditioning alone, and causal context may already capture part of what
the hyper-latent carries. **Whether it survives alongside M11/M13 is Phase B and
is not assumed here.**

**As a drop-in entropy improvement it is worth single-digit BD-rate points.** The
residual is ~80% of the stream, so +2.46% of it is ~2% of total bytes. On this
transform the whole refit moved 7.6-11.1% of bytes for 33.9 BD-rate points -
about 3.7 points per percent - so 2% of bytes projects to roughly **7 BD-rate
points**, taking +214.2% to about +207%. That is a real gain and it is nowhere
near the 3.07x the codec needs to close.

**Its value is that it is differentiable.** This is the reason the milestone
exists: nothing the project owns can put a content-adaptive rate model in the
training loss - the static table is not differentiable, M10K/M11 are fitted
post-hoc, and `rate_estimator.py` is one Laplace per channel. The gate's job was
to establish, cheaply and on held-out data, that per-element content adaptation
genuinely carries information before committing weeks to end-to-end R-D training.
It does, and the controls make that hard to argue with.

**Measured at frozen transform weights** that were fitted under a per-channel
Laplace rate term, so this is a LOWER bound on what joint training would give.
`h_a` is also given the dequantized signal rather than the continuous one, which
biases the gate down again.

## One asymmetry found along the way

The Laplace training proxy **understates** the real side cost by 8-23%
(proxy/real 0.918 / 0.922 / 0.772 at 5/4/3-bit). Training therefore
under-penalised side information slightly. The reported net uses the real
histogram cost throughout, so the numbers above are honest; and a Phase B that
trains against a tighter proxy would find a better trade-off, not a worse one.

A second finding, pinned by a test: Laplace smoothing puts a **floor** on the
side cost. Every symbol in the hyper alphabet keeps at least one count - the
zero-probability safety every table in this codec relies on - so on 2**8 symbols
even a perfectly predictable channel pays −log2((n+1)/(n+256)) per value, ~3.1
bits at n=32. The side cost is driven by the hyper alphabet as much as by the
hyper-latent's real entropy, which makes `--hyper-bits` a live lever.

## Recommendation for Phase B

1. **Test `hyper_plus_zref` against the FULL deployed stack**, not M10K's family.
   This is the one result that decides whether the +1.57-2.46% is additive or
   already inside M11's causal context. Cheapest decisive step, and it must come
   first.
2. **If additive: joint training.** Put the hyperprior in the R-D loss and
   retrain the transform against it. That is the step this gate exists to
   de-risk, and the frozen-weight measurement is its lower bound.
3. **For intra, try causal context on the latent, not a hyperprior.** The static
   histogram is near the marginal optimum; the gap is conditioning, and causal
   context is free.
4. **Re-open the 192-channel rejection** once a hyperprior is in the loss. That
   latent was rejected because cost was decoupled from content - the exact defect
   a hyperprior addresses - so the rejection was conditional on an entropy model
   the codec no longer has to keep.

## A correction in this report

`correction_note` in the JSON records it: the run printed a plain
`gain − control`, which credits an arm for its control being bad. With every
residual control losing 6.5-8.5%, that turned measured gains of
+1.57/+2.46/+2.16% into reported +9.95/+8.65/+10.10%. The derived figures were
recomputed under M11's rule (`gain − max(control, 0)`), whose own comment settles
it: "A negative control (the cost of a useless context) is NOT added back." No
measurement changed - only the statistic derived from them - and
`net_of_control_before_correction` keeps the original value in the file.

## Files

- `m23_offline_gate.json` - the report, all arms, controls and training histories
- `gate.log` - the run (not committed; `*.log` is ignored repo-wide)

---

# M23 Phase B1 - the hyperprior IS additive to the full deployed stack

2026-10-25. `scripts/m23_phaseb_gate.py`, VAL-B, side channel charged, net of a
whole-split permuted control. TEST never opened.

Phase A's open question was whether its +1.57/+2.46/+2.16% survived once the
parent was the real shipped stack - M11-G16's causal context plus the K=512
codebook plus M13 - rather than M10K's family. **It does.**

| bits | arm | full stack | net b/sym | side | gain | control | **NET** | verdict |
|---|---|---|---|---|---|---|---|---|
| 5 | **`scale_map`** | 3.04611 | 3.00347 | 0.39% | +1.3999% | -0.3847% | **+1.3999%** | **meaningful** |
| 5 | `learned` | 3.04611 | 3.03951 | 0.02% | +0.2165% | -0.0241% | +0.2165% | weak |
| 4 | **`scale_map`** | 2.13300 | 2.09526 | 0.56% | +1.7691% | -0.5494% | **+1.7691%** | **meaningful** |
| 4 | `learned` | 2.13300 | 2.12704 | 0.03% | +0.2791% | -0.0340% | +0.2791% | weak |
| 3 | **`scale_map`** | 1.33700 | 1.31308 | 0.89% | +1.7889% | -0.8765% | **+1.7889%** | **meaningful** |
| 3 | `learned` | 1.33700 | 1.33204 | 0.05% | +0.3707% | -0.0543% | +0.3707% | weak |

## How much of Phase A survived

| bits | Phase A (over M10K) | Phase B1 (over the full stack) | retained |
|---|---|---|---|
| 5 | +1.5734% | **+1.3999%** | **89%** |
| 4 | +2.4588% | **+1.7691%** | **72%** |
| 3 | +2.1599% | **+1.7889%** | **83%** |

So M11-G16's causal context was already capturing 11-28% of what the transmitted
map carries, and the rest is genuinely new information. That was the one thing
Phase A could not establish, and it is the result Phase B existed to produce.

## The arm that passed needs no network at all

`scale_map` is the hyperprior stripped to its mechanism: **mean |residual| per
position, average-pooled to stride 2, binned on TRAIN quantiles, transmitted and
charged.** 64 values per frame against 16,384 symbols, 8 bins, 4,096 joint
contexts against the 512 prototypes. Nothing is learned.

That matters for two reasons. It separates "does a transmitted scale map help"
from "did our analysis transform learn a good one" - and it means the gain is
reachable without any training at all, by a change to the container and the
coder rather than to the model.

**The charge can decide the verdict and is reported so it can be checked.**
Pre-charge the gains are +1.78/+2.32/+2.67%; the flat 0.01172 bits/symbol map
costs 0.38/0.55/0.88 percentage points. The charge is constant while the stream
gets cheaper, so its relative weight grows as the quantizer coarsens - at lower
rates it would eventually eat the gain, which makes stride and bin count
rate-dependent levers rather than fixed choices.

## The control validates the machinery exactly, not just the result

Every control came out negative, so under M11's rule nothing is subtracted. But
the numbers say something sharper: **the control's loss is precisely its side
charge**, to seven digits.

| bits | control measured | predicted from the charge alone | control symbol bits vs parent |
|---|---|---|---|
| 5 | -0.3847% | -0.3847% | 3.046109 vs 3.046109 (6e-7) |
| 4 | -0.5494% | -0.5494% | 2.132996 vs 2.132996 (2e-8) |
| 3 | -0.8765% | -0.8765% | 1.336996 vs 1.336996 (1e-7) |

A permuted context carries no information, the hierarchical smoothing therefore
selects the maximum strength in the grid (1.05e6) and the child distribution
collapses all the way back to its parent - so the control pays the charge and
gets nothing for it, which is exactly what it should do. Both halves of the gate
are confirmed at once: the fallback works, and the charge is applied identically
to both arms.

## The `learned` arm is a construction failure, not a finding about h_a

+0.22/+0.28/+0.37%, weak at every point. **This does not show that Phase A's h_a
learned nothing**, and must not be read that way. The tell is the side cost:

| | side bits/symbol | at 5-bit | at 4-bit | at 3-bit |
|---|---|---|---|---|
| `scale_map` | | 0.01172 | 0.01172 | 0.01172 |
| `learned` | | 0.00073 | 0.00073 | 0.00073 |

**16x cheaper, and identical at every rate point.** A transmitted map whose cost
does not change at all between rate points is a nearly constant map - it is
transmitting almost nothing. The reduction is why: `learned_signal` quantizes
h_a's 8-channel 4x4 hyper-latent and then averages it to a single scalar per
position, discarding seven of the eight dimensions Phase A charged for and
trained.

So this arm tests "h_a collapsed to one scalar map", not "h_a". The honest
reading is that the reduction is inadequate, and the fix is a multi-channel
discrete context (one context axis per hyper-channel, or a product-quantized
index) or keeping Phase A's network predictor and adding the prototype index to
its conditioning. Logged for Phase B2 rather than reported as a negative result.

## What this is worth on the stream, as a projection

The residual is 79.6-84.2% of the bytes, so +1.77% of it is **~1.5% of the total
stream**. At the ~3.7 BD-rate points per percent of bytes measured in
`outputs/benchmarks/parity_refit/README.md`, that projects to roughly **5-6
BD-rate points**: +214.2% to about +209%.

**That is a projection from VAL-B ideal bits, not a measurement.** Two things
stand between it and a claim: ideal bits have to be realised in coded bytes
(M13's realisation ratio was ~100%, so the precedent is good), and it has to hold
on DAVIS TEST. Neither has been done.

It also needs a **container change**, which is a frozen item in every milestone
so far: the coarse map is side information the `.nvct` v2 format has nowhere to
put. That is a deliberate scope decision for Phase B2, not an oversight.

## Phase B2, in order

1. **Coded validation of `scale_map`** - real arithmetic-coded bytes on VAL-B,
   with the map in a real side-information block, to check the ideal-bits gain
   realises. This is the step that turns ~1.5% of bytes from a projection into a
   number.
2. **Then TEST, once.**
3. **Fix the `learned` arm** with a multi-channel context, to find out whether a
   learned analysis beats the hand-made magnitude summary. If it does not, the
   hand-made map is the better engineering answer anyway - it needs no training
   and no network at inference.
4. **Joint R-D training** remains the larger prize and is now better founded: the
   mechanism is confirmed additive to the real stack, not just to M10K.

## Files

- `m23_phaseb_gate.json` - the report, both arms, controls, h_a histories
- `phaseb.log` - the run (not committed; `*.log` is ignored repo-wide)

## A harness check worth recording

Phase B1 rebuilds the M13-recalibrated baseline from scratch through a different
script, and it reproduces the M13 refit gate run earlier the same day:

| bits | deployed, here | M13 gate | recalibrated, here | M13 gate | diff |
|---|---|---|---|---|---|
| 5 | 3.07555 | 3.07555 | 3.04611 | 3.04604 | 6.9e-5 |
| 4 | 2.19005 | 2.19005 | 2.13300 | 2.13296 | 3.6e-5 |
| 3 | 1.39555 | 1.39555 | 1.33700 | 1.33688 | 1.2e-4 |

The deployed figures are identical and the recalibrated ones agree to ~1e-4,
which is the independent check that this gate's parent really is the shipped
stack. The 4-bit `scale_map` row also reproduced bit-for-bit between a smoke run
and the full run in separate processes.
