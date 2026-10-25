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
