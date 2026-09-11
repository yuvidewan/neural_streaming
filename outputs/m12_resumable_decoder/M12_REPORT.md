# M12 — RESUMABLE ARITHMETIC DECODER + SPATIAL CAUSAL ENTROPY: REPORT

## 1. Phase 0 reproducibility status

M11's determinism infrastructure (`deterministic_kernels()` guard in
`m10h_motion_compensation.py`, fixed calibration, seeded RNGs) was not
touched by M12 and is inherited unchanged. `outputs/m11_autoregressive_entropy/reproducibility_fixed.json`
already shows 19/19 fingerprinted fields identical across two independent
M11 processes.

For the M12-specific addition (the resumable decoder), the same class of
check was re-run for a representative configuration — M11-G16, 4-bit, five
real validation frames, encode → legacy prefix-redecode → resumable
decode — as **two independent Python processes** (fresh interpreter each
time, no shared state). A combined SHA-256 over each process's payload
bytes, ideal bit count, legacy symbols, resumable symbols and codebook
frequencies came out **byte-identical** across both runs:

```
5fa8f40e2c68bca00c42b06436cb02b450fbf63dd44a0798dfcc756ef758bd5e
```

(`outputs/m12_resumable_decoder/m12_reproducibility_cross_process.json`).
Phase 0: **intact**.

## 2. Phase A decoder audit

`src/nvc/compression/_native/range_coder.c`'s `rc_decode` held its entire
state in four local variables for the whole call: `low`, `high`, `value`
(the interval integers, `int64_t`) and a `BitReader` (`data` pointer,
`length`, `position`, `bit` — i.e. which bit of the payload comes next).
Nothing else determines what the next `decode_symbols` call would produce.
`decode_symbols`/`rc_decode` never persisted this state past a single call,
so continuing a decode meant re-deriving it from scratch: call again with a
longer symbol count, walking every bit of the payload from position 0.

This is exactly what `m11_ar_entropy.decode_frame` (unmodified by M12) does
per channel group: `decode_symbols(payload, count, ...)` with
`count = (g+1)*G*H*W`, so group *g* re-decodes every earlier group's symbols
too. For *G* groups over *N* symbols that is O(N·G) bit-level work instead
of O(N) — negligible at M11's deployed G=16 (4 groups), prohibitive at G=1
(64 groups, measured 18–27× M10L's decode latency in M11), and the reason a
context needing many more, finer-grained groups (one per row, one per
position) was never attempted.

## 3. Existing decoder state

Confirmed by direct inspection of `range_coder.c`/`range_coder.py` (no
guessing):

| State | Type | Owner today |
|---|---|---|
| `low`, `high` | `int64_t` | locals inside `rc_decode` |
| `value` | `int64_t` | locals inside `rc_decode` |
| bit position | `BitReader{data, length, position, bit}` | local inside `rc_decode` |
| "which table applies next" | `table_index[]` | supplied fresh by the *caller* every call — not decoder state |

The last row matters: which frequency table each symbol uses was already an
external input, not something the decoder itself remembers. Only the first
three rows needed to be made persistent.

## 4. Resumable decoder design

Added to `range_coder.c` (purely additive — `rc_encode`/`rc_decode`'s
signatures and behavior are untouched):

```c
void   *rc_decoder_open(const uint8_t *payload, int64_t payload_len);
int32_t rc_decoder_decode(void *handle, int64_t n,
                           const int64_t *cumulative, int64_t table_width,
                           const int64_t *table_index, int64_t *out_symbols);
void    rc_decoder_close(void *handle);
```

`rc_decoder_open` mallocs an `RCDecoderState{BitReader, low, high, value}`
and runs the same initial-`PRECISION`-bits read `rc_decode` used to. Each
`rc_decoder_decode` call runs the *identical* per-symbol loop body
`rc_decode` used to, reading from and writing back into that struct instead
of locals, so it can be called again later and pick up exactly where it left
off. `rc_decoder_close` frees it.

**`rc_decode` is now a thin wrapper**: `open` → one `decode` call for the
whole stream → `close`. This makes the zero-loss requirement true *by
construction*: `decode_symbols(payload, n, ...)` and
`ResumableDecoder(payload).decode_group(...)` called once for all `n`
symbols run the exact same code, not two implementations that happen to
agree.

Python-side (`src/nvc/compression/range_coder.py`), the smallest API the
spec asked for:

```python
decoder = ResumableDecoder(payload)
first  = decoder.decode_group(cumulative_1, table_index_1)
second = decoder.decode_group(cumulative_2, table_index_2)
decoder.close()                      # or: with ResumableDecoder(payload) as decoder: ...
```

No internal implementation detail (the C handle, the interval integers) is
exposed — `decode_group` takes exactly the same `cumulative`/`table_index`
shape `decode_symbols` already used, scoped to one group.

`_native/__init__.py` gained ctypes bindings for the three new exports
(`argtypes`/`restype`), following the file's existing pattern exactly.

## 5. Legacy/resumable equivalence

Proven twice over:

- **By construction** (§4): `rc_decode` calls the resumable primitives
  internally.
- **By test**: `tests/test_m12_resumable_decoder.py`, 31 tests — single-symbol
  groups, one-group vs many-group splits, arbitrary/random group boundaries,
  pause/resume across separate statements, two independent decoder handles
  agreeing at the same boundary, all three deployed bit depths (5/4/3),
  random/peaked/uniform frequency tables, an M10L/M11-style shared-codebook
  table shape (not just per-channel tables), truncated-payload and
  empty-payload edge cases, a full encode→decode→dequantize→reconstruct
  round trip through the real autoencoder (identical `torch.equal`
  reconstruction), and API robustness (closed-decoder, empty-group,
  overflow, context-manager-on-exception). **31/31 pass.**
  `tests/test_m12_spatial_offline_gate.py` adds 9 more for the Phase B
  parent-swap code specifically. Both new files ran clean inside the full
  suite: **1205/1205 tests pass**, no regressions.
- **On real streams**: the Phase A4 benchmark (§7) additionally checks
  legacy-vs-resumable symbol equality on 10 real M11-G16 validation frames
  per bit depth (30 frames total) before timing anything — 0 mismatches.

## 6. Bitstream compatibility

**No format change.** `.nvct` v2 is untouched — the resumable decoder
changes *how* an existing payload is walked, never what bytes it contains or
what the entropy-model identity binds to. Every existing M10H/M10J/M10K/
M10L/M11 stream decodes identically through either path. No `.nvct v3`.

## 7. Decoder speedup

Measured on real M11-G16 payloads (5 validation P-frames encoded, then
decoded 40× each for timing, GPU: RTX 5060 Laptop), comparing legacy
prefix-redecoding (`m11_ar_entropy.decode_frame`, untouched) against the new
resumable path at the SAME group size, isolating only the coder step:

| bits | G | legacy coder (ms) | resumable coder (ms) | coder speedup | legacy total (ms) | resumable total (ms) | total speedup |
|---|---|---|---|---|---|---|---|
| 5 | 16 | 1.828 | 0.832 | **2.2×** | 6.484 | 5.429 | 1.19× |
| 5 | 1 | 23.756 | 3.323 | **7.1×** | 94.958 | 69.965 | 1.36× |
| 4 | 16 | 1.427 | 0.686 | **2.1×** | 5.791 | 5.089 | 1.14× |
| 4 | 1 | 19.096 | 3.191 | **6.0×** | 87.502 | 71.513 | 1.22× |
| 3 | 16 | 1.130 | 0.552 | **2.0×** | 5.455 | 4.908 | 1.11× |
| 3 | 1 | 15.259 | 3.054 | **5.0×** | 84.028 | 72.341 | 1.16× |

(full data: `outputs/m12_resumable_decoder/m12_decoder_benchmark.json`)

**Yes — resumability removes the redundant prefix-decoding work**: the
coder step alone is 2.0–2.2× faster at today's deployed G=16, and 5.0–7.1×
faster at G=1 (where legacy redundancy is worst — 64 re-decodes of a
growing prefix). The gap widens as G shrinks toward 1, exactly matching the
O(N·G) vs O(N) prediction in §2.

**But total decode time is dominated by the network, not the coder**, once
G is small: at G=1, 64 sequential forward passes cost ~65–70 ms regardless
of decoder — matching M11's own finding that one decode step is
kernel-launch-bound at ~0.7 ms whether it covers 1 channel or 64. This is
the load-bearing number for §29's classification below: making the coder
free does not make *fine-grained autoregression* free, because the network
forward pass is the sequential bottleneck, not the arithmetic coder.

## 8. Phase B causal context design

Reused `m11_causal_context.py` unmodified — its scan-order audit, causal
dependency graph, magnitude/context functions (`left`, `up`, `left_up`,
`left_up_upleft`, `neighbourhood`, ...) and permutation-control methodology
are exactly the machinery M11 already proved causal and unbiased. What M12
adds is `scripts/m12_spatial_offline_gate.py`, which asks the sharper
question M11 could only gesture at: **does spatial context add anything
once the parent is the actual deployed M11-G16 model**, not M10L or a toy
channel-count context?

`m11_g16_prototype_indices` runs M11-G16 with the TRUE previously-decoded
symbols (one parallel forward pass — valid because coding is lossless and
this is exactly what the encoder/`nll_bits` already do) and assigns every
position to its real 512-entry codebook prototype *k*. That *k* becomes the
gate's parent grouping variable, replacing M10L's.

## 9. Causality proof

Every spatial context tested is proven causal by `cx.check_causality`
before being scored — perturbing every symbol at flat index ≥ *i* and
requiring context *i* to be unchanged, exercised at every channel/row
boundary plus 48 random probes. All 5 candidates passed at all 3 bit depths
(`causality.causal: true`, `leaking_positions: []` in the gate JSON). The
underlying causality checker and its planted-leak tests
(right/down/self/next-channel/down-right neighbours) are M11's, covered by
`tests/test_m11_causal_context.py`; M12's own test suite adds coverage for
the NEW code (`m11_g16_prototype_indices` ordering/batching/determinism,
§25) rather than re-testing what M11 already proved.

## 10. Random-control methodology

Identical to M11's: contexts are permuted across a WHOLE split at once
(train, VAL-A, VAL-B separately), not per-frame — M11 measured that a
per-frame shuffle leaks a frame-level "how busy is this frame" signal
(+0.06% to +0.15% spurious gain), while a whole-split shuffle agrees with an
independent uniform-random-label check at −0.08% to −0.09%. Every control in
this run landed at **±0.001–0.009%**, an order of magnitude inside the
0.1% tolerance, at every bit depth and every context — the estimator itself
is trustworthy.

## 11. Offline entropy gate

Full results: `outputs/m12_spatial_offline_gate/m12_spatial_offline_gate.json`.
VAL-A tunes (smoothing strength, context choice), every reported number is
VAL-B (disjoint validation sequences, 144 P-frames from 9 DAVIS sequences),
TEST never touched.

| bits | recalibration (not context) | left | up | left_up | left_up_upleft | **neighbourhood** (chosen) |
|---|---|---|---|---|---|---|
| 5 | +1.604% | +0.468% | +0.360% | +0.497% | +0.354% | **+0.540%** |
| 4 | +2.916% | +0.554% | +0.459% | +0.563% | +0.336% | **+0.905%** |
| 3 | +5.077% | +0.464% | +0.366% | +0.467% | +0.311% | **+0.910%** |

All values are NET of the permuted control (which was ≈0 everywhere, §10).
`neighbourhood` (a 4-bucket sum of left/up/up-left/up-right magnitude) was
selected on VAL-A at every rate point and is the number that matters.

## 12. M11-G16 vs M12 held-out entropy

The chosen context's net gain over the **recalibrated** M11-G16 baseline —
i.e. the entropy reduction attributable to spatial information the model
does not already have — is **+0.54% (5-bit), +0.91% (4-bit), +0.91%
(3-bit)**. Against the model **as actually deployed** (not recalibrated),
`neighbourhood` alone would be worth +2.14% / +3.79% / +5.94% — but most of
that is recalibration (§13), not spatial context.

Against M11's pre-registered thresholds:

| bits | net gain | verdict |
|---|---|---|
| 5 | +0.540% | marginal |
| 4 | +0.905% | marginal |
| 3 | +0.910% | marginal |

**Every rate point lands in the 0.5–1.0% "marginal" band. None crosses the
1.0% "meaningful" threshold.** The gate technically "passes" by its own
mechanical rule (≥0.5% net, controls within tolerance, at every point), but
the verdict is consistently *marginal*, never *meaningful* — see §29 for
why that distinction, plus the latency evidence from §7, is what drives the
classification.

## 13. Recalibration baseline

Reported separately per §12's requirement not to confuse it with context.
Recalibrating M11-G16's own P(s|k) against TRAIN's true symbols (same
histogram-smoothing estimator M11 used for M10L) is worth **+1.6% to
+5.1%** on its own, growing sharply at lower bit depths (3-bit: +5.08%).
This is larger than M11's own M10L-recalibration finding (+0.06% to +1.9%)
— plausibly because M11-G16's 512 prototypes were fitted by Lloyd's
algorithm against the *network's* predicted distributions, not against
observed TRAIN symbol frequencies directly, leaving more room for a direct
histogram refit to close. This is a genuine, cheap, zero-latency
improvement opportunity for M13 (§36) — it needs no new context, no new
model, just refitting the existing codebook's frequency tables — but it is
explicitly **not** a finding about spatial context and is not what §12's
numbers are net of confusing it with.

## 14. Ideal bits

Reported as bits/symbol throughout §11–13 (the estimator's natural unit);
not converted to a full-stream ideal-bit total because Phase B4/B6 (the
learned spatial model and the coded-byte/DAVIS benchmark) were not run —
see §29.

## 15. Actual coded bytes

Not measured. Phase B5 (actual arithmetic-coded validation) is gated on
Phase B3 clearing the "meaningful" bar (§7 of the M12 brief: "DO NOT start
spatial model training until resumable decoding is proven correct" and "DO
NOT start the full DAVIS benchmark until the spatial entropy gate passes").
It did not clear that bar (§12), so no learned spatial model exists to
measure coded bytes for.

## 16. Coder overhead

Not applicable for the same reason as §15 — there is no spatial-context
coded stream to measure overhead on. The RESUMABLE coder's own overhead
(vs. its ideal) is unchanged from the legacy coder's, since §4's refactor
does not alter the coder's bit-level algorithm at all, only how its state
is threaded between calls; M11's own coder-overhead measurements (0.01–0.03%
at G=16, `outputs/m11_autoregressive_entropy/m11_benchmark.json`'s
`ideal_vs_deployed`) apply unchanged.

## 17. Full DAVIS results

**Not run.** Gated on the offline entropy gate clearing "meaningful"
(§29 of the M12 brief; also this report's §29). It did not.

## 18. BPP / 19. PSNR / 20. MS-SSIM / 21. BD-rate / 22. Per-sequence results

Not applicable — no new coded stream exists (§15–17). M11-G16's own DAVIS
numbers (`outputs/m11_autoregressive_entropy/m11_benchmark.json`) are
unchanged by M12, since M12 touched no motion, residual, quantization or
model-architecture code path that would affect them, and Phase A's decoder
change is bit-exact (§5–6).

## 23. Encode latency

Unaffected. M12 changes decode-side state management only; `encode_frame`
(`m11_ar_entropy.py`) was not modified and was not part of the benchmark.

## 24. Decode latency

See §7's full table. Headline: resumability gives a real, measured 2.0–2.2×
coder-only speedup at the deployed G=16 operating point, and 5.0–7.1× at
G=1 — but G=1 total decode time (~70–95 ms) remains dominated by 64
sequential network forward passes (~65–70 ms of that total), which
resumability does not touch. Total decode speedup at G=1 is therefore a more
modest 1.16–1.36×, not the coder's own 5–7×.

## 25. Sequential steps

- **G=16 (deployed M11-G16):** 4 sequential steps/frame (one per channel
  group of 16), unaffected by Phase A — resumability changes the cost
  *per* step's coder portion, not the step count.
- **G=1 (Phase A4's stress case):** 64 sequential steps/frame — the same
  step count the legacy decoder already used; what changes is that each
  step's coder cost drops from a growing-prefix redecode to a genuinely
  incremental one.
- **Spatial context (not built, estimated from §7's scaling):** the
  candidate contexts (`left`, `up`, `left_up`, `neighbourhood`) all read
  within-CHANNEL neighbours, so a group must additionally be row-sequential,
  not just channel-sequential — at minimum ~4 channel groups × 16 rows = 64
  steps for row-granularity, or up to 4 × 256 = 1024 steps for full
  per-position granularity. Since decode latency is kernel-launch-bound per
  step at ~0.7–1.1 ms regardless of step size (M11's own finding, confirmed
  by G=1 costing about the same per-step network time as G=16), even the
  coarsest row-level spatial grouping would cost roughly what G=1 already
  measured (~70 ms/frame) at minimum, and per-position grouping would be
  substantially worse.

## 26. Parallelism

What stays parallel: within a decode step, every position of the group's
channel(s) is predicted and decoded in one batched forward pass + one
`decode_group` call — unchanged by M12. What is sequential: the *number of
groups*, because group *g*'s context depends on group *g−1*'s decoded
symbols. Phase A made the SEQUENTIAL steps' coder cost cheap; it does not
and cannot reduce the *count* of sequential steps a given context
architecture requires — that is fixed by which symbols the context reads,
which is a modeling decision (§9), not a decoder one.

## 27. Parameter count

Unchanged from M11: M11-G16 checkpoints have 12,744–13,536 trainable
parameters (56–60 KB) across the three bit depths (verified directly from
the checkpoints, matching the M11 report). M12 added zero trainable
parameters — Phase A is pure decoder infrastructure, and Phase B did not
reach model training (§29).

## 28. Memory

`ResumableDecoder` allocates one small fixed-size C struct per handle
(`RCDecoderState`: a `BitReader` plus three `int64_t`s — well under 100
bytes) instead of `rc_decode`'s equivalent locals; no per-group buffer
growth, no behavior change versus legacy at any group size. The gate's
prototype-index computation processes frames in batches of 8 to bound GPU
memory (§8); no new persistent memory structure was introduced.

## 29. Provenance

Phase A touches no model, so no new provenance binding was needed — the
resumable decoder is validated against the SAME `.nvct` v2 entropy-model
identity, calibration signature and codebook M11-G16 already uses
(`outputs/m11_autoregressive_entropy/m11_benchmark.json`'s `provenance`
block: e.g. 4-bit `m11_op` identity `1687b85e55d12934`, calibration
`eab9082596c6f980`). Phase B's gate is diagnostic-only (no deployed
model), so it carries its own `context_definition_id`s (from
`m11_causal_context.context_definition_id`, unmodified) for the record but
binds no new stream identity.

## 30. Reproducibility

Confirmed for a representative M12 configuration (M11-G16, 4-bit) across
two independent processes — §1. Determinism of the resumable decoder
itself (repeated decodes of the same payload, two independent handles at
the same boundary) is additionally covered by dedicated unit tests
(§5, `tests/test_m12_resumable_decoder.py`).

## 31. Compatibility

`.nvct` v2 unchanged (§6). No entropy-model identity, codebook or container
field needed to change, because Phase A is bit-exact and Phase B never
reached deployment.

## 32. Tests

40 new tests, all passing inside the full 1205-test suite (0 regressions):

- `tests/test_m12_resumable_decoder.py` — **31 tests**: single-symbol/
  one-group/multi-group/arbitrary-boundary decoding, pause/resume,
  independent-handle agreement, exact stream position, all 3 deployed bit
  depths, random/peaked/uniform distributions, an M10L/M11-style shared
  codebook table shape, a full reconstruct-through-the-real-autoencoder
  round trip, deterministic repeats, truncated/empty payloads, and API
  robustness (closed decoder, empty group, overflow, context-manager
  exception safety).
- `tests/test_m12_spatial_offline_gate.py` — **9 tests**: the new
  `m11_g16_prototype_indices` function's correctness (range, batching
  agreement, C-major ordering, determinism, true-symbol dependence), the
  gate's recalibration/context/control arithmetic on synthetic data with a
  planted-informative and a planted-uninformative context, and VAL-A/VAL-B
  split-isolation discipline (disjoint, sequence-level, never reads TEST).

## 33. Files modified/created

**Modified (shared arithmetic-coder infra — purely additive; `rc_encode`/
`decode_symbols`'s signatures, behavior and every existing caller are
unchanged):**
- `src/nvc/compression/_native/range_coder.c` — added
  `rc_decoder_open`/`_decode`/`_close`; `rc_decode` refactored to call them
  internally (§4).
- `src/nvc/compression/_native/__init__.py` — ctypes bindings for the 3 new
  exports.
- `src/nvc/compression/range_coder.py` — added the `ResumableDecoder` class.

**Created:**
- `scripts/m12_resumable_decode.py` — Phase A4 benchmark (legacy vs.
  resumable, correctness + latency).
- `scripts/m12_spatial_offline_gate.py` — Phase B1–B3 offline gate.
- `tests/test_m12_resumable_decoder.py`, `tests/test_m12_spatial_offline_gate.py`.
- `outputs/m12_resumable_decoder/m12_decoder_benchmark.json`,
  `m12_reproducibility_cross_process.json`, this report.
- `outputs/m12_spatial_offline_gate/m12_spatial_offline_gate.json`.

No M10A–M11 script was rewritten; all reused via the repo's existing
`_load_script` pattern. No commits were made.

**Unrelated environment note (flagged, not part of M12's scope):** this
machine's PATH lists a 32-bit MinGW `gcc` (`C:\MinGW\bin`) before a working
64-bit one, so `_native.load()`'s automatic rebuild (triggered here because
editing `range_coder.c` made its mtime newer than the compiled `.dll`)
silently produced an unloadable 32-bit DLL against a 64-bit Python
(`WinError 193`). Rebuilding explicitly with the 64-bit compiler fixed it
for this session, but the underlying PATH ordering will break the *next*
edit to `range_coder.c` on this machine the same way. Worth a separate,
narrow fix (reordering PATH, or having `_native._build()` prefer a compiler
whose reported target arch matches Python's).

## 34. Classification

**B — RESUMABLE DECODER SUCCESS, WEAK SPATIAL SIGNAL**

Phase A fully succeeds: the decoder is resumable, bit-exact with the legacy
path by construction and by 31 tests, and measurably faster where it
matters (2.0–2.2× the coder step at G=16, 5–7× at G=1). Phase B's honest
answer is that spatial context, once measured against the model actually
deployed (not a weaker stand-in), is real — every control is ≈0, so this
is not noise — but consistently **marginal** (0.31–0.91% net, chosen
context 0.54–0.91%), never crossing the pre-registered 1.0% "meaningful"
line at any of the three rate points. §29 explains why "marginal" here
does not clear M12's stated higher bar.

## 35. Bottleneck

For Phase A: none remaining that Phase A itself could address — the coder
is no longer the redundant-work bottleneck it was. The bottleneck for going
further with context granularity is now squarely the **network forward
pass's per-step, kernel-launch-bound latency** (§7, §25): each sequential
step costs ~0.7–1.1 ms regardless of how much work it does, so the cost of
finer-grained context is set by *how many groups* a context needs, not by
anything the arithmetic coder does. Spatial context of the kind gated here
needs within-channel row- or position-level grouping, which multiplies step
count well beyond G=16's 4 steps toward G=1's 64 (measured ~70–95 ms/frame,
dominated by network) or worse.

## 36. Recommendation for M13

1. **Ship nothing new from Phase B.** The spatial-context gate does not
   clear the bar M12 itself set, and forcing it (§29 of the brief) would
   trade a sub-1% entropy gain for a decode-latency regression of roughly
   10–20× at the coarsest workable granularity — a bad trade against a
   codec that already has a practical 7–9 ms/P-frame operating point.
2. **The resumable decoder is worth keeping regardless of Phase B's
   outcome.** It is bit-exact, tested, and already measurably faster at
   the deployed G=16 point (coder step 2.0–2.2×) — a real, free win, and
   the natural building block if a future milestone ever does want
   finer-grained (but still channel-only, sequential-step-limited) context.
3. **Recalibration (§13) is the actual free lunch this milestone
   surfaced.** M11-G16's own codebook, refit against TRAIN's true symbol
   histogram rather than the network's predicted distributions, is worth
   +1.6% to +5.1% at zero decode-latency cost — no context, no new model,
   no new sequential steps. This is a much better M13 starting point than
   spatial context: same "don't force it" discipline applies, but the
   cost side of this trade is essentially zero, unlike spatial context's.
4. If spatial context is revisited later, do it only alongside a change
   that removes the network's per-step latency floor (e.g. batching
   multiple groups' network passes so the sequential dependency is on the
   CODER alone, which is now cheap) — otherwise the same latency argument
   will recur regardless of how informative the context turns out to be.
