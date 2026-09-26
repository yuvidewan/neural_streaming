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

## 2026-09-27 — The Stage 1 gate harness: measuring a new transform without retraining the entropy stack

`scripts/benchmark_intra_gate.py` measures intra-only BD-rate for **any**
analysis/synthesis transform. It exists because measuring Stage 1 looked blocked
behind weeks of retraining, and wasn't.

### The blocker, and why it was not one

`benchmark_parity.py --gop 1` reaches the intra numbers through
`m21.prepare_rate_point`, which rebuilds the entire deployed stack: the M11-G16
context model, the M10K learned entropy model, the K=512 codebooks, M14's motion
table, and a provenance check on each. Every one is fitted to a **64-channel**
latent. Stage 1's has 192, and `ChannelContextEntropyModel`'s
`nn.Embedding(latent_channels, hidden)` alone makes the trained G16 checkpoint
unloadable against it. Read literally: retrain the entropy stack before measuring
anything — weeks of work, on the entropy side, which is not where the parity gap is.

But **an I-frame touches none of it.** In `m21.encode_sequence_refined` the
P-branch uses the context model, codebooks and motion; the I-branch uses
`intra_params` and `intra_entropy_model` and nothing else. At GOP 1 there are no
P-frames — which is exactly why `nvc_deployed` and `nvc_m22` came out
byte-identical in the `--gop 1` run.

So the harness needs one thing: `calibrate_grids`, which runs the autoencoder over
TRAIN frames and fits the grids. It is architecture-agnostic. **No trained entropy
model, no codebook, no provenance gate, no retraining.** A test AST-walks the
script's calls and imports to keep it that way, rather than substring-matching the
file — the module docstring legitimately names what it avoids.

### Verified, not asserted

Run with `--verify` on the deployed checkpoint it reproduces the committed
intra-only run's own totals, PSNR and MS-SSIM to 1e-6, through a completely
different code path built on the promoted package (`nvc.video.container`,
`nvc.compression.codec`) rather than on the research scripts. Artifacts in
`outputs/benchmarks/intra_gate_baseline/`.

The x264 all-intra curve is read from the committed `parity_intra.json` rather
than re-encoded, so the reference side is the same curve `+176.2%` came from and
FFmpeg never runs. Scoring imports `benchmark_parity`'s own `score_frames` /
`aggregate_quality` / `bd_rate`; a second scorer would end the one property that
makes these numbers comparable across runs.

### The bug this cost, worth recording

The first version defaulted `--checkpoint` to
`outputs/checkpoints/vimeo_epoch17_best.pt` — where the checkpoint lineage
*starts* (Vimeo training, before the DAVIS fine-tune), not the deployed M10F
model. It coded **38% more bytes at 2.5 dB lower PSNR**, which looked exactly like
a bug in the new coding path.

What exposed it was the M11 provenance gate: the calibration signature came out
`4c3bd029…` where the deployed 4-bit checkpoint records `eab90825…`. Along the way
`calibrate_grids` was confirmed deterministic — two fresh-seeded calls produced
identical grids — which ruled out RNG state and pointed at an input instead. Two
tests now pin the default: equal to `benchmark_parity`'s, and not the lineage start.

### Header fields an intra stream cannot fill honestly

`.nvct` v2 records `residual_entropy_model_id`, `motion_entropy_model_id` and
`motion_bits` whether or not the stream has P-frames. The deployed path fills the
residual one from the M11-G16+M13 identity, which does not exist here; this fills
it from `calibrate_grids`'s own static residual model. That changes 8 bytes of a
**fixed 56-byte** header and nothing else — the ids are truncated/padded to 8
bytes, and the quantization blocks after them are sized by `numel`, from the same
calibration either way. Total stream bytes are unchanged, which is what `--verify`
proves.

### What this changes about the plan

Measuring a trained Stage 1 checkpoint is now one command and one calibration.
Recalibrating the G16 context model, codebooks and motion table is still needed
for a **full-video** number, but not for the Stage 1 gate, and should be scoped
only once Stage 1 clears that gate.

10 tests, including a bit-exact intra round trip on a miniature model and a
negative control that the decoder refuses a stream containing a P-frame.

### The training schedule: a final LR decay, and early stopping made relative

Two things in `scripts/train_vimeo_stage1.py` that would each have cost a 20-hour
re-run to correct, found by reviewing the settings before launching rather than after.

**There was no learning-rate decay at all.** `1e-4` with Adam is the right starting
rate — it is the standard for this architecture class — but every reference recipe
for these models *ends* with a decay to `1e-5`, worth a few tenths of a dB that no
amount of extra epochs at the undecayed rate recovers. New `--lr-decay-chunks`
(default 2) and `--lr-decay-factor` (default 0.1) run the last chunks of the
schedule at the decayed rate.

It is keyed to the chunk's **position in the schedule** rather than to a torch
scheduler's state, for two reasons. This script is built to be interrupted and
resumed, and a position-derived rate is correct after a resume with no scheduler
state to checkpoint. And a plateau scheduler would be the wrong instrument here:
validation loss is measured on each chunk's own held-out split, so it is a different
dataset every chunk — "no improvement" often means "harder chunk", not "converged".
Only the model's parameter group is decayed; the rate estimator keeps `--rate-lr`,
whose 2*C scalars are fitting a density and need O(1) movement (M9C.1).

**Early stopping used an absolute threshold on a metric that shrinks by orders of
magnitude.** `--early-stop-min-delta 1e-5` was ~1% of the validation MSE after one
epoch (measured at 0.000969 in the chunk-1 trial) but ~5% once MSE reached 2e-4, so
late chunks would stop at the patience floor whether or not they were still
learning — the metric's scale deciding rather than convergence. Replaced by
`--early-stop-min-improvement`, a fraction of the chunk's current best, default
0.002. The first epoch's `inf` best is special-cased, because `inf * 0.0` is NaN and
every comparison against NaN is False, which would have counted the opening epoch as
a stall.

Each epoch's learning rate is now recorded in `history.json`, so the decay is
visible rather than something to infer from the chunk schedule.

8 new tests. One consequence worth noting: with the easier relative bar, chunks are
more likely to use the full epoch ceiling, so a ten-chunk Phase A is nearer the
upper end of its time estimate than before.

### `colab_train_stage1.ipynb` corrected

The notebook's section 7 told the reader to measure the gate with
`benchmark_parity.py --gop 1` and warned that the intra grids, G16 context model and
codebooks would all need recalibrating "before the number means anything". Both were
wrong, and wrong in the way that costs most: the command **raises a provenance error**
on a 192-channel Stage 1 checkpoint rather than producing a number, and the reader
would have found that out after a 13-hour training run. It now points at
`benchmark_intra_gate.py`, says explicitly why not to use the parity harness here, and
scopes the refitting as needed for a full-video number only.

Three tests pin the corrected instructions, including one asserting the old
"before the number means anything" claim does not come back.

---

## 2026-09-25 — The Stage 1 training path: Vimeo-90K on Colab

The Stage 1 transform existed but nothing could train it on Vimeo, and the Vimeo
images are not on the development machine — the earlier runs were on Kaggle/Colab.
This is the path from the architecture to a trained checkpoint. **Nothing is trained
yet**; this is the machinery, not a result.

### `scripts/train_vimeo_stage1.py`

Chunked, resumable training of `ResidualGDNAutoencoder` over the ten Kaggle Vimeo-90K
chunks. `train_vimeo_qat_combined.py` (Milestone 8B) is **not** modified — its chunk
machinery (Kaggle download, collision-reconciling extraction, one-chunk-at-a-time
symlinking, per-chunk split lists, manifest building, progress bookkeeping) is reused
read-only through the usual `_load_script` helper, so there is one implementation of
the awkward parts and this script only adds what Stage 1 needs.

**Two phases, because the rate proxy needs a bin width and a bin width needs a trained
model.** A from-scratch network has no meaningful latent scale to calibrate, so this
runs the way the baseline lineage did: Phase A distortion-only from scratch, then
calibrate on a train-split manifest, then Phase B on `D + lambda*R` continuing from
Phase A's weights. The optimizer's parameter list changes between phases (the rate
estimator's own `loc`/`log_scale` join it), so a cross-phase resume restarts the
optimizer and says so rather than failing — M9C's reasoning exactly.

### `colab_train_stage1.ipynb`

A thin wrapper: mount Drive, cache the Kaggle token, install, run the script. All the
logic is in the tested script rather than in cells. The run cells build an argument
list and call `subprocess.run(..., check=True)` instead of `!python ... \` shell
continuations, which IPython does not reliably continue and which break on any Drive
path containing a space. A sanity cell reports measured ms/step before committing to a
long run.

### Checkpoints now record their architecture

`save_checkpoint` writes `architecture` for any non-default model and
`load_model_from_checkpoint` dispatches on it, so a Stage 1 checkpoint rebuilds as a
`ResidualGDNAutoencoder` rather than being silently mis-loaded. Baseline checkpoints
are unaffected: the field is omitted for them, so every M1–M22 checkpoint is
byte-identical and loads exactly as before. `scripts/train_autoencoder.py` gained
`--architecture {baseline,stage1}`, `--base-channels` and `--residual-blocks`, all
defaulting to the previous behavior.

### A pre-existing bug, fixed in passing

`python scripts/train_autoencoder.py --help` crashed with
`AttributeError: 'tuple' object has no attribute 'strip'`. A stray trailing comma
inside `help=( ... )` on `--rate-track-scale` made that help string a one-element
tuple. Present on master since M9F; the CLI's own help has been unusable since.

### Tests

21 new: 13 for the trainer (both objectives, early stopping, checkpoint reload,
refusal to start Phase B without a calibration, and that the M8B script is imported
rather than copied or modified) and 8 for the notebook (every code cell parses, every
flag it passes exists in a parser it calls, the output directory cannot collide with
the baseline runs' `progress.json`).

### Two bugs the first real chunk found

The trial run on chunk 1 (locally, not Colab — the machine already had the Kaggle
token, symlink permission and the disk) got through download, extraction,
symlinking and the split (8,958 train / 995 test sequences) and then died:

- **`KeyError: 'frame_directory'`.** Vimeo chunks produce a *sequence* manifest
  (`sequence_id` + `frame_filenames`), and the script used the frame loaders,
  which want a `frame_directory` per item. `train_vimeo_qat_combined.py` uses
  `create_sequence_*_loader`; this now does too. The tests missed it because they
  passed loaders in directly, so nothing exercised the manifest→loader path — it
  is now `build_loaders()` with a test that builds a miniature Vimeo tree, runs it
  through the same `build_chunk_manifests`, and constructs the loaders.
  Negative-controlled: the frame loader raises exactly that `KeyError` on that
  manifest.
- **A re-run re-downloads the whole chunk.** `download_and_extract_chunk` wipes
  its scratch directory before downloading, so a crash mid-chunk costs the full
  6–10GB again — and `--keep-chunk` does not help, since it only governs deletion
  *after* training. New `--reuse-chunk` skips the download when the frames are
  already extracted, falling back to downloading if it finds a half-extracted tree
  rather than silently training on a fraction of a chunk. Off by default: on Colab
  the VM is wiped between sessions, so reuse would be a lie.

With both fixed the chunk trains: 3,920 steps per epoch at batch 16 (8,958
sequences x 7 frames), ~3.3 it/s on the RTX 5060, so about 20 minutes an epoch.

One operational note from the same run: the 9GB download dropped its connection
once and resumed. The Kaggle CLI retries up to 5 times on its own, but a ten-chunk
run wants supervision rather than fire-and-forget.

### Not done

The run itself, the `lambda` sweep, and recalibrating the intra grids, G16 context
model and codebooks against the new latent — all of which were fitted against the
baseline transform and are invalidated by a new one. The gate number means nothing
until that recalibration happens.

---

## 2026-09-25 — The intra-only scoreboard: the Stage 1 gate's denominator

Stage 1 is gated on intra-only BD-rate against the current intra codec, and that
number did not exist. `scripts/benchmark_parity.py --gop 1` now measures it, on the
same 719 DAVIS TEST frames, with the same scorer and the same primary convention as
the Stage 0 scoreboard. Results in `outputs/benchmarks/parity_intra/`.

**The current codec all-intra is +176.2% BD-rate (PSNR) / +167.6% (MS-SSIM) against
x264 all-intra.** Across the four reporting conventions, +127.5% to +176.2%.

### What it changed about the plan

It splits the Stage 0 deficit into a transform half and a temporal half for the
first time. Against x264 forced into NVC's own structure: **+176.2% all-intra,
+306.2% at GOP 10.** Measured the other way — what each codec loses when forced
all-intra, at matched PSNR — x264 needs **2.8x** the bits it needed with P-frames,
NVC only **1.8x**. Both get worse without temporal prediction; x264 gets worse
faster, which is exactly why the intra-only gap is the smaller of the two.

The transform is still the largest single deficit, and +176% with no motion
involved at all is what Stage 1 targets. But NVC's motion compensation is now
measurably the weaker half, and Stage 3 is carrying more of the remaining gap than
the roadmap assumed.

### One arm is broken and must not be cited

The report contains `nvc_*_vs_h265_intra` at +31.3%. **Unusable.** x265 configured
all-intra needs 0.5326 bpp at 29 dB where x264 all-intra needs 0.3026 — x265
performing worse than x264 is backwards for any correctly-configured encoder, and
its rate curve has a floor (0.3772 bpp at crf38 to 0.3388 at crf44 while PSNR falls
25.4 to 23.4 dB). Undiagnosed; `keyint=1` is the suspect. Stage 0's x265 arms
behave normally and are unaffected.

### Script changes

GOP 1 did not work before this. Three fixes, all in `scripts/benchmark_parity.py`:

- **Calibration and coding GOPs are now separate.** The residual grid, context
  model, codebooks and motion table are fitted on P-frame data; at GOP 1 there are
  none, and `calibrate_grids` raised on an empty collection. The intra tables this
  measurement uses are fitted on I-frame latents and do not depend on the GOP, so
  calibration stays at the deployed GOP 10 and only the coding GOP changes — which
  is also what makes this *the current intra codec* rather than a re-tuned one.
- **The M22 reproduction check is skipped away from the deployed GOP.** M22 recorded
  its totals at GOP 10; elsewhere a mismatch is the point of the run, not a failure.
  It now reports `null` with a reason instead of marking the run invalid.
- **At GOP 1 the forced arms are named `h264_intra` / `h265_intra`**, because
  calling an all-intra encode "lowdelay" invites the one misreading that matters
  here. `--gop` does not reach the default `h264` / `h265` arms at all — they keep
  their B-frames, so `nvc_*_vs_h264` (+920.3%) is context, not the gate.
  `compare()` now finds classical arms by name from the report rather than from a
  hardcoded list, which would have silently dropped the renamed arms.

Six new tests in `tests/test_benchmark_parity.py` (18 total) cover the renaming, the
all-intra encoder flags, the default arms keeping their B-frames, arm discovery in
`compare()`, and both GOP rules.

`nvc_deployed` and `nvc_m22` came out byte-identical at all three rate points, as
expected — M22's residual re-centering only touches P-frame residuals. Decode was
bit-exact everywhere.

---

## 2026-09-23 — Stage 1 begins: a real analysis/synthesis transform

Stage 0 established that the gap to H.264 is in the transform, not the entropy
coder. This is the replacement transform. It is **not trained yet** — no coded byte
changes, no BD-rate moves, and the deployed codec is untouched. What exists is the
architecture, under test, ready for the training run that is the actual long pole of
the stage.

### `nvc.models.gdn` — GDN and IGDN

Generalized Divisive Normalization (Ballé, Laparra and Simoncelli 2016): each
channel is divided by a norm pooled across all channels at the same spatial
position, `y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2)`. IGDN multiplies by the
same root. Unlike ReLU it is smooth and it mixes channels, which is the point — it
removes statistical dependence between channels rather than making the network
represent it, so the entropy coder's per-channel factorized model becomes a less
wrong assumption about the latent.

`beta` and `gamma` are stored as square roots with an asymmetric lower bound
(`LowerBound`): the gradient is blocked only when it would push a parameter further
below the floor, never when it would bring it back. A plain `torch.clamp` pins a
parameter at the floor permanently, and a negative value under the square root is a
NaN several hours into a training run.

### `nvc.models.residual_transform` — the transform itself

`AnalysisTransform` / `SynthesisTransform` / `ResidualGDNAutoencoder`: the same
four-stage stride-16 structure, with GDN in place of ReLU, identity-initialized
residual blocks between the downsampling stages, and 192 channels instead of 32/64.

**8,437,827 parameters against the baseline's 593,411** — inside the 8–12M band
Stage 1 asks for, and pinned by a test so the defaults cannot drift out of it.

Three things are deliberately unchanged, because Stage 1's lever is the transform
and nothing else: stride 16 exactly (the latent grids, the G16 context model, the
motion block size and `nvc.video.codec.AUTOENCODER_STRIDE` are all written against
it), the sigmoid that keeps reconstructions in `[0, 1]`, and the
`encode`/`decode`/`config_dict`/`num_parameters` surface, so existing training,
checkpoint and evaluation code takes the new model without edits. No BatchNorm: a
codec cannot have a frame's latent depend on what else was in the batch.

### Tests (`tests/test_residual_gdn_transform.py`, 34 tests)

Beyond shapes: GDN with zero gamma is exactly the identity (which pins the pedestal
arithmetic); it contracts large activations more than small ones, and IGDN expands
where it contracts; parameters stay non-negative after a hostile update; every
parameter receives a finite gradient; twenty Adam steps reduce the loss without a
NaN. The residual block is the identity at initialization, with a negative control
that it stops being one once trained.

One honest caveat, tested to a bound rather than asserted away: a frame's latent is
bit-identical across repeated calls, but encoding it inside a batch of 4 differs
from encoding it alone in the last couple of mantissa bits (~7e-9). That is
PyTorch's 3x3 and 1x1 convolution kernels choosing a different code path per batch
size, not this model's arithmetic — the baseline's 5x5 stride-2 convolutions happen
not to. It is ~7 orders of magnitude below one quantizer step and the codec encodes
a frame at a time, so it cannot change a symbol; the test bounds it at 1e-6, where
real batch leakage would land far above.

### What is not done

Training on full Vimeo-90k (~91,701 sequences; the deployed checkpoints saw 10
chunks) and the intra-only BD-rate gate. Until those run, this is a better
architecture on paper only.

---

## 2026-09-16 — Stage 0: the H.264 scoreboard, and the video codec promoted into `nvc.video` (GATE MET)

The first two items of `PARITY_ROADMAP.md` Stage 0. Neither changes a single coded
byte; both change what the project can measure and ship.

### The scoreboard (`scripts/benchmark_parity.py`, `outputs/benchmarks/parity_s0/`)

One pass on DAVIS TEST: every codec saw the same frames and was scored by the same
code on the 8-bit pixels a decoder delivers. Against default `libx264` the M22 codec
has a **BD-rate of +450.7% on PSNR and +378.3% on MS-SSIM** (~5.5x the bits). The
"4.8x" quoted since 15 September was only the 3-bit point: the gap widens with
quality, because each rate doubling buys this codec 1.3 dB against H.264's 2.8 dB.
Forcing x264 into this codec's GOP structure still leaves +300.5%.

Two things the run found. The August comparison had scored H.264 and NVC with
different PSNR definitions (all four common conventions are now reported;
+451% to +507%, same conclusion). And `nvc.evaluation.perceptual_metrics.msssim`
read up to +0.005 too high on channels-last CUDA tensors, which was fixed in
`8968b8e`. No earlier result was affected.

### The promotion (`src/nvc/video/`)

The video codec lived only in `scripts/`, assembled at run time by `importlib`, and
rebuilt its state from TRAIN data on every run. It is now a package:

- `nvc.video.motion`, `.container`, `.entropy`: the inference paths of M10H, M10L,
  M11 and M13, ported with the arithmetic unchanged.
- `nvc.video.bundle.CodecBundle`: one file per operating point holding everything
  the codec needs. It is loaded with `weights_only=True` and recomputes every
  identity on load, so a drifted bundle is refused.
- `nvc.video.VideoCodec`, plus `nvc.encode(frames, bundle)` and
  `nvc.decode(data, bundle)`.
- Hardening the research reader lacked: quantization-block sizes are checked
  against the latent channel count, decoded motion vectors are range-checked, and
  a stream is refused before any payload is read unless its structure and all
  three identities match the bundle.

`scripts/export_codec_bundle.py` froze six bundles (deployed and M22, at 5/4/3 bits),
cross-checking each identity against the research rig's own.
`scripts/verify_promoted_codec.py` then ran the package and the research path side
by side on **all 9 DAVIS TEST sequences for all 6 bundles: 54 of 54 streams are
byte-identical, reconstructions are identical, package decode is bit-exact, and
every total equals M22 Phase 19's recorded bytes** (1,900,744 / 3,009,891 /
4,292,887 deployed; 1,867,362 / 2,970,488 / 4,247,581 M22).

`tests/test_video_codec.py` (25 tests) pins the same contract on a miniature codec
in seconds. Its central test was checked with a negative control: routing prototypes
through the wrong codebook makes it fail.

The research scripts are unchanged and remain the record of how each piece was
derived. Bundles (`*.pt`) are regenerable and not committed; `bundles.json` records
their digests and identities.

**CI** (`.github/workflows/tests.yml`) now runs the full suite on Ubuntu for every push and
pull request. A dry run on a fresh clone found one test that assumed git-ignored M22
checkpoints were present; it now checks digests for the checkpoints that exist and the
committed provenance records everywhere. **Stage 0 is complete.**

---

## 2026-09-15 — M22: residual-quantizer freeze-lift (MEANINGFUL, REPLICATES ON DAVIS TEST)

**Source:** M17 measured a real residual oracle gap and called it unimplementable. M18 (intra
precision), M20 (codebook hysteresis) and M21 (causal reference refinement) each attacked the
decoded *reference* and each failed to convert a measured mechanism into coded bytes. M22 lifted
the one freeze none of them touched: the residual quantizer itself.
**Tests:** 1534 passing (full suite), zero regressions. 98 new tests.
**Scope:** experiment only — `src/nvc/` untouched; GOP, motion, `.nvct` v2, range coder, M11-G16
architecture, K=512, M13 mechanism and λ=3e-4 all frozen; the autoencoder was never retrained.

### Re-centering the residual grid is worth 1.1-1.8% of the total stream, on held-out TEST

Eight grids were pre-registered before any VAL-B coded result was read, spanning the step-size
axis in both directions. The winner, `symmetric_p01`, forces the grid symmetric about zero
(half-width = max(|p0.1|, |p99.9|)) instead of spanning the raw (0.1, 99.9) percentile range. Its
`scale` is essentially unchanged — ratio **1.02** — and what actually moves is `zero_point`, on
**37 of 64 channels**. That realignment drops 0th-order residual symbol entropy from **1.568 to
1.239 bits** at 3-bit: 21% fewer bits for the same residuals at the same step and the same quality.

Full 719-frame DAVIS TEST, against the frozen production baseline (whose totals reproduced M14's
record byte-for-byte at all three rate points):

| bits | baseline | candidate | total stream | P-residual | dPSNR | dMS-SSIM |
|---|---|---|---|---|---|---|
| 5 | 4,292,887 | 4,247,581 | **+1.0554%** | +1.2866% | **+0.0213** | +0.000074 |
| 4 | 3,009,891 | 2,970,488 | **+1.3091%** | +1.6484% | **+0.0320** | -0.000036 |
| 3 | 1,900,744 | 1,867,362 | **+1.7563%** | +2.3296% | -0.0126 | -0.000412 |

**BD-rate -2.4746% PSNR / -1.0968% MS-SSIM.** At 5- and 4-bit it is a strict Pareto improvement:
fewer bytes *and* higher PSNR. All 9 TEST sequences improve individually (+1.14% to +3.79%); none
regresses. I-frame bytes are identical to the byte at every rate point and motion moves by at most
162 bytes, so the entire gain sits in the P-residual channel — exactly where the freeze was lifted.

### It generalizes, which is the part M21 failed

VAL-B predicted +1.6271%; TEST delivered **+1.7563%** — a gap of **-0.1292 points in the
candidate's favour**, with 87%/105%/116% retention at 5/4/3-bit. M21's locked candidate went
+0.7141% to +0.1411% at this identical step, retaining 20%. Runtime is unchanged (the change adds
no operations, parameters or memory), decode is exact on 9/9 sequences at all three rate points,
and an independent process reproduces the locked candidate with **zero** differences across 13
aggregate and 24 per-sequence fields.

### The mechanism is NOT the one M17-M21 were chasing

Phase 13's four-level decomposition (real vs oracle reference, on VAL-B) is unambiguous: gap
closure at the coded-byte level is **-5.23% / -0.33% / +0.22%** at 5/4/3-bit — zero within
measurement, slightly *negative* at 5-bit. Symbol agreement barely moves (70.18% -> 70.51%).

Both arms get cheaper: a better-aligned grid helps the oracle reference exactly as much as the real
one, so the ratio between them is unmoved while both absolute costs fall. **M17's oracle gap is
still open and still unexploited**; M22 found an orthogonal axis. The GOP-boundary anomaly is
likewise untouched — the gain is uniform across GOP position (+1.3035% boundary vs +1.2844%
ordinary at 5-bit), and the boundary's oracle gap stays at 38.7% against 16.3% for ordinary
positions at 3-bit.

### The quantizer is welded to the entropy stack, and that was priced

`calibration_signature` hashes only the residual scale/zero_point, so any grid change moves it and
the deployed G16 checkpoint rejects it with a `ProvenanceError`. The residual quantizer therefore
has **no interface at which it can change alone** — it drags M10K, G16, the K=512 codebook and the
M13 frequencies with it. M22 ran two explicitly separated arms rather than papering over this.

Refitting the downstream stack is worth **+57.61% of the total stream** averaged over 7 variants.
Phase 15 decomposed why, without repairing anything: under the new grid **99.7630%** of positions
route to a different K=512 prototype, and coding the new symbols with the deployed tables costs
**+103.1045%** more bits. Recalibration is not a separately schedulable phase — it is an
inseparable part of the same atomic change, and the calibration signature already refuses to let a
partial deployment exist.

Attribution was closed with a full 2x2 (grid x refitted stack). The grid's main effect is
**+1.6271%**; the refit's own effect is **-0.12% to +0.12%**, i.e. nothing, at every rate point.
The gain is the grid's, not the fitting procedure's.

### The bigger lead, correctly excluded

`broad_p001` (percentiles 0.01/99.99) scores **BD-rate -8.2372%** — four to seven times the
selected candidate — but reaches it by shifting the quality-per-bit-depth operating point
(-0.7873 dB at fixed 3-bit depth), failing the distortion guard declared before any result was
read. The guard was not relaxed to admit it.

An interim reading during the sweep called that candidate "not a representation improvement, just a
lower quality point." **That was wrong**, and BD-rate refutes it: it genuinely needs ~8% fewer bits
at equal quality. What the fixed-depth guard rejects is narrower — its unsuitability as a *drop-in*
replacement — and conflating the two would have buried the largest measurement in the milestone.

### Classification: A — meaningful residual-representation success

Graded on held-out DAVIS TEST, not on the selection set. **M23 should optimize the residual
quantizer specifically (option B):** re-derive the inherited (0.1, 99.9) percentiles against
BD-rate rather than fixed-depth bytes, separate the centering and range axes that M22 only varied
jointly, decide the quality operating point explicitly, and revisit the integer rounding of
`zero_point` — which leaves even the "symmetric" grid asymmetric by up to half a step.

### Two bugs found in the research rig, both caught by guards rather than luck

**A wrong origin.** The first full sweep reported a control of 897,872 bytes where Phase 0 had
established 723,381 — a 24% error. `collect_train_residuals` walked I-frames as
`model.decode(latent)` while production performs the full intra round trip, shifting the rebuilt
3-bit mean step from 1.73229 to 1.7266. Every delta was being measured from the wrong place. Fixed,
and `verify_deployed_grid()` now **raises** if the TRAIN stack does not rebuild the deployed grid
exactly, rather than quietly reporting a control that is not the control.

**A closure writing into a deleted file.** Incremental report persistence was added after a crashed
run lost 1.5 hours of refits; its `persist()` closure wrote through a variable named `path`, which
a later `for path in stream_dir.glob(...)` rebound — so every write landed in a just-deleted
`.nvct` file. Fixed by renaming both, with an AST-level regression test that fails if any name
`persist()` closes over is also a loop target in `main`.

### Files changed

`scripts/m22_residual.py` · `scripts/m22_baseline.py` · `scripts/m22_diagnostics.py` ·
`scripts/m22_sweep.py` · `scripts/m22_analysis.py` · `scripts/m22_mechanism.py` ·
`scripts/m22_codebook.py` · `scripts/m22_reproduce.py` · `scripts/m22_davis.py` ·
`tests/test_m22_residual_freeze_lift.py` · `outputs/m22_residual_freeze_lift/m22_report.md`

## 2026-09-13 — M21: causal reference refinement (VAL-B MARGINAL, DID NOT REPLICATE ON TEST)

**Source:** M17 proved the decoded reference costs real bytes; M18 that shrinking its aggregate
error does not recover them; M19 that the dominant mechanism is the reference error changing the
residual SYMBOL; M20 closed the codebook-routing branch. M21 tested the one remaining direct
mechanism: a small, causal, decoder-available refinement of the previous reconstruction.
**Tests:** 1439 passing (full suite), zero regressions. 60 new tests.
**Scope:** experiment only - every M10-M20 identity, checkpoint, quantizer, codebook, motion
estimator, GOP, lambda and `.nvct` v2 frozen; no production change; `src/nvc/` untouched.

### The deployed reference is a local optimum at 5- and 4-bit

18 pre-registered candidates (12 pixel-domain, 5 latent-domain, plus an identity control) were
measured open-loop against M16/M17's oracle reference before anything was built. At 5-bit and
4-bit, across 136 comparisons (17 transforms x 4 metrics x 2 rate points), **not one candidate
improved pixel MSE, latent MSE, symbol agreement or coded bytes**. Latent-domain smoothing is
catastrophic (up to -57.8% of the residual channel); the theoretically motivated autoencoder
re-projection nearly doubles pixel error, because the autoencoder is lossy and a round trip of an
already-good reconstruction just adds a second dose of its own loss.

### A real 3-bit crossover that did not generalize

At 3-bit - where the residual quantizer is coarsest and the reference carries the most
quantization noise - three mild pixel smoothers improve **all four metrics together**, which is
the opposite of M18's pattern. In the real closed loop `px_median3_a50` (a half-strength 3x3
median on the decoded previous frame) reached **+0.7141% of the total stream on VAL-B at 3-bit
with PSNR and MS-SSIM both up** - a pure win clearing the 0.5% gate.

Locked by the declared Phase 7 rule before DAVIS was opened, it delivered **+0.1411% on the full
719-frame DAVIS TEST** - a fifth of the VAL-B estimate, below the production line - while costing
-0.8228% at 5-bit and -0.6341% at 4-bit. BD-rate applied at every rate point is a wash (+0.059%
PSNR / -0.181% MS-SSIM). **Classification: C - no coded-rate improvement**, graded on the held-out
set because VAL-B is the set the candidate was selected on.

The gap is sequence variance, not a bug: 6 of 9 TEST sequences improve (three by 1.5-1.9%), but
`schoolgirls` alone costs +7,971 bytes and flips the aggregate. VAL-B's four sequences contained
no comparable case - an honest sign the offline gate is thin for an effect this size.

### Two findings that do survive

- **The gain is not picture quality.** At 3-bit the winner makes pixel MSE against the oracle
  *worse* (-2.7%) while improving motion SAD (+12.0% of the way to the oracle), residual RMS
  (+23.9%) and coded bytes (+5.0%). It is a better PREDICTION TARGET, not a better picture. Any
  future refinement trained against reconstruction error would have rejected it.
- **The gain is GOP-boundary-located.** On TEST at 3-bit the entire effect is at position 1
  (+2.4954% of its residual bytes) while positions 2-9 get slightly worse (-0.2177%),
  independently replicating the boundary spike M16/M17/M19 all found and localising a real
  mechanism to it.

### Decoder compatibility, provenance, reproducibility

All 18 candidates are decoder-compatible: 158 sequence-level round trips with symbols,
reconstructions AND refined reference latents identical every time, zero side information, no
`.nvct` change. A negative control proves the refinement is load-bearing. The identity arm
reproduces `m13_closed_loop.encode_multi` byte-for-byte including the container file, and
reproduces M14's recorded DAVIS totals **exactly at all three rate points**. The 3-bit VAL-B sweep
reproduced byte-for-byte in an independent process.

### What this means for M22

M21 was the last cheap structural experiment available under the current freezes. M18, M19, M20
and M21 now agree: aggregate reference error is not the lever, routing is not exploitable, and the
reference cannot be improved by causal post-processing. All four point at the thing none of them
was allowed to touch - **the residual quantizer and the autoencoder that define what a symbol is**.
M22 should be scoped as an explicit, pre-registered freeze-lift there, optimising the reference as
a prediction target rather than as a picture.

Full analysis: `outputs/m21_reference_refinement/m21_report.md`.

---

## 2026-09-13 — M20: codebook-assignment hysteresis (NEGATIVE RESULT — BRITTLENESS IS NOT EXPLOITABLE)

**Source:** M19 found the 512-entry codebook's assignment is brittle (it flips for 35-55% of
positions even in the smallest reference-error decile) and that 9-12% of the reference-error
excess bits come from positions where the residual symbol is unchanged and only the entropy
table moves. M19 recommended exactly one M20 experiment: a stability margin on the assignment.
M20 ran it and the answer is a clean negative.
**Tests:** 1379 passing (full suite), zero regressions. 56 new tests.
**Scope:** diagnostic only - every M13/M14/M15/M17/M18/M19 identity, checkpoint, quantizer,
codebook, motion estimator, GOP, `.nvct` v2 frozen; DAVIS TEST untouched; no learned model,
no production change.

### Hysteresis is decoder-compatible and costs bytes everywhere

Three pre-declared previous-assignment definitions (temporal / previous channel group / raster)
x eight non-zero margins were swept on all 246 VAL-B P-frames at 5/4/3-bit, against a margin=0
identity control proven byte-identical to the deployed `m13.encode_frame_recalibrated`. The best
of 72 (state, margin, bit-depth) results is **+0.0007% of the total stream** - eight bytes out of
908,404 - against the project's 0.5% "weak" line. The response is monotonically harmful in the
margin at every state and every bit depth, reaching -2.4%/-4.0%/-6.8% at the largest margin
tested. **Classification: C - hysteresis does not improve coded rate.**

### Why it fails, measured rather than guessed

Holding a position on its previous assignment makes the actual code length **worse 47-74% of the
time**: the argmin under a noisy predicted distribution is a weak but genuinely positive
predictor, and a stability margin is a slightly-losing bet taken tens of millions of times. The
decision is also far tighter than expected - the median position prefers its prototype over the
runner-up by **0.003 bits**, which is M19's "brittleness" in numerical form and explains why a
margin cannot help: the decision is marginal, but so is the consequence.

**14 configurations reduce routing churn while increasing coded bytes** - the failure mode the
milestone flagged in advance, observed directly (starkest: previous-channel-group @ 0.500 at
5-bit cuts churn 7.8 points and routing-only cases 19.4% while costing 6.7% of residual bytes).
Churn reduction was measured and reported but explicitly refused as a success criterion, which
mattered here.

### Two doors closed alongside it

- **Routing by M13's recalibrated coding tables** (a genuinely decoder-available alternative rule,
  measured as a labelled reference point, never a candidate) is also worse: -1.39%/-1.93%/-3.37%
  of the total stream. M13's one-way assign/coding split is retrospectively validated as the
  better choice, not an oversight.
- **The non-causal per-symbol "oracle routing" bound** looks like +57% but is vacuous: it collapses
  to 7-17 distinct tables and needs 1.7-3.0 bits/position of assignment entropy to save 0.8-1.8
  bits/position - it costs more side information than it saves at every bit depth. Any routing
  bound that reads the symbol it routes is measuring a side channel, not an opportunity.

### Decoder compatibility, invariants, provenance

654 encode/decode round trips (54 in the Phase B probe, 600 in the sweep) across every state and
margin: assignments and symbols identical every time, **zero side information, no `.nvct` change**.
The symbol-change rate is bit-identical across all 27 configurations at every bit depth, so
residual symbols - and therefore G16 context and the reconstruction - are provably untouched by
the rule. The 5-bit sweep reproduced byte-for-byte in an independent process. All 117 existing
M13/M14/M15 `.nvct` streams re-parse with the unmodified reader, are still format version 2, and
every entropy identity in them is attributable to an earlier milestone's own recorded JSON.

### What this means for M21

M20 does **not** solve the reference bottleneck and must not be read as doing so: it addressed only
M19's 9-12% routing-only component, leaving the dominant 88-91% mechanism - reference error
changing the residual symbol itself - entirely untouched. Combined with M19's own bound, the
expected value of further work on the 512-prototype assignment is close to zero. M21 should target
the 88-91%: a reference-refinement step applied identically at encoder and decoder (the only thing
that could move symbols rather than tables), or an explicit, pre-registered freeze-lift.

Full analysis: `outputs/m20_codebook_hysteresis/M20_REPORT.md`.

---

## 2026-09-12 — M19: reference error shape diagnostic (MULTI-PART MECHANISM, PRECISELY QUANTIFIED)

**Source:** M18 showed shrinking the reference error's aggregate magnitude doesn't reliably recover
M17's bytes. M19 asks what KIND of error actually matters - spatial structure, channel structure,
codebook routing, or G16 prediction - by decomposing the SAME real-vs-oracle gap at the position
level, never assuming an answer.
**Tests:** 1323 passing (full suite), zero regressions. 13 new tests.
**Scope:** diagnostic only - every M13/M14/M15/M17/M18 identity, checkpoint, quantizer, codebook,
motion estimator, GOP, `.nvct` v2 frozen; DAVIS TEST untouched; no learned model, no production
change.

### The finding does not fit a single pre-registered category, and says so

Spatial structure is real (autocorrelation ~0.83-0.88 at every bit depth, growing edge-concentration
1.36->1.55) but a magnitude-preserving spatial shuffle recovers 46%/60%/65% of M17's gap at
5/4/3-bit on its own - so magnitude is the larger factor, more so at coarser quantization. Channel
concentration is weak and actually SHRINKS toward uniform at coarser bit depths (top-10% channels
hold only 11-15% of error/churn against a 10% uniform baseline) - ruling out "a few bad channels."
The 512-entry codebook's assignment is surprisingly brittle - it flips 35-55% of the time even for
the SMALLEST error decile - but a precise 2x2 decomposition (symbol changed? x assignment changed?)
shows this routing-only effect contributes only ~9-12% of TOTAL excess bits, consistently across
all three bit depths. The dominant driver (88-91% of excess bits, every bit depth) is the reference
error changing the coded RESIDUAL SYMBOL itself - upstream of both codebook routing and G16's
conditional prediction, a mechanism the milestone's own A-E taxonomy has no clean label for.

### Verdict: F - no single mechanism dominates as pre-registered, but the actual mechanism is precisely quantified

Classified F not for lack of evidence but because forcing A-E would misrepresent a real, consistent,
cross-bit-depth-robust finding. Reproduced byte-for-byte (including the seeded shuffle control)
across two independent processes.

### M20 recommendation

Exactly one experiment: a stability margin (hysteresis) on `SharedCodebook.assign_tensor`, so a
position only reassigns to a different prototype when the alternative's advantage exceeds a fixed
threshold. Causal, decoder-compatible (a fixed rule both sides can compute), minimal (no retraining,
no new model), and targeted precisely at the ~9-12%-of-excess-bits mechanism this report isolated -
with an explicit, pre-stated ceiling: it cannot touch the larger 88-91% mechanism, which would
require addressing the reference itself (M18 already showed the quantizer angle on that fails).

---

## 2026-09-12 — M18: intra quantizer precision audit (DOES NOT EXPLAIN THE M17 GAP)

**Source:** M17 confirmed a large, real, deployed-pipeline residual-rate bottleneck (+2.0% to
+14.9% total-stream upper bound vs an unattainable oracle reference) and decomposed ~85% of it to
reference PIXEL quality. M18 tests the smallest plausible realizable fix: recalibrating the intra
quantizer's scale/percentile parameters - never its architecture, alphabet, or any frozen
downstream stage.
**Tests:** 1310 passing (full suite), zero regressions. 10 new tests.
**Scope:** every M13/M14/M15/M17 identity, checkpoint, quantizer family, codebook, motion
estimator, GOP, `.nvct` v2 frozen; DAVIS TEST untouched. Zero modifications to any existing
tracked file.

### The audit rejects the naive premise before any candidate is even tested

Normalized (step/std, SNR - never raw scale) comparison of the deployed intra vs residual
quantizers shows intra is already the RELATIVELY better-calibrated one at every bit depth (e.g.
5-bit SNR 20.24dB vs residual's 13.43dB; near-identical clipping fractions, 0.44% vs 0.44%). Intra's
larger ABSOLUTE error comes from latents with inherently higher variance (whole-image content vs a
motion-compensated difference), not a fixable calibration defect.

### Two realistic candidates, tested through the real deployed pipeline anyway

Broader TRAIN coverage (best offline MSE candidate at 5/4-bit, +37.7%/+12.4% latent MSE) and
tighter percentile clipping (best at 3-bit, +19.8% latent MSE) were each pushed through the real
M11-G16+M13 closed loop, holding everything except I-frame coding frozen. Neither closes the gap:
broader coverage recovers a negligible 1.3%/-2.0% of M17's oracle gap (net total-stream
+0.194%/+0.104% - real but an order of magnitude below the 0.5% gate); tighter clipping recovers a
real 6.8% on the P-frame channel alone at 3-bit, but its own I-frame byte cost (+27%) MORE than
erases it - net total-stream **-3.244%**, worse than doing nothing. Where a candidate helps at all,
the effect is boundary-position-concentrated, matching M16/M17's own GOP-position finding exactly.

### The important secondary question, confirmed

Per the milestone's own explicit framing: intra reconstruction genuinely improved (latent and, in
most configurations, image-space MSE) under every tested candidate, while actual downstream
residual bytes barely moved or moved the wrong way once I-frame cost was honestly netted out. The
bottleneck is real (M17) but is NOT simply "bad intra PSNR" - aggregate reconstruction quality does
not reliably predict downstream byte cost through the context-conditioned G16 + codebook pipeline.

### Verdict: C - INTRA QUANTIZER DOES NOT EXPLAIN / CLOSE THE M17 GAP

No candidate reached the 0.5% coded-validation gate; Phase F/G were correctly never triggered. This
strengthens rather than weakens the case for M17's own larger-scope recommendation - a causal
reference-refinement mechanism targeted at the SHAPE of the reference error, not just its
magnitude, since magnitude improvements alone (this milestone's entire candidate set) do not
transfer to byte savings.

---

## 2026-09-12 — M17: residual oracle audit through the deployed M11-G16 + M13 pipeline (MEANINGFUL BOTTLENECK CONFIRMED)

**Source:** M16's simplified per-channel proxy estimated a residual-side reference-quality gap of
+1.3% to +12.7% and explicitly flagged it as *not decision-grade* - a lead, not a finding, since
the real deployed residual coder (M11-G16 causal context + 512-entry codebook + M13's recalibrated
frequencies) is far more sophisticated than a per-channel model. M17 redoes the same real-vs-oracle
comparison through the actual deployed pipeline, never a proxy.
**Tests:** 1300 passing (full suite), zero regressions. 11 new tests.
**Scope:** every M13/M14/M15/M16 identity, checkpoint, quantizer, codebook, motion estimator, GOP,
`.nvct` v2 frozen; DAVIS TEST untouched. Zero modifications to any existing tracked file.

### The result reverses the expected direction

Contrary to the natural assumption that a sophisticated, context-conditioned model would already
capture what a crude proxy misses (making the proxy an *overestimate*), the real pipeline shows an
even LARGER gain than M16's proxy: +2.47% / +8.47% / +19.61% channel-level at 5/4/3-bit (vs the
proxy's +1.29% / +4.52% / +12.69%). The mechanism: G16's assignment to one of 512 codebook
prototypes depends on the model's context-conditioned prediction, which itself depends on the
reference latent - so a degraded reference doesn't just shift the residual's marginal distribution
(what a per-channel model would catch), it also misroutes positions to badly-matched prototypes.
At 3-bit, up to 73% of all codebook assignments differ between the real and oracle reference.

### Decomposition and GOP-position shape

~85% of the effect at every bit depth is reference PIXEL quality alone (motion vectors held fixed);
only ~15% comes from additionally re-selecting motion vectors against the oracle reference - the
same decomposition M16 established matters little for motion's own channel matters a great deal
here, because residual is 76-82% of total stream bytes where motion is only 4-9%. The effect is
boundary-dominated (M16's own GOP-position shape, confirmed again): position 1 shows a 2-3x larger
gap than any other position (38.7% at 3-bit vs 13.9-18.2% elsewhere), never boundary-only or
monotonically accumulating.

### Total-stream upper bound, and the coded-validation gate

Translated to total-stream bytes via the real M13/M14 byte-share breakdown: **+2.03% / +6.75% /
+14.86%** at 5/4/3-bit - clearing the 1% "meaningful" line by a wide margin at every rate point
(3-bit alone exceeds M13's entire original deployed recalibration gain). This triggered Phase F's
coded-validation gate: 243/243 oracle-variant payloads round-tripped exactly through the unmodified
entropy decoder, and the 5-bit diagnostic reproduced byte-identical totals in a second, independent
process. Both corroborate the numbers are genuine, real, arithmetic-coded bytes - not a computation
artifact.

### Why nothing ships

The oracle reference is, by construction, unavailable to any real decoder - it requires the raw,
uncoded previous frame, which a real system never transmits twice. Every mechanism that could
plausibly capture part of this gap (a reference-refinement model, a redesigned quantizer) is a new
model or architecture change, explicitly outside this audit milestone's freezes. M17 confirms
*whether* a bottleneck exists; it does not attempt to close it.

### Verdict: A - RESIDUAL ORACLE GAP IS A MEANINGFUL TOTAL-STREAM BOTTLENECK

Confirmed under the real deployed pipeline, corroborated by round-trip decode and cross-process
reproducibility - the strongest possible evidentiary standard this project's audit milestones use,
and the first of M14-M17's audit chain to land on "yes, real, and large" rather than "confirmed real
but too small to matter."

---

## 2026-09-12 — M16: GOP-boundary reference / motion-calibration asymmetry audit (REAL EFFECT, NOT A BOTTLENECK)

**Source:** M14/M15 documented a bit-depth asymmetry in the *calibration-time helper
functions* (`calibrate_grids`, `collect_motion_symbols`), concentrated at GOP boundaries.
M16 asks whether the real deployed coder has a matching, materially-sized inefficiency -
audit first, before touching anything.
**Tests:** 1289 passing (full suite), zero regressions. 12 new tests.
**Scope:** every M13/M14/M15 identity, checkpoint, quantizer, codebook, motion estimator,
GOP, `.nvct` v2 frozen. Zero modifications to any existing tracked file.

### The trace, and a correction made mid-audit

Traced `m13_closed_loop.encode_multi`/`decode_sequence` directly - the real coder, not the
calibration shortcuts. Every frame's reference (I or P) is a real, quantized pixel
reconstruction; there is no boundary-specific code branch anywhere. Phase A's own first
conclusion from this was that "GOP-boundary bit-dependence" is purely a calibration-function
artifact. Phase B/C's actual measurements then showed that conclusion was only half right:
the *code* has no boundary special-case, but the *data* does - intra quantization is
measurably coarser (relative to signal) than residual quantization at matched bit budgets,
so the one reference transition through intra quantization (I -> first P) takes a much
bigger one-time hit than any transition through residual quantization (P -> next P). Stated
plainly rather than silently smoothed over, since catching your own structural read being
incomplete is the point of an audit.

### The measurement

Real vs. an ideal (true-latent, oracle-only, never deployed) reference, VAL-B, per GOP
position (gop_size=10): position 1 (boundary) shows a SAD gap of +5.2% / +11.9% / +34.6% at
5/4/3-bit - roughly 3-4x every other position in the GOP (which stays comparatively flat at
+0.8-12.7%, not a monotonically growing "accumulating error" curve). Up to 30.5% of all
motion vectors (45.3% at the boundary position specifically, 3-bit) differ from what a
perfect-reference encoder would choose. Motion channel entropy gain reaches "meaningful"
under the real entropy-model class: +1.04% (4-bit), +3.98% (3-bit).

### Why nothing ships

Translated to total-stream bytes using motion's actual ~4-9% byte share, the oracle upper
bound for motion is **at most +0.36% of total bytes (3-bit, best case, 100% unrealizable
oracle)** - weak at every bit depth. Per the milestone's own gate ("if the oracle is not
meaningful, STOP"), no implementation was attempted; Phase F/G/H were correctly never
reached. A parallel residual-channel proxy measurement showed a much larger apparent gap
(up to +12.7% channel / +9.6% total-stream at 3-bit) but only under a simplified per-channel
model, not the real deployed G16 + M13 entropy coder - flagged as an M17 lead, not acted on,
since it is not decision-grade evidence as measured.

### Verdict: REFERENCE DISCREPANCY CONFIRMED AND GOP-BOUNDARY-CONCENTRATED, NOT A BOTTLENECK

A real, precisely-characterized, mechanistically-understood effect was found and correctly
rejected once translated to the metric that actually matters (total-stream bytes) rather
than chased on an eye-catching channel-level or motion-vector-level number.

---

## 2026-09-12 — M15: broad-TRAIN calibration policy, a root-cause experiment (COVERAGE CONFIRMED, NO PRODUCTION CHANGE)

**Source:** M14 found and fixed a motion-table calibration gap and named a suspected general
mechanism: `calibrate_grids`'s sequential, first-N-frames TRAIN sampling under-covers the
72-sequence TRAIN population. M15 tests that mechanism directly as a controlled,
same-budget experiment, instead of shipping another one-off recalibration.
**Tests:** 1277 passing (full suite), zero regressions. 23 new tests.
**Scope:** every M13/M14 identity, checkpoint, quantizer, codebook, motion estimator, GOP,
`.nvct` v2 frozen. Zero modifications to any existing tracked file — every M15 change is
new, untracked infrastructure (confirmed via `git status`).

### Four policies, one shared abstraction

`scripts/m15_calibration_policy.py` — entirely separate from `calibrate_grids` — returns,
for any of four named policies, a list of `BenchmarkSequence`s truncated to an unmodified
**prefix** of each sequence's own frames, so the *existing*, unmodified M14 collectors can
fit a table from any of them identically to how they fit one from `calibrate_grids`'s own
list:

- **A — current**: sequential, manifest order, stop at 400 total (the deployed policy, made explicit).
- **B — uniform**: flat allocation across all 72 sequences, same 400-frame budget as A.
- **C — M14 broad**: 8 frames/sequence, 576 total (M14's own shipped recipe).
- **D — shuffled uniform**: seed-42 shuffled sequence order, then B's allocation rule.

At the same 400-frame budget, A's per-sequence-count stdev (17.71, touching 6/72
sequences) is **35× B/D's** (0.50, touching 72/72) — the coverage gap this milestone set
out to test, measured directly rather than assumed.

### The central result

Held-out VAL-B bits/symbol for motion (bit-depth-independent, one table per policy):

| policy | gain vs A (root cause) | gain vs deployed C (Phase G) |
|---|---|---|
| B — uniform, 400 | **+11.91% (meaningful)** | −0.38% (weak) |
| C — M14 deployed, 576 | +12.24% (meaningful) | — |
| D — shuffled, 400 | +12.04% (meaningful) | −0.22% (weak) |

400 uniform (B) recovers 97.3% of 576 broad's (C) entire gain over sequential (A) despite
31% fewer frames; shuffling order (D) changes almost nothing. **Coverage, not sample
count or manifest order, explains essentially all of M14's motion gain** — the strong-
evidence pattern the milestone was designed to distinguish from the weaker alternatives.
Intra stayed weak (+0.35–0.43%) under every policy at every rate point, generalizing
M14's original rejection beyond its one tested recipe.

Confirmed on real arithmetic-coded bytes (3 validation sequences, frozen M13 residual
arm): current (A-intra + C-motion, M14's exact deployed state) vs broad_motion (A-intra +
B-motion) differ by **+0.0033% to +0.0083%** total-container bytes - negligible, and in
the opposite direction from the offline VAL-B comparison (expected sample noise at ~90
P-frames). Every invariant held: symbols, motion, reconstruction, PSNR/MS-SSIM
bit-identical between combos at every rate point.

### Why nothing ships

M14's own deployed recipe (8 frames/sequence, exactly 576/72 = 8, zero remainder) is
*itself* already a perfectly flat, fully-covering allocation - a degenerate special case
of "uniform." M15 confirms the *general principle* (coverage) is what made M14 work, but
the *specific instance* already in production was already a good one - there was no gap
left for a more "principled" same-family policy to close. The full 719-frame DAVIS
benchmark was deliberately skipped: the milestone's own gate promotes a candidate to that
compute only if it beats the *current deployed baseline* meaningfully, which none did.

Per-sequence diagnostic (VAL-B): 3 of 4 sequences improve substantially and consistently
under every broad policy; `pigs` regresses under B, C, *and* D alike - a real,
consistent exception, reported rather than smoothed over.

### Verdict: ROOT-CAUSE CONFIRMED (COVERAGE), NO PRODUCTION CHANGE WARRANTED

A clean, reusable, deterministic, TRAIN-only, provenance-compatible calibration-policy
abstraction now exists for any future milestone that needs it. Nothing in the deployed
codec changes, and the report says so plainly rather than manufacturing a win from a
well-run negative result.

---

## 2026-09-11 — M14: entropy-table calibration audit across the codec (MOTION TABLE RECALIBRATED, INTRA CORRECTLY REJECTED)

**Source:** M13 recalibrated one static entropy table (the deployed M11-G16 residual codebook) and
shipped a real coded-byte gain. M14 asks the obvious follow-up: where else does this calibration gap
exist? Audit every static table in the deployed codec before recalibrating any of them.
**Tests:** 1251 passing (full suite), zero regressions. 19 new tests (`test_m14_recalibration.py`:
13, `test_m14_closed_loop.py`: 6).
**Scope:** λ=3.0e-4, M13's recalibrated residual table, checkpoints, quantizer, codebook
prototypes/assignments, motion estimator, GOP, `.nvct` v2 all frozen. One correctness fix to the
decoder's caller (below) — no format or bitstream change.

### Phase A — the audit

Of the `.nvct` v2 header's three entropy-model identity slots, two were still live and never
recalibrated: `intra_entropy_model_id` and `motion_entropy_model_id`. A third table —
`calibrate_grids`'s plain per-channel residual grid — turned out to be dead weight: computed every
run, never read by the deployed M13 closed loop (residual coding goes through M11-G16 + M13's
codebook instead). Excluded from candidacy — recalibrating a table nothing reads can't change a
deployed byte.

**The root cause differs from M13's.** Intra and motion were already direct empirical fits, never
neural-predicted, so M13's "predicted vs actual" story doesn't apply. Instead: TRAIN has 72
sequences / 4,826 frames, but `calibrate_grids`'s `max_frames=400` budget is spent *sequentially*
(walk sequences in manifest order, stop at N frames) — so the deployed tables are fitted from ~6 of
72 sequences, about 8% of TRAIN. A coverage gap, not a bias gap.

### Phase B — offline gate (held-out VAL-A; pre-registered thresholds: <0.5% weak, 0.5–1.0%
marginal, ≥1.0% meaningful)

- **intra**: +0.37% to +0.41% across 5/4/3-bit — weak, rejected before any further compute was spent.
- **motion**: +10.55% to +11.89% across 5/4/3-bit — meaningful, proceeds to coded validation.

(Caught a real bug first: an early draft fed the broadly-sampled TRAIN list into the
*deployed-baseline* calibration too, making both sides draw from the same narrow sample and
producing nonsensical negative "gains." Fixed by separating the full, untruncated sequence list used
to reproduce the deployed baseline from the per-sequence-capped list used to fit the recalibrated
candidate.)

### Phase D — actual arithmetic-coded validation (90 VAL frames, 3 sequences)

| bits | total bytes, baseline → motion | total container gain |
|---|---|---|
| 5 | 587,939 → 582,935 | +0.8511% |
| 4 | 418,691 → 413,773 | +1.1746% |
| 3 | 268,449 → 263,903 | +1.6934% |

All invariants (symbols, reconstruction, motion, PSNR, MS-SSIM) identical baseline vs motion at
every rate point.

### Phase E — full DAVIS TEST (719 frames, 9 sequences)

| bits | motion-channel bytes | motion gain | TOTAL container bytes | TOTAL gain |
|---|---|---|---|---|
| 5 | 181,118 → 164,010 | +9.446% | 4,309,995 → 4,292,887 | +0.3969% |
| 4 | 184,298 → 166,842 | +9.472% | 3,027,347 → 3,009,891 | +0.5766% |
| 3 | 188,896 → 171,612 | +9.150% | 1,918,028 → 1,900,744 | +0.9011% |

Motion is only 4–6% of total stream bytes, so the two numbers answer different questions and both
are reported — the channel-level gain is the honest measure of the fix itself; the total-stream gain
is the honest measure of its deployed impact. BD-rate: **−0.683%** (PSNR), **−0.681%** (MS-SSIM).
PSNR/MS-SSIM bit-identical baseline vs motion at every rate point. Motion's table carries no neural
network, so recalibration structurally cannot move latency — confirmed, not just expected, by
per-stage timing at every rate point.

### A correctness gap found and fixed along the way

`m13_closed_loop.decode_sequence` checked only `residual_entropy_model_id` against the stream
header — never `intra_/motion_entropy_model_id`. Invisible in M13 (which never varied those tables
across a call), directly exploitable once M14 started shipping two different motion tables: decoding
a recalibrated-motion stream against the wrong table would have silently produced corrupted motion
vectors with no error, only a wrong reconstruction. Fixed: all three `.nvct` v2 identities are now
checked before a symbol is decoded. Zero regressions across the full 1251-test suite.

### Verdict: CALIBRATION GAP CLOSED (MOTION), CORRECTLY REJECTED (INTRA)

Audit-first discipline paid for itself: intra was rejected at the cheapest possible stage (the
offline gate) instead of being chased through coded validation and a full DAVIS run first. Motion
cleared every gate through to a full-DAVIS, byte-accounted, invariant-checked confirmation.

**M15 candidate (from the report):** don't chase more tables one at a time —
`calibrate_grids`'s own sequential (not per-sequence) frame-selection strategy under-samples TRAIN
for *any* table it fits. Fixing that directly addresses the shared root cause behind this gap and
any others like it.

---

## 2026-09-11 — M13: deployed M11-G16 table recalibration (REAL, MEANINGFUL, DEPLOYED COMPRESSION GAIN)

**Source:** M12's spatial-context offline gate incidentally measured that recalibrating M11-G16's
512-entry codebook frequencies — fitting them from TRAIN's actual symbol histogram instead of the
network's predicted distribution — gained +1.6% to +5.1% offline, with zero spatial context at all.
M13 asks whether that survives to real, deployed, arithmetic-coded bytes.
**Tests:** 27 new (`test_m13_recalibration.py`: 14, `test_m13_closed_loop.py`: 13); full suite green,
zero regressions.
**Scope:** codebook prototypes, symbol-to-prototype assignment, quantizer, motion estimator, model,
arithmetic coder, GOP, λ=3.0e-4, `.nvct` v2 all frozen. Only the residual entropy *frequency table*
changes.

### The split that makes this safe

A `SharedCodebook`'s assignment (which prototype a position routes to) must never be computed from a
recalibrated table — only the original, deployed codebook may decide assignment. A separate codebook
object carries the recalibrated frequencies, used only for coding. Two tests pin this:
`test_assignment_never_uses_the_recalibrated_codebook`,
`test_recalibration_does_not_mutate_the_original_codebook`. Because assignment is untouched, symbols
and reconstruction are provably identical old vs new — recalibration can only change which frequency
table *encodes* an already-decided symbol, never which symbol gets chosen. Provenance comes for
free: `model_identity()`/`codebook_id()` hash weights, calibration *and* frequencies, so the
recalibrated table gets its own distinct 8-byte `.nvct` identity automatically.

### Offline gate (Phase C, independent reproduction through the real integer-quantized tables)

+1.609% / +2.927% / +5.110% at 5/4/3-bit.

### Deployed — DAVIS TEST, 719 frames, every byte counted

Byte gain **+1.42% / +2.64% / +4.04%** at 5/4/3-bit, realized gain 99.97–100.02% of the offline
estimate — the coder captures essentially all of the modelled gain. BD-rate **−2.372%** (PSNR),
**−2.366%** (MS-SSIM). PSNR and MS-SSIM bit-identical old vs new at every rate point; symbols,
motion and reconstruction identical; byte accounting closes.

### A correctness gap found and fixed along the way

Phase D/E's first draft loaded the M11-G16 checkpoint without calling M11's own `check_provenance()`
guard against fresh calibration — a real silent-stale-model risk. Fixed; the guard passes cleanly
(confirms provenance was valid all along, just previously unverified).

### Verdict: REAL, MEANINGFUL, DEPLOYED COMPRESSION GAIN

Not just an offline estimate — confirmed on real, arithmetic-coded bytes at three independent scales
(VAL-A tuning, VAL-B/Phase C offline reproduction, full DAVIS TEST), with quality and every
non-frequency invariant untouched.

---

## 2026-09-10 — M12: a resumable arithmetic decoder, and a spatial-context ceiling check (RESUMABLE DECODER SUCCESS, WEAK SPATIAL SIGNAL)

**Source:** M11's channel-autoregressive entropy model (G16) decodes each group via a prefix
redecode — correct, but the coder re-walks every already-consumed bit from the start on each call.
M12 asks two independent questions: (1) can the decoder be made resumable — stateful across calls —
without touching the bitstream; and (2) once resumable, does spatially-causal context (left / up /
neighbourhood) add anything beyond M11-G16's channel context?
**Tests:** 40 new (`test_m12_resumable_decoder.py`: 31, `test_m12_spatial_offline_gate.py`: 9); full
suite green, zero regressions.
**Scope:** M10H motion estimator, quantization, M10K/M10L entropy models, M11-G16 operating point,
GOP, λ=3.0e-4, calibration, `.nvct` v2 all frozen. No bitstream change — the resumable decoder is
byte-exact to the existing one by construction, not by comparison.

### Part 1 — resumable decoding

Split the decoder's state (`low`/`high`/`value`, bit-reader position) from its loop:
`rc_decoder_open` / `rc_decoder_decode` / `rc_decoder_close` in the native range coder, wrapped by a
Python `ResumableDecoder`. The legacy `rc_decode` becomes a thin wrapper over the same three calls,
so equivalence to the old path holds by construction. Coder-step speedup: **2.0–2.2×** at G=16
(M11's deployed group size), **5.0–7.1×** at G=1.

### Part 2 — is there anything left for spatial context to find?

Reused M11's own causal-context offline-gate machinery, parent distribution swapped from M10L to
M11-G16's own codebook, with a strategic rule fixed before measuring: M11-G16 already ships and
works, so a new context needs to clear a higher, pre-registered bar (≥1.0% = "meaningful") to be
worth the resumable-decoder cost of deploying it. Net gain (whole-split, permutation-controlled):
**0.31–0.91%** at 5/4/3-bit — real, but never crosses 1.0% at any rate point.

### Verdict: RESUMABLE DECODER SUCCESS, WEAK SPATIAL SIGNAL

The resumable decoder ships as infrastructure — a strict win (byte-exact, faster, no format change)
independent of what ends up conditioning on it. Spatial context is measured, not deployed: below its
own pre-registered bar, reported as a negative result rather than force-fit into a deployment. (M13
found a better lever on the same codebook: recalibrating its *frequency table*, not its context,
gained +1.6–5.1% offline with zero spatial context at all.)

---

## 2026-09-10 — M11: decoded residuals know where motion compensation failed (AUTOREGRESSIVE ENTROPY SUCCESS)

**Source:** M10K/M10L model P(R | z_ref, channel). M11 asks whether residual symbols the decoder has
*already decoded* add enough information to justify a sequential dependency in the entropy model.
**Tests:** 1168 passing (full suite), zero regressions. 110 new tests.
**Scope:** no `src/nvc/` change, no coder change, no container change. `.nvc`, `.nvcs`, `.nvct` v1/v2
untouched; every M10A–M10L script byte-identical. λ frozen at 3.0e-4.

### Phase 0 — the calibration determinism fix, measured on a GPU

The `calibrate_grids` guard landed in `61dd8434` on a CPU-only machine, which could only confirm the
guard was *active*. Measured here on the GPU, with the real checkpoint and a DAVIS validation clip,
two independent processes:

| | pre-fix (`8d22f419`) | fixed |
|---|---|---|
| fingerprinted fields that differ | **9 / 19** | **0 / 19** |
| residual grid, max \|diff\| | 2.6e-5 (7.7e-5 relative) | 0 |
| stream bytes (container / residual / motion) | +29 / +39 / −10 | 0 / 0 / 0 |

Worth knowing: the fix does not just remove noise around the old value. Deterministic cuDNN
kernels round differently from the default ones (7.0e-4 per pixel on decode), so calibration moves
to a new, *stable* point — median residual-scale shift 0.05%, max 1.0% on one channel, zero-points
unchanged, stream bytes ~+0.05%. **Absolute byte counts after the fix therefore differ from the
published M10H–M10L figures by about that much**; within-run comparisons were never affected. The
benchmark re-verified the fix at full scale: its fresh 400-frame calibration matched the cached one
from a different process, byte for byte, at every rate point. `scripts/m11_reproducibility.py` and
`tests/test_m11_reproducibility.py` make the two-process check permanent — calibration, z_ref,
residual symbols, stream bytes, reconstruction, and M11's own probabilities, tables and payload.

### The audit findings that shaped everything

- **Coding order is C-major raster**, `i = c·H·W + y·W + x` (pinned against the codec by test).
  When symbol (c, y, x) is decoded the decoder holds *every* position of channels 0…c−1 — including
  positions spatially ahead of the current one — plus rows above and pixels to the left in channel c.
- **`rc_decode` is stateless and batch-only.** Decoding a prefix is exact, so a model conditioned
  only on earlier *channels* can decode one channel per step through the unmodified coder with zero
  rate overhead. A spatially conditioned model needs a table per symbol: ~16,384 prefix decodes,
  O(N²), seconds per frame. Measured: 64 channel-prefix decodes cost 28.3 ms vs 0.9 ms for one full
  decode (31.6×), bit-exact; 64 separate per-channel streams would instead cost +0.64% in
  terminations before any length table.

### Offline gate — baselined on M10L, not on the marginal

Count tables conditioned on M10L's 512-prototype index, so z_ref's information is already in the
baseline and any gain is information z_ref did **not** provide. Tuned on VAL-A, reported on VAL-B
(disjoint validation sequences — all 9 validation sequences used, where M10K/M10L's gates used 3).

Three findings along the way, each of which would have misled if taken at face value:

1. **Recalibration is not context.** Refitting P(R | prototype) on TRAIN symbols alone gains
   +0.06 / +0.75 / +1.91% at 5/4/3-bit with no residual context at all. Reported separately.
2. **My first random control was wrong.** Shuffling contexts *within a frame* keeps each frame's
   histogram — a "how busy is this frame" statistic built partly from future positions — and showed
   a spurious +0.147% for a skewed context. A whole-split shuffle and M10J's uniform labels agreed at
   −0.08%, the honest price of a useless context. Fixed; a regression test pins it.
3. **The smoothing grid was too small** and its optimum sat on the edge; extended, the controls
   tightened to +0.000–0.009% and 5-bit recalibration shrank from an apparent +0.40% to +0.06%.

| context (family) | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| left + up (spatial) | +2.94% | +3.58% | +4.59% |
| prev_channel (channel) | +3.21% | +4.28% | +5.41% |
| channel_activity (channel) | +2.43% | +6.46% | +12.13% |
| neighbourhood × channel (both) | +3.79% | +7.87% | +12.53% |

(held-out VAL-B vs deployed M10L; every permuted control within +0.01%)

**Why it works:** the residual is `latent − E(warp(prev))`. Where motion compensation fails, the
residual is large in every channel at once — and z_ref, which describes the reference, cannot know
where the warp failed. Once a few channels are decoded, their activity at a position says exactly
that. It dominates at 3-bit, where coding is mostly a zero/non-zero decision.

### The model

M10K's network plus three causal input planes per channel — previous-group magnitude, running
activity over every earlier group, and an availability flag — **12,744–13,536 parameters**, 56–60 KB.
Warm-started from M10K with the new weights at zero, so step 0 *is* M10K; a no-context ablation
trained identically separates what context contributes (it contributes all of it: the ablation
moves −0.01 / +0.06 / +0.14%). Channels are decoded in **groups of G**: decode time is set by the
number of sequential steps (a step costs ~0.7 ms whether it handles 1 channel or 64 — launch-bound),
so G trades context for steps. Operating point chosen by a rule fixed before any G was measured:
the largest G keeping ≥ 50% of the G = 1 VAL-A gain at every rate point → **G = 16, four steps**, with
its own 512-entry codebook (which costs nothing — at 5-bit it is marginally better than per-position
tables).

### Deployed — DAVIS test, 719 frames, every byte counted

| arm | bits | P residual | TOTAL | BPP | vs M10L (P resid) | decode ms/P |
|---|---|---|---|---|---|---|
| M10L | 5 | 3,813,564 | 4,595,790 | 0.7803 | | 3.8 |
| **M11-op (G16)** | 5 | **3,578,734** | **4,360,960** | **0.7404** | **−6.16%** | **7.0** |
| M11 (G1) | 5 | 3,508,074 | 4,290,300 | 0.7284 | −8.01% | 102.9 |
| M10L | 4 | 2,631,880 | 3,261,023 | 0.5536 | | 4.7 |
| **M11-op (G16)** | 4 | **2,463,204** | **3,092,347** | **0.5250** | **−6.41%** | **8.8** |
| M11 (G1) | 4 | 2,410,267 | 3,039,410 | 0.5160 | −8.42% | 119.9 |
| M10L | 3 | 1,588,393 | 2,066,495 | 0.3508 | | 5.4 |
| **M11-op (G16)** | 3 | **1,500,471** | **1,978,573** | **0.3359** | **−5.54%** | **8.5** |
| M11 (G1) | 3 | 1,476,433 | 1,954,535 | 0.3318 | −7.05% | 99.3 |

**PSNR and MS-SSIM identical to every digit across all six arms**; symbols, motion and
reconstruction identical; byte accounting closes; coder realises 100% of every modelling gain
(overhead ≤ 0.026%).

| BD-rate | PSNR | MS-SSIM |
|---|---|---|
| **M11-op vs M10L** | **−4.81%** | **−4.81%** |
| M11 (G1) vs M10L | −6.25% | −6.25% |
| M11 (G1) vs intra | −38.44% | −47.19% |

M11-op beats M10L on **all 9 test sequences** (−2.40% to −7.80% at 4-bit), most on the
motion-sensitive ones — schoolgirls −7.80%, bmx-bumps −6.69%, drone −6.68% — which is the mechanism
above showing up per sequence. Test tracked validation closely: 5.5–6.4% on test against 6.4–7.1%
on VAL-B.

### What it costs

M11-op decodes in 7.0–8.8 ms/P against M10L's 3.8–5.4 — **1.6–1.9×**, for −4.81% BD-rate. Four
sequential steps; per step the full network runs once (parallel over every channel and position),
a prototype assignment, and a prefix decode. Encode is a single parallel pass (3.5–6.8 ms/P). Full
channel autoregression (G = 1) buys another 1.4 points for 18–27× M10L's decode time — not a
practical trade with this coder.

### Two things this does not claim

- **The learned model is not this family's ceiling.** At 3-bit a count table
  P(R | M10L prototype, channel-activity bucket) beat it on the same validation frames (+12.1% vs
  +8.9%): the learned no-context arm recovered only +0.14% of the +1.9% recalibration the tables
  found, and every context arm selected its final or penultimate epoch — training had not converged
  within M10K's matched 20-epoch budget.
- **Spatial context is measured, not deployed.** It adds 0.4–1.4 points on top of channel context
  in the gate, and needs a resumable decoder to deploy.

### Verdict: AUTOREGRESSIVE ENTROPY SUCCESS

Held-out gain far past the 1% "meaningful" line at every rate point, realised as actual bytes on the
test set (−4.81% BD-rate over M10L at the practical operating point), with symbols, motion,
reconstruction and quality bit-identical, provenance enforced, and decode latency within 2× of M10L.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen model: M10F λ=3e-4 seed 42 `best.pt`.
Entropy models warm-started from M10K, trained 20 epochs on 536 TRAIN P-frames per rate point,
selected on VAL-A, reported on VAL-B; test read once. Closed-loop symbols cached in the system temp
directory under a key covering the checkpoint bytes, the M10H source, every codec setting and the
sequence lists. All commands need the project venv (`./.venv/Scripts/python.exe`).

---

## 2026-09-10 — Per-channel INT8 activation quantization for NVC-ACCEL: real improvement, not a full fix

**Source:** `hardware/ARCHITECTURE.md`'s own open risk #1 and RESEARCH_NOTES_NEXT_STEPS.md §C measured a
real ~1 dB PSNR cost from per-tensor INT8 activation quantization (one scale for a whole layer) and
named the standard fix - per-channel activation scales, on direct precedent from this project's own
latent quantizer, which is already per-channel - but never implemented it. Isolated entirely to the
hardware-accelerator validation track; cannot touch the real running codec (no `src/nvc/` change).
**Tests:** no dedicated test file (`hardware/` sits outside TESTING.md's rule, matching
`test_parallel_entropy_poc.py`'s existing precedent); correctness verified instead by re-running the
original per-tensor path after the refactor and confirming it reproduces the prior measurement
(−0.998 dB here vs. −0.997 dB originally - noise-level, confirms the refactor changed nothing about
the existing path).

### What changed

`hardware/int8_activation_validation.py` gets a new `--activation-quant per-tensor|per-channel` flag.
`per-tensor` (the original behavior) stays the default; `per-channel` computes one percentile-calibrated
INT8 scale per input channel instead of one for the whole layer, mirroring
`_fake_quantize_weights_per_output_channel`'s existing per-output-channel treatment of weights. Both
paths share one calibration/evaluation loop so they're directly comparable from one script.

### Result — real QAT checkpoint, full 719-frame DAVIS test split, both modes measured

| activation quant | Mean PSNR | Δ vs. float32 | Mean MS-SSIM | Δ vs. float32 |
|---|---|---|---|---|
| per-tensor (original) | 28.750 dB | −0.998 dB | 0.9611 | −0.0119 |
| **per-channel** | **28.941 dB** | **−0.807 dB** | **0.9674** | **−0.0056** |

Per-channel recovers ~19% of the PSNR cost and ~53% of the MS-SSIM cost versus per-tensor - a real,
measured improvement, not asserted from the general principle alone. **But it does not close the
gap**: −0.807 dB is still roughly 6x the 0.134 dB QAT-alone 8-bit→6-bit drop this project already
treats as an acceptable cost, so per-channel activation quantization alone does not make INT8
activations free for this design. Calibration fit stayed healthy (0.1389% clipped, threshold 2%) and
entropy coding stayed lossless every frame, exactly as the original per-tensor run.

### What's still open

INT16 activations - the other fix RESEARCH_NOTES_NEXT_STEPS.md §C already named - remain unimplemented
and unmeasured. Whether −0.807 dB is an acceptable V1 cost, or whether closing the rest of the gap is
worth INT16's larger SRAM budget, is a real engineering call this measurement informs but doesn't
settle by itself.

### Disposition

Documentation updated in `hardware/ARCHITECTURE.md` (§10 risk #1, §11) and `RESEARCH_NOTES_NEXT_STEPS.md`
§C with the same numbers. Code lives on `master` directly (not a throwaway branch) - unlike the
speculative chroma-subsampling/decode-speed investigations, this is a straightforward, correct,
isolated improvement to validation tooling with no downside and zero risk to the real codec.

---

## 2026-09-10 — M10L: a 512-entry shared codebook keeps M10K's gain at a tenth of the cost (PRACTICAL SUCCESS)

**Source:** M10K won ~1% of residual bytes over M10J with a 12k-parameter learned entropy model, but
profiling put 97–99% of its cost in building 16,384 integer frequency tables per P-frame — a pure
implementation problem, not a modelling one. M10L asks whether a small shared codebook of tables can
carry the same gain.
**Tests:** 1056 passing (full suite), zero regressions. 72 new tests.
**Scope:** no `src/nvc/` change, no container change, no coder change. `.nvc`, `.nvcs`, `.nvct` v1/v2
untouched; every M10A–M10K script byte-identical. λ frozen at 3.0e-4.

### The measured bottleneck, before touching anything

Stage-by-stage profile of the real M10K P-frame path (verified against `frame_entropy_model` on every
frame — same frequencies, same `table_index`, same payload):

| stage | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| network forward | 0.81 ms (2.5%) | 0.57 ms (4.8%) | 0.61 ms (8.0%) |
| probability normalization | 0.15 | 0.12 | 0.14 |
| **float → int frequencies** | **26.52 ms (80.7%)** | **8.78 ms (74.4%)** | **5.29 ms (69.3%)** |
| table validation + alloc | 2.01 | 0.83 | 0.58 |
| cumulative materialization | 0.008 | 0.007 | 0.007 |
| arithmetic encoding | 0.97 (3.0%) | 0.62 (5.2%) | 0.48 (6.3%) |
| device → host transfer | 2.39 (7.3%) | 0.86 (7.3%) | 0.51 (6.7%) |
| **TOTAL (deployed)** | **32.88** | **11.79** | **7.63** |

Table construction (float→int + validation + cumsum) is **86.8 / 81.5 / 77.0%** of the path. This is
also a more complete accounting than M10K's own figure, which excluded the host transfer and the
coder — and M10K's benchmark JSON predates its vectorisation fix, so those timings describe code that
no longer exists. Separately timed here and found *not* to be deployment work: the ideal-bits
diagnostic (1.2–3.1 ms), which sat inside M10K's timed region.

### The idea

Fit K prototype distributions on TRAIN once, quantize them into coder frequencies once, and at encode
time replace the whole table build with a lookup:

    z_ref -> network -> probabilities -> nearest prototype -> table_index

A row is assigned to the prototype minimising its **expected code length**, `sum_s p(s) * -log2 q(s)`
— the quantity that actually costs bits. Two consequences: minimising cross-entropy is identical to
minimising KL (they differ by H(p), a per-row constant), and the whole assignment is one GEMM, so the
cheap representation is also the fast one. Clustering is Lloyd's algorithm with the centroid each
metric actually requires — the mean under KL, the component-wise median under L1.

### The zero-loss reference (the STOP condition)

Before measuring any K: a codebook holding each frame's own 16,384 distributions, mapped to itself,
must reproduce M10K **byte for byte**. At all three rate points — frequencies identical, `table_index`
identical, payload identical, round trip exact — and the real nearest-prototype *search* against that
same codebook also reproduces M10K's payload, which validates the assignment code and not just the
plumbing.

### Offline gate — held-out validation, before any test data

Every K at every rate point, fitted on TRAIN, scored on 179 validation P-frames:

| K | 5-bit vs M10K | 4-bit vs M10K | 3-bit vs M10K | table time saved |
|---|---|---|---|---|
| 16 | +0.39% | +0.50% | +0.67% | 94–98% |
| 32 | +0.20% | +0.31% | +0.30% | 93–97% |
| 64 | +0.11% | +0.19% | +0.21% | 93–97% |
| 128 | +0.07% | +0.12% | +0.13% | 93–97% |
| 256 | +0.03% | +0.07% | +0.06% | 55–97% |
| **512** | **+0.01%** | **+0.04%** | **+0.03%** | **82–95%** |

K=512 selected at every rate point: the best held-out rate among candidates clearing the
pre-registered runtime bar (≥50% table-time reduction, ≤0.5% rate penalty), ties to the smaller
codebook. Deliberately not "smallest K that passes" — the runtime bar is a threshold already met, and
the frontier shows the smallest passing K is not even the fastest.

### Deployed — DAVIS test, 719 frames, every byte counted

| arm | bits | P residual | motion | TOTAL | BPP | PSNR | MS-SSIM | vs M10J | vs M10K |
|---|---|---|---|---|---|---|---|---|---|
| learned (M10K) | 5 | 3,813,563 | 181,160 | 4,595,831 | 0.7803 | 29.271 | 0.9737 | −0.80% | |
| **codebook (M10L)** | 5 | **3,813,242** | 181,160 | **4,595,510** | **0.7802** | 29.271 | 0.9737 | **−0.81%** | **−0.01%** |
| learned (M10K) | 4 | 2,631,255 | 184,164 | 3,260,264 | 0.5535 | 28.975 | 0.9676 | −0.93% | |
| **codebook (M10L)** | 4 | **2,632,000** | 184,164 | **3,261,009** | **0.5536** | 28.975 | 0.9676 | **−0.91%** | **+0.02%** |
| learned (M10K) | 3 | 1,588,596 | 188,977 | 2,066,779 | 0.3509 | 27.945 | 0.9473 | −1.49% | |
| **codebook (M10L)** | 3 | **1,588,650** | 188,977 | **2,066,833** | **0.3509** | 27.945 | 0.9473 | **−1.49%** | **+0.00%** |

**PSNR and MS-SSIM identical to every digit across all four temporal arms**, motion bytes identical,
residual symbols identical, reconstruction identical, byte accounting closes. BD-rate **codebook vs
learned +0.01%** — the two are the same codec at different cost. Coder overhead stays at +0.01–0.02%
of the ideal code length in both.

### What it cost, and what it bought

| bits | arm | tables | model ms/P | coder ms/P | total ms/P | vs M10K |
|---|---|---|---|---|---|---|
| 5 | learned | 16,384 | 23.34 | 0.82 | 24.16 | |
| 5 | **codebook** | **512** | **2.30** | 0.72 | **3.02** | **−87.5%** |
| 4 | learned | 16,384 | 13.40 | 0.74 | 14.14 | |
| 4 | **codebook** | **512** | **2.91** | 0.68 | **3.58** | **−74.6%** |
| 3 | learned | 16,384 | 7.86 | 0.57 | 8.43 | |
| 3 | **codebook** | **512** | **2.02** | 0.53 | **2.55** | **−69.8%** |

("model ms" is everything before the coder, and for both learned and codebook that includes the
shared 0.3 ms network forward; the offline gate reports the split.)

Coder-facing table memory drops **95.3 / 93.8 / 91.0%** — 8.52 MB → 266 KB at 5-bit, rebuilt every
frame before, loaded once now. The codebook itself is 24–82 KB on disk. Fitting is a one-off 9–23 s,
offline, on TRAIN.

M10K was ~14× M10J's residual-coding time; **M10L is 1.6–2.2×**, for the same −1.03% BD-rate over
M10J that M10K bought at −1.04%.

### Per-sequence (all deltas vs M10K)

Within ±0.15% everywhere, at every rate point — noise, not structure. M10L is marginally worse on 6–7
of 9 sequences and better on the rest, but it wins on **bmx-bumps and gold-fish at all three rate
points**: the two sequences M10K found hardest are the ones a shared, averaged prototype helps.
M10J's bmx-bumps regression (458,343 vs marginal's 450,488) stays reversed, at 449,102.

### A determinism finding, verified rather than assumed

This run's absolute totals differ from M10K's published run in the fifth significant figure (motion
181,160 vs 181,207, a 0.026% difference). Cause, measured directly: **`model.decode` is not
bit-reproducible without `deterministic_kernels()`** (max abs difference 4.7e-4 on a fixed latent),
and `calibrate_grids` calls it outside that guard — so the quantization grid differs slightly between
*processes* and propagates into reconstruction → motion estimation → motion bytes. Nothing in M10L
touches motion; within a single run all four arms share one calibration and one motion field, which
the invariants confirm. Every M10K-vs-M10L comparison above is within one run and therefore exact.
Worth fixing in `calibrate_grids` eventually — it is M10H code, frozen for this milestone.

### Verdict: PRACTICAL SUCCESS

M10L retains essentially all of M10K's entropy gain (BD-rate +0.01%, and at 5-bit marginally
*better*) while removing 70–88% of the residual-coding time and 91–95% of the table memory, with
symbols, motion, reconstruction and quality bit-identical and no change to the coder or container.

The cumulative arithmetic is worth stating plainly: hand-designed conditioning (M10J) got 2–3%, a
12k-parameter learned model (M10K) got another 1%, and M10L kept that 1% while making it affordable.
None of it needed a bigger model — M10I's 498k-parameter conditional *transform* remains the only
thing in this thread that moved rate the wrong way.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU, seed 42. Frozen model: M10F λ=3e-4 seed 42 `best.pt`.
Codebooks fitted on 100,000 TRAIN-sampled distributions per rate point, 15 Lloyd iterations (the cap
— they had not fully converged, so the ≤0.04% penalty is if anything an overestimate), selected on
179 validation P-frames. All commands require the project venv (`./.venv/Scripts/python.exe`).

---

## 2026-09-09 — M10K: a 12k-parameter learned entropy model beats the lookup tables (LEARNED ENTROPY SUCCESS)

**Source:** M10J showed the warped reference carries exploitable conditional information and captured
2–3% of it with a hand-designed 4-bucket lookup. M10K asks whether a small learned model captures
more.
**Tests:** 984 passing (full suite), zero regressions. 43 new tests.
**Scope:** no `src/nvc/` change, no container change, no coder change. `.nvc`, `.nvcs`, `.nvct` v1/v2
untouched; every M10A–M10J script byte-identical. λ frozen at 3.0e-4.

### The audit finding that made this deployable unchanged

The arithmetic coder already picks a frequency table per symbol via `table_index`, and **nothing
constrains how many tables there are.** A model predicting a different distribution at each of the
64×16×16 = 16,384 positions is therefore expressible as 16,384 tables with `table_index = arange`.
Measured before building anything: exact round trip, 4 ms encode / 2 ms decode, 4.1 MB of cumulative
array. So M10K needed **no new coder, no format change and no container version bump** — the `.nvct`
v2 residual entropy model id already distinguishes the arms.

### The model

    z_ref -> (each latent channel as its own sample)
    conv 1->32 (3x3) -> ReLU -> conv 32->32 (3x3) -> ReLU
      + learned per-channel embedding
    -> conv 32->alphabet (1x1)  ->  logits at every position

**11,880–12,672 trainable parameters** (alphabet-dependent), 54–57 KB checkpoints. The 3×3 stack
gives each prediction a 5×5 receptive field on `z_ref` — the same locality M10J's `local_activity4`
bucketed by hand, except the network learns what to extract. Channel identity enters through an
embedding, matching the information M10H/M10J get from having one table per channel.

Objective is **pure rate**: `L = -log2 P(R | z_ref, channel)` on the existing M10H symbols. No
reconstruction loss, no λ — M10I already demonstrated what happens when a rate question is posed to a
distortion-dominated objective. Fitted on TRAIN, selected on VALIDATION, measured on TEST once.

**Causality:** the predictor's only input is `z_ref`, which the decoder rebuilds before touching the
residual payload. No autoregressive dependence on decoded symbols, deliberately — that would add a
sequential dependency to the coder and is not needed to answer this question.

### Offline gate (held-out validation, before any benchmark)

| bits | M10H marginal | M10J local_activity4 | M10K learned | K vs J |
|---|---|---|---|---|
| 5 | 3.21763 | 3.15588 | **3.11839** | **+1.19%** |
| 4 | 2.26931 | 2.21796 | **2.18596** | **+1.44%** |
| 3 | 1.41047 | 1.37238 | **1.34754** | **+1.81%** |

bits/symbol. Gate threshold was >0.5% on held-out data; passed at all three rate points.

### Deployed — DAVIS test, 719 frames, every byte counted

| arm | bits | P residual | motion | TOTAL | BPP | PSNR | MS-SSIM | vs M10H | vs M10J |
|---|---|---|---|---|---|---|---|---|---|
| marginal | 5 | 3,928,075 | 181,207 | 4,710,390 | 0.7997 | 29.271 | 0.9737 | | |
| local_activity4 | 5 | 3,848,919 | 181,207 | 4,631,234 | 0.7863 | 29.271 | 0.9737 | −1.75% | |
| **learned** | 5 | 3,813,506 | 181,207 | **4,595,821** | **0.7803** | 29.271 | 0.9737 | **−2.54%** | **−0.80%** |
| marginal | 4 | 2,725,811 | 184,254 | 3,354,910 | 0.5696 | 28.976 | 0.9676 | | |
| local_activity4 | 4 | 2,659,788 | 184,254 | 3,288,887 | 0.5584 | 28.976 | 0.9676 | −2.09% | |
| **learned** | 4 | 2,631,062 | 184,254 | **3,260,161** | **0.5535** | 28.976 | 0.9676 | **−3.00%** | **−0.93%** |
| marginal | 3 | 1,669,303 | 188,912 | 2,147,421 | 0.3646 | 27.946 | 0.9473 | | |
| local_activity4 | 3 | 1,617,619 | 188,912 | 2,095,737 | 0.3558 | 27.946 | 0.9473 | −2.66% | |
| **learned** | 3 | 1,589,353 | 188,912 | **2,067,471** | **0.3510** | 27.946 | 0.9473 | **−4.12%** | **−1.50%** |

**PSNR and MS-SSIM are identical to every digit across all three temporal arms**, and motion bytes
are identical. Verified per rate point: residual symbols identical, reconstruction identical, motion
identical, metrics identical, byte accounting closes.

### The coder realises 100% of it

| bits | arm | ideal P bits | vs M10H | P residual bytes | vs M10H | coder overhead | realised |
|---|---|---|---|---|---|---|---|
| 5 | learned | 30,505,210 | +2.92% | 3,813,506 | +2.92% | +0.01% | **100%** |
| 4 | learned | 21,045,611 | +3.48% | 2,631,062 | +3.48% | +0.01% | **100%** |
| 3 | learned | 12,711,762 | +4.79% | 1,589,353 | +4.79% | +0.02% | **100%** |

No discretization bottleneck: the float→integer conversion loses nothing measurable, and the
arithmetic coder stays within 0.02% of the ideal code length.

### BD-rate over three rate points

| comparison | PSNR | MS-SSIM |
|---|---|---|
| local_activity4 vs marginal | −2.11% | −2.10% |
| **learned vs marginal** | **−3.13%** | **−3.13%** |
| **learned vs local_activity4** | **−1.05%** | **−1.04%** |
| marginal (M10H) vs intra | −32.21% | −41.84% |
| local_activity4 (M10J) vs intra | −33.64% | −43.11% |
| **learned (M10K) vs intra** | **−34.34%** | **−43.74%** |

### Cost — and an implementation finding that changes the verdict

The first measurement said M10K cost 38.7–129.5 ms per P-frame against M10J's 1.66–1.90 ms, i.e.
23–68× slower. Profiling showed why, and it was not the model:

| | 5-bit | 4-bit | 3-bit |
|---|---|---|---|
| network forward pass | 0.28 ms | 0.26 ms | 0.29 ms |
| float→int table build | 72.88 ms | — | — |

**The 12k-parameter network costs 0.28 ms. Essentially 100% of the cost was an un-vectorised Python
loop distributing the rounding residual across 16,384 tables.** Vectorising it (verified
bit-identical on real data plus uniform / peaked / random adversarial cases, so every byte count
above is unchanged) gives:

| bits | params | network | tables | total ms/P | M10J ms/P |
|---|---|---|---|---|---|
| 5 | 12,672 | 0.28 | 19.34 | **19.62** | 1.90 |
| 4 | 12,144 | 0.26 | 8.50 | **8.76** | 1.82 |
| 3 | 11,880 | 0.29 | 4.48 | **4.76** | 1.66 |

So the honest cost is **2.9–10× M10J**, not 23–68×, and the remaining overhead is still table
construction rather than inference. It scales with alphabet size, which is why 3-bit is cheapest.

### Per-sequence (4-bit)

M10K beats M10J on 7 of 9: surf −3.01%, cat-girl −2.27%, bmx-bumps −2.02%, drone −1.89%,
car-turn −0.29%, cows −0.07%, drift-chicane −0.04%. It **loses on 2**: schoolgirls +1.33%,
gold-fish +0.43%.

Notably M10K reverses M10J's one regression: on bmx-bumps the hand-designed `local_activity4` was
*worse* than the marginal model (458,090 vs 450,356 bytes), while the learned model recovers to
448,848 — better than both. The losses are not obviously systematic at n=2.

### One deployment property worth knowing

The learned model is **coupled to the quantization calibration it was fitted under.** Scored against
symbols from a different grid it silently costs bits rather than failing — an early trial run with a
mismatched calibration made it look 14.8% *worse* than the marginal model. The deployment pins the
calibration, and a test records the coupling so it cannot be forgotten. M10J's tables have the same
dependency but degrade far more gently.

### Verdict: LEARNED ENTROPY SUCCESS

M10K beats M10J on held-out validation at all three rate points (+1.19 / +1.44 / +1.81%) and realises
that as actual bytes (−0.80 / −0.93 / −1.50% residual), for **−1.05% BD-rate**, with symbols, motion,
reconstruction and quality bit-identical.

Whether it is *worth deploying over M10J* is a separate judgement the numbers support either way:
−1.05% BD-rate for 2.9–10× the residual-coding time and a 12k-parameter model that must be shipped
and version-matched, against M10J's essentially free lookup. For an offline encode the learned model
is clearly better; for a real-time decoder the table-build cost still needs work before it is.

The cumulative picture is worth stating plainly: hand-designed conditioning got 2–3%, a learned model
of **12 thousand parameters** got another 1%, and M10I's **498 thousand-parameter** conditional
transform got −1.1% (the wrong way). Parameter count has not been the binding constraint at any point
in this thread; what is being optimised has.

### Next milestone

Two candidates, both pointed at by measurement rather than ambition:

1. **Make the gain cheaper.** The table build is 97–99% of M10K's cost and is a pure implementation
   problem — GPU-side construction, or quantising the predicted distributions to a modest codebook of
   shared tables, which would also shrink the 4.1 MB cumulative array.
2. **Then, and only then, more context.** The obvious untested source is an autoregressive dependence
   on already-decoded residual symbols, which the gate methodology can evaluate offline before any
   coder change. It would add a sequential dependency to the decoder, so it should be costed against
   (1) before being built.

Not recommended: a larger entropy network. 12k parameters already beat a hand-designed context, and
the evidence says the constraint is elsewhere.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU, deterministic cuDNN, seed 42. Frozen model: M10F λ=3e-4
seed 42 `best.pt`. Entropy models trained 40 epochs on 536 TRAIN P-frames per rate point, selected on
179 validation P-frames by validation NLL, best-validation weights restored before any measurement.
All commands require the project venv (`./.venv/Scripts/python.exe`).

---

## 2026-09-09 — Why NVC decode is slower than encode, and closing the gap where possible

**Source:** NVC's own decode is measurably slower than its encode - real 719-frame DAVIS
measurements (chroma-subsampling entry, below) put it at 36-43 ms/frame decode vs 14-20 ms/frame
encode, all software, all CPU. Investigated why, then investigated fixes for each cause found -
built and tested everything that does NOT need retraining; one real, no-retraining fix for the
network side does not currently exist on this machine, and is documented as such rather than
skipped silently.
**Tests:** 962 passing (full suite), zero regressions. 12 new tests.
**Scope:** lives on the unmerged `decode-speed` branch (`a050d41`) - see "Disposition" below.

### Two independent causes, both confirmed by direct measurement

**1. The decoder's `ConvTranspose2d` is 1.72x slower than `Conv2d` for the identical FLOP count.**
Confirmed with a controlled A/B: `ConvTranspose2d(64→128, 16×16→32×32)` timed against its exact
mathematical transpose, `Conv2d(128→64, 32×32→16×16)` - same weight shape, same MAC count by
construction, only the op direction differs. 2.21 ms vs 1.28 ms. Not folklore; measured on this
machine.

**2. Arithmetic decoding has to search for the symbol; encoding never does.** From the actual C
source (`range_coder.c`): `rc_encode` already knows which symbol to code, so it looks up
`cum[symbol]`/`cum[symbol+1]` directly - O(1). `rc_decode` doesn't know the symbol yet - it finds
it via `bisect_right_i64`, a binary search over the cumulative table (up to 8 comparisons at
8-bit) that encoding skips entirely.

### Fix attempted and shipped: table-based decode (entropy coder)

Standard production-range-coder technique: since `TOTAL_FREQUENCY` is fixed at exactly 65536 for
every table (`EmpiricalEntropyModel`'s own constructor invariant), a precomputed array mapping
every possible interval position directly to its symbol turns the search into one array read.

- New `rc_decode_lut` in `range_coder.c`, structurally identical to `rc_decode` except the bisect
  call is replaced by a lookup; `rc_decode` itself is untouched.
- `build_symbol_lut(cumulative)` in `range_coder.py`; `decode_symbols(..., lut=...)` is a new
  optional keyword - omitting it (the default) is byte-for-byte the same as before.
- `EmpiricalEntropyModel.symbol_lut`: lazily built, cached per model.
- Verified bit-exact against the existing bisect path across bit depths, table counts, and random
  data.

**Real, measured speedup - NOT wired in as the default, because it is not a blanket win:**

| bits | table width | speedup |
|---|---|---|
| 8 | 257 | **1.06x faster** |
| 6 | 65 | ~1.00x (a wash) |
| 4 | 17 | **0.81x - slower** |

At low bit depths the tiny cumulative table already fits in L1 cache, so the search is nearly
free; the 65536-entry LUT does not fit cache at any bit depth, so replacing a cheap cache-resident
search with a cache-unfriendly lookup can lose. This project's own default bit depths (4/6/8) span
exactly the range where the answer flips sign, so it ships as an explicit opt-in
(`decode_symbols(lut=...)`, `EmpiricalEntropyModel.symbol_lut`), matching how
`bits_per_channel`/`companding_gamma` were handled - not a change to the default decode path.

### Fixes investigated for the network side: none usable without retraining

**Architecture change that works, but needs retraining (out of scope here):** `Conv2d(k=1) +
PixelShuffle` in place of `ConvTranspose2d`, verified 1.61x faster end-to-end in a decoder-stack
timing test - at 4x fewer parameters (75K vs 297K), so it is cheaper because it is a smaller
function, not a free win. Needs real retraining and an RD-quality check before it could be trusted;
not attempted here since this pass was scoped to no-retraining fixes only.

**No-retraining options tried, in order of theoretical upside - none usable on this machine today:**

1. **`torch.compile`** - fails outright. Its Windows CPU (Inductor) backend emits MSVC-style
   compiler flags (`/I`, `/DLL`, `/MD`) unconditionally, regardless of which compiler binary `CXX`
   points at; this machine has MinGW g++, not real MSVC, so compilation fails immediately
   (`CppCompileError`). Confirmed by actually pointing `CXX` at g++ and reading the resulting
   command line - not a guess. Other registered backends (`tvm`, `openxla`) need large additional
   installs (`apache-tvm`, `torch_xla`) not reasonable to add for this check.
2. **ONNX Runtime** (the project's own OPTIMIZATION_ANALYSIS.md S5 lever) - exported cleanly and
   verified numerically identical (max abs diff `1.5e-7` vs PyTorch eager on the real
   `vimeo_qat_noise_best.pt` decoder). But measured **consistently slower** than PyTorch eager
   across all four ONNX Runtime graph-optimization levels and both single- and multi-threaded
   configurations (0.48x-0.96x - i.e. 4% to over 2x slower), not a tuning miss.
3. **Thread-count tuning** - PyTorch's own default (4 threads, this machine has 8 logical cores)
   is already at the empirical optimum; both fewer (1-2) and more (8) threads measured worse.

### Disposition: entropy-coder fix built and tested; network side stays open

The table-based decode is real, tested, working code - kept on the unmerged `decode-speed` branch
(`a050d41`) rather than merged to `master`, per instruction to document only this pass. The
network-side gap remains: the one real fix identified (`PixelShuffle`) needs retraining, and no
no-retraining alternative tested here actually helps on this machine. Revisit `torch.compile` if
Visual Studio Build Tools (real MSVC) are ever installed - the failure mode measured here is
specifically the missing compiler, not a deeper incompatibility.

---

## 2026-09-09 — Chroma subsampling investigated: no measured benefit (CLOSED)

**Source:** OPTIMIZATION_ANALYSIS.md Q4 and `codecs.py`'s own docstring both flag that H.264/H.265
subsample chroma (`yuv420p`) while NVC codes full-resolution RGB - a documented asymmetry counting
against NVC in every benchmark. Investigated as a candidate quick win before committing to any
retraining.
**Tests:** 964 passing (full suite), zero regressions. 46 new tests.
**Scope:** no `src/nvc/compression/` or `src/nvc/models/` change, no retraining, no `.nvc`/`.nvcs`
format change. Lives entirely on the unmerged `chroma-subsampling` branch (commit `a638e55`) - see
"Disposition" below for why the code isn't on `master`.

### What was built

The cheapest possible version, usable with the EXISTING trained model and calibration, no
retraining: `src/nvc/data/color.py` converts RGB → YCbCr, box-averages the two chroma planes 2×2,
bilinear-upsamples them back to full resolution, and converts back to RGB - still a full-resolution
`[3, H, W]` tensor, fed into the unmodified encoder. Wired as `NVCCodec(chroma_subsampled=True)`
and a `--codecs nvc-chroma420` option in `benchmark_rd.py` (opt-in, not in the default codec set).
Deliberately documented as a bounded, lesser version of the real thing: a true half-resolution
chroma path would need a new dual-resolution architecture and retraining.

### Result 1 - the cheap NVC version: no measurable win

DAVIS test, 719 frames, frame-weighted, `vimeo_qat_noise_best.pt`:

| bits | plain bpp | chroma420 bpp | Δ bitrate | PSNR | MS-SSIM |
|---|---|---|---|---|---|
| 8 | 1.8956 | 1.8952 | −0.020% | 28.705 → 28.690 dB | 0.9646 → 0.9644 |
| 6 | 1.3933 | 1.3929 | −0.027% | 28.626 → 28.611 dB | 0.9630 → 0.9628 |
| 4 | 0.8802 | 0.8798 | −0.044% | 27.381 → 27.371 dB | 0.9338 → 0.9336 |

Real chroma detail was genuinely removed (verified in `test_color.py`), but the entropy coder
essentially didn't reward it - bitrate barely moves and quality drops by about the same tiny
amount, the signature of pure information loss with no compensating compression benefit. Consistent
with the design caveat stated up front: this version doesn't reduce what the network actually
processes (still 3×H×W in, 3×H×W out), only whatever detail-removal the entropy coder happens to
exploit on its own.

### Result 2 - the real-codec reference check: also no measured win, and initially backwards

To calibrate what "the real thing" should be worth, H.264/H.265 were measured at `yuv420p` vs
`yuv444p` (same 719 frames, same CRF values). Comparing at matched CRF is not matched quality
though - CRF targets quality *within* one pixel format, and PSNR differed by up to 0.7 dB between
the two at "the same" CRF. Redone properly as a piecewise-linear matched-quality comparison (the
same method M10's own BD-rate analysis uses), interpolating each codec's 3-point RD curve:

| codec | yuv420p bitrate vs yuv444p, at matched RGB quality |
|---|---|
| H.264 | **+9.05%** (yuv420p costs *more*) |
| H.265 | **+5.32%** (yuv420p costs *more*) |

The opposite of the textbook "4:2:0 saves bits" result. Investigated rather than accepted at face
value: a targeted 170-frame Y-only-PSNR check (this project scores full RGB everywhere, including
NVC; most codec literature scores luma only) found the RGB-PSNR quality gap between yuv420p/yuv444p
at CRF28 is 0.359 dB, of which only 0.166 dB survives when scored on luma alone - so roughly half
of the apparent penalty is specifically a chroma-domain scoring effect, real but only a partial
explanation. Even Y-only PSNR still favours yuv444p slightly on this sample, so the honest
conclusion is narrower than "it's just the metric": on this test set, at these quality levels,
mature encoders are not clearly winning bits from chroma subsampling either way, whichever metric
is used.

### Result 3 - software-only speed (the fairness question from earlier this session)

All CPU, all software (`libx264`/`libx265`, no hardware encode/decode path):

| codec | encode/frame | decode/frame |
|---|---|---|
| NVC | 14-20 ms | **36-43 ms** |
| H.264 | 4.5-5.2 ms | 8.7-11.4 ms |
| H.265 | 11.8-18.5 ms | 8.8-10.8 ms |

NVC's decode is clearly the slowest of the three on pure CPU - the real, measured version of the
"H.265 gets a hardware decoder NVC doesn't have" concern, independent of the chroma question.

### What this does not establish, and the one fact that still stands

The reference number this investigation needed - "how much does chroma subsampling save in a
mature codec" - came back near-zero-to-negative on this test set and this project's own RGB-based
scoring convention, so no bounded retraining-upside estimate could honestly be given; inventing one
without that anchor would have been exactly the kind of unaccounted-for number this investigation
was trying to avoid. The one fact that isn't in question is arithmetic, not measurement: 4:2:0 has
exactly 50% the raw sample count of 4:4:4 (1.5×H×W vs 3×H×W) - a true ceiling on how much data
shrinks, but nothing measured here shows that translating into compressed bits saved for this
codec, on this data.

### Disposition: CLOSED, branch not merged

The infrastructure (`color.py`, the `nvc-chroma420` arm, 46 tests) is correct, tested, and
harmless - but the premise it exists to chase isn't supported by measurement, on either the cheap
version or the real-codec reference. Per the plan agreed before starting ("if need be revert back
to current"), the `chroma-subsampling` branch is left unmerged - nothing destroyed, fully
recoverable (`git log chroma-subsampling`), just not part of the main line. A full retrained
dual-resolution chroma architecture is not recommended on this evidence.

---

## 2026-09-09 — M10J: conditioning the entropy model works (CONDITIONAL ENTROPY SUCCESS)

**Source:** M10I conditioned the residual TRANSFORM and improved distortion rather than rate. M10J
targets rate directly, where the entropy model controls it, and changes nothing else.
**Tests:** 941 passing (full suite), zero regressions. 42 new tests.
**Scope:** no `src/nvc/` change, no container change, no training, no new weights, no checkpoint
selection, no λ sweep. `.nvc`, `.nvcs`, `.nvct` v1 and v2 all untouched; every M10A–M10I script
byte-identical.

### The one conceptual change

    M10H:  residual symbol            ->  per-CHANNEL frequency table       (64 tables)
    M10J:  residual symbol + context  ->  per-(CHANNEL, CONTEXT) table      (256 tables)

Same motion estimator, same motion quantization, same warp, same `z_ref`, same residual, same
residual quantization, same arithmetic coder, same container, same GOP. Only which probability table
codes each symbol differs.

### The gate: measure predictability before building anything

Run first, deliberately, so a null result would have cost one script rather than a milestone. Two
methodological points decided what "a gain" means:

**The baseline is H(R | channel), not H(R).** The deployed coder already has one table per channel,
so crediting the reference with that gain would manufacture a result out of nothing.

**Adding contexts always lowers a plug-in entropy estimate**, even for a random context. So every
scheme was measured against a RANDOM context of equal cardinality and re-scored on a held-out
validation split. Test frames were never touched.

| bits | H(R\|channel) held-out | best context | net held-out reduction | random control |
|---|---|---|---|---|
| 5 | 3.2170 | local_activity4 | **+1.94%** | −0.01% |
| 4 | 2.2685 | local_activity4 | **+2.28%** | −0.01% |
| 3 | 1.4108 | sign_x_magnitude4 | **+3.37%** | −0.01% |

The random controls sit at ~0.00%, so the estimator is unbiased at this sample size (536 train
P-frames, 8.8M symbols per rate point). The signal is real, and it grows as the rate falls.

### What was deployed

Two contexts survived to deployment, both cardinality 4, both deterministic and computable by the
decoder from `z_ref` alone:

    magnitude4        which per-channel |z_ref| quantile band the position falls in
    local_activity4   how much z_ref varies in a 3x3 neighbourhood, bucketed

Thresholds are per-channel quantiles fitted on TRAINING references only. Contexts with fewer than
1,000 training samples fall back deterministically to that channel's marginal table, so a rare
context never becomes an unstable tiny histogram. Tables use the project's existing Laplace
smoothing.

**No container change was needed.** `.nvct` v2 already stores an 8-byte residual entropy model id
and the decoder verifies it; a 256-table conditional model hashes differently from a 64-table
marginal one, so a marginal decoder handed a conditional stream fails loudly, and an M10H stream
stays decodable with its own declared model. Both are tested.

### The ablation is structural, not asserted

All three arms are coded in ONE closed-loop pass per sequence: motion, warp, `z_ref`, the residual
and its quantized symbols are computed once and shared, and each arm only re-codes those same
symbols with its own tables. The arms *cannot* differ in anything but their probability model,
because only the tables are computed more than once.

Verified at every rate point: **symbols identical, reconstruction identical, motion identical.**

### Results — DAVIS test, 719 frames, every byte counted

| arm | bits | I bytes | P residual | motion | TOTAL | BPP | PSNR | MS-SSIM | Δresid |
|---|---|---|---|---|---|---|---|---|---|
| intra | 5 | 5,611,995 | 0 | 0 | 5,628,186 | 0.9555 | 29.376 | 0.9684 | |
| marginal (M10H) | 5 | 584,917 | 3,928,402 | 181,199 | 4,710,709 | 0.7998 | 29.272 | 0.9737 | |
| magnitude4 | 5 | 584,917 | 3,877,428 | 181,199 | 4,659,735 | 0.7911 | 29.272 | 0.9737 | **−1.13%** |
| **local_activity4** | 5 | 584,917 | 3,849,496 | 181,199 | **4,631,803** | **0.7864** | 29.272 | 0.9737 | **−1.75%** |
| marginal (M10H) | 4 | 428,654 | 2,725,590 | 184,255 | 3,354,690 | 0.5696 | 28.976 | 0.9676 | |
| magnitude4 | 4 | 428,654 | 2,680,106 | 184,255 | 3,309,206 | 0.5618 | 28.976 | 0.9676 | **−1.44%** |
| **local_activity4** | 4 | 428,654 | 2,659,732 | 184,255 | **3,288,832** | **0.5584** | 28.976 | 0.9676 | **−2.09%** |
| marginal (M10H) | 3 | 273,015 | 1,668,788 | 188,891 | 2,146,885 | 0.3645 | 27.946 | 0.9473 | |
| magnitude4 | 3 | 273,015 | 1,627,357 | 188,891 | 2,105,454 | 0.3575 | 27.946 | 0.9473 | **−2.13%** |
| **local_activity4** | 3 | 273,015 | 1,616,961 | 188,891 | **2,095,058** | **0.3557** | 27.946 | 0.9473 | **−2.67%** |

**PSNR and MS-SSIM are identical to every reported digit**, which is exactly the predicted signature:
the reconstruction cannot change when only the probability model does. Motion bytes are identical.
Total BPP falls 1.68% / 1.97% / 2.41%.

### The coder is not the bottleneck

| bits | arm | ideal P bits | vs marginal | P residual bytes | vs marginal | coder overhead | realised |
|---|---|---|---|---|---|---|---|
| 5 | local_activity4 | 30,793,025 | +2.01% | 3,849,496 | +2.01% | +0.01% | **100%** |
| 4 | local_activity4 | 21,274,949 | +2.42% | 2,659,732 | +2.42% | +0.01% | **100%** |
| 3 | local_activity4 | 12,932,781 | +3.11% | 1,616,961 | +3.11% | +0.02% | **100%** |

"Ideal P bits" is the Shannon cost of the same coded symbols under each arm's own tables. The
arithmetic coder lands within **0.02%** of it and the byte reduction realises **100%** of the
modelling gain — so this is definitively not an entropy-coder bottleneck, and the deployed result
tracks the offline prediction (2.01 vs 1.94, 2.42 vs 2.28, 3.11 vs 2.71) closely.

### BD-rate over three rate points

| comparison | PSNR | MS-SSIM |
|---|---|---|
| magnitude4 vs marginal | −1.55% | −1.55% |
| **local_activity4 vs marginal** | **−2.11%** | **−2.10%** |
| marginal (M10H) vs intra | −32.22% | −41.86% |
| **local_activity4 vs intra** | **−33.65%** | **−43.13%** |

### Per-sequence (4-bit, best of the two contexts)

Every sequence improves. drift-chicane **−11.66%**, surf −7.52%, car-turn −5.11%, drone −2.40%,
gold-fish −1.45%, cows −1.37%, schoolgirls −1.16%, cat-girl −1.06%, bmx-bumps −0.93%.

Context choice is content-dependent: on bmx-bumps `magnitude4` beats `local_activity4`
(446,011 vs 457,957 bytes), the only sequence where the stronger context loses. bmx-bumps remains
the hardest sequence for every temporal method tried so far.

### Verdict: CONDITIONAL ENTROPY SUCCESS

Every success criterion is met: identical symbols, identical reconstruction, identical motion, lower
residual entropy AND lower residual bytes, lower total BPP, improvement at all three rate points,
train-only calibration, exact causal decoder symmetry.

The honest scale: this is a **~2% bitrate reduction**, free at inference (a bucket lookup per symbol)
and requiring no training, no new weights and no format change. Small, but clean, cheap, and it is
the rate reduction M10I set out to find and did not get.

Worth stating plainly against M10I: a learned 498k-parameter conditional transform trained for a
milestone moved rate **+1.1%** (the wrong way); a 256-entry lookup table conditioned on the same
reference moves it **−2.1%**. The information was there all along — M10I's objective simply had no
reason to spend capacity on rate.

### Next milestone — from the measured bottleneck

The gate answered its question, so a NEURAL conditional entropy model is now warranted and is the
natural next step: the discrete 4-bucket context captures 2-3% and is the crudest possible use of
`z_ref`, so a learned per-symbol probability model conditioned on the full reference has clear
headroom. Two things should go with it: context cardinality is currently untuned (4 buckets was the
first thing tried), and the fallback threshold is a fixed 1,000 samples.

Also worth noting for whoever picks this up: the gain grows monotonically as rate falls
(1.75 → 2.09 → 2.67%), so the low-rate operating points are where conditioning pays most.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU, deterministic cuDNN. Frozen model: M10F λ=3e-4 seed 42
`best.pt`. Tables fitted on 536 TRAIN P-frames per rate point; thresholds from the same references;
zero fallbacks triggered at deployment scale. All commands require the project venv
(`./.venv/Scripts/python.exe`).

---

## 2026-09-09 — M10I: conditioning improves distortion, not rate (MODEL CAPACITY BOTTLENECK)

**Source:** M10H left the motion-compensated residual coded against its MARGINAL statistics — one
per-channel grid and one frequency table for every residual, regardless of what the reference looked
like. The hypothesis was that conditioning on the warped reference would make residuals cheaper to
code.
**Tests:** 899 passing (full suite), zero regressions. 34 new tests.
**Scope:** no `src/nvc/` change. `.nvc`, `.nvcs`, M10G's `.nvct` v1 and M10H's v2 all untouched;
`train_autoencoder.py` and every M10A–M10H script byte-identical. λ frozen at 3.0e-4; no sweep.

### The mechanism (conditioning enters both transforms)

    analysis    w     = r     + f_a([r,     z_ref])
    synthesis   r_hat = w_hat + f_s([w_hat, z_ref])

`f_a` and `f_s` are 3-layer convolutional networks taking the concatenation of the residual with the
reference latent, so every output position can depend on the reference at that position. **`w`, not
`r`, is what gets quantized and entropy-coded.** Tests pin that the conditioning is load-bearing:
the same residual with a different reference produces a different coded tensor, gradients flow back
through the reference path, and the effect is spatially local (a bounded receptive field), not a
global summary.

**Trainable: 498,176 (the conditional codec only). Frozen: 593,411 (encoder + decoder at λ=3e-4),
plus the block-matching motion estimator.**

The last convolution of each branch is **zero-initialised**, so at step 0 `w = r` and `r_hat = w_hat`
— the codec is exactly M10H, bit for bit. A test asserts the two produce byte-identical streams at
initialisation. That is what makes this an ablation of the conditioning mechanism rather than a
comparison of two unrelated codecs.

### Training

`L = D + 3.0e-4·R`: D is pixel MSE after the frozen decoder, R is the project's differentiable
Laplace proxy on `w` with QAT noise at 4-bit and scale tracking. 4,323 causal (z_t, z_ref) training
pairs and 594 validation pairs, precomputed from the M10H closed loop over the DAVIS train/val
splits — an open-loop approximation, exactly on-policy at step 0 and drifting only as training moves
away from the baseline.

Checkpoint selected by the M10G convention: **epoch 28 by `val_loss`, validation only, before any
test evaluation**; final epoch 30, best-vs-final gap 0.20%. (An earlier run reported epoch 28 while
saving epoch 30's weights — a real defect, since the convention exists precisely to deploy the
selected checkpoint. Fixed by keeping the best state during training; the deployed weights are now
verified distinct from the final ones.)

### Three rate points, chosen from measurement

A bit-depth probe of the M10H MC path found the RD curve is **vertical above 5-bit**: 8→5 bit halves
the rate for 0.08 dB. 8/6/5 could not support a BD-rate integration. **5/4/3** spans 1.42 dB over a
2.1× rate range.

### Results — DAVIS test, 719 frames, every byte counted

| arm | bits | I bytes | P residual | motion | TOTAL | BPP | PSNR | MS-SSIM | resid bits/P | motion bits/P |
|---|---|---|---|---|---|---|---|---|---|---|
| intra | 5 | 5,611,995 | 0 | 0 | 5,628,186 | 0.9555 | 29.376 | 0.9684 | 0 | 0 |
| m10h | 5 | 584,917 | 3,933,408 | 181,150 | 4,715,666 | **0.8006** | 29.271 | 0.9737 | 48,862 | 2,250 |
| m10i | 5 | 584,917 | 3,983,630 | 180,914 | 4,765,652 | 0.8091 | **29.336** | **0.9743** | 49,486 | 2,247 |
| intra | 4 | 4,113,820 | 0 | 0 | 4,130,011 | 0.7012 | 28.560 | 0.9516 | 0 | 0 |
| m10h | 4 | 428,654 | 2,737,122 | 184,282 | 3,366,249 | **0.5715** | 28.975 | 0.9676 | 34,002 | 2,289 |
| m10i | 4 | 428,654 | 2,775,031 | 183,324 | 3,403,200 | 0.5778 | **29.051** | **0.9686** | 34,472 | 2,277 |
| intra | 3 | 2,622,383 | 0 | 0 | 2,638,574 | 0.4480 | 26.190 | 0.8952 | 0 | 0 |
| m10h | 3 | 273,015 | 1,695,452 | 188,952 | 2,173,610 | **0.3690** | 27.947 | 0.9473 | 21,062 | 2,347 |
| m10i | 3 | 273,015 | 1,714,465 | 188,149 | 2,191,820 | 0.3721 | **28.033** | **0.9488** | 21,298 | 2,337 |

### The ablation — the hypothesis failed

| bits | Δmotion | Δresidual | Δtotal | ΔBPP | ΔPSNR | ΔMS-SSIM |
|---|---|---|---|---|---|---|
| 5 | −236 B | **+50,222 B** | +49,986 B | +1.06% | +0.066 dB | +0.0006 |
| 4 | −958 B | **+37,909 B** | +36,951 B | +1.10% | +0.077 dB | +0.0009 |
| 3 | −803 B | **+19,013 B** | +18,210 B | +0.84% | +0.086 dB | +0.0016 |

The success signature was *motion unchanged, residual DOWN, quality held or improved*. **Residual
bits went UP at every rate point**, and down on only **1 of 9** sequences (cat-girl, −0.25%). The
conditioning hypothesis, as stated, fails — and the brief is explicit that this is decisive
regardless of overall results.

**On the flagged motion difference:** the evaluator marks the ablation confounded because Δmotion is
not exactly zero. It is 0.13–0.52% of motion bytes, while the residual effect is **24–213× larger**.
The cause is inherent rather than a bug: motion is estimated from the previous *reconstruction*, and
a different residual model produces a different reconstruction. A controlled test confirms motion is
byte-identical when the reconstructions match (the identity-initialised codec). The confound is real,
quantified, and far too small to affect the conclusion — but it is reported rather than absorbed.

### What training actually optimised

| | D | R (proxy) | λ·R | rate share of objective |
|---|---|---|---|---|
| epoch 1 | 1.2246e-03 | 0.5038 bpp | 1.511e-04 | 11.0% |
| epoch 28 (selected) | 1.1856e-03 | 0.5062 bpp | 1.519e-04 | 11.4% |

**Distortion improved 3.18%; the rate proxy moved +0.47%** — i.e. the model spent its entire capacity
on distortion and never reduced rate. At the frozen λ = 3e-4 the rate term is only ~11% of the
objective in this residual setting, so that is where the gradient pointed. This is not the deployed
entropy coder failing to capture a gain: **there was no rate gain in the differentiable proxy
either**, which rules out an entropy-coding bottleneck directly.

### RD: a modest improvement, in the opposite currency

| comparison | PSNR BD-rate | MS-SSIM BD-rate |
|---|---|---|
| m10h vs intra | −31.75% | −41.40% |
| **m10i vs intra** | **−32.70%** | **−42.14%** |
| **m10i vs m10h** | **−3.43%** | **−2.08%** |

So the conditional model does improve the RD operating point by ~3.4% BD-rate, consistently across
all three rate points and with quality up at every one — it simply gets there by buying quality with
bits rather than by making the residual cheaper. That is a real but small gain, and it is not the
gain the milestone set out to test.

### Per-sequence (4-bit)

PSNR improved on 7 of 9 sequences (largest: surf +0.17, drift-chicane +0.16, car-turn +0.15);
gold-fish (−0.13) and drone (−0.03) regressed slightly. Residual bytes rose almost everywhere,
most on gold-fish (+3.83%) and drift-chicane (+3.10%).

**bmx-bumps — the hard sequence — barely moved: 26.13 → 26.17 dB (+0.04) for +0.22% residual.** It
remains ~2.75 dB below intra at this rate point. Conditioning did not touch the high-motion deficit
that M10H left open.

### Causal invariants — all verified

Encoder/decoder reference symmetry bit-exact at every arm and rate point; decoder reproduces the
encoder's reconstruction exactly; byte accounting closes everywhere (motion + residual + overhead =
file size); motion estimated only against the reconstruction (verified by recording every call);
no lookahead; I-frames carry no motion; sequence boundaries reset; truncated residual payloads and
mismatched entropy models rejected.

### Verdict: MODEL CAPACITY BOTTLENECK

The conditioning mechanism is live, trainable and measurably load-bearing, and it produced a
consistent −3.4% BD-rate gain. But it did **not** reduce residual rate at any rate point, which is
the hypothesis this milestone existed to test.

Not **C (entropy model)**: the differentiable proxy showed no rate reduction either, so there was no
gain for the deployed coder to lose. Not **D (motion/reference)**: the reference is the same one that
gave M10H a −19.5% BD-rate over intra. Not **E**: there is a small, consistent RD improvement. Not
**A**: residual rate rose.

The measured limitation is that a 498k-parameter, 3-layer conditional transform trained open-loop
against an objective whose rate term is ~11% of the loss will improve distortion, because that is
where the gradient is — and that is what it did.

### Next milestone — from the measured bottleneck

Two candidates, both pointed at directly by the evidence, neither started:

1. **Make rate the thing being optimised.** The cleanest test of the original hypothesis is a
   conditional ENTROPY model — per-symbol context selected from `z_ref` — rather than a conditional
   transform. The project's arithmetic coder already accepts a per-symbol `table_index`, so context
   modelling needs no new coder and no new container, and it targets rate directly instead of
   competing with distortion inside a distortion-dominated loss.
2. **If the transform route is kept**, it needs closed-loop training and materially more capacity,
   and the residual objective's rate weight has to be revisited — which is a separate decision from
   the frozen intra λ and should not be conflated with it.

bmx-bumps remains the standing diagnostic: any future temporal work should be judged on whether it
moves that sequence.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU, deterministic cuDNN, seed 42. Frozen model: M10F λ=3e-4
seed 42 `best.pt`. 30 epochs, batch 8, model LR 1e-4, rate LR 1e-2. Calibration: TRAIN split only,
400 frames, fitted per arm and per rate point — the conditional arm's grid fitted to `w`, the tensor
it actually codes. All commands require the project venv (`./.venv/Scripts/python.exe`).

---

## 2026-09-08 — M10H: motion compensation works, and pays for itself (SUCCESS)

**Source:** M10G's temporal baseline cut bitrate 9.2% but lost 0.95 dB, almost entirely on three
high-motion sequences, and showed the loss was NOT error accumulation. The remaining suspect was that
`x_hat_{t-1}` is simply misaligned with `x_t` when content moves.
**Tests:** 865 passing (full suite), zero regressions. 46 new tests.
**Scope:** no `src/nvc/` change at all. `.nvc`, `.nvcs` and M10G's `.nvct` v1 all untouched;
`train_autoencoder.py` and every M10A–M10G script byte-identical. No lambda sweep; the intra
operating point stays frozen at λ = 3.0e-4.

### Design — and motion is paid for

    x_warp = Warp(x_hat_{t-1}, mv_t)      pixel-space, integer-pel block warp
    z_ref  = E(x_warp)
    dz     = z_t - z_ref                  coded with the RESIDUAL grid
    decoder: same warp from the DECODED motion, z_hat_t = z_ref + dz_hat

Warping happens in pixel space (where motion means something); the residual stays in the latent
domain, which M10G established as the only causal formulation available here (the decoder's Sigmoid
cannot emit signed pixel residuals).

**Every motion vector is quantized, entropy-coded, written into the stream and counted in the
reported bitrate.** No ground-truth flow, no encoder-only side information. The encoder warps with
the *decoded* motion, so the decoder's reference is bit-identical by construction. `.nvct` v2 carries
explicit motion and residual lengths per frame, which is what makes motion bytes attributable rather
than hidden inside the residual.

| | |
|---|---|
| estimator | full-search block matching, SAD |
| block size | 16×16 px — at 256×256 that is exactly one vector per latent position |
| precision | **integer pixel**, no interpolation, so the warp is bit-exact |
| range | ±16 px per component |
| alphabet | 6 bits (33 values in a 64-symbol table) |
| boundary | replicate |
| entropy coding | project arithmetic coder, 2 tables (dy, dx) |
| tie-break | min SAD, then min \|dy\|+\|dx\|, then min dy, then min dx — deterministic, biased to zero |

### Results — DAVIS test, 719 frames, every byte counted

| arm | bits | I bytes | P residual | motion | overhead | TOTAL | BPP | PSNR | MS-SSIM |
|---|---|---|---|---|---|---|---|---|---|
| intra | 8 | 10,061,586 | 0 | 0 | 16,191 | 10,077,777 | 1.7110 | 29.630 | 0.9739 |
| prev | 8 | 1,049,055 | 8,218,907 | 0 | 16,191 | 9,284,153 | 1.5762 | 28.679 | 0.9726 |
| **mc** | 8 | 1,049,055 | 7,842,324 | **179,509** | 16,191 | **9,087,079** | **1.5428** | **29.365** | **0.9759** |
| intra | 4 | 4,113,820 | 0 | 0 | 16,191 | 4,130,011 | 0.7012 | 28.560 | 0.9516 |
| prev | 4 | 428,654 | 3,111,332 | 0 | 16,191 | 3,556,177 | 0.6038 | 27.820 | 0.9523 |
| **mc** | 4 | 428,654 | 2,737,258 | **184,219** | 16,191 | **3,366,322** | **0.5715** | **28.976** | **0.9676** |

Byte accounting closes exactly for every rate-accounted arm (motion + residual + overhead = file
size on disk), and encoder/decoder reference symmetry is bit-exact everywhere.

### Motion pays for itself, roughly 2:1

| | motion bytes | share of stream | per P-frame | residual saved vs prev | **net** |
|---|---|---|---|---|---|
| 8-bit | 179,509 | 1.98% | 279 B | +376,583 B | **+197,074 B** |
| 4-bit | 184,219 | 5.47% | 286 B | +374,074 B | **+189,855 B** |

Motion costs about 2% of the stream and buys back more than twice that in residual. This is the
question M10H existed to answer, and the answer is unambiguous.

### Motion compensation improves BOTH bitrate and quality over M10G

At 8-bit, against M10G's previous-frame baseline: **BPP 1.5762 → 1.5428 (−2.1%)** *and* **PSNR
28.679 → 29.365 (+0.686 dB)** *and* MS-SSIM 0.9726 → 0.9759. Not a trade — strictly better on all
three.

Against the intra baseline at 8-bit: −9.8% bitrate for −0.27 dB PSNR (M10G's prev-frame arm was
−0.95 dB), and **MS-SSIM is now HIGHER than intra** (0.9759 vs 0.9739).

BD-rate over the two rate points, versus intra:

| arm | PSNR BD-rate | MS-SSIM BD-rate |
|---|---|---|
| prev (M10G) | **+100.15%** | −9.84% |
| **mc (M10H)** | **−19.49%** | **−44.86%** |

`mc vs prev` BD-rate reports `n/a`, and that is the strongest possible outcome rather than a gap:
the two curves have **no overlapping quality range**. mc spans 28.976–29.365 dB, prev spans
27.820–28.679 dB, so mc is better than prev at every measured point while also being cheaper. There
is nothing to integrate over because one curve dominates the other outright. Concretely, **mc at
4-bit (0.5715 BPP, 28.976 dB) beats prev at 8-bit (1.5762 BPP, 28.679 dB) — higher quality at a
third of the bitrate.**

Two rate points only, so the BD integration is directional rather than precise.

### The high-motion failure is substantially repaired

M10G's three failures, prev → mc at 8-bit:

| sequence | prev PSNR | mc PSNR | change | mc BPP vs prev |
|---|---|---|---|---|
| bmx-bumps | 23.83 | 26.30 | **+2.48 dB** | −4.16% |
| drone | 28.84 | 30.99 | **+2.15 dB** | −4.93% |
| cat-girl | 28.17 | 29.33 | **+1.16 dB** | −6.72% |

Seven of nine sequences improve against prev-frame; the two that do not (drift-chicane, gold-fish)
are already at parity on quality and simply pay ~6% more bytes for motion they do not need, which is
the expected cost of an unconditional per-block motion field on static content.

bmx-bumps is repaired but not fixed: at 26.30 dB it is still 3.8 dB below intra's 30.09. That
residual gap is the next bottleneck, and the oracle says where it is *not*.

### Oracle diagnostic — this is NOT a motion-estimation bottleneck

| | prev | mc (coded) | oracle (dense flow, NOT transmitted) |
|---|---|---|---|
| 8-bit PSNR | 28.679 | **29.365** | 29.110 |
| 4-bit PSNR | 27.820 | **28.976** | 28.758 |

The oracle warps with dense Farneback flow that is never coded, so its bitrate is not a compression
result and it is excluded from every ranking (`is_rate_accounted("oracle")` is False, and an oracle
stream refuses to decode). Its purpose is purely to bound what better motion could buy.

**It does not beat block matching.** Dense flow produces a cheaper residual but slightly lower PSNR,
so a free, dense, sub-pixel motion field would not recover the remaining gap to intra. Integer-pel
16×16 block matching is already capturing essentially all the available motion gain at this
resolution. The remaining deficit therefore lives in the **residual representation**, not in motion
estimation or motion coding.

### Determinism is a correctness requirement here, not a nicety

Motion compensation puts the NETWORK in the reference path — both sides compute
`z_ref = E(Warp(x_hat_{t-1}))`, and `x_hat_{t-1}` itself came from `model.decode(...)`. Measured on
this machine: `encode()` is bit-exact on GPU by default but **`decode()` is not** (its transposed
convolutions select nondeterministic algorithms), so encoder and decoder drifted apart along the
P-chain and symmetry failed. The codec now forces `cudnn.deterministic=True, benchmark=False` for the
duration of every encode and decode, after which symmetry is exact. M10G's default path never needed
this because its reference was a pure dequantize with no network in it.

### Verdict: MOTION COMPENSATION SUCCESS

Motion compensation improves the RD operating point (−19.49% BD-rate vs intra on PSNR, −44.86% on
MS-SSIM, and outright dominance over M10G's temporal baseline), pays for its own bitrate roughly
2:1, and materially reduces the high-motion penalty that M10G identified.

### Next milestone — chosen from the measured bottleneck

**The residual codec, not motion.** The oracle rules out motion estimation and representation as the
limiting factor, and motion already costs only 2% of the stream. What remains is that a
motion-compensated latent residual is still coded by a grid and entropy model fitted to its marginal
statistics, with no conditioning on the reference. bmx-bumps' remaining 3.8 dB deficit is the
concrete target.

That points at a **learned temporal residual model** — conditioning the residual's coding on the
warped reference — rather than at optical flow, sub-pixel motion, variable block sizes, or GOP
tuning. Adding a second and third rate point would also turn the directional BD-rate into a real one.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU, deterministic cuDNN. Model: M10F λ=3e-4 seed 42 `best.pt`
(best-validation, per the M10G Part A convention). Calibration: TRAIN split only, 400 frames, 0.1/99.9
per-channel percentiles, fitted separately per arm and per rate point; all three entropy model ids
recorded in every stream header. All commands require the project venv (`./.venv/Scripts/python.exe`).

---

## 2026-09-08 — M10G: checkpoint convention fixed, and a causal temporal baseline that works

**Source:** two things M10F left open — the final-snapshot evaluation convention that corrupted its
own control, and the fact that NVC is still intra-only.
**Tests:** 819 passing (full suite), zero regressions. 54 new tests.
**Scope:** no `src/nvc/` change at all. `.nvc` and `.nvcs` remain bit-exact; `train_autoencoder.py`
and every M10A–M10F script left byte-identical. Historical M10A–M10F results are untouched and are
NOT restated.

---

### Part A — the checkpoint evaluation convention (declared for all future experiments)

    PRIMARY   : best VALIDATION checkpoint under the experiment's declared training objective
    SECONDARY : final training checkpoint, retained for convergence diagnostics

Three properties make it safe, and all three are enforced in code rather than asserted in prose:

1. **Deterministic** — minimum objective wins, exact ties go to the earliest epoch.
2. **Declared before training** — the objective key is an input, not a post-hoc choice.
3. **Structurally unable to see the test set** — `select_checkpoint` takes only validation history,
   and every record is screened against a forbidden-key list. A test metric appearing in a history
   file raises `TestMetricLeakError` rather than quietly influencing the choice. Selecting on
   deployed BD-rate or test PSNR would be selection on the evaluation set.

For rate-aware training the objective is `val_loss` = D + λ·R — the quantity actually optimised, not
proxy bitrate, which would prefer a model that discards quality to save rate. Stale records from a
different objective (carried in by `--resume-model-only`) are excluded, the same rule M9C.1
established.

### Part B — how often the old convention actually bit (non-destructive)

Re-read the existing M10E and M10F artifacts. No retraining, no re-benchmarking, nothing overwritten:

| experiment | runs | degraded final snapshot (>2%) | worst gap |
|---|---|---|---|
| M10F | 10 | 2 — `CTRL@s42` (3.04%), `UPPER_ANCHOR@s43` (2.64%) | 3.04% |
| M10E | 10 | 1 — `UPPER_REFERENCE@s43` (4.08%) | 4.08% |
| **total** | **20** | **3 (15%)** | |

`best.pt` exists for all 20 runs, so the new convention is fully implementable — it needs only
validation history, which every run already records. **Historical results remain valid under the
final-snapshot convention they declared before running.**

---

### Part C — temporal coding: three design decisions, documented

**1. Residuals live in the LATENT domain, not pixel space.** `Decoder` ends in `nn.Sigmoid()`, so a
reconstruction is bounded to [0,1] while a pixel residual lies in [−1,+1] — the decoder cannot emit
one. `Encoder` ends on a bare `Conv2d` whose latent is explicitly "an unconstrained real-valued
tensor", so latent differences are naturally signed and the existing percentile calibration, uniform
quantizer and arithmetic coder handle them unchanged. No new activation, no new head, no training.

    I-frame:  z_t = E(x_t)        -> INTRA grid
    P-frame:  dz  = z_t - z_ref   -> RESIDUAL grid

Two grids, separately calibrated: intra latents and latent residuals have different distributions,
and one shared grid would misallocate levels for both.

**2. The reference is a decoded quantity.** `z_ref` is the previously *decoded* latent, never the
original frame's. The encoder runs closed-loop — it dequantizes its own symbols and predicts from
that — so encoder and decoder hold bit-identical reference state by construction. A `reencode` mode
(`z_ref = E(D(ẑ))`) is also implemented; both are causal and symmetric.

**3. A separate prototype container, `.nvct`.** `.nvcs` v1 carries exactly one quantization block for
the whole file and has no per-frame type field, so it cannot express a stream needing two grids and
I/P marking. Extending it in place would change shipped format semantics. `.nvct` deliberately
follows `.nvcs`'s conventions — same magic+version prologue, same length-prefixed records, same
truncation and trailing-data strictness — so it continues that direction rather than competing with
it. 44-byte header, GOP=10, every sequence starts with an I-frame.

### Causal invariants — verified, not assumed

| invariant | status |
|---|---|
| decoder reaches exactly the encoder's reference latents | **exact equality** |
| encoder's closed-loop reconstruction == decoder's output | **exact equality** |
| decoding is deterministic (same latents twice) | **exact equality** |
| same input encodes to identical bytes | **exact equality** |
| every sequence starts with an I-frame; no reference crosses a boundary | verified |
| P-frame without a reference | rejected (`CausalityViolationError`) |
| truncated header / truncated payload / trailing data | rejected |
| unknown frame-type byte, bad magic, unknown version | rejected |
| mismatched entropy model | rejected |
| decoder sees only the bitstream — no original or future frames | verified |

### Benchmark — intra-only vs temporal, DAVIS test, 719 frames, all bytes counted

Both arms use the same frozen model (λ = 3.0e-4, `best.pt` per Part A), the same intra grid, and the
same container. Arm A is the identical coder at GOP=1, so the two differ *only* in whether P-frames
exist and container overhead is accounted identically on both sides.

| | A intra-only | B temporal | change |
|---|---|---|---|
| I-frames / P-frames | 719 / 0 | 75 / 644 | |
| I-frame payload | 10,061,586 B | 1,049,055 B | −89.57% |
| P-frame payload | 0 | 8,085,297 B | |
| container overhead | 13,207 B | 13,207 B | +0.00% |
| **TOTAL container bytes** | **10,074,793** | **9,147,559** | **−9.20%** |
| **stream BPP** | **1.7105** | **1.5531** | **−9.20%** |
| compression ratio | 14.03 | 15.45 | +10.14% |
| mean PSNR | 29.630 dB | 28.676 dB | **−0.953 dB** |
| mean MS-SSIM | 0.9739 | 0.9726 | −0.0013 |

Per-frame: I = 13,987 B, P = 12,555 B — P-frames are ~10% cheaper than I-frames.

**This is one operating point per arm, so it is NOT a BD-rate result and no RD superiority is
claimed.** Rate falls 9.2% and PSNR falls 0.95 dB; whether that trade is favourable needs multiple
rate points, which is future work.

### The result is strongly motion-dependent

| sequence | BPP change | PSNR change | helps? |
|---|---|---|---|
| schoolgirls | −29.22% | −0.04 dB | yes |
| drift-chicane | −28.51% | 0.00 dB | yes |
| gold-fish | −20.47% | **+2.96 dB** | yes |
| surf | −12.53% | −0.19 dB | yes |
| car-turn | −8.92% | −0.06 dB | yes |
| cows | −7.48% | 0.00 dB | yes |
| drone | +2.15% | −3.48 dB | no |
| bmx-bumps | +2.90% | **−6.33 dB** | no |
| cat-girl | +6.51% | −1.44 dB | no |

Six of nine sequences improve, several dramatically and at no quality cost. The three failures are
the high-motion ones, and they fail badly. **Frame differencing has no way to model motion**, so when
content translates, the "residual" is not small — it is two misaligned copies of the scene, which is
more expensive than coding the frame outright. The aggregate −0.95 dB is almost entirely those three
sequences.

### The drift diagnostic: no error accumulation

| GOP position | 0 (I) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean PSNR dB | 29.695 | 29.300 | 29.042 | 29.013 | 29.112 | 29.056 | 29.120 | 29.126 | 29.215 | 29.230 |

PSNR steps down **0.40 dB once** at the first P-frame and is then flat — it actually recovers
slightly (−0.07 dB across the entire chain, against 0.29 dB of wobble). **Error is not compounding.**
That is the closed loop working as designed: the encoder forms every residual against the *true*
current latent, so each frame re-targets `z_t` and the previous frame's error is corrected rather
than carried.

This matters for what to do next, because the two failure modes need opposite fixes. The analysis
classifies the shape from the data rather than assuming one:

- **Accumulation** (a falling chain) would be fixed by a shorter GOP or a refresh mechanism.
- **A fixed per-P-frame penalty** (one step, then flat) is what we actually have, and **a shorter GOP
  would not help it** — the cost is the residual representation itself.

### Verdict

The temporal pipeline is technically sound: causal, symmetric, deterministic, strict about malformed
streams, and honest about bytes. It delivers a real **−9.2% bitrate** reduction with all I-frame
overhead included, at −0.95 dB PSNR concentrated in three high-motion sequences.

That is exactly the baseline M10G was meant to produce — and it identifies the next lever precisely.
The gap is **motion**, not GOP length and not entropy coding.

### Next milestone (proposed, not started)

**Motion compensation, warping the reference before differencing.** The evidence points at it
directly: frame differencing already wins on low-motion content, and its only failures are the
sequences where content moves. Everything else about the pipeline is measured and passing, so a
learned temporal model can be evaluated against a baseline that is known to be correct.

A second rate point per arm would also let this become a real BD-rate comparison instead of a
single-point one.

### Reproducibility

torch 2.13.0+cu130, RTX 5060 Laptop GPU. Model: M10F λ=3e-4 seed 42 `best.pt`. Calibration: TRAIN
split only, 400 intra frames + 360 residual frames, 0.1/99.9 per-channel percentiles, both entropy
model ids recorded in every stream header. All commands require the project venv
(`./.venv/Scripts/python.exe`).

---

## 2026-09-08 — M10F: the lower λ boundary is closed — λ = 3e-4 (NO USEFUL REFINEMENT)

**Source:** M10E left the converged optimum on the lower boundary for the second milestone running
— its best arm (3e-4) was the smallest λ it tested. M10F asks one question: **does the RD basin
keep improving below λ = 3e-4?** It does not.
**Tests:** 765 passing (full suite), zero regressions. 27 new tests.
**Scope:** no architecture, quantizer, entropy-coder or `.nvc` change; no `src/nvc/` change at all;
`train_autoencoder.py`, `m10a_pilot.py`, `m10c_convergence.py`, `m10d_lambda_refinement.py` and
`m10e_lambda_lock.py` all left byte-identical.

### Design — a boundary-closing experiment, not another sweep

Five λ × two seeds (42, 43) = **10 runs**, 18,120 steps each, all from M8-QAT `90d51157…`.
Only λ and seed vary.

| arm | λ | role |
|---|---|---|
| CTRL | 0 | matched control |
| VERY_LOW | 1.0e-4 | new territory |
| LOW | 2.0e-4 | new territory |
| **BRIDGE** | 3.0e-4 | M10E's best, **repeated on purpose** |
| UPPER_ANCHOR | 4.5e-4 | M10E's working-point candidate |

24 fail-closed preflight checks passed, including two M10E could not make: the **manifest is
hashed** (`b72b5317…`), so "identical split" is verified rather than asserted; and the calibration
and benchmark constants are **read off M10E's own evaluator** and compared, because the bridge arm
is meaningless if the two experiments evaluate differently.

All 30 calibrations passed the 2% guard at **0.118–0.195%**. All 10 benchmarks completed with zero
round-trip failures.

### The methodology problem, stated first because it shapes everything below

**The pre-registered primary metric was compromised by one run — and that run was a control.**

Two of ten runs exceeded the 2% best-vs-final gap: `UPPER_ANCHOR@s43` (2.64%) and, critically,
`CTRL@s42` (**3.04%**). CTRL@s42's final epoch 70 dipped ~0.17 dB below its epoch-68 best, and the
fixed final-snapshot convention locked that in. Against M10E's control, which is configuration- and
seed-identical:

| bits | seed | M10E CTRL PSNR | M10F CTRL PSNR | Δ |
|---|---|---|---|---|
| 8 | 42 | 29.998 | 29.834 | **−0.164** |
| 8 | 43 | 30.053 | 30.051 | −0.002 |

Seed 43's control reproduces to **0.002 dB**. Seed 42's is 0.164 dB worse. Because BD-rate is paired
per seed, a degraded control inflates *every* seed-42 candidate, and it widened the
control-to-control gap from M10E's 0.054 dB to 0.216 dB — driving the measured noise floor from
**1.62 to 4.86 points**.

With a 4.86-point floor every pairwise λ comparison is "tied", and the automatic boundary verdict
came back `D_FLAT_WITHIN_NOISE` (spread 4.18 ≤ floor 4.86). **That verdict is an artifact of the
inflated floor, not a finding.** The convention was not changed after the fact — it is the same one
M10D and M10E used, and the final-snapshot results are retained as primary throughout.

### The boundary question, answered by an analysis the contamination cannot touch

Within a seed the control is *shared by every arm*, so scoring each λ against **BRIDGE (3e-4) of its
own seed** removes the control from the comparison entirely:

| λ | vs 3e-4, seed 42 | vs 3e-4, seed 43 | mean | seeds agree? |
|---|---|---|---|---|
| **1.0e-4** | **+5.59%** | **+5.70%** | **+5.65%** | yes — to 0.11 points |
| 2.0e-4 | −0.86% | +3.92% | +1.53% | no, sign differs |
| 4.5e-4 | +0.70% | +13.71% (contaminated) | — | s42 only: +0.70% |

And a floor that does not involve the control at all — the BRIDGE arm against M10E's
configuration-identical 3e-4 run at the same seed — is **0.10–1.79 points**, consistent with M10D
(1.41) and M10E (1.62). That is the trustworthy floor; 4.86 is the contaminated one.

Against ~1.8 points:

- **λ = 1e-4 is worse than 3e-4 by 5.65 points — 3.1× the clean floor, and the two seeds agree to
  0.11 points.** This is the most reproducible single number in the experiment.
- λ = 2e-4 (+1.53) and λ = 4.5e-4 (+0.70 on its clean seed) are **tied** with 3e-4.

**The basin is bracketed below.** Combined with M10E (6e-4 and above are worse), the useful region is
roughly **2e-4 – 4.5e-4**, flat inside itself, with clear degradation on both flanks.

### Deployed `.nvc` (DAVIS test, 719 frames, fresh per-run calibration)

| λ | 8-bit BPP / PSNR | 6-bit | 4-bit | BD PSNR | BD MS-SSIM |
|---|---|---|---|---|---|
| CTRL | 1.8447 / 29.943 | 1.3424 / 29.804 | 0.8314 / 27.978 | — | — |
| 1.0e-4 | 1.7988 / 30.046 | 1.2964 / 29.951 | 0.7873 / 28.569 | −13.94% | −13.90% |
| 2.0e-4 | 1.7797 / 30.081 | 1.2770 / 29.963 | 0.7688 / 28.655 | −16.72% | −17.89% |
| **3.0e-4** | 1.7688 / 30.096 | 1.2663 / 29.980 | 0.7592 / 28.647 | **−18.12%** | −19.27% |
| 4.5e-4 | 1.7593 / 29.885 | 1.2568 / 29.785 | 0.7501 / 28.497 | −14.11% | **−19.39%** |

Every rate-aware λ beats the control at every depth. The CTRL and 4.5e-4 PSNR figures are depressed
by the two contaminated runs.

### Bridge: the pipeline itself is stable

λ = 3e-4, M10E vs M10F — independently trained, independently calibrated, one milestone apart:

| seed | ΔBPP | ΔPSNR |
|---|---|---|
| 42 | −0.01…−0.02% | +0.006…+0.000 dB |
| 43 | −0.01…−0.03% | −0.027…−0.034 dB |

Bitrate reproduces to **0.03%** and PSNR to **0.034 dB**. The 0.216 dB control gap is therefore not
general run-to-run noise — it is one bad final epoch, and the rest of the pipeline is far tighter
than that.

### Proxy — four questions, kept apart

1. **Lower λ → lower proxy R?** No, and correctly so: proxy R rises as λ falls (0.6479 → 0.6950 from
   4.5e-4 to 1e-4), because weaker rate pressure means a more expensive latent.
2. **Does proxy R order actual bitrate?** **Yes, at all three depths** — cross-λ ordering is
   perfectly monotonic. The `rank_agreement=False` flag comes from 8 inversions that are *all*
   between the two seeds of the same λ, magnitudes 0.0011–0.0417%, i.e. 3–100× below the
   control-to-control BPP noise. Noise-level ties, not ordering failures.
3. **Does the proxy-minimising λ minimise PSNR BD-rate?** **No** — lowest proxy R is 4.5e-4, best
   PSNR BD-rate is 3e-4. Reproduces M10D and M10E.
4. **Does it minimise MS-SSIM BD-rate?** Yes here (both 4.5e-4) — but 4.5e-4 and 3e-4 differ by only
   0.12 MS-SSIM points, so this is a coin-flip, not corroboration.

### Verdict: NO USEFUL REFINEMENT

Nothing below 3e-4 improves on M10E's operating region, and 1e-4 is measurably worse. M10E's region
stands; M10F's contribution is closing the flank that made it provisional.

**This is not classified METHODOLOGICAL FAILURE**, despite the contaminated control, because
fairness, calibration provenance, checkpoint identity and benchmark integrity all verified clean;
the contamination's cause, direction and magnitude are all identified; and the central question was
answered by a control-independent analysis that the contamination cannot affect, with the two seeds
agreeing to 0.11 points.

### Operating point: λ = 3.0e-4

Chosen on deployed RD behaviour, reproducibility and noise — not on being the smallest λ, the
highest PSNR, or the lowest proxy R:

- **Interior to a now-bracketed basin**: 1e-4 is worse below (M10F), 6e-4 worse above (M10E). Unlike
  M10E's recommendation, this is no longer a boundary point.
- **Best PSNR BD-rate in both experiments** that measured it (M10E −16.78%, M10F −18.12%).
- **The most stable arm in the experiment**: zero best-vs-final gap on *both* seeds, the only arm
  besides LOW@s42 to achieve that, and the smallest MS-SSIM seed spread (0.14).
- **Replicated across independent experiments** to 0.03% bitrate and 0.034 dB.

It is not measurably better than 2e-4 or 4.5e-4 — the basin is genuinely flat between them — so this
is a defensible choice within a flat region, not a measured optimum. That distinction should survive
into any downstream work.

### Methodology finding for future milestones

Evaluating the **final** snapshot is now known to have bitten twice in two experiments (M10E: 1 of
10 runs; M10F: 2 of 10, one a control). When it hits a control it corrupts the primary metric for
that whole seed and inflates the noise floor ~3×. Future experiments should either evaluate
`best.pt`, or average the last few epochs, **applied uniformly and decided before results are seen**.
Changing it retroactively here was deliberately avoided. Recording `best_vs_final` per run at
training time (new in M10F) is what made the problem visible immediately rather than by hand.

### Reproducibility

seeds 42/43, torch 2.13.0+cu130, RTX 5060 Laptop GPU, ~11.3 min/run, ~1.9 h for the grid.
Per-snapshot SHA256 in `snapshots.csv`; snapshots retained at 604/1,812/4,832/10,268/18,120 for all
10 runs. Best-checkpoint selection used only current-objective history in every run. All commands
require the project venv (`./.venv/Scripts/python.exe`).

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
