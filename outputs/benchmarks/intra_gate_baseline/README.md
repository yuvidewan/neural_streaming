# Intra-only gate harness - verification against the committed denominator

`scripts/benchmark_intra_gate.py --verify`, run 2026-09-27 on the deployed M10F
checkpoint. **All three rate points reproduce `outputs/benchmarks/parity_intra/`
exactly**, through a different code path:

| bits | bytes (this run) | bytes (recorded) | ΔPSNR | ΔMS-SSIM | reproduces |
|---|---:|---:|---:|---:|:---:|
| 5 | 5,628,186 | 5,628,186 | 0.00e+00 | 0.00e+00 | yes |
| 4 | 4,130,011 | 4,130,011 | 0.00e+00 | 0.00e+00 | yes |
| 3 | 2,638,574 | 2,638,574 | 0.00e+00 | 0.00e+00 | yes |

BD-rate against x264 all-intra: **+176.2%** (primary convention, PSNR) - the same
figure `parity_intra` reported. Decode bit-exact at every point, `valid: true`.

```
./.venv/Scripts/python.exe scripts/benchmark_intra_gate.py --verify \
    --output-dir outputs/benchmarks/intra_gate_baseline \
    --output-name intra_gate_baseline.json
```

## Why this harness exists

`benchmark_parity.py --gop 1` produced the denominator, but it gets there through
`m21.prepare_rate_point`, which rebuilds the whole deployed stack: the M11-G16
context model, the M10K learned entropy model, the K=512 codebooks, M14's motion
table, and a provenance check on each. All of them are fitted to a **64-channel**
latent. The Stage 1 transform's latent has 192 channels, and
`ChannelContextEntropyModel`'s `nn.Embedding(latent_channels, hidden)` alone makes
the trained G16 checkpoint unloadable against it.

Read literally that says "retrain the entropy stack before you can measure
anything" - weeks of work, on the entropy side, which is not where the parity gap is.

**It isn't true, because an I-frame touches none of it.** In
`m21.encode_sequence_refined` the P-branch uses the context model, the codebooks
and motion; the I-branch uses `intra_params` and `intra_entropy_model` and nothing
else. At GOP 1 there are no P-frames - which is why `nvc_deployed` and `nvc_m22`
came out byte-identical in the `--gop 1` run, M22's residual re-centering being
inert without residuals.

So this harness needs one thing: `calibrate_grids`, which runs the autoencoder over
TRAIN frames and fits the grids. It is architecture-agnostic. Measuring a trained
Stage 1 checkpoint is one command and one calibration.

## What makes the numbers comparable

- **The reference curve is read, not re-encoded.** The x264 all-intra curve comes
  straight out of the committed `parity_intra.json`, so the reference side is
  bit-for-bit the curve `+176.2%` came from and FFmpeg never runs.
- **One scorer.** `score_frames`, `aggregate_quality`, `summarize_point`, `curve`
  and `bd_rate` are imported from `benchmark_parity` rather than reimplemented.
- **The coding path is the promoted package** (`nvc.video.container`,
  `nvc.compression.codec`), not the research scripts - which is what makes the
  byte-for-byte agreement above a real cross-check rather than a tautology.

## Two things to know before citing a number from here

- **Calibration runs at GOP 10, coding at GOP 1.** `calibrate_grids` fits the
  residual grid and motion table on P-frames and raises on an empty collection, so
  it cannot run at GOP 1. The intra half it produces - the only half used for
  coding - is fitted on I-frame latents and does not depend on the GOP.
- **Three header fields are filled differently from the deployed path.** `.nvct` v2
  records `residual_entropy_model_id`, `motion_entropy_model_id` and `motion_bits`
  whether or not the stream has P-frames. The deployed path fills the residual one
  from the M11-G16+M13 identity, which does not exist here; this fills it from
  `calibrate_grids`'s own static residual model. That changes 8 bytes of a fixed
  56-byte header and nothing else, which is why the totals above match exactly.

## The bug that cost a debugging cycle

The first version defaulted `--checkpoint` to
`outputs/checkpoints/vimeo_epoch17_best.pt`, which is where the checkpoint lineage
*starts* - Vimeo training, before the DAVIS fine-tune - not the deployed M10F
model. It coded 38% more bytes at 2.5 dB lower PSNR, which looked exactly like a
bug in the lean coding path.

The M11 provenance gate is what exposed it: the calibration signature came out
`4c3bd029…` where the deployed 4-bit checkpoint records `eab90825…`.
`calibrate_grids` was also confirmed deterministic along the way (two fresh-seeded
calls produced identical grids), which ruled out RNG state and pointed at an input.
`tests/test_benchmark_intra_gate.py` now pins the default equal to
`benchmark_parity`'s, and asserts it is not the lineage start.
