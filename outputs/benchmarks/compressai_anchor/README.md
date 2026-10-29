# Learned intra anchor: CompressAI bmshj2018 on DAVIS TEST

2026-10-26. `scripts/benchmark_compressai_anchor.py`, full DAVIS TEST (9 sequences,
719 frames), the same scorer, conventions and piecewise-linear BD-rate as
`benchmark_parity.py`. Real bytes: every string `compress()` emits, with frames
decoded by `decompress()` from those strings only and rounded to uint8.

CompressAI 1.2.8 has no Windows wheel; it was built from source with MSVC into a
scratch directory and passed in with `--compressai-path`. Weights are the zoo's
MSE-trained `bmshj2018-factorized` and `bmshj2018-hyperprior`, qualities 1-8,
trained on Vimeo-90K.

## Result (primary convention, positive = more bits than the anchor)

| test | anchor | PSNR | MS-SSIM |
|---|---|---|---|
| bmshj2018 factorized | x264 all-intra | **-23.3%** | -23.8% |
| bmshj2018 hyperprior | x264 all-intra | **-31.8%** | -27.3% |
| 64ch + QAT, all-intra | bmshj2018 factorized | **+90.9%** | +99.1% |
| 64ch + QAT, all-intra | bmshj2018 hyperprior | +115.2% | +109.8% |
| 593k, all-intra | bmshj2018 factorized | +246.7% | +219.7% |

The harness check: computed from the stored points, the two NVC rows against x264
all-intra come out at +176.2% and +46.5%, the recorded gate figures exactly.

Computed separately from `m23_deploy_test.json`: the full final codec (GOP 10,
with the scale map) is **+7.8% / -1.7%** against the factorized model and
**+23.7% / +5.9%** against the hyperprior, although both code every frame
independently.

## What it says

The factorized model is the closest published design to ours: a GDN transform at
stride 16, trained on Vimeo-90K, no context model and no hyperprior. It differs in
fixed-step rounding, a learned per-channel density and one model per lambda trained
for rate. It needs about half our all-intra bits, so the remaining intra gap sits
in the range-fitted quantizer and the missing rate term more than in transform
capacity. That is the next milestone's target.

## Files

- `compressai_anchor.json` - 16 points with per-sequence rows, and every comparison
  under all four conventions
- `run.log` - the run, download progress removed
