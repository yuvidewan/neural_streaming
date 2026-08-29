# Milestone 8 Close-Out: QAT vs. Control vs. M7 Baseline

**Question:** Does quantization-aware training (noise-relaxed) improve the actual rate-distortion
performance of the neural codec, especially at lower quantization bit depths?

**Answer, up front:** Yes, in this implementation, on this evaluation set. QAT beats the
same-budget control model at every bit depth on both PSNR and MS-SSIM, at equal or lower
bitrate, and the margin is largest at 4-bit — consistent with the hypothesis. Full numbers,
methodology, and caveats below.

---

## 1. Files created

- [scripts/compare_m8_models.py](scripts/compare_m8_models.py) — small dedicated script to overlay
  several `benchmark_rd.py` run directories on one PSNR/MS-SSIM-vs-BPP axis, keyed by model
  identity (`plot_rate_distortion.py` only groups by codec within one run).
- [tests/test_scripts_compare_m8_models.py](tests/test_scripts_compare_m8_models.py) — 4 tests
  for the new script (end-to-end, payload-bpp invariant, argument validation, missing-run error path).
- `outputs/calibration/qat_combined_control.json` (+ `_6bit`/`_4bit`) — fresh calibration for the
  control checkpoint.
- `outputs/calibration/qat_combined_noise.json` (+ `_6bit`/`_4bit`) — fresh calibration for the QAT
  checkpoint.
- `outputs/metrics/qat_combined_control/`, `outputs/metrics/qat_combined_noise/`,
  `outputs/visualizations/qat_combined_control/`, `outputs/visualizations/qat_combined_noise/` —
  per-model symbol-distribution stats/plots from calibration (kept out of each other's and M7's
  way via `--metrics-dir`/`--visualizations-dir`).
- `outputs/benchmarks/m8_qat_close_out/{baseline,control,qat}/` — three independent
  `benchmark_rd.py` run directories (results.json/.csv, aggregate.csv, metadata.json, plots/).
- `outputs/benchmarks/m8_qat_close_out/comparison/` — `comparison.json`, `comparison.csv`,
  `rd_psnr_vs_bpp.png`, `rd_msssim_vs_bpp.png`.
- `outputs/benchmarks/m8_calibration_fit_precheck.json` — standalone Task 3 fit-check output.
- `outputs/benchmarks/m8_checkpoint_hashes_before.json` — Task 8 integrity baseline.
- This file, `MILESTONE_8_RESULTS.md`.

All new `outputs/` artifacts are local-only (the whole `outputs/` tree is gitignored, matching
the project's existing convention — nothing here changes what gets committed).

## 2. Files modified

- [src/nvc/evaluation/rd_benchmark.py](src/nvc/evaluation/rd_benchmark.py) — `check_calibration_fit()`
  now also returns `clipped_low_percent`, `clipped_high_percent`, `clipped_total`, `total_values`,
  surfacing fields `count_clipped()` already computed but the function previously discarded. No new
  metric invented; existing callers (`benchmark_rd.py`, `require_calibration_fit()`) are unaffected
  since only new dict keys were added.
- [tests/test_rd_benchmark.py](tests/test_rd_benchmark.py) — added
  `test_check_calibration_fit_reports_low_high_clip_split` (imports `check_calibration_fit`,
  which wasn't previously imported in this file).
- [TESTING.md](TESTING.md) — added `compare_m8_models.py` to the Layer 2 script-test table.

No files under `src/nvc/models/`, `src/nvc/compression/quantization.py`,
`src/nvc/compression/nvc_format.py`, or any training script were touched. Architecture and
codec format are unchanged.

## 3. Exact checkpoints compared

| | Model | Path | Epoch (in `best.pt`) | val MSE | val PSNR | SHA256 |
|---|---|---|---|---|---|---|
| A | M7 baseline | `outputs/checkpoints/vimeo_epoch17_best.pt` | 17 | 0.0009153 | 31.26 | `bc0e89e5...decfff8` |
| B | Control (new) | `outputs/qat_combined/checkpoints_qat_control/best.pt` | 36 | 0.0004574 | 34.33 | `613652d4...9506b02` |
| C | QAT (new) | `outputs/qat_combined/checkpoints_qat_noise/best.pt` | 40 | 0.0004554 | 34.35 | `90d51157...aadaa776` |

**Caveat found during Task 1 and worth restating plainly:** your brief described 46 epochs
(control) / 51 epochs (QAT) across 10/10 chunks. The `best.pt` files that actually exist only
embed history up to **epoch 36 / chunk 7 (control)** and **epoch 40 / chunk 7 (QAT)** — `best.pt`
is a lowest-validation-loss snapshot, so training continued through chunks 8–9 without ever
beating that val loss again. No `latest.pt`/chunk-9 checkpoint exists in either `qat_combined`
folder (only `best.pt` was transferred from the GPU machine), so this evaluation is necessarily
of the chunk-7 "best" snapshots, not chunk-9 "final" snapshots. This is expected behavior for a
best-checkpoint RD comparison, not a bug — flagged so it isn't mistaken for one.

Also: `outputs/checkpoints/vimeo_qat_noise_best.pt` (epoch 14) is a **different, older** QAT
checkpoint from earlier M8A infra validation, already benchmarked once in
`outputs/benchmarks/vimeo_qat_noise_vs_milestone7/`. It is not Model C and was not touched.

## 4. Exact calibration methodology

**Manifest/split decision (confirmed with you before running anything):** no Vimeo train
manifest exists anywhere in the repo — `data/external/vimeo_septuplet/` is empty, and
`train_vimeo_qat_combined.py` deletes each chunk after training it to save disk space, so nothing
persists after a finished run. The only manifest in the repo is `data/processed/manifest.json`,
which is the **DAVIS** manifest (90 sequences, split counts 72/9/9). Checking `vimeo_epoch17.json`'s
own metadata and, more importantly, its already-executed fit-check against real DAVIS test frames
(0.05–0.08% clip, order of magnitude under the 2% guard) confirmed the M7 baseline's calibration
was, in practice, already built from the DAVIS train split — not genuine Vimeo frames — and fits
cleanly. You confirmed reusing this same source for control/QAT, over downloading a fresh Vimeo
chunk (the `kaggle` CLI isn't even installed on this machine).

**Method (identical across all three models, unchanged from M7):**
- Script: `scripts/calibrate_quantizer.py`, method `per_channel_percentile`, percentiles
  `[0.1, 99.9]` (project defaults).
- Manifest: `data/processed/manifest.json`, split `train` (DAVIS train, disjoint from the DAVIS
  test split used for benchmarking — no test leakage into calibration).
- 400 calibration frames (50 batches × batch size 8, both project defaults).
- `--seed 42` (project default, `configs/default.json: random_seed`).
- Bit depths: 8, 6, 4; mode `per_channel`.
- M7's own calibration files (`outputs/calibration/vimeo_epoch17*.json`) were **not** regenerated
  or touched — Task 2 only required recalibrating control/QAT.

Commands executed (control):
```powershell
python scripts\calibrate_quantizer.py --checkpoint outputs\qat_combined\checkpoints_qat_control\best.pt --bits 8 --output outputs\calibration\qat_combined_control.json      --metrics-dir outputs\metrics\qat_combined_control --visualizations-dir outputs\visualizations\qat_combined_control --seed 42
python scripts\calibrate_quantizer.py --checkpoint outputs\qat_combined\checkpoints_qat_control\best.pt --bits 6 --output outputs\calibration\qat_combined_control_6bit.json --metrics-dir outputs\metrics\qat_combined_control --visualizations-dir outputs\visualizations\qat_combined_control --seed 42
python scripts\calibrate_quantizer.py --checkpoint outputs\qat_combined\checkpoints_qat_control\best.pt --bits 4 --output outputs\calibration\qat_combined_control_4bit.json --metrics-dir outputs\metrics\qat_combined_control --visualizations-dir outputs\visualizations\qat_combined_control --seed 42
```
Commands executed (QAT): identical, with `--checkpoint outputs\qat_combined\checkpoints_qat_noise\best.pt`,
`--output outputs\calibration\qat_combined_noise*.json`, `--metrics-dir/--visualizations-dir ...qat_combined_noise`.

Calibration metadata schema (`calibration_metadata` in each output file) already carried
checkpoint path/epoch, bits, mode, split, frame count, percentile range, and entropy model ID —
no schema changes were needed for Task 2's identification requirement.

## 5. Calibration/clipping results

**At calibration time** (400 DAVIS-train frames, self-consistent by construction — this measures
whether the percentile grid landed sanely, not generalization):

| Model | Bits | Clipped % | Below % | Above % |
|---|---|---|---|---|
| Control | 8 | 0.1932 | 0.0964 | 0.0968 |
| Control | 6 | 0.1741 | 0.0875 | 0.0866 |
| Control | 4 | 0.1155 | 0.0566 | 0.0589 |
| QAT | 8 | 0.1929 | 0.0969 | 0.0961 |
| QAT | 6 | 0.1738 | 0.0870 | 0.0868 |
| QAT | 4 | 0.1168 | 0.0616 | 0.0552 |

**At benchmark time**, against real, unseen DAVIS **test** frames (the actual generalization
check — this is what `check_calibration_fit()`/`require_calibration_fit()` gate on, and what the
Task 3 low/high-split extension made visible):

| Model | Bits | Clipped % | Below % | Above % | Latent range | Guard |
|---|---|---|---|---|---|---|
| Baseline (M7) | 8 | 0.0816 | 0.0397 | 0.0420 | [-11.12, 10.73] | OK |
| Baseline (M7) | 6 | 0.0759 | 0.0355 | 0.0404 | [-11.12, 10.73] | OK |
| Baseline (M7) | 4 | 0.0526 | 0.0183 | 0.0343 | [-11.12, 10.73] | OK |
| Control | 8 | 0.1148 | 0.0568 | 0.0580 | [-11.92, 13.02] | OK |
| Control | 6 | 0.1030 | 0.0507 | 0.0523 | [-11.92, 13.02] | OK |
| Control | 4 | 0.0793 | 0.0374 | 0.0420 | [-11.92, 13.02] | OK |
| QAT | 8 | 0.1438 | 0.0717 | 0.0721 | **[-59.74, 64.67]** | OK |
| QAT | 6 | 0.1293 | 0.0629 | 0.0664 | **[-59.74, 64.67]** | OK |
| QAT | 4 | 0.0954 | 0.0511 | 0.0443 | **[-59.74, 64.67]** | OK |

**This answers the "did QAT change the latent distribution" half of Task 3 directly, and the
answer is yes, substantially:** the QAT model's raw latent dynamic range is roughly **5× wider**
than both the control and M7 baseline (±60–65 vs. ±11–13). Despite that, fresh percentile
calibration adapts to it and clips at essentially the same low rate as the others (all three
models sit an order of magnitude under the 2% guard threshold, all `fits: true`, no
`--allow-calibration-mismatch` needed anywhere). Clipping is symmetric (low% ≈ high%) for every
model/bit-depth — no evidence of one-sided distribution shift.

## 6. Benchmark commands executed

Full DAVIS test split (9 sequences, 719 frames), no `--max-sequences`/`--max-frames-per-sequence`
limits, `--codecs nvc` only (per Task 4 — H.264/H.265 are unchanged and were not rerun), no
`--allow-calibration-mismatch` (not needed):

```powershell
python scripts\benchmark_rd.py --manifest data\processed\manifest.json --split test --codecs nvc `
  --checkpoint outputs\checkpoints\vimeo_epoch17_best.pt --calibration outputs\calibration\vimeo_epoch17.json `
  --nvc-bits 8 6 4 --output-dir outputs\benchmarks\m8_qat_close_out --run-name baseline --device cpu --seed 42

python scripts\benchmark_rd.py --manifest data\processed\manifest.json --split test --codecs nvc `
  --checkpoint outputs\qat_combined\checkpoints_qat_control\best.pt --calibration outputs\calibration\qat_combined_control.json `
  --nvc-bits 8 6 4 --output-dir outputs\benchmarks\m8_qat_close_out --run-name control --device cpu --seed 42

python scripts\benchmark_rd.py --manifest data\processed\manifest.json --split test --codecs nvc `
  --checkpoint outputs\qat_combined\checkpoints_qat_noise\best.pt --calibration outputs\calibration\qat_combined_noise.json `
  --nvc-bits 8 6 4 --output-dir outputs\benchmarks\m8_qat_close_out --run-name qat --device cpu --seed 42

python scripts\compare_m8_models.py --run baseline_m7=outputs\benchmarks\m8_qat_close_out\baseline `
  --run control=outputs\benchmarks\m8_qat_close_out\control --run qat=outputs\benchmarks\m8_qat_close_out\qat `
  --output-dir outputs\benchmarks\m8_qat_close_out\comparison
```

Every number below comes from a real `.nvc` encode → decode → reconstruction over all 719 test
frames — nothing estimated from latent entropy or training loss.

## 7. Full 3-model × 3-bit result table

| Model | Bits | Total BPP | Payload BPP¹ | Compression ratio | Mean PSNR (dB) | Mean MS-SSIM | Pooled PSNR | Mean bytes/frame | Calib. clip % |
|---|---|---|---|---|---|---|---|---|---|
| baseline_m7 | 8 | 1.9146 | 1.8475 | 12.54 | 27.581 | 0.9512 | — | 15,684 | 0.0816 |
| control | 8 | 1.8762 | 1.8092 | 12.79 | 29.458 | 0.9686 | — | 15,370 | 0.1148 |
| **qat** | **8** | **1.8587** | **1.7917** | **12.91** | **29.748** | **0.9730** | — | 15,227 | 0.1438 |
| baseline_m7 | 6 | 1.4128 | 1.3458 | 16.99 | 27.513 | 0.9493 | — | 11,574 | 0.0759 |
| control | 6 | 1.3749 | 1.3079 | 17.46 | 29.311 | 0.9662 | — | 11,264 | 0.1030 |
| **qat** | **6** | **1.3566** | **1.2895** | **17.69** | **29.614** | **0.9710** | — | 11,113 | 0.1293 |
| baseline_m7 | 4 | 0.9000 | 0.8330 | 26.67 | 26.529 | 0.9184 | — | 7,373 | 0.0526 |
| control | 4 | 0.8635 | 0.7965 | 27.79 | 27.419 | 0.9248 | — | 7,073 | 0.0793 |
| **qat** | **4** | **0.8446** | **0.7776** | **28.41** | **27.855** | **0.9375** | — | 6,921 | 0.0954 |

¹ *Payload BPP* = total BPP minus the fixed `.nvc` header (`FIXED_HEADER_SIZE=37` bytes +
64 channels × 8 bytes/channel = 549 bytes/frame, from `nvc_format.py`'s documented header
layout) — a derived arithmetic quantity from the real measured `total_bytes`, not re-measured
or estimated. *Pooled PSNR* omitted from the table for space; identical pattern to mean PSNR,
full values in `outputs/benchmarks/m8_qat_close_out/{name}/results.json`.

**Entropy coding efficiency** (from the calibration runs, aggregate order-0 vs. per-channel
static model, at 8-bit — mean over 400 calibration frames):

| Model | Fixed-width | Aggregate empirical entropy | Per-channel model | Headroom vs. fixed |
|---|---|---|---|---|
| baseline (from M7's calibration file) | 8.0 | — | — | — |
| control | 8.0 | 7.2144 | 7.1031 | 0.8969 bits/symbol |
| qat | 8.0 | 7.1915 | 7.0829 | 0.9171 bits/symbol |

Encode/decode speed is essentially identical across all three models at every bit depth
(e.g. 8-bit: ~0.019 s/frame encode, ~0.05 s/frame decode for all three) — confirms no
architectural change occurred, as required.

## 8. PSNR vs. BPP analysis

![PSNR vs BPP](outputs/benchmarks/m8_qat_close_out/comparison/rd_psnr_vs_bpp.png)

QAT's curve sits strictly above control's at all three points, which sits strictly above M7's.
The **QAT vs. control** gap in PSNR: **+0.29 dB (8-bit), +0.30 dB (6-bit), +0.44 dB (4-bit)** — the
gap widens at the lowest bit depth, in absolute dB. Critically, QAT achieves this at *lower* BPP
than control at every point (0.8446 vs 0.8635 at 4-bit, etc.) — this is not a quality-for-rate
trade, QAT dominates control on both axes simultaneously.

## 9. MS-SSIM vs. BPP analysis

![MS-SSIM vs BPP](outputs/benchmarks/m8_qat_close_out/comparison/rd_msssim_vs_bpp.png)

Same ordering. **QAT vs. control** gap in MS-SSIM: **+0.0044 (8-bit), +0.0048 (6-bit), +0.0127
(4-bit)**. The 4-bit gap is roughly 3× the 6-/8-bit gap in absolute MS-SSIM terms — a clearer,
more strongly bit-depth-dependent effect than the PSNR gap, and it points the same direction:
QAT's advantage grows as quantization gets coarser.

## 10. 8-bit interpretation

QAT preserves quality at 8-bit, as expected — the brief noted a major improvement wasn't expected
here since the float-to-8-bit gap was already small. Measured: QAT is +0.29 dB PSNR and +0.0044
MS-SSIM over control, +2.17 dB PSNR and +0.0218 MS-SSIM over the M7 baseline, at slightly lower
BPP than both. The QAT-vs-control gap is real but modest, consistent with 8-bit quantization
being fine-grained enough that noise-relaxation training has less to correct for.

## 11. 6-bit interpretation

Does QAT improve the RD trade-off vs. the same-budget control at 6-bit? Yes: +0.30 dB PSNR,
+0.0048 MS-SSIM, at lower BPP (1.3566 vs 1.3749). The gap is essentially the same size as at
8-bit in PSNR terms, and only marginally larger in MS-SSIM — 6-bit does not show a dramatically
different story from 8-bit here.

## 12. 4-bit interpretation — the main hypothesis

This is where the brief said to look hardest, and it's where the effect is clearest:

- **PSNR:** QAT +0.44 dB over control (largest of the three bit depths), +1.33 dB over M7 baseline.
- **MS-SSIM:** QAT +0.0127 over control (roughly 3× the 6-/8-bit gap), +0.0191 over M7 baseline.
- **Quality loss relative to 8-bit:** control loses 2.04 dB PSNR / 0.0438 MS-SSIM going 8-bit→4-bit;
  QAT loses 1.89 dB PSNR / 0.0355 MS-SSIM over the same drop. QAT's degradation curve going into
  4-bit is measurably gentler than control's, on both metrics.
- **BPP:** QAT is lower than control at 4-bit too (0.8446 vs 0.8635) — not a quality-for-rate trade.
- **Clipping/entropy:** clip rate at 4-bit is low and symmetric for both (control 0.0793%, QAT
  0.0954%, both far under threshold); entropy-model efficiency is essentially the same between
  control and QAT (headroom 0.90 vs 0.92 bits/symbol at 8-bit) — the quality gain is not coming
  from a cheaper entropy code, it's coming from the reconstruction itself surviving quantization
  better.

**QAT improves PSNR, improves MS-SSIM, reduces the quality loss relative to 8-bit, and does so at
equal-or-lower BPP, with the largest margin exactly at 4-bit** — every sub-question Task 5 asked
for 4-bit comes back affirmative in this run.

## 13. Did QAT actually help?

Yes, measured through the full `.nvc` encode → entropy-code → decode → reconstruct path, on the
real held-out DAVIS test split, not training loss or latent-space proxies. The training-loss
near-tie the brief flagged (0.000455 vs 0.000457 val MSE) is *not* what determined this — those
two numbers are close enough to be noise, and codec-level PSNR/MS-SSIM is measurably,
consistently, and monotonically better for QAT once quantization and real entropy coding are
in the loop. This is the distinction Task 9 asked to be strict about: training-loss parity said
nothing here; actual `.nvc` RD performance is where the QAT effect shows up.

## 14. Comparison against the same-budget control

QAT beats control at all three bit depths, on both PSNR and MS-SSIM, at equal-or-lower bitrate.
The control model itself is a legitimate, well-trained same-budget baseline (34.33 val PSNR at
training time, comfortably ahead of M7) — QAT isn't winning because control is weak, it's winning
against a control that already substantially outperforms M7. The margin over control specifically
grows at 4-bit (both metrics), which is the result that actually answers Milestone 8's question.

## 15. Comparison against the original M7 baseline

Both control and QAT substantially outperform M7 at every bit depth (e.g. at 4-bit: M7 26.53 dB /
0.9184 MS-SSIM vs. QAT 27.86 dB / 0.9375 MS-SSIM). This is expected and not itself evidence about
QAT — M7 is a much shorter/earlier-stopped Vimeo run (epoch 17) than either new model (epoch 36/40),
so this gap reflects additional training, not the QAT mechanism. The QAT-specific claim rests on
the QAT-vs-control comparison in §12–14, not the QAT-vs-M7 gap.

## 16. Plots generated

- `outputs/benchmarks/m8_qat_close_out/comparison/rd_psnr_vs_bpp.png`
- `outputs/benchmarks/m8_qat_close_out/comparison/rd_msssim_vs_bpp.png`
- Per-model individual plots also exist (unchanged `plot_rate_distortion.py` output) under each
  run's own `plots/` directory:
  `outputs/benchmarks/m8_qat_close_out/{baseline,control,qat}/plots/`.
- No M7 plots were overwritten — `outputs/benchmarks/vimeo_vs_h264_h265_davis/plots/` and
  `outputs/benchmarks/vimeo_qat_noise_vs_milestone7/plots/` are untouched; the M8 comparison lives
  entirely under the new `outputs/benchmarks/m8_qat_close_out/` directory.

## 17. Test-suite result

```
464 passed in 63.14s (0:01:03)
```
Run via `.venv\Scripts\python.exe -m pytest` from the project root, after all calibration/benchmark
work above. 460 tests existed before this milestone; 4 new (`test_scripts_compare_m8_models.py`)
plus 1 new regression test in `test_rd_benchmark.py` (`test_check_calibration_fit_reports_low_high_clip_split`).
Zero failures, zero skips beyond the usual FFmpeg-guarded ones.

## 18. Model/checkpoint integrity verification

SHA256 recorded before Task 2 and re-verified after Tasks 2–6, for every checkpoint and every
M7 calibration file that must not change:

```
OK  outputs/checkpoints/vimeo_epoch17_best.pt
OK  outputs/checkpoints/davis_baseline_best.pt
OK  outputs/checkpoints/latest.pt
OK  outputs/checkpoints/vimeo_qat_noise_best.pt
OK  outputs/qat_combined/checkpoints_qat_control/best.pt
OK  outputs/qat_combined/checkpoints_qat_noise/best.pt
OK  outputs/calibration/vimeo_epoch17.json
OK  outputs/calibration/vimeo_epoch17_6bit.json
OK  outputs/calibration/vimeo_epoch17_4bit.json
```
All nine unchanged. No historical checkpoint or M7 calibration file was overwritten. Codec format
(`nvc_format.py`) and arithmetic coding (`range_coder.py`/native C backend) were not modified —
only read from, via the identical `benchmark_rd.py`/`calibrate_quantizer.py` paths M7 used.

## 19. Bugs/issues encountered

- **`check_calibration_fit()` dropped the low/high clip split** that `count_clipped()` already
  computed internally — this is the one code change made this milestone (§2), a minimal
  additive fix, not a rewrite. Covered by a new regression test.
- **No Vimeo train manifest exists on disk** (see §4) — resolved by your explicit confirmation to
  reuse the DAVIS train split, matching M7's own actual (if not obviously labeled) methodology.
- **`best.pt` reflects chunk 7 of 9, not the final chunk 9**, for both control and QAT (see §3) —
  not a bug, but worth carrying into any future re-run: if you want the *final* checkpoints
  compared instead of the *best-val* ones, those would need to be saved and transferred
  separately (they weren't, in this run).
- **`compare_m8_models.py`'s first draft raised a bare `SystemExit` on a missing run directory**
  instead of following this project's `[ERROR]` + `return 1` convention — caught immediately by
  its own test (`test_compare_m8_models_missing_run_dir_fails_cleanly`) and fixed before merge.
- No other issues. No retraining was performed; no architecture, codec format, or entropy coding
  semantics were touched, per the milestone's explicit constraints.

## 20. Exact next recommended milestone

The measured result — QAT's advantage growing specifically at 4-bit, at equal-or-lower bitrate,
with clean symmetric calibration fit throughout — is exactly the profile that justifies the next
step the brief explicitly deferred rather than one that was assumed going in:

**Add the rate-distortion loss term (the milestone this evaluation deliberately did not touch).**
QAT-with-noise-relaxation already improves the RD trade-off without any RD-aware loss; the natural
next milestone is training with an explicit rate term so the model can trade bits for quality
directly, using this milestone's 4-bit result as the baseline to beat. Before that, two smaller,
cheap follow-ups are worth doing first since they're direct loose ends from this run:
1. Re-run (or at least re-save) chunk 8–9 checkpoints for control/QAT so a "final" vs. "best"
   comparison can be made — right now only "best" (chunk 7) was evaluable.
2. Consider whether calibrating from genuine Vimeo frames (vs. the DAVIS-train stand-in used here
   and by M7) changes anything materially — low priority given how well the DAVIS-train
   calibration already fits, but worth a one-time check before it becomes an assumed default.

Temporal coding and architecture changes remain correctly out of scope per the brief.
