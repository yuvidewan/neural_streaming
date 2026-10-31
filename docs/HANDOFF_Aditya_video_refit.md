# Handoff for Aditya: carry the Phase B transform into the video codec

From Yuvraj's session, 31 Oct 2026. Replaces the QAT-gate handoff (you ran that one: +46.5%,
`cd4c35e`). Short version: **Phase B beat x264 on intra (-12.3%), but the video codec still runs
on run4's transform. The job now is the P-frame refit, starting with a one-model pilot.**

## Where things stand

| | intra BD-rate vs x264 all-intra | full video vs x264 default |
|---|---|---|
| deployed 593k codec | +176.2% | +450.7% |
| run4, 64-ch QAT + your refit | +46.5% | **+214.2%** (`outputs/benchmarks/parity_refit/README.md`) |
| **run5, Phase B lambda sweep** | **-12.3%** | **not measured: this handoff** |

All DAVIS TEST, `sequence_mean/psnr`.

## What Phase B is

`colab_train_stage1.ipynb` with `RUN_PHASE_B = True`, `QUANTIZER = 'unit_step'`, `UNIT_STEP = 1.0`,
`RATE_LAMBDAS = [1e-4, 3e-4, 1e-3]`: three fine-tunes of run4's `best.pt` on D + lambda*R, with
bmshj2018's fixed-step rounding instead of the percentile grid, Vimeo chunks 1-7, no NaN.
Each lambda is ONE operating point (one model), not a bit depth of one model.

### The intra TEST gate

`outputs/stage1_vimeo/colab_run5_phase_b/gate_phase_b_unitstep.json`, run on Yuvraj's CPU with the
new `scripts/benchmark_intra_gate.py --quantizer unit_step` (commit `55e875c` / `ab69244`).
9 sequences, 719 frames, 400 TRAIN calibration frames, `run valid: True`, decode-exact everywhere.

| lambda (step 1.0) | bytes | bpp | PSNR | MS-SSIM | clipped values |
|---|---|---|---|---|---|
| 1e-4 | 5,350,268 | 0.908 | 33.431 dB | 0.9869 | 581 |
| 3e-4 | 4,044,930 | 0.687 | 32.830 dB | 0.9840 | 467 |
| 1e-3 | 2,752,733 | 0.467 | 31.410 dB | 0.9751 | 1,468 |

| BD-rate of the sweep against | PSNR | MS-SSIM |
|---|---|---|
| x264 all-intra (primary) | **-12.3%** | -19.4% |
| x264, the other three conventions | -13.3% / -11.7% / -2.4% | about -19.5% each |
| run4 QAT, same frames | **-31.5%** | -32.0% |
| bmshj2018-factorized (committed anchor) | +17.6% (was +90.9%) | +4.2% |

Caveats: three points, overlap with x264 only 31.4-32.9 dB, and `global_pooled/psnr` is just -2.4%.

### How the gate codes the unit step (you'll reuse this)

The unit step is the codec's own affine grid with `scale = step` and an integer zero point, so
dequantization is exactly `round(z / step) * step` and the `.nvct` format, range coder and
per-channel histogram tables are unchanged. `calibration.unit_step_quantization_params(latents,
step=...)` fits only the alphabet (each channel's TRAIN symbol span sets `zero_point`; the widest
span sets `bits`, capped at 10). Tests in `tests/test_unit_step_calibration.py`.

## Get the checkpoints

Not in git (`*.pt`). Yuvraj is sending `phase_b_checkpoints_for_aditya.zip` (260 MB, the three
`best.pt` files only). Unzip it at the repo root and they land at

```
outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda{1e-4,3e-4,1e-3}/checkpoints/best.pt
```

(`history.json` for each is already committed there).

## The video refit: pilot on one model first

Why one model first: every lambda has a different latent, so each needs its own full P-frame stack
(about three hours each on your GPU, going by the run4 refit). Three blind refits are nine hours;
a pilot tells us in three whether the intra gain survives into video at all.

**Use lambda 3e-4** (the middle point, 32.83 dB intra).

**What to expect.** P-frame residuals are ~80% of the stream and I-frames ~12% (M24 Phase A2),
and today's -12.3% is about the I-frames only. The better transform should help residuals too, since they live in its
latent, but by how much is the open question. Expect a smaller gain than intra.

### Step 1: decide the residual quantizer BEFORE refitting anything

M22 taught us the residual quantizer cannot change after the refit: it invalidates M10K -> M10L ->
M11 -> M13 in that order (`m22_residual.py` docstring). So choose it first.

```bash
.venv/Scripts/python.exe scripts/m24_residual_gate.py --checkpoint outputs/stage1_vimeo/colab_run5_phase_b/phase_b_unitstep_lambda3e-4/checkpoints/best.pt --output-dir outputs/stage1_refit/phase_b_lambda3e-4 --output-name m24_residual_gate.json
```

On run4 this gate found the percentile grid winning at 3-bit (+6.4%) and the unit step winning
above 4-bit (`outputs/m24_quantizer/README.md`, Phase A2). This model was trained for the unit
step and its latent is a third of run4's size (std ~3 vs ~9), so re-measure rather than assume.
Note `--step-multipliers` are relative to the 4-bit reference step the script prints at start;
make sure the sweep includes an absolute step of 1.0, the step the model was trained at.

- **unit step wins at matched quality:** residuals go to the unit step. The rate point is then the
  model itself, and the alphabet width comes from `unit_step_quantization_params`.
- **percentile grid wins:** keep the plain grid (`nvc_deployed`) for residuals, and its 3/4/5-bit
  depths stay a rate knob within this one model.

### Step 2: put the unit step on the video path's I-frames

Every I-frame in the video codec should be coded the way the intra gate just measured it, with
`unit_step_quantization_params` at step 1.0 instead of `calibrate_grids`' percentile intra grid.
No format change (same affine params in the header). Where it plugs in: `intra_params` /
`intra_entropy_model` from `calibrate_grids` in `benchmark_parity.py` and `m21.encode_sequence_refined`.
M14 rejected an intra recalibration on run4; that verdict was about the percentile grid and doesn't
carry over here.

### Step 3: refit the P-frame stack, in the run4 order

Same scripts and gates as `outputs/stage1_refit/README.md`, against the lambda 3e-4 checkpoint and
the residual quantizer from step 1, into `outputs/stage1_refit/*_phase_b_lambda3e-4/` so nothing of
run4's refit is overwritten:

1. **M10K**: z_ref-conditioned entropy model (gate: >0.5% over M10J on held-out validation)
2. **M10L**: K=512 codebook refit
3. **M11**: G16 channel-autoregressive context model (offline gate, then trained, round-trip exact)
4. **M13**: codebook frequency recalibration
5. **M14**: motion-table recalibration

One plumbing difference: the stack is keyed by bit depth (`learned_entropy_{3,4,5}bit.pt`, etc.).
If residuals go unit-step, a rate point is a checkpoint and the "bits" is the fitted alphabet width,
which can be wider than 5. Check the K=512 codebook and the M10K output head size for that before
running them.

### Step 4: the full-video number

`benchmark_parity.py` on full DAVIS TEST, GOP 10, same harness as `parity_refit`, compared with
run4's **+214.2%**:

- residuals on the percentile grid: one model gives a 3-point curve (5/4/3-bit), so a BD-rate;
- residuals on the unit step: one model is one point, so compare at matched quality with
  `benchmark_parity.rate_ratio_at_quality` against run4's refitted curve.

**Failure condition, stated in advance:** if the pilot point does not need clearly fewer bits than
run4's refitted curve at the same quality, the intra gain is not reaching the stream. Then the
bottleneck is P-frame coding, and we stop and report before refitting the other two lambdas.

### Step 5: if the pilot wins, the other two

Repeat steps 1-4 for lambda 1e-4 and 1e-3 (about three hours each) and measure all three together,
as one curve (each model one point, or three points each if residuals stay on the grid).

## Later, not now (Yuvraj, on JupyterLab)

Two more lambdas (3e-5 and 3e-3) to widen the intra curve and settle the 32.6 dB ceiling question,
then gating all five with `--quantizer unit_step`. Not blocking the refit.

## Repo state

- `c-motion-search` and `master` both carry the unit-step gate (`55e875c` / `ab69244`) and the
  trainer resume fix (`c0c4cbd` / `a72508a`).
- Uncommitted on Yuvraj's side, unrelated: demo code (`src/nvc/demo/`, `scripts/*_demo.py`, their
  tests) and paper material in `docs/`.
- `tests/test_m18_intra_quantizer.py::test_m18_scripts_do_not_modify_any_production_source` fails
  whenever `src/nvc/` has uncommitted edits; it's a working-tree check, not a bug.
