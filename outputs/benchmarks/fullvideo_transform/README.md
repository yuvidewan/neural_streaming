# The transform's full-video worth: −280 BD-rate points

2026-10-24. `scripts/benchmark_fullvideo_transform.py`, GOP 10, full DAVIS TEST
(9 sequences, 719 frames), `decode-exact=True` at every point in both arms.

The Stage 1 work was all intra-only. This measures what the transform is worth on
**full video** — the thing the project is judged on — by holding the entropy
stack fixed and changing only the transform.

## Result

| anchor | 593k control | **64ch + QAT** | delta |
|---|---|---|---|
| x264 default, PSNR | +528.5% | **+248.1%** | **−280.4** |
| x264 default, MS-SSIM | +444.4% | **+237.0%** | −207.5 |
| x265 default, PSNR | +567.6% | **+269.8%** | −297.8 |
| x264 I-10 no-B, PSNR | +357.1% | **+152.4%** | −204.7 |
| x265 I-10 no-B, PSNR | +397.5% | **+175.2%** | −222.3 |

**The transform more than halves the full-video deficit.** Both arms ran through
the identical entropy stack, so the delta is attributable to the transform and
nothing else.

| bits | control bpp / PSNR | QAT bpp / PSNR | Δ bytes | Δ PSNR |
|---|---|---|---|---|
| 5 | 0.8007 / 29.812 | 0.8500 / 32.858 | +6.2% | **+3.05 dB** |
| 4 | 0.5715 / 29.456 | 0.6136 / 32.289 | +7.4% | **+2.83 dB** |
| 3 | 0.3690 / 28.337 | 0.4016 / 30.560 | +8.8% | **+2.22 dB** |

## Read this correctly

The entropy stack here is **`calibrate_grids` only** — intra grid, residual
grid, motion table and an empirical entropy model for each. No M10K
z_ref-conditioned model, no M10L codebook, no M11 channel-autoregressive context
model, none of the M13/M14/M22 recalibrations. All of those are fitted to the
deployed latent and are refused by the provenance check against a new transform,
which is why they are absent rather than merely unused.

So **the absolute numbers here are not comparable to the paper's Table II.** The
control arm proves the size of the gap: the deployed codec with its full stack
sits at +450.7% against x264, and the same transform through this simple stack
sits at +528.5% — the learned stack is worth **77.8 BD-rate points** on the old
transform. That the control reproduces the deployed codec's PSNR almost exactly
at 11% more bytes (0.8007 bpp / 29.812 dB against 0.7211 / 29.82) is the check
that this harness is measuring what it claims.

**That projection has since been MEASURED, and it was optimistic by 44 points.**
The estimate was that if the learned stack were worth a similar 77.8 points on
the new transform, the full stack would land near **+170%** against x264. The
refit was done and measured on the same single-pass harness: **+214.2%**
(`outputs/benchmarks/parity_refit/README.md`). The stack transfers at only 48% of
its old value, -33.9 points against -70.0 on the `nvc_deployed` arm, because a
transform that decorrelates more leaves less redundancy for an entropy model to
collect. **Do not quote the +170% figure**; it is superseded.

## A second finding: motion's share falls

| bits | control | QAT |
|---|---|---|
| 5 | 3.8% | 3.6% |
| 4 | 5.5% | 5.0% |
| 3 | 8.7% | 7.8% |

Motion costs relatively less of the new transform's stream at every operating
point, because the residual grew. That is consistent with a higher-fidelity
latent whose residuals carry more information, and it means the motion model is
a slightly smaller target than it was — worth knowing before anyone invests in
learned optical flow.

## What this changes about the plan

The deployed refit is six milestones of work — M10K, M10L, M11, then M13/M14/M22
— because each is fitted to the old latent. Before this measurement that was a
speculative investment. It is no longer: the transform is worth over 280 BD-rate
points on full video through a fixed stack, so refitting the stack around it is
committing effort to a win that is already demonstrated rather than hoped for.

Order, cheapest first:

1. **M10K** — the z_ref-conditioned entropy model on the new transform's
   residuals. Everything downstream warm-starts from it.
2. **M10L** — refit the 512-prototype codebook from M10K's predictions.
3. **M11** — the channel-autoregressive context model (worth −4.81% on the old
   transform).
4. **M13/M14/M22** — the recalibrations, each worth 1–4%.
5. Re-run `benchmark_parity.py` for the real full-video number.

## Files

- `fullvideo_deployed.json` — the 593k control
- `fullvideo_64ch_qat.json` — the 64-channel QAT transform
- `both_arms.log` — both runs
