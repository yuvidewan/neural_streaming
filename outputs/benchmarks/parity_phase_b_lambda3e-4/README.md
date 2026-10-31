# Full-video benchmark: Phase B lambda 3e-4 pilot

`benchmark_parity.py`, full DAVIS TEST (9 sequences, 719 frames), GOP 10, decode-exact, run
valid, `--arms nvc_deployed --rate-points 8` on the unit-step refit of
`colab_run5_phase_b/phase_b_unitstep_lambda3e-4`. One model is one point:
0.4644 bpp, 32.699 dB PSNR, 0.98386 MS-SSIM (sequence_mean).

At matched sequence_mean PSNR it needs **0.651x the bits of run4's refitted codec** (−34.9%,
`parity_refit/`) and 2.29x x264 default. The full table, the refit chain behind it and the
caveats are in `outputs/stage1_refit/phase_b_lambda3e-4/README.md`.
