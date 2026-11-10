# Better motion, gate 1: the motion oracle (2026-11-10)

`scripts/p2_motion_oracle.py`, gate 1 of `docs/PLAN_learned_motion.md`. It asks how much a
better `x_warp` would help the gate 3 learned P-coder, before anything is built.

**Setup:**
- VAL-B (car-roundabout, drift-straight, pigs, stunt), GOP 10.
- Every P-frame is chained on its own arm's previous reconstruction and coded by the gate 2
  stage 2 P-coder for that lambda.
- **Rates:** ideal code lengths. The P-coder's Laplace is within 0.01% of real bytes (gate 3).
  Paid motion is scored against an empirical dy/dx table fitted on its own vectors.
- Only P-frames are scored.

| lambda | arm | residual bits/P | motion bits/P | P PSNR | residual vs block16 |
|---|---|---|---|---|---|
| 1e-3 | block16 (deployed) | 15,626 | 1,866 | 31.678 | 0 |
| 1e-3 | block8 | 14,171 | 8,091 | 31.868 | -9.3% |
| 1e-3 | block4, free | 11,195 | 0 | 32.603 | -28.4% |
| 1e-3 | **dense flow, free** | 16,372 | 0 | 31.680 | **+4.8%** |
| 1e-3 | dense flow from clean frames, free | 16,337 | 0 | 31.680 | +4.5% |
| 1e-3 | zero motion | 36,954 | 0 | 30.284 | +136.5% |
| 3e-4 | block16 (deployed) | 25,470 | 1,832 | 32.782 | 0 |
| 3e-4 | block8 | 23,788 | 7,862 | 32.965 | -6.6% |
| 3e-4 | block4, free | 20,472 | 0 | 33.655 | -19.6% |
| 3e-4 | **dense flow, free** | 27,184 | 0 | 32.809 | **+6.7%** |
| 1e-4 | block16 (deployed) | 37,690 | 1,815 | 33.220 | 0 |
| 1e-4 | block8 | 35,942 | 7,737 | 33.416 | -4.6% |
| 1e-4 | block4, free | 33,341 | 0 | 34.123 | -11.5% |
| 1e-4 | **dense flow, free** | 38,801 | 0 | 33.267 | **+2.9%** |

P-frame BD-rate vs block16 (motion + residual): block8 **+9.1%**; block8 free -23.0%;
block4 free -56.6%; **dense flow free -4.1%**; dense flow from clean frames free -4.5%.

**GATE 1 NOT PASSED.** The bar, fixed in the plan before any result, is that free dense
flow saves at least 25%. It saves 4.1%, and all of that is the motion bits it does not pay
for.
- **Dense flow makes the residual bigger (+3% to +7%), even for free.** Estimating the flow
  between the clean original frames changes nothing (-4.5%), so this is not a flow-quality
  problem at Farneback's level.
- **Part of the cause is training:** the P-coder only ever saw block-shifted references. A
  smooth warp is new to it, so this oracle understates what a jointly trained flow+P system
  could do. It cannot say by how much.
- **8x8 blocks** cut the residual 5-9% and add 0.2 dB, but their motion costs 4x as much,
  so they lose overall (+9.1%).
- **Free 4x4 blocks are not an achievable target.** A free 4x4 vector, chosen by searching the
  frame being coded, carries about 10 bits about that frame per 16 pixels. Its -56.6% is
  partly the frame smuggled through the motion, not better prediction. It is reported as a
  diagnostic, not as the gate.
- **The real prize is not motion:** zero motion costs 1.6-2.4x block16, so block motion
  already captures most of what motion can give this P-coder.

**Do not build the learned-motion pilot on this evidence.** The one open question is the
training caveat. A cheap check is to fine-tune the P-coder for a few thousand steps on
dense-warped references, then rerun this oracle. If dense flow still cannot reach -25%,
the motion line is closed.

`gate1_oracle.json` holds every number. `gate1_oracle.log` is not committed.
