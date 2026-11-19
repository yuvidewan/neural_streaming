# B-frames oracle — NOT PASSED (2026-11-19)

Plan and gate, fixed before any code: [`docs/PLAN_bframes.md`](../../docs/PLAN_bframes.md).
Code: [`scripts/bframes.py`](../../scripts/bframes.py), plus `--bframes` and `--b-model` in
[`scripts/p4_gate3.py`](../../scripts/p4_gate3.py).

## Result

Each λ was scored on VAL-B (4 videos, real bytes, decode-exact) by MSE + λ·bpp, with no
B-frames (M = 1) as one of the options. **No B-frames won at every λ**, so TEST and UVG were
not coded: the oracle codec is the current codec, at 0.0% BD-rate.

**Verdict: NOT PASSED.** Per the plan, B-frames close for now and the next roadmap item
(6, MS-SSIM models) starts.

VAL-B cost (MSE + λ·bpp, lower is better):

| λ | M=1 (now) | M=2 same | M=2 coarser | M=4 same | M=4 coarser |
|---|---|---|---|---|---|
| 1e-2 | **0.002507** | 0.002599 | 0.002599* | 0.002754 | 0.002754* |
| 3e-3 | **0.001369** | 0.001402 | 0.001442 | 0.001480 | 0.001587 |
| 1e-3 | **0.000844** | 0.000856 | 0.000875 | 0.000900 | 0.000950 |
| 3e-4 | **0.000496** | 0.000504 | 0.000526 | 0.000536 | 0.000586 |
| 1e-4 | **0.000264** | 0.000265 | 0.000290 | 0.000275 | 0.000324 |

\* λ 1e-2 has no coarser model, so coarser = same.

M=2 same vs M=1: the bits change by +10.1% / +5.4% / +2.8% / +0.8% / +0.1% from 1e-2 down
to 1e-4, with PSNR within ±0.05 dB. The loss is largest at low rate.

## Why (VAL-B, λ 1e-3)

| | P bytes/frame | P PSNR | B bytes/frame | B PSNR | motion bytes |
|---|---|---|---|---|---|
| M=1 | 2,322 | 32.68 | — | — | 35,420 |
| M=2 same | 2,711 (+17%) | 32.39 | **2,053** | **32.94** | 59,811 (+69%) |
| M=4 same | 3,448 (+48%) | 31.67 | 2,245 | 32.81 | 75,309 |

- **The B-frames themselves work.** At M=2 a B-frame costs 12% fewer bytes than an M=1
  P-frame, and is 0.26 dB better. The average of both sides is the most used mode (45%),
  ahead of forward (27%) and backward (29%).
- **The anchors pay for it.** A P-frame that predicts from 2 frames back (4 at M=4) costs
  17% (48%) more and loses 0.3 dB (1.0 dB). That frame is also the reference the B-frames use.
- **Motion nearly doubles.** A B-frame sends two fields plus a mode map, and anchor motion is
  larger. At low rate motion is a big share of the stream, which is why the loss is largest
  there.
- Coding B-frames with the coarser model makes them much cheaper (1,120 B at 31.8 dB), but
  the quality drop outweighs the saving at every λ.

## Notes for later (not part of the verdict)

- The P-coder was trained on one-frame-back predictions. Anchors two frames back are outside
  its training. A B-coder trained on two-sided inputs, and a P-coder trained on longer
  distances, are the obvious ways to recover this. Under the plan, that training only went
  ahead if this oracle won, so it is not started.
- The container now carries B records (frame type 2). The decoder rebuilds display order from
  the frame types alone. All-P streams decode exactly as before: the M=1 path is unchanged,
  and every stream in this run was decode-exact.
