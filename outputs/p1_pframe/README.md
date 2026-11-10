# Gate 1: learned P-frame coder, local pilot (2026-11-05)

`docs/PLAN_learned_pframe.md`, gate 1. Code: `scripts/p1_pframe.py`, tests `tests/test_p1_pframe.py`.
Run with `run_pilot.sh` (the 3e-4 model trained first; the rest ran via `resume_pilot.sh`).

Setup: one P-step on DAVIS VAL-B (4 sequences, 271 pairs), reference = the intra codec's
reconstruction of the previous frame, integer 16x16 block motion. Intra and hand-built rates use
static per-channel histograms fitted on TRAIN; the learned coder uses its own conditional Laplace
prior. All rates are proxies (ideal code lengths), not `.nvct` bytes.

Each P-coder is warm-started from its intra checkpoint, 20k steps, batch 8, 256x256 crops.
From-scratch training collapsed to the zero-rate solution (x_hat ~ warped frame, 26 dB).

| lambda | arm | bits/frame | PSNR | x intra at same PSNR |
|---|---|---|---|---|
| 1e-3 | intra | 30000 | 30.90 | 1.000 |
| 1e-3 | hand-built | 18762 | 30.89 | outside range |
| 1e-3 | **learned** | **16989** | **31.84** | **0.427** |
| 3e-4 | intra | 44496 | 32.20 | 1.000 |
| 3e-4 | hand-built | 30716 | 32.20 | 0.691 |
| 3e-4 | **learned** | **27356** | **32.90** | outside range |
| 1e-4 | intra | 59079 | 32.74 | 1.000 |
| 1e-4 | hand-built | 43236 | 32.74 | 0.732 |
| 1e-4 | **learned** | **40135** | **33.33** | outside range |

BD-rate: hand-built vs intra -32.6%, **learned vs intra -56.1%**, **learned vs hand-built -37.2%**.

**GATE 1 PASSED** (threshold 0.45x), with caveats:
- Only one learned point (1e-3) falls inside intra's PSNR range, so the 0.427 ratio rests on one
  point. The BD-rate (-56.1% vs intra, i.e. ~0.44x) is consistent with it.
- The learned coder sits 0.6-0.9 dB above the other arms at every lambda, so lambdas are not
  quality-matched; a larger lambda sweep would put more points in range.
- Proxy rate, one P-step, intra-quality reference. Error growth over a GOP (gate 2, stage 2) and
  real bytes (gate 3) are untested.

`gate1.json` holds the numbers. Models (`lambda*/model.pt`) and logs are not committed.

## Gate 2: trained on Vimeo-90K, laptop RTX 5060 (2026-11-08 to 11-10)

`run_gate2_local.sh` runs the Colab notebook's two stages on the laptop instead. One P-coder per
lambda, warm started from that lambda's intra checkpoint, with the same VAL-B evaluation as gate 1:
- **Stage 1:** Vimeo chunks 1 and 2, 10,000 steps each, unroll 1, lr 1e-4 with x0.1 on the last chunk.
- **Stage 2:** chunk 3, 4,000 steps at unroll 3, starting from stage 1. It runs on the coder's own
  detached reconstructions, at lr 1e-5.

That is 3 of the 10 Vimeo chunks. Each chunk was deleted from disk once all three lambdas had used it.

| lambda | arm | bits/frame | PSNR | stage 1: x intra | stage 2: x intra |
|---|---|---|---|---|---|
| 1e-3 | **learned** | 16642 | 31.57 | 0.454 | **0.453** |
| 3e-4 | **learned** | 26192 | 32.66 | 0.474 | **0.464** |
| 1e-4 | **learned** | 38238 | 33.09 | outside range | outside range |

Bits/frame and PSNR are the stage 2 values. The intra and hand-built rows are unchanged from gate 1.

BD-rate, stage 2 (stage 1 in brackets): learned vs intra **-52.8%** (-52.2%); learned vs
hand-built **-31.8%** (-30.9%).

**GATE 2 NOT PASSED, by a narrow margin.** The threshold is <= 0.45x intra at every in-range point,
and the measured ratios are 0.453 and 0.464.
- Two learned points are now inside intra's PSNR range; gate 1 had one. At 1e-3 the ratio moved
  from 0.427 to 0.453. Training is now on Vimeo and testing on DAVIS VAL-B, so the test is out of
  domain, which gate 1's training was not.
- The learned coder is still clearly better than the hand-built P path: -31.8% BD-rate.
- This gate scores one P-step from an intra-quality reference. Stage 2 trains for chained frames,
  which this gate does not measure, so stage 2 barely moves the numbers here.
- Next: either measure the coder in the real chained codec with real bytes (gate 3), or train
  longer on chunks 4-10 with `--init-from gate2_local/stage2`.

`gate2_stage1.json` and `gate2_stage2.json` hold the numbers. `gate2_local/*/lambda*/history.json`
and `progress.json` hold the training curves. Models (`*.pt`) and logs are not committed.
