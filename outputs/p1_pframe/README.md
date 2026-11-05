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
