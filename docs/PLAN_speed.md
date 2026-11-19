# Plan: speed, phase 1 (lossless), fixed 2026-11-19 before any change

Roadmap item 4 (`docs/ROADMAP_2026-11-17.md`).

## What the profile says (`scripts/p7_speed_profile.py` → `outputs/p7_speed/profile.json`)

Measured on the laptop RTX 5060. One GOP (31 frames), second pass reported, CUDA synced
around every stage:

| | DAVIS 256x256 | UVG 1920x1072 |
|---|---|---|
| encode | 0.30-0.42 s/frame | 1.07-1.11 s/frame |
| of which block motion search | **95-96%** | **52-53%** |
| decode | 0.011 s/frame (~89 fps) | 0.38-0.39 s/frame (2.6 fps) |
| of which synthesis + refine network | 52% | 66% |
| of which y prior network | 16% | 22% |
| of which range decoding | 10% | 7% |

- The motion search is a Python loop over all 33x33 = 1,089 candidate vectors, each a handful
  of tiny GPU calls. At 256x256 the time is almost all call overhead.
- Range coding is under 2% of encode and 7-10% of decode. Batching it is not worth doing first.
- Decode at 1080p is the networks. Making them faster (half precision, fused layers) changes
  the reconstruction, so it changes the bits and needs its own quality gate. That is phase 2,
  planned separately.

## Phase 1: a vectorised motion search with the same output

Replace the candidate loop in `m10h_motion_compensation.estimate_block_motion` with a batched
version: many candidate shifts per GPU call, then one deterministic selection with the same
rule (lowest SAD, then smallest |dy|+|dx|, then the first in (dy, dx) order).

## Gate (fixed before any change)

1. **Same output:** the new search returns exactly the same motion field as the old one on
   every P-frame of the full DAVIS TEST set (719 frames) and the first 31 frames of every UVG
   sequence, using the codec's real references. Any difference fails the gate.
2. **Same streams:** `p4_gate3.py` at λ 1e-3 on DAVIS TEST gives the same bytes, bit for bit,
   as `outputs/p4_wide_latent/gate3_a2_wide_gop30.json` (same total bytes, same PSNR to the
   last digit, decode-exact).
3. **Faster:** encode time per frame drops by **at least 3x on DAVIS** and **at least 1.5x on
   UVG**, measured with `p7_speed_profile.py` as above.

PASS needs all three. If 1 or 2 fail, the change is dropped. If only 3 fails, the change is
kept (it is free) and reported as too small.
