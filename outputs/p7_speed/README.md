# Speed phase 1: batched motion search, NOT PASSED (2026-11-19), change kept

Plan, gate fixed before the change: [`docs/PLAN_speed.md`](../../docs/PLAN_speed.md).

## Result

| check | result |
|---|---|
| 1. Same motion fields | **PASS**: 692 P-frames of DAVIS TEST and 203 of UVG (first 31 frames of each video), real references, 0 differences (`check_*.search_check.json`) |
| 2. Same streams | **PASS**: λ 1e-3 on DAVIS TEST gives 1,757,043 B, identical per video and in PSNR to `gate3_a2_wide_gop30.json`, decode-exact |
| 3. Faster (≥3x DAVIS, ≥1.5x UVG) | DAVIS **8.0-11.5x** (pass); UVG **0.9x** (fail) |

So the gate is **NOT PASSED** on the UVG speed bar. As the plan says, the change is free, so it is
kept.

Encode time per frame (`profile.json` before, `profile_after.json` after; laptop RTX 5060):

| | before | after |
|---|---|---|
| DAVIS 256x256, λ 1e-2 | 0.296 s | **0.037 s** (motion search 280 → 16.5 ms) |
| DAVIS 256x256, λ 1e-3 | 0.420 s | **0.037 s** (404 → 16.5 ms) |
| UVG 1920x1072, λ 1e-2 | 1.108 s | 1.190 s (589 → 582 ms) |
| UVG 1920x1072, λ 1e-3 | 1.072 s | 1.196 s (558 → 584 ms) |

- On small frames the old search was almost all Python and GPU call overhead, and batching
  removes it. DAVIS now encodes at about 27 fps, untuned.
- At 1080p the full search is limited by memory traffic. It reads the frame 1,089 times
  whatever the batching, so it stays at about 0.58 s per frame. The run-to-run spread is
  about 10%: decode, which did not change, also moved from 0.38 to 0.44 s.
- 1080p encode speed needs a smaller search, such as a hierarchical or predicted search. That
  changes the motion field, so it belongs with roadmap item 5 and needs a rate gate.
- 1080p decode (0.38-0.45 s per frame) is the networks. That is phase 2: half precision or
  fused layers, also with a quality gate.

## Found on the way: motion is most of the bits at 1080p

The profile also splits P-frame bytes (UVG Beauty, first GOP):

| λ | motion bytes per P-frame | residual bytes per P-frame | motion share |
|---|---|---|---|
| 1e-2 | 11,091 | 3,937 | **74%** |
| 1e-3 | 10,395 | 14,695 | 41% |

On DAVIS (256 blocks per frame) motion is about 335 bytes per P-frame. At 1080p there are
8,040 blocks, each coded with about 11 bits, independently and with no rate term in the
search. This is the fixed cost behind the weak UVG low rate in
[`../p6_low_rate/README.md`](../p6_low_rate/README.md). It is one video, so it is a lead, not a
measurement over UVG.

## Files

- `scripts/m10h_motion_compensation.py`: `estimate_block_motion` is the batched search on CUDA
  and falls back to `_estimate_block_motion_loop` (the original) on CPU.
- `tests/test_motion_compensation.py::test_batched_search_matches_loop_on_cuda`.
- `scripts/p7_check_search.py`: checks 1 and 2. `scripts/p7_speed_profile.py`: check 3 and the
  byte split. `run_gate.sh` ran all three.
