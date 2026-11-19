# Plan: B-frames, oracle first (fixed 2026-11-19, before any code or result)

Roadmap item 3 (`docs/ROADMAP_2026-11-17.md`): "Oracle first: the P-coder with a two-sided
reference. Then train a B-coder." The current codec is the one that passed
`docs/PLAN_motion_rate.md`: A2 wide-latent P-coder + wide I-coder, GOP 30, bit-aware motion
(`--motion-lambda "1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625"`). Its reports:
`outputs/p8_motion/test_motion.json` (DAVIS TEST) and `outputs/p8_motion/uvg_motion.json` (UVG).

## Why

Every frame now predicts from the frame before it. x264's default (the anchor we score against)
also uses B-frames, which predict from a past and a future frame. Things that appear, are
uncovered or move unevenly are often visible in one of the two, and the average of two
predictions has less noise. Our worst videos (cows, schoolgirls, HoneyBee, Jockey) have exactly
this kind of content.

## The oracle (no training; the learned coders are unchanged)

1. **Structure.** Inside each GOP of 30 (closed: no frame references across an I-frame),
   anchors are every M-th frame plus the GOP's last frame (and the video's last frame). An
   anchor is a P-frame that predicts from the previous anchor, as now. The frames between two
   anchors are B-frames:
   - M = 2: one non-reference B in the middle.
   - M = 4: hierarchical. The middle B (offset 2) uses the two anchors; offsets 1 and 3 use
     their nearest decoded neighbours on each side.
   - M = 1 is today's codec.
   Frames are coded and written in decoding order (each anchor, then its B-frames). The header
   records M, and the decoder rebuilds the order and outputs frames in display order.
2. **Two-sided prediction.** For each B-frame:
   - A forward field (from the past reference) and a backward field (from the future
     reference), each found with the same bit-aware search and λ_m as the P-frames.
   - Per 16×16 block a mode: forward, backward, or the average of both warps. The mode is
     the lowest `SAD + λ_m · mode bits`.
   - Vectors a block does not use are set to their predictor, in raster order, so they cost
     about nothing.
   - Both fields are coded with `motion_pred`. The mode map is range-coded with a per-frame
     3-symbol table, sent as three 16-bit frequencies.
   - Every motion and mode byte is counted in the bitrate.
3. **Coding the B-frame.** The existing P-coder codes the B-frame with the two-sided prediction
   in place of `warped`. Two options for which model codes it:
   - **same:** the model of this λ.
   - **coarser:** the next coarser λ model, like x264's B-frame QP offset. For λ 1e-2 there is
     no coarser model, so this equals same.
4. Every stream must decode bit for bit.

## Choosing the settings (VAL-B only)

- Grid: M in {1, 2, 4} × B model in {same, coarser}. M = 1 has no B-frames, so it is today's
  codec.
- For each λ, pick the setting with the lowest VAL-B `MSE + λ · bpp`: global pooled MSE on
  [0, 1], real bytes, as in `docs/PLAN_motion_rate.md`.
- If M = 1 wins for a λ, that λ keeps no B-frames.
- Everything is fixed before TEST or UVG are coded.

## Gate (fixed before any result)

The score uses real bytes, GOP 30, untuned, all five λ, decode-exact. It is the BD-rate PSNR
(sequence mean, piecewise linear) of the oracle codec against the current codec on the same
frames:

- **Primary, UVG 1080p:** all 7 videos, 672 frames.
- **Guard, DAVIS TEST:** 9 videos.

Verdicts:
- **PASS:** UVG **≤ −5%** and DAVIS TEST **≤ −5%**, every stream decode-exact. The oracle
  becomes the default. The next step is training a dedicated B-coder, which gets its own plan
  and gate.
- **SMALL WIN:** both ≤ 0%, and at least one ≤ −2%. Kept. B-coder training still goes ahead,
  because a coder trained on two-sided predictions should only add to it.
- **NOT PASSED:** anything else. The oracle is dropped, B-frames close for now, and the next
  roadmap item (6, MS-SSIM models) starts.

Always reported:
- The same comparisons against x264/x265: 4:2:0 and 4:4:4, PSNR and MS-SSIM, low band and
  5-point, per video.
- The low-delay anchors: B-frames add delay, so they are no longer like-for-like and are
  labelled that way.
- The B-frame byte share, and B vs P bytes per frame.
- The mode mix (forward / backward / average).
- Encode and decode time.
