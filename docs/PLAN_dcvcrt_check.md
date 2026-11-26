# Plan: one-day check with the published DCVC-RT (fixed 2026-11-26, before any run)

**Status (2026-11-26): done, GO.** Results: [`outputs/p14_dcvcrt_check/README.md`](../outputs/p14_dcvcrt_check/README.md). DCVC-RT wins 16/16 videos against every x264 arm and needs 54.5% (DAVIS) and 76.2% (UVG) fewer bits than our codec, within its own quality range, which tops out 0.9-4.1 dB below ours. *Correction (2026-11-26, after the run):* the "larger latent at 1/8" below is wrong as written; DCVC-RT's code shows a 128-channel coded latent at 1/16 (0.5 values per pixel), with 256-channel features at 1/8 carrying the temporal context. See `docs/PLAN_v2_codec.md`.

## Why

The goal is to beat H.264 on every video, in real time. Today we beat x264 (normal settings,
4:2:0) on 27 of 46 test videos, and x264 low-delay on 35 of 46. The videos we lose are
detailed ones (cows, schoolgirls, MCL-JCV SRC20/SRC13), where our codec hits a quality
ceiling. At 1080p we compress at 0.8 frames/s and decode at 2.5 frames/s.

The next big step would be a second-generation design along the lines of DCVC-RT (Microsoft,
CVPR 2025): no motion search, a larger latent at 1/8 resolution, built for speed. That is weeks
of work. This check runs the **published** DCVC-RT, with Microsoft's own code and weights, on
our test videos and our laptop first. It answers two questions before any of that work:

1. Does a codec of this design beat x264 on the videos we lose?
2. How fast does it run on the RTX 5060?

DCVC-RT is a measuring stick only. It is not our codec, and its numbers are never quoted as
ours.

## What runs

- **Code and weights:** github.com/microsoft/DCVC, folder `DCVC-family/DCVC-RT` (MIT licence),
  with `cvpr2025_image.pth.tar` and `cvpr2025_video.pth.tar` from the README's link. Nothing is
  changed or retrained. Kept outside the repo, in `C:/dcvc_scratch`.
- **Build:** only the C++ entropy coder (`MLCodec_extensions_cpp`), with the MSVC Build Tools
  already installed. The fused CUDA kernels are **not** built: that needs the CUDA toolkit,
  which is a system install we avoid. DCVC-RT then falls back to plain PyTorch, which gives the
  same output but runs slower. So **speed here is a lower bound**.
- **Videos:** the same PNG frames our codec codes.
  - DAVIS TEST: 9 videos, 256×256, 719 frames.
  - UVG: 7 videos, 1920×1072, 96 frames each.
  - MCL-JCV is left out (its frames were deleted; re-preparing them takes hours). It can follow
    if this check says go.
- **Settings:**
  - DCVC-RT's RGB path, the same input and metric as ours.
  - An I-frame every 30 frames, the same as ours and the GOP-30 x264 anchors.
  - Its own default otherwise: `reset_interval` 32.
  - Six rate points from its own uniform sampling (`--rate_num 6`).
  - Real bitstreams (`--write_stream`); bytes are the written stream sizes.
- **Scoring:** our usual BD-rate (RGB PSNR, sequence mean; bpp = bytes × 8 / pixels), against
  the same anchor files as the current headlines:
  - DAVIS: the 30 fps anchors.
  - UVG: the `*_realfps` anchors.

  It is compared per video and pooled, against x264 4:2:0 (default, medium), x264 low-delay
  (GOP 30), x264 4:4:4, and our current codec (`outputs/p13_audit_fixes/test_motion.json`,
  `uvg_motion.json`).
- **Coverage first:** for every video, the overlap of DCVC-RT's quality range with each anchor's
  is listed before any BD-rate is read. A video with no overlap is reported as such, not scored.

## Decision rule (fixed now)

**GO** (write the plan for our own second-generation codec along DCVC-RT lines) only if both
hold:
1. DCVC-RT beats x264 low-delay on at least **15 of 16** videos, and x264 4:2:0 default on at
   least **14 of 16**.
2. DCVC-RT needs at least **25% fewer bits than our current codec** (pooled BD-rate PSNR) on
   **both** DAVIS and UVG.

Otherwise **NO-GO**. A rebuild is then not clearly worth weeks. We report which videos still
lose and look at targeted fixes on the current codec instead.

**Also reported, either way:**
- Whether DCVC-RT still loses to x264 on cows and schoolgirls. If it does, a new design alone
  will not give "every video".
- Speed (report only): encode and decode frames/s on UVG 1080p, and on DAVIS, against the
  30 frames/s target. It is measured with the PyTorch fallback, next to the paper's own numbers
  (1080p at about 125 frames/s encode and 113 decode, on an A100).

## Known limits

- DCVC-RT was trained by Microsoft with more data and compute than we have. Our own version
  could land weaker than theirs, so a GO says the design can do it, not that we will.
- DCVC-RT is tuned mainly for YUV 4:2:0. Its RGB path may understate it a little.
- DAVIS and UVG have been examined video by video. A final "every video" claim still needs a
  fresh test set.
