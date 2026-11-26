# Requirements: what "done" means (2026-11-26)

This is the written definition of the goal. Every plan says which requirements it moves. Every
result is checked against the whole list, and an unmet requirement is reported as a blocker, not
a footnote. A requirement is added here the moment it is found, before any work that depends on
it.

The user's choices (2026-11-26) set the targets:
- **Decoding devices:** any PC with a GPU, integrated graphics included. Changed on 2026-11-26
  from "any PC, including PCs without a GPU" after conflict C1 (below).
- **Use:** live streaming.
- **Content:** camera video, screen sharing and text, and animation and games.

Status key: **fails**, **untested**, **partial**, **met**. Measurements quoted are from
`docs/FLAW_CHECK_2026-11-26.md` and from the checks run on 2026-11-26.

## Open conflicts (decide before v2 work starts)

**C1. Decoder cost: resolved on 2026-11-26.** Real-time 1080p at 30 frames/s means decoding 62
million pixels per second. This laptop's CPU runs our network at about 125 billion multiply-adds
per second, which allows only about 2,000 multiply-adds per pixel. Our decoder is about 300 times
over that, and DCVC-RT's (about 172,000 per pixel) about 86 times. No known trained neural video
codec is that cheap and beats H.264, so the user widened R7 from "no GPU" to "any GPU, integrated
graphics included".

Integrated GPUs differ a lot. This laptop's own Intel UHD Graphics (the small iGPU of the HX
chips) is close to CPU class, about 3,000 multiply-adds per pixel at 1080p30. A mainstream Intel
Iris Xe (about 2 TFLOPS in 16-bit) allows about 10,000, and an AMD Radeon 780M about 35,000, at
realistic efficiency. The budget is therefore fixed as:

- **The whole decoder, at most 10,000 multiply-adds per pixel** (temporal context, entropy
  parameters and reconstruction together, per P-frame and per I-frame).
- That gives real-time 1080p30 on an Iris Xe-class iGPU and real-time 720p30 on weaker iGPUs like
  this laptop's UHD Graphics.
- DCVC-RT's decoder would have to be about 17 times lighter, and ours about 40 times.
- Measured in three ways:
  - the exact count (PyTorch's FLOP counter);
  - real-time frames per second at 720p on this laptop's UHD Graphics through DirectML;
  - before any final claim, 1080p on a real Iris Xe-class or Radeon-iGPU machine (none is
    available yet).

This is a heavy constraint on compression. Whether a decoder this light can beat H.264 is the
main research risk of v2, and the plan has to test it early.

**C2. Content against training data.** Every model so far was trained on Vimeo-90K camera footage,
and every test set is camera footage. R6 needs screen, text, animation and game content in both
training and testing.

**C3. Low bitrates against our rate range.** Our lowest 1080p point is 2.4 to 3.3 Mbps. Streaming
ladders go down to about 0.4 Mbps at lower resolutions (R3).

## Requirements

| # | Requirement | How it is measured | Status today | Covered by |
|---|---|---|---|---|
| R1 | Beats H.264 on PSNR on every test video, over the full range and in the streaming band | `docs/PLAN_fair_eval.md` protocol: original YUV to the anchors, YUV PSNR, x264 medium `-tune psnr`, per-video BD-rate | **fails**: fair RGB check vs x264 4:4:4 is +37% (UVG), +62% (MCL-JCV), +64% (DAVIS) per video; YUV protocol not yet run | fair eval, then v2 gates 2, 3, 6 |
| R2 | Beats H.264 on what viewers see: VMAF per video, against x264 with its default visual tuning | VMAF (and MS-SSIM) per video, same protocol | **untested** | to add to the fair eval |
| R3 | Codes and wins at every rung of a live-streaming ladder (typical, at 30 frames/s): 1080p 3–8 Mbps, 720p 1.5–4, 480p 0.8–1.5, 360p 0.4–0.8 | per-rung BD-rate in the bitrate window | **fails / untested**: only 1080p tested; lowest point 2.4–3.3 Mbps; 720p, 480p, 360p never coded | fair eval (1080p band); v2 |
| R4 | Any frame size: 360p to 4K, landscape and portrait, sizes not divisible by 16 | encode and decode exact at each size; memory fits the target device | **fails**: crashes when a side does not divide by 16; a 4K frame needs 10.8 GiB (8 GiB card), 34 s per frame | fair eval adds padding; v2 needs tiling or a lighter model |
| R5 | Frame rates 24 to 60 (and 120) | coded and scored at real frame rate | **partial**: 24–120 coded at 1080p; rate control missing (R11) | |
| R6 | Content: camera video, screen sharing and text, animation and games, each with its own test set | per-category BD-rate and win counts on sets fixed before use | **fails / untested**: camera only, in training and testing | test sets to choose in a plan; v2 training data |
| R7 | Decodes in real time on any PC with a GPU, integrated graphics included: decoder at most 10,000 multiply-adds per pixel; 1080p30 on an Iris Xe-class iGPU, 720p30 on weaker iGPUs | exact multiply-add count; frames/s at 720p on this laptop's UHD Graphics (DirectML); 1080p on an Iris Xe or Radeon iGPU machine before any final claim | **fails**: decoder about 400,000 multiply-adds per pixel, 2 frames/s at 1080p on the RTX 5060, about 0.1 frames/s on the CPU | v2 amendment 2 (design re-sized) |
| R8 | Encodes in real time on the streamer's PC (assumed: a PC with a consumer NVIDIA GPU such as an RTX 3060), with under 2 s of delay | encode frames/s and delay at 1080p | **fails**: 0.8 frames/s at 1080p | v2 gate 4 |
| R9 | Decodes to identical frames on every device and software version, NVIDIA, Intel and AMD GPUs included | bit-identical decode on a second machine, on the CPU and on an integrated GPU | **fails**: same GPU only (CPU 0 of 10) | v2 amendment 1; the integer arithmetic has to give the same result through DirectML on Intel and AMD as through CUDA |
| R10 | Handles scene cuts | the cut clips of the fair eval; an I-frame (or equivalent) at cuts | **fails**: fixed I-frame every 30 frames, never measured | fair eval measures; v2 |
| R11 | Rate control: holds a target bitrate with a buffer limit (CBR or capped VBR) | bitrate error and buffer use on streaming clips | **fails**: five fixed quality points | v2 |
| R12 | Damaged data is detected, and a viewer can join or resync at a random-access point at least every 2 s | checksums per frame; join a stream mid-way | **partial**: I-frames every 30 frames; no checksum | v2 |
| R13 | 8-bit 4:2:0 BT.709 in and out (the live-streaming format) | YUV in and out, scored in YUV | **partial**: works only in RGB, converted at both ends | fair eval measures the cost; v2 should code YUV directly |
| R14 | Claims are made only under the fair protocol, on video never used for any choice | fresh sets fixed before work starts | **partial**: protocol written, not run; fresh camera set fixed (22 Xiph); no fresh screen or animation sets | fair eval; v2 gate 6 |
| R15 | Models and code can be rebuilt and shipped: weights versioned and backed up, a decoder that installs on another PC | release artefacts; a clean-machine install test | **fails**: 346 MB of weights only on this laptop; never installed elsewhere | to plan |
| R16 | 10-bit and HDR video | | **not required now**, listed as a known gap | |
| R17 | Training data licensed for the intended use | licence check of each dataset | **partial**: Vimeo-90K is research-only, fine for research, not for a product | to check before any release |

## How this list is used

- A plan states which R-numbers it moves and how it will measure them.
- A result is never quoted without the status of R1, R2, R7 and R9 next to it.
- 2026-11-26: R7 widened from CPU-only to any GPU, integrated included, with a 10,000
  multiply-add per pixel decoder budget (C1).
- Any requirement that turns out wrong or missing is fixed here first, with the date.
