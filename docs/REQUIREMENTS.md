# Requirements: what "done" means (2026-11-26)

This is the written definition of the goal. Every plan says which requirements it moves. Every
result is checked against the whole list, and an unmet requirement is reported as a blocker, not
a footnote. A requirement is added here the moment it is found, before any work that depends on
it.

The user's choices (2026-11-26) set the targets:
- **Decoding devices:** any PC, including PCs without a GPU.
- **Use:** live streaming.
- **Content:** camera video, screen sharing and text, and animation and games.

Status key: **fails**, **untested**, **partial**, **met**. Measurements quoted are from
`docs/FLAW_CHECK_2026-11-26.md` and from the checks run on 2026-11-26.

## Open conflicts (decide before v2 work starts)

**C1. CPU-only decoding against the v2 design.** Real-time 1080p at 30 frames/s means decoding 62
million pixels per second. This laptop's CPU (16 threads) runs our network at about 125 billion
multiply-adds per second, so a CPU-only decoder can spend about 2,000 multiply-adds per pixel, with
nothing left for entropy decoding or colour conversion. Weaker PCs have less; integer (int8)
arithmetic might give 2 to 4 times more.

| Decoder | Multiply-adds per pixel | Against the ~2,000 budget |
|---|---|---|
| Ours (encoder and decoder) | about 585,000 | about 300× over (measured: 9.8 s per 1080p P-frame on the CPU) |
| DCVC-RT, decoder only | about 172,000 | about 86× over |

The v2 plan (`docs/PLAN_v2_codec.md`) follows DCVC-RT, so it cannot meet requirement R7 as
written. Neural codecs with decoders near 1,000 to 3,000 multiply-adds per pixel exist (Cool-chic,
C3), but they fit a network to each video, so encoding takes minutes per frame. That cannot work
for live streaming. I know of no trained neural video codec at that decoding cost that beats
H.264.

The options are:
- keep R7 and design an ultra-light decoder, accepting a high risk of not beating H.264;
- widen R7 to PCs with any GPU, including integrated graphics, which raises the budget to roughly
  10,000 to 40,000 multiply-adds per pixel depending on the chip (DCVC-RT's decoder would still
  have to be 4 to 17 times lighter);
- keep CPU-only decoding for lower resolutions only.

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
| R7 | Decodes in real time on any PC, including without a GPU (at least 1080p at the stream's frame rate) | frames/s of the full decoder on a CPU-only run, on this laptop and on a weaker PC | **fails**: about 0.1 frames/s on the CPU, 2 frames/s on the GPU; see C1 | open decision |
| R8 | Encodes in real time on the streamer's PC (assumed: a PC with a consumer NVIDIA GPU such as an RTX 3060), with under 2 s of delay | encode frames/s and delay at 1080p | **fails**: 0.8 frames/s at 1080p | v2 gate 4 |
| R9 | Decodes to identical frames on every device and software version | bit-identical decode on a second machine and on the CPU | **fails**: same GPU only (CPU 0 of 10) | v2 amendment 1 |
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
- Any requirement that turns out wrong or missing is fixed here first, with the date.
