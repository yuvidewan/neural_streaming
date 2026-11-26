# DCVC-RT check: GO (2026-11-26)

Plan and decision rule, fixed before any run: [`docs/PLAN_dcvcrt_check.md`](../../docs/PLAN_dcvcrt_check.md).
Frames and config: [`prepare.py`](prepare.py) → [`dcvc_config.json`](dcvc_config.json). Scoring:
[`report.py`](report.py) → [`results.json`](results.json). DCVC-RT's own per-run log:
[`dcvcrt.json`](dcvcrt.json), [`run.log`](run.log).

DCVC-RT is Microsoft's published codec (CVPR 2025), run with their code and weights
(`cvpr2025_image.pth.tar`, `cvpr2025_video.pth.tar`, MIT licence), unchanged. It is a measuring
stick for the design, not our codec. Settings as the plan fixed them:
- the RGB path;
- an I-frame every 30 frames;
- `reset_interval` 32;
- six rates from its own sampling (qp 0, 13, 25, 38, 50, 63);
- real streams.

The fused CUDA kernels were not built (no CUDA toolkit on this laptop), so it ran on the plain
PyTorch fallback. PSNR is measured on frames decoded from the written stream, and bytes are the
stream file sizes.

## Result

BD-rate PSNR, DCVC-RT against each rival, pooled. Negative means DCVC-RT needs fewer bits.

| Rival | DAVIS TEST (9) | UVG 1080p (7) | Videos won |
|---|---|---|---|
| x264 4:2:0, default | −50.5% | −71.9% | 16 / 16 |
| x264 4:2:0, low-delay | −57.6% | −77.9% | 16 / 16 |
| x264 4:4:4, default | −47.9% | −65.7% | 16 / 16 |
| Our current codec | **−54.5%** | **−76.2%** | 16 / 16 |

Decision rule:
1. Wins against x264 low-delay ≥ 15/16 and against x264 4:2:0 ≥ 14/16. Measured: 16 and 16, **pass**.
2. At least 25% fewer bits than our codec on both sets. Measured: −54.5% and −76.2%, **pass**.

**Verdict: GO.** By the plan, the next step is to write the plan for our own second-generation
codec along DCVC-RT lines.

The videos we lose worst are won too. Against x264 4:2:0, DCVC-RT is −31.9% on cows and −28.4%
on schoolgirls, where our codec is +145.5% and +167.7%. It also wins HoneyBee (−70.1%) and Jockey
(−65.7%), our two UVG losses.

## The limit: DCVC-RT stops at lower quality

Its highest rate point (qp 63) tops out below ours on every video:

| | DCVC-RT top (pooled) | Our top (pooled) |
|---|---|---|
| DAVIS | 35.2 dB at 0.280 bpp | 37.9 dB at 1.014 bpp |
| UVG | 38.4 dB at 0.084 bpp | 40.9 dB at 0.592 bpp |

Per video, its top is 0.9 to 4.1 dB under ours. Examples: Beauty 34.7 dB against our 38.1, cows
31.6 against 33.9, and schoolgirls 33.4 against 34.3. Every BD-rate above is computed over DCVC-RT's own range. Against our codec
that range runs, for example, from 32.6 to 34.7 dB on Beauty and 33.2 to 38.9 dB on Jockey. So
DCVC-RT wins heavily at low and middle quality, and this check says nothing about quality above
its top point. That is where our codec's 1080p wins against x264 come from.

The ceiling may come from the RGB path. DCVC-RT is tuned for YUV 4:2:0, and its 64 quality
levels were set for that. The plan fixed the RGB path, and the YUV path was not run. For our own
codec, a design like this would need rate points trained up to about 41 dB on 1080p.

## Speed (RTX 5060 laptop, PyTorch fallback, no fused kernels)

| | DCVC-RT encode | DCVC-RT decode | Ours encode | Ours decode |
|---|---|---|---|---|
| 1920×1072 | 95 ms (10.6 fps) | 101 ms (9.9 fps) | 1,274 ms | 456 ms |
| 256×256 | 11.7 ms (85 fps) | 10.5 ms (95 fps) | 37 ms | 12 ms |

At 1080p it encodes 13× and decodes 4.5× faster than our codec, on the same laptop, without its
fused kernels. That is still a third of real time (30 fps). The paper reports 125 fps encode and
113 fps decode at 1080p on an A100 with the fused kernels. Timing uses DCVC-RT's own
`time.time()` per frame, averaged after 10 warm-up frames.

## Limits

- **Their training, not ours.** Microsoft trained these weights with more data and compute than
  we have. A GO says the design can beat x264 on these videos, not that our own version will.
- **Only DAVIS and UVG.** MCL-JCV was left out because its frames were deleted.
- **Unrounded output.** DCVC-RT scores PSNR on its float output before rounding to 8 bits; our
  codec and x264 are scored on 8-bit frames. At these qualities the difference is below about
  0.02 dB.
- **Default `reset_interval` 32** with I-frames every 30: as the plan fixed, not tuned.

## Files outside the repo

- `C:/dcvc_scratch/DCVC-RT`: the code.
- `C:/dcvc_scratch/checkpoints`: the weights, 266 MB.
- `C:/dcvc_scratch/streams`: the streams.
- `C:/dcvc_scratch/data`: hard links to the test frames, so they take no extra space.
