# Plan: a fair evaluation against H.264 and H.265 (fixed 2026-11-26, before any code or run)

## Why

The whole-project flaw check (`docs/FLAW_CHECK_2026-11-26.md`) found that our comparisons with
x264 were not fair:

- **x264 4:2:0 was handicapped.** Our 1080p sources are 4:2:0 YUV. We converted them to RGB,
  FFmpeg converted that back to 4:2:0 with a different matrix, and both codecs were scored in RGB.
  x264 4:2:0 could never pass 42–44 dB on UVG, and our "wins" sat right under that ceiling.
- **The averaging was not the standard one.** Headlines used one pooled curve. Per-video
  averaging gives MCL-JCV +19.3%, not −2.3%.
- **The anchors were not tuned for PSNR,** although we score PSNR.
- **The codec cannot code a 1080-row frame,** so every 1080p test was on 1072-row crops.
- **The streaming bitrates were never judged on their own,** and that is where we lose most.

This plan fixes the method and re-measures the current codec once, to give an honest baseline.
The same protocol then judges v2: its gates 3 and 6 get rewritten to use it in v2 amendment 2,
after this baseline exists.

Nothing is trained or tuned. The codec runs exactly as shipped:
- the MSE model set (`outputs/p4_wide_latent/vimeo` and `outputs/p6_low_rate/vimeo`, λ 1e-2 to 1e-4);
- GOP 30 with wide I-frames;
- rate-constrained motion with `--motion-lambda 1e-2:0.25,3e-3:0.25,1e-3:0.125,3e-4:0.125,1e-4:0.0625`;
- no encode-time tuning.

## The protocol

### Test video

- **UVG:** 7 videos (Beauty, Bosphorus, HoneyBee, Jockey, ReadySetGo, ShakeNDry, YachtRide), from
  the original 8-bit 4:2:0 YUV files.
- **MCL-JCV:** all 30 videos, from the original 8-bit 4:2:0 YUV sources.
- **Frames:** the first 96 of each, at full 1920×1080 with no crop, each at its real frame rate.

DAVIS is not YUV video and is downscaled to 256×256. It becomes a development set only: no claim
against H.264 is made on it, and it is not re-run here. The fresh Xiph set in `docs/PLAN_v2_codec.md`
is not touched.

### How each codec gets the video

x264 and x265 get the original YUV directly, with no conversion. They write raw elementary
streams (`.264`, `.265`, no MP4 container), which are decoded back to 4:2:0 YUV.

Our codec works in RGB, so it pays for its own conversions:

1. YUV to RGB with `scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int`
   (the conversion our prepare scripts already use).
2. Code and decode.
3. RGB back to YUV 4:2:0 with `scale=out_color_matrix=bt709:in_range=pc:out_range=tv:flags=accurate_rnd+full_chroma_int`.
4. Score against the original YUV.

These two conversions are part of our codec in real use, so their loss counts against it. The
loss of the conversions alone, with no coding, is measured and reported as our codec's ceiling in
this protocol.

The codec gets padding. A frame whose sides do not divide by 16 is padded at the right and bottom
by repeating the edge pixels, coded at the padded size, and cropped back after decoding. The
stream header records the original size. The bits spent on padding count, and quality is scored
on the original 1920×1080 only.

### Anchors

The headline anchors are four arms, all at preset medium with `-tune psnr`:

| Arm | Structure |
|---|---|
| x264, default | x264's own GOP (keyint 250), B-frames, scene-cut detection on |
| x264, low-delay | `-bf 0`, an I-frame every 30 frames (our intra period), scene-cut detection on |
| x265, default | x265's own GOP and B-frames, scene-cut detection on |
| x265, low-delay | `bframes=0:keyint=30`, scene-cut detection on |

Three more arms are reported, not headline:
- x264 medium with its default visual tuning (the old anchor), default structure;
- x264 veryslow with `-tune psnr`, both structures, on UVG only (veryslow is about 10 times
  slower to run).

CRF runs from 8 to 46 in steps of 2. Coverage rule, fixed now: if on any video an arm's curve
does not reach past our highest and lowest quality points, more CRFs are added in steps of 2 at
that end until it does. This is mechanical, not a choice. The FFmpeg, x264 and x265 versions and
the CPU thread count go into the report.

### Metrics

PSNR is computed per frame for each plane (Y, U, V; 8-bit, peak 255) and averaged over the frames
of a video. The combined score is PSNR_YUV = (6·PSNR_Y + PSNR_U + PSNR_V) / 8, the weighting used
for 4:2:0 in JVET and in the DCVC papers. Rate is the size of the whole stream file in bits,
divided by the luma pixel count (bits per pixel). Each point is also given in Mbps at the video's
real frame rate and at 30 frames/s.

Reported alongside, not headline:
- PSNR_Y on its own;
- MS-SSIM on Y;
- an RGB track: our RGB output against the RGB source, compared with x264 4:4:4 (medium,
  `-tune psnr`) fed the same RGB through an explicit BT.709 full-range 4:4:4 conversion.

### BD-rate and what counts as a win

BD-rate is computed per video:
- log-rate is interpolated against PSNR_YUV with monotone piecewise-cubic (PCHIP) curves;
- the difference is averaged over the quality range where both curves exist;
- the result is (10^mean − 1) × 100%;
- the project's old piecewise-linear method is computed as a check, and any video where the two
  differ by more than 3 points is flagged.

The headline is the mean of the per-video BD-rates, with the median and the number of videos won.
A video is won against an arm if its BD-rate is below 0, and won robustly if it is −3% or better.
Our codec "beats H.264 on a video" only if it wins against both headline x264 arms there. Pooled
curves are reported last and are never the headline.

There are two bands:
- **Full range:** all five of our rate points.
- **Streaming band**, per video: the quality interval between our codec's quality at 0.016 bpp and
  at 0.129 bpp (1 and 8 Mbps at 1080p, 30 frames/s). It is read off our curve, clipped to our
  tested range, and starts at our lowest point if that is above 0.016 bpp. BD-rate is averaged
  over that interval only. A video whose band is narrower than 1 dB is reported as not scored.

### Extra checks (report only)

- **Scene cuts.** Three clips are made by joining the first 48 frames of one UVG video to the first
  48 of another: Beauty then Bosphorus, HoneyBee then Jockey, ReadySetGo then YachtRide. The cut
  lands at frame 49, in the middle of our GOP. Each clip is coded by our codec and the headline
  arms. Reported: the BD-rate per clip, and the bits and PSNR of the five frames after the cut.
- **Long clips.** The first 300 frames of UVG Bosphorus and Jockey, coded by our codec and the two
  x264 headline arms, to see whether the 96-frame results hold over a longer clip.
- **Speed.** On this laptop, with nothing else running:
  - x264 and x265 (the headline arms): frames per second for encoding raw YUV from the SSD to an
    elementary stream, and for decoding to null, at CRF 22, 30 and 38;
  - our codec: frames per second from its own encode and decode timers, model loading excluded.

  The median over videos is reported.

## Proofs before any number is quoted

If any of these fails, the run stops and the failure is reported.

1. **Padding changes nothing where it is not needed.** UVG Beauty at 1072 rows, λ 1e-2 and 1e-4,
   encoded with the code before and after the padding change, gives byte-identical streams.
2. **Padding works.** Unit tests code and decode frames of 72×64, 70×98 and 1080×1920 exactly. A
   real 1920×1080 UVG clip round-trips decode-exact.
3. **The anchor path adds nothing.** x264 with `-qp 0` and x265 with `lossless=1` on one UVG video
   decode to frames identical to the source YUV, byte for byte.
4. **Our conversion ceiling** (YUV to RGB to YUV, no coding) is measured per video, per plane.
5. **The metric code is right.** Unit tests check PSNR per plane and the 6:1:1 weighting on
   synthetic frames, and PCHIP BD-rate on known cases (identical curves give 0%; a curve at twice
   the rate gives +100%).
6. **Coverage** is checked for every video and every arm, after the extension rule.
7. **Every stream of ours decodes bit-exactly in a separate process,** and every rate is
   recomputed from the file sizes.

## Order (one heavy job at a time, in one detached chain that resumes)

1. Code: padding in the codec, `scripts/fair_eval.py`, and their tests. About a day.
2. Proofs 1, 2, 3 and 5.
3. Downloads, asking first:
   - the 7 UVG archives from ultravideo.fi, 4.92 GiB in total, deleted once extracted;
   - the first 96 frames of the 30 MCL-JCV sources, range-read from Hugging Face as before, about
     9 GB as YUV.

   The YUV files are kept until the report is written, and then deleting them is asked.
4. Anchors on the CPU, about 4 hours (veryslow is most of it).
5. Our codec on the GPU: UVG, MCL-JCV, the scene-cut clips and the long clips. That is about
   22,000 coded frames, about 11 hours.
6. Proofs 4, 6 and 7, then the report.

## What happens with the result

The report goes to `outputs/p15_fair_eval/README.md`. The main README and the paper replace their
H.264 headlines with the new per-video numbers. The old 4:2:0 RGB numbers stay, labelled as
measured against a handicapped anchor. Whatever the result, it is reported as it comes out.

The headline sentence takes this form: "Against x264 (preset medium, tuned for PSNR, given the
original 4:2:0 video), the codec needs X% more (or fewer) bits per video on average on UVG and Y%
on MCL-JCV (YUV PSNR), wins on N of 37 videos, and needs Z% more (or fewer) in the streaming
band."

v2 amendment 2 then rewrites v2's gates 3 and 6 on this protocol, using these baseline numbers.
It also takes up the flaw check's other points: integer I-frames, any frame size, scene cuts,
rate control, and whether v2 should code 4:2:0 YUV directly, as DCVC-RT does, instead of RGB.

## What this plan does not do

- No training, tuning or new setting for the codec.
- It does not touch the fresh Xiph test set.
- It does not fix rate control, error detection or decoding on other devices. Those are v2's
  requirements. Scene cuts are only measured here.
