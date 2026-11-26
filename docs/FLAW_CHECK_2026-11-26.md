# Whole-project flaw check (2026-11-26)

The user asked for a check of the whole project for flaws, before anything moves forward. This
check is against the project's real goal: a codec that beats H.264 on any real video, in real
time, on any machine. Earlier audits asked whether our numbers were computed correctly. They did
not ask whether the codec meets that goal, and known limits were listed at the bottom of reports
instead of being raised as blockers. That is how the same-GPU decoding problem sat in the second
audit's "known limits" until the user found it. Everything below is ranked by how much it
changes what we can claim or build.

Scripts and numbers: `outputs/p13_audit_fixes/` (current results), the anchor files in
`outputs/benchmarks/`, and the check script kept in this session's scratch folder (its output is
quoted below).

## 1. The 1080p "beats H.264" result comes from how x264 was set up, not from the codec

Our test videos (UVG, MCL-JCV) are 4:2:0 YUV sources (BT.709). We converted them to full-colour
RGB, then fed that RGB to FFmpeg, which converts it back to 4:2:0 with a different matrix
(BT.601, limited range) before x264 sees it (`scripts/benchmark_parity.py`, `ffmpeg_round_trip`,
lines 316–330). Both codecs are then scored in RGB. Re-subsampling chroma that was already
upsampled, through a second matrix, loses information x264 can never get back, so x264 4:2:0 has
a hard quality ceiling in this setup:

| Set | x264 4:2:0, best setting | x264 4:4:4, best setting | Our top point |
|---|---|---|---|
| UVG (CRF 0) | 41.7–44.3 dB at 4.34 bpp | 52.0–52.1 dB | 38.1–42.9 dB |
| MCL-JCV (CRF 8) | 37.0–43.6 dB | 41.8–47.7 dB | 34.8–45.5 dB |
| DAVIS (CRF 8) | 36.5–41.8 dB | 40.7–44.4 dB | 33.9–40.3 dB |

Our 1080p wins over x264 4:2:0 sit at 38–41 dB, where x264's curve is pressed flat against that
ceiling. With the ceiling removed (x264 in 4:4:4, scored in RGB, which is the fair comparison for
an RGB codec), we lose on every set:

| Against x264 4:4:4 | Pooled | Per-video mean | Videos won |
|---|---|---|---|
| DAVIS | +54.7% | +64.3% | 3 / 9 |
| UVG | +18.0% | +37.1% | 2 / 7 |
| MCL-JCV | +38.7% | +62.1% | 6 / 30 |

The other fair comparison, closer to real use, would give x264 the original 4:2:0 sources and
score both codecs in YUV. It has not been run (the UVG and MCL-JCV YUV files were deleted after
conversion). It removes x264's handicap, so the 4:2:0 wins above cannot be expected to survive
it; how it compares with the 4:4:4 result is unknown until it is measured.

So, measured fairly, the current codec does not beat H.264. It needs roughly 1.2 to 1.6 times
x264's bits on 1080p and 1.5 to 1.6 times on DAVIS. The main README, the paper (abstract, Sections
V-B and VI) and the v2 plan all state the 4:2:0 result as the headline.

## 2. At normal streaming bitrates we lose heavily, even to the handicapped anchor

Our five rate points at 1080p, 30 frames/s:

| | λ 1e-2 | 3e-3 | 1e-3 | 3e-4 | 1e-4 |
|---|---|---|---|---|---|
| UVG | 2.4 Mbps | 3.8 | 7.1 | 16.5 | 36.6 |
| MCL-JCV | 3.3 Mbps | 5.0 | 8.4 | 19.3 | 36.4 |

Real-time 1080p streaming and video calls run at about 2 to 8 Mbps, our three lowest points. Over
those points, per video, we need 73% (UVG) and 81% (MCL-JCV) more bits than x264 4:2:0, about
99–116% more than x264 4:4:4, and 186–193% more than x265. Every 1080p win is at 16 to 37 Mbps,
far above what a real-time stream uses.

## 3. The headline BD-rates use a non-standard average that flatters us

Our headline curves pool bytes over all videos and average PSNR over videos, then take one
BD-rate. The standard practice (JVET common test conditions, and most learned-codec papers) is a
BD-rate per video, then the mean. Against x264 4:2:0:

| | Pooled (quoted) | Per-video mean | Per-video median |
|---|---|---|---|
| DAVIS | +26.7% | +38.8% | +34.1% |
| UVG | −20.2% | −14.1% | −14.6% |
| MCL-JCV | −2.3% | **+19.3%** | −6.8% |

By the standard method we lose on MCL-JCV even against the handicapped anchor.

## 4. Streams decode correctly only on the GPU that made them

Known (second audit: CPU decode matched 0 of 10). The v2 plan now requires integer decoding
(amendment 1). One more unknown on top: streams record no PyTorch, cuDNN or driver version, and
nothing has tested whether a software update on the same GPU breaks decoding. It very likely can.

## 5. The codec cannot code standard frame sizes

The network and the 16×16 block motion assume width and height divide by 16. A 1920×1080 frame
crashes the coder ("Expected size 9 but got size 10", tested today on 72-row input; 1080/16 is
67.5). So does 854×480, DAVIS's own size. Every 1080p result in this project is on frames
cropped to 1920×1072. A real codec pads and crops internally; ours does neither.

## 6. It is far from real time

On this laptop, x264 medium encodes 1080p at 69–117 frames/s (CPU). Ours encodes at about 0.8
frames/s and decodes at about 2, so it is about 100 times slower to encode. At 256×256 it encodes
in 35–41 ms per frame, 24–29 frames/s, which is below real time; the paper's sentence "The codec
is real time at 256×256" is wrong for encoding.

## 7. No scene-cut handling

The frame types are a fixed pattern, an I-frame every 30 frames
(`scripts/m10h_motion_compensation.py`, `gop_frame_types`). At a scene cut the next P-frame is
predicted from an unrelated picture. None of our test clips has a cut, so the cost has never been
measured. x264 detects cuts and inserts an I-frame. The v2 plan's fresh test set also leaves out
the one Xiph sequence with cuts (rush_field_cuts), which avoids exactly this case.

## 8. No rate control

The codec has five fixed quality points. It cannot hit a target bitrate or follow a changing
network, which real-time streaming needs. x264 has CRF, CBR and VBV.

## 9. No error detection

A flipped bit is not detected (no checksum; second audit). A stream damaged in transit decodes
to garbage without warning.

## 10. The anchors are not tuned for the metric we score

x264 and x265 run with their default visual tuning (psy-rd, adaptive quantization), which costs
PSNR, and we score them on PSNR. The fair PSNR anchor uses `-tune psnr`, and many papers also
quote preset veryslow. Both would make x264 stronger. The size of the effect is not measured.

## 11. The decoder needs 346 MB of models that exist only on this laptop

Ten model files (five rate points × I and P coders) are excluded from git (`*.pt` in
`.gitignore`). They exist on this laptop and its OneDrive sync, nowhere else. Nobody else can
decode our streams, and a lost laptop loses the codec.

## 12. Test data and protocol limits

- DAVIS and UVG were reused across many decisions, and MCL-JCV has now been examined video by
  video.
- 1080p clips are 96 frames: 0.8 s of UVG at 120 frames/s.
- DAVIS is downscaled to 256×256, which is not a standard test condition.
- Most results use one training seed.

## 13. Flaws in the v2 plan written today

- Gates 3 and 6 judge v2 against x264 4:2:0 in the same flawed setup (flaw 1), with pooled
  averaging (flaw 3), and do not judge the streaming bitrates separately (flaw 2).
- The plan contradicts itself: amendment 1 requires the whole decoder to run in integers by
  gate 2, while gates 1 and 2 keep our five floating-point wide I-coders. Their I-frames would not
  decode identically on the CPU, and every P-frame depends on them.
- It has no requirement for arbitrary frame sizes, scene cuts or rate control.

## 14. The tests check consistency, not the goal

The full suite passes (2,051 passed, 14 skipped for missing optional packages or old stage
outputs, 5 min 51 s on 2026-11-26). But no test feeds a frame size that does not divide by 16,
decodes on a second device, or puts a scene cut in a clip. The tests confirm that the code agrees
with itself, which is why none of the flaws above ever showed up as a failing test.

## What stays valid

The comparisons between our own codec versions were all made in one fixed pipeline, so the
improvements measured from step to step stand (for example, motion rate −10.7% on UVG, the
detail latent −20.6% P-frame bits). The trained models are not wasted. What is wrong is the
standing against H.264. The DCVC-RT check also stands: DCVC-RT beat x264 4:4:4 by 48–66%, which
does not depend on the 4:2:0 handicap.

## What should happen before v2

1. Fix the evaluation protocol in a plan of its own, before any run:
   - x264 and x265 fed the original 4:2:0 sources (UVG and MCL-JCV downloaded again), scored in
     YUV against those sources, with our codec's output converted to 4:2:0 BT.709 for scoring;
   - the RGB comparison against x264 4:4:4 kept next to it;
   - `-tune psnr` anchors next to the default ones;
   - per-video BD-rate averaging as the headline;
   - the streaming band (about 2–8 Mbps at 1080p30) judged on its own.
2. Re-measure the current codec under that protocol. That number is the honest baseline.
3. Correct the main README and the paper's claims to match it.
4. Amend the v2 plan:
   - its gates use the fixed protocol;
   - the I-frames are integer too;
   - it pads to any frame size;
   - it detects scene cuts, and the fresh set includes rush_field_cuts;
   - it has rate control to a target bitrate.
5. Keep a written list of what a usable codec needs (any device, any frame size, real time,
   scene cuts, rate control, error detection, fair anchors, standard averaging, streaming
   bitrates, unseen test video). Every future result is checked against that list, and an
   unmet item is reported as a blocker, not as a footnote.
