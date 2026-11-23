# Plan: fixing the second audit's two material findings (fixed 2026-11-24, before any run)

The second audit ([`outputs/p12_audit2/README.md`](../outputs/p12_audit2/README.md)) found two
things over its 1-point rule. Both make our quoted numbers stricter than they should be. The
user asked to fix both. The audit plan's rule for a material finding: re-measure, and re-quote
old next to new.

No model is trained or changed, and no setting is tuned. The direction of both fixes is known
(they can only lower our BD-rate). Their size is what gets measured, and it is reported as it
comes out. If any number moves the other way, that is reported too.

## Fix 1: drop the unused quantization blocks from `.nvct` streams

**What is wrong.** Every stream from `p4_gate3.py` writes two per-channel quantization blocks
after the 56-byte header (64 channels × 8 bytes × 2 = 1,024 B). They belong to the old
I-frame coder. With wide I-frames (every current headline) the decoder never reads them.

**The change.** When wide I-frames are used, the header declares 0 intra and 0 residual
quantization parameters, and the blocks are not written. The header fields already exist, so
this is still format version 2 and every old stream still reads. The reader returns no
parameters for a count of 0. An old-I-frame decode of such a stream raises an error instead of
guessing. Old-I-frame streams are unchanged.

**Proof, fixed now. All of it must pass, or the change is not shipped and the old numbers
stand:**
1. **Unit tests:** a zero-block stream round-trips; a stream with blocks still reads as
   before; an old-I-frame decode of a zero-block stream raises.
2. **The 12 audit streams** (DAVIS bmx-bumps and car-turn × 5 λ; UVG Beauty × 1e-2, 1e-4) are
   re-encoded with the new code. Each new file must equal the kept old file with the two count
   fields set to 0 and bytes 56–1,079 removed, i.e. exactly 1,024 B smaller with byte-identical
   payloads. Each must decode bit-identically in a fresh process.
3. **Real re-runs** under the new container: the DAVIS TEST MSE set (as `p8_motion/test_motion`),
   the DAVIS TEST MS-SSIM set (as `p10_msssim/test_msssim`) and the UVG MSE set (as
   `p8_motion/uvg_motion`). Every video's stream must be exactly 1,024 B smaller than in the
   old run, with identical PSNR and MS-SSIM, and decode-exact.

**Re-quote.** Those three sets are quoted from the real re-runs. The UVG MS-SSIM set and both
MCL-JCV sets are not re-coded (about 11 h for no new information once 2 and 3 pass). Their bpp
is re-computed as (total bytes − 1,024 × videos) × 8 / pixels, which 2 and 3 show is exact.
They are labelled *derived*.

## Fix 2: feed the anchors at the real frame rate

**What is wrong.** `benchmark_parity.ffmpeg_round_trip` feeds x264/x265 every video at 30 fps.
x264's rate control depends on the frame rate. UVG is 120 fps, and on Beauty the real rate
needs 1.1–1.6% more bits for the same PSNR. So our 30 fps anchor is the harder one.

**The change.** A new `--real-fps` option takes each video's frame rate from its manifest
(`fps` per item) and fails if any video has none. The frame rate used goes into the report's
configuration. `prepare_uvg.py` records 120 fps for all 7 UVG videos (the UVG 1080p sources
are all 120 fps), and the existing UVG manifest gets the same field. MCL-JCV already records
24/25/30 per video. **DAVIS records no frame rate, so DAVIS anchors stay at 30 fps.** This is
stated wherever DAVIS is quoted.

**Runs.** UVG x264/x265 4:2:0 and 4:4:4 at 120 fps. MCL-JCV 4:2:0 and 4:4:4 at each video's
real rate. They use the same CRF lists and `--sequence-major`, and go to new folders
(`*_realfps`). The old anchor files are kept. **Check:** the 18 MCL-JCV videos that really are
30 fps must give byte-identical anchor files to the old run. That shows nothing but the frame
rate changed.

## What gets re-quoted

The current-codec headlines: DAVIS and UVG MSE set (p8) against x264 4:2:0 and 4:4:4, and the
UVG low band; the MS-SSIM set rows; the four MCL-JCV rows. The table gives old, after fix 1,
and after both fixes. Dated gate results (A2 gate 3, encode tuning, low-rate, the motion gate
itself) are records of what each gate measured at the time, and stay as they were.

## Order (one heavy job at a time, one detached script)

| Step | Where | Time |
|---|---|---|
| Unit tests, then proof 2 | GPU | about 15 min |
| DAVIS TEST MSE and MS-SSIM re-runs | GPU | about 15 min |
| UVG anchors at 120 fps, 4:2:0 then 4:4:4 | CPU | about 3 h |
| UVG MSE re-run | GPU | about 2 h |
| MCL-JCV anchors at real fps, 4:2:0 then 4:4:4 | CPU | about 11 h |
| Re-quote report | | minutes |

If proof 1 or 2 fails, the chain stops before any re-run. MCL-JCV goes last because it moves
numbers least. Stopping there leaves every other result complete.
