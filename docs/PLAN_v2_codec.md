# Plan: our second-generation codec, v2 (fixed 2026-11-26, before any code or run)

> **Partly superseded (2026-11-26), read first.** `docs/REQUIREMENTS.md` R7 now caps the whole
> decoder at 10,000 multiply-adds per pixel, so that it runs in real time on integrated graphics.
> DCVC-RT's decoder is about 172,000 per pixel, about 17 times over. The "Model size" paragraph
> below (about 20 M parameters, or a small variant with 192/256 channels) is void. The design
> gets re-sized in amendment 2, written after the fair evaluation (`docs/PLAN_fair_eval.md`), and
> that amendment also moves gates 3 and 6 to the fair protocol. No v2 code before then. Whether a
> decoder this light can beat H.264 at all is now the first thing v2 must test.
> Gate 2's bar (25% fewer bits than the current codec) has to go too: a decoder about 40 times
> lighter than the current one is unlikely to beat it, and the goal is H.264, not our old codec.
> Amendment 2 judges v2 against H.264 under the fair protocol at every gate.

## Why

The goal is to beat H.264 on every video, in real time. The current codec
(`outputs/p13_audit_fixes/README.md`) needs 20.2% fewer bits than x264 4:2:0 on UVG and 2.3%
fewer on MCL-JCV, but 26.7% more on DAVIS (x264 at preset medium, scored in RGB PSNR). It wins on
27 of 46 test videos. It loses at low rates, against x264 4:4:4, and on textured scenes such as
cows, schoolgirls and MCL-JCV SRC13 and SRC20. It decodes 1080p at about 2 frames/s.

The published DCVC-RT (`outputs/p14_dcvcrt_check/README.md`) beat x264 on all 16 DAVIS and UVG
videos, against every arm including 4:4:4. It needed 54.5% (DAVIS) and 76.2% (UVG) fewer bits
than our codec. On the laptop, with no fused kernels, it encoded 1080p 13 times faster than ours
and decoded it 4.5 times faster. Its limit was quality: its top rate point stays 0.9 to 4.1 dB
below ours, so all of those wins are at low and middle quality.

DCVC-RT's code (`C:/dcvc_scratch/DCVC-RT/src/models`) shows what it does differently. It sends no
motion vectors. The decoder keeps the previous frame's decoded features (256 channels at 1/8
resolution), and the next frame is coded given them. Our own motion oracle pointed the same way:
a free dense flow field saved only 4.1% of P-frame bits (`outputs/p2_motion`). Its coded latent
is 128 channels at 1/16 resolution, 0.5 values per pixel, plus a hyperprior of 128 channels at
1/64. Ours codes 0.75 values per pixel, so the gain comes from the transform, the temporal context
and the training, and not from coding more values. Its layers are depthwise-separable blocks on
input that is pixel-unshuffled by 8 in one step. One model covers 64 quality levels through
learned per-level scale vectors, and the network runs in integer arithmetic so that decoding is
identical across devices.

A correction to earlier notes: `docs/PLAN_dcvcrt_check.md`, the first version of the 2026-11-26
paper and the 2026-11-14 research survey say DCVC-RT codes "a larger latent at 1/8". That is
wrong as written. 1/8 is its feature resolution; the coded latent is at 1/16.

## What gets built

A new codec, v2, lives in `src/nvc/v2/` with scripts named `scripts/v2_*.py`. The current codec
stays the shipped codec and the baseline until v2 passes gate 2.

P-frames use conditional coding in the feature domain, along DCVC-RT's lines. The input frame is
pixel-unshuffled by 8 and depthwise blocks work at 1/8. The temporal context is the previous
frame's decoded features, with no motion search and no motion bits. The coded latent is at 1/16
with a hyperprior at 1/64. The prior predicts a mean and a scale from the hyperprior and the
temporal context, in two spatial steps (checkerboard), and the latent is rounded with a unit step
around the predicted mean.

One model covers every rate, with a learned scale vector per quality level. Training samples *λ*
over a range that covers our current 10^-2 to 10^-4 and goes beyond it at the top, so that 1080p
can reach at least 41 dB. DCVC-RT's ceiling is the thing to beat here.

Gates 1 and 2 keep our five grey-reference wide I-coders, one per *λ*. For each of the five rate
points, the v2 quality level is the one whose P-frame PSNR is closest to that I-coder's, picked on
VAL-B. A new variable-rate I-model is built only if I-frames are more than 25% of the stream at any
rate point at gate 2; the share is reported either way. The GOP stays at 30, low-delay, with no
B-frames.

The bitstream is a new `.nvct` version carrying the model identity and one quality level per
frame, coded by the existing C range coder with scale tables. Every stream at every gate must
decode bit-exactly on the same GPU in a separate process.

Layer definitions may be taken from github.com/microsoft/DCVC under its MIT licence, with the
notice kept. No DCVC weights are used, copied or distilled. Every v2 weight is trained by us, from
random initialization or from our own models.

## Training (laptop RTX 5060, 8 GB)

The data is Vimeo-90K septuplets in the same 10 chunks as before. Each chunk is downloaded, used
and deleted before the next. Crops are 256×256, the largest the 448×256 frames allow. Training
runs in two stages: first one P-step from an I-coded reference, then cascaded steps unrolled over
3 and then 6 frames on the model's own reconstructions, so that it learns to stop error growth
across a GOP. It uses bf16 autocast and gradient-norm clipping at 1.0 (the stage 1 run of
October diverged without it), saves a checkpoint every 30 minutes, and every stage resumes.

The model starts at DCVC-RT's layer sizes, about 20 M parameters for P-frames. If the step rate
measured in the first hour puts gate 1's budget above 4 days, it switches to a small variant with
192 feature channels and 256 reconstruction channels. That choice is made on speed alone, before
any quality number exists. Rough budgets: gate 1 takes 3 Vimeo chunks and 2 to 4 days; gate 2
takes all 10 chunks and 2 to 3 weeks. Both are re-estimated from the first day's step rate and
reported.

## Gates

They run in order and stop at the first failure. Every bar is fixed now.

1. Pilot on VAL-B. Closed loop, GOP 30, ideal code lengths. Both arms share the same wide
   I-frames; the A2 arm pays for its motion bits and v2 has none. It passes if v2's P-frame
   BD-rate (PSNR) against the current A2 P-coder is −15% or better over the overlap. The A2 coder
   had all 10 chunks of training and v2 only 3, so the bar favours the baseline. v2's top P-frame
   PSNR and its 1080p decode speed are reported, not judged.
2. The real codec on DAVIS TEST and UVG, in real bytes, decode-exact. It passes if two things
   hold: the pooled BD-rate (PSNR) against the current codec is −25% or better on both sets, and
   the top pooled point reaches at least our current top, 37.9 dB on DAVIS and 40.9 dB on UVG. The
   second condition is the lesson of the DCVC-RT check: wins below a low ceiling do not count.
3. Per video against x264 on DAVIS and UVG, with the current anchors (UVG at its real frame rate,
   preset medium) and coverage checked per video first. It passes if v2 beats x264 4:2:0 default
   on at least 15 of the 16 videos and x264 4:2:0 low-delay on all 16. x264 4:4:4 and x265 are
   reported. MCL-JCV is downloaded again and measured, report only.
4. Speed: 1080p encodes and decodes at 30 frames/s or more on this laptop. This gets its own plan,
   where fp16, fused kernels, CUDA graphs or TensorRT are allowed and the output must stay
   decode-exact. It is report only until gate 3 passes.
5. The same output on another device: decoding on a second device (the CPU, or another GPU) is
   bit-identical. This gets its own plan (integer arithmetic) and is report only until gate 3
   passes.
6. The final claim, on video nobody has looked at: the fresh set below plus MCL-JCV. "Beats H.264
   on every video" may be written only if v2 beats both x264 4:2:0 default and x264 4:2:0
   low-delay (preset medium, real frame rate) on every single video of both sets, with coverage
   checked and each video scored over its full overlap. x264 4:4:4 and x265 are reported next to
   it. If any video loses, the count of wins is the result.

## Fresh test set (fixed now, not downloaded yet)

The set is 22 camera-shot 1080p sequences from Xiph's derf collection
(media.xiph.org/video/derf/):

aspen, blue_sky, controlled_burn, crowd_run, dinner, ducks_take_off, factory, in_to_tree, life,
old_town_cross, park_joy, pedestrian_area, red_kayak, riverbed, rush_hour, snow_mnt, speed_bag,
station2, sunflower, touchdown_pass, tractor, west_wind_easy.

The animations (big_buck_bunny, elephants_dream, sita_sings_the_blues, sintel_trailer) and
rush_field_cuts, which has scene cuts, are left out. Each sequence is prepared like UVG and
MCL-JCV: the first 96 frames, converted to RGB and cropped to 1920×1072. Before any coding, the
frames are hash-checked against the 641,907 stored Vimeo-90K hashes and against UVG and MCL-JCV;
any sequence that matches is dropped and listed. Nothing is tuned or chosen on this set, and its
frames are not opened before gate 6 except for that check.

## What does not change

The current codec, its numbers and the paper stay as they are until gate 3 passes. The working
rules stay too: plans are committed before their runs, bars never move, failures are reported,
heavy jobs run one at a time, and commits carry no Co-Authored-By trailer.

## Risks

Microsoft trained DCVC-RT on several GPUs, with longer sequences and 512×512 crops. On one laptop
GPU, v2 may land well short of it; gate 1 is sized to find that out in days. The 256×256 crops may
also undertrain fine 1080p detail. If gate 2 fails only on the ceiling condition, an extra stage
on higher-resolution training video (BVI-DVC, for example) gets its own plan; it is not added to
this one after the fact. The whole plan takes weeks, and every stage resumes from its
checkpoints.

## Amendment 1 (2026-11-26, before any code or run): the same output on every device

A stream has to decode to the same frames on any machine, or the codec cannot be used outside
the computer that made it. The current codec decodes identically only on the GPU that encoded
it: in the second audit, decoding on the laptop's CPU matched 0 of 10 times. The cause is
floating point. Two devices add the same numbers in different orders, the predicted scale of a
latent value moves in its last digits, a value near a table boundary picks a different table,
and the range decoder loses its place. Each decoded frame is also the next frame's reference, so
small differences grow through the GOP. The original plan checked this at gate 5, after the
compression gates. Converting a finished floating-point model to integers at the end could cost
compression that gates 2 and 3 had already counted, so the requirement moves to the start.

What changes in the design. Every part of the decoder runs in exact integer arithmetic at
inference: the hyperprior synthesis, the prior that predicts means and scales, the temporal
context, and the reconstruction that becomes the next frame's reference. Weights and activations
in these parts are quantized to integers with fixed scales, sums are exact, and each rescaling
step uses an integer multiply and a fixed rounding rule, so that the result does not depend on
the order of the arithmetic. The table index for each latent value is computed from integers.
The encoder may run in floating point for its analysis transform, but it rebuilds every
reference with the same integer decoder, so the two sides stay in step. Layers are chosen to be
integer-friendly from the start. Training runs in floating point with the integer scheme
simulated (quantization-aware) for the last part of each stage, as in Ballé et al. (integer
networks) and DCVC-RT. The implementation may use int8 matrix products with int32 accumulation,
or floating-point products of small integers whose sums stay exact; either way, cuDNN algorithms
that change the arithmetic (TF32, Winograd) are switched off in this path. The gates measure the
outcome, not the method.

What changes in the gates.
- Gate 1 adds a report: the P-frame BD-rate cost of the simulated integer decoder against the
  floating-point one on VAL-B. It is reported, not judged, so that the cost is known early.
- Gate 2 adds a third condition: every gate 2 stream decodes bit-identically on two devices,
  the laptop GPU and the laptop CPU. One mismatch fails the gate. The −25% bar is measured with
  the integer decoder, so the integer cost gets no allowance.
- Gate 5 changes from "report only" to a requirement before gate 6: the gate 3 streams decode
  bit-identically on a second physical machine with a different GPU model (the RTX 3060 desktop
  offered on 2026-11-26, or any other), as well as on its CPU.
- Gate 4 (speed) is measured on the integer decoder, since that is the decoder people would run.
