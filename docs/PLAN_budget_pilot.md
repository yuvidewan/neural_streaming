# Plan: the 10,000-operation decoder pilot (fixed 2026-11-27, before any code or run)

## Why this comes first

`docs/REQUIREMENTS.md` R7 caps the whole decoder at 10,000 multiply-adds (MAC) per pixel, so that
it decodes in real time on integrated graphics. The feasibility review
(`docs/RESEARCH_light_codecs_2026-11.md`, checked at source in its Section 11) found no published
codec that meets this budget, encodes in real time and beats H.264. The lightest decoder shown at
VVC low-delay quality is about 72,000 MAC per pixel. So whether R7 and R1 (beat H.264) can be met
together is the biggest unknown in the project. If they cannot, the device target has to change,
and that changes everything after it.

This pilot answers that question in about five days, before any long run. It is a test, not the
v2 codec, and its stop rule is fixed below.

## What gets built

A small conditional codec, written in this repository (`src/nvc/pilot/`, script
`scripts/pilot_budget.py`). One network codes both kinds of frame: an I-frame is coded with an
empty temporal context, the same trick our current codec uses with its grey reference, so the
whole decoder is inside the budget.

The design follows DCVC-RT's lines (the review's Direction 1):
- the RGB frame is pixel-unshuffled by 8, and depthwise-separable blocks work at 1/8 resolution;
- the temporal context is the previous frame's decoded features at 1/8, with no motion vectors;
- the coded latent is at 1/16, with a hyperprior at 1/64;
- the prior predicts a mean and a scale from the hyperprior and the temporal context, in two
  spatial (checkerboard) steps;
- rounding is unit-step around the predicted mean;
- one model covers every rate, with learned per-level scale vectors, trained over a λ range wide
  enough to reach 41 dB at 1080p.

Layer code may come from microsoft/DCVC under its MIT licence. No DCVC weights are used.

### The budget, enforced in code

- **Decoder:** at most 10,000 MAC per pixel for an I-frame and for a P-frame, counting everything
  the decoder runs per frame: hyperprior synthesis, prior networks, the temporal-context path and
  the reconstruction.
- **How it is counted:** PyTorch's FLOP counter at 1920×1088, with MAC = FLOP / 2.
- **The check:** a unit test fails if either frame type goes over the budget. A pilot whose decoder
  is over budget is not a valid result.
- **Encoder:** at most 60,000 MAC per pixel, so that real-time encoding on a desktop GPU stays
  possible later.
- **Layer widths:** chosen to fit the budget and written into a config file committed before
  training starts. The choice is made on the budget alone, before any quality number exists.

## Training

- **Data:** Vimeo-90K septuplets, at most 2 chunks. Each chunk is downloaded, used and deleted
  (asking first, with sizes).
- **Settings:** 256×256 crops, bf16, gradient clipping at 1.0, checkpoints every 30 minutes.
- **Stage 1:** I-frames and one P-step, λ sampled across the range.
- **Stage 2:** three unrolled P-steps on the model's own reconstructions.
- **Hard cap:** 72 GPU hours in total on the laptop's RTX 5060. Whatever exists at the cap is
  evaluated. No extension is allowed after seeing a result.

## Evaluation

### Development video

The pilot is judged on video that no test or decision has used. The UVG collection has 9 videos
outside our 7-video test set, and they are only available in 4K. Four are fixed now, chosen for
variety before any result:

| Video | Content | 4K archive |
|---|---|---|
| CityAlley | city street, motion | 3.16 GiB |
| FlowerKids | people, fine texture | 2.87 GiB |
| RaceNight | night, fast motion | 3.68 GiB |
| SunBath | outdoor, people | 1.17 GiB |

That is 10.88 GiB to download, asking first. Each archive is deleted once its frames are out.

Each video is prepared the same way:
- the first 96 frames, at 50 frames/s;
- downscaled from 3840×2160 to 1920×1080 in 4:2:0 with
  `scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int`;
- kept as YUV 4:2:0 for the anchors.

Downscaled 4K is sharper and less noisy than native 1080p, and the write-up says so. This set is
never used to claim anything against H.264. It only decides this pilot.

### Protocol

The fair protocol of `docs/PLAN_fair_eval.md` applies:
- x264 and x265 get the original YUV, at preset medium with `-tune psnr`, in both the default and
  the low-delay structure;
- scoring is in YUV PSNR, weighted 6:1:1;
- BD-rate is computed per video with PCHIP, after the coverage check.

The pilot codes RGB, so it pays for the conversion both ways. It uses an I-frame every 30 frames
and P-frames only. The current codec is coded on the same videos as a reference point.

### Rate

The pilot's rate is the ideal code length from its prior. To keep this honest, one video at two
rate points is also coded with the real range coder, and the real size must be within 1% of the
ideal one. If it is not, all pilot rates are scaled by the measured ratio before scoring.

### Speed (reported)

- Decoder frames per second at 1080p on the RTX 5060 (PyTorch, fp16).
- Decoder frames per second at 720p on the laptop's Intel UHD Graphics, through ONNX Runtime with
  DirectML. R7's target for weak integrated GPUs is 720p at 30 frames/s, and the gap to it is
  reported.

## Verdict (fixed now)

The score is the mean, over the four videos, of the per-video BD-rate (YUV PSNR) against x264
low-delay with `-tune psnr`. That is the like-for-like anchor for a P-only codec with an I-frame
every 30 frames.

| Mean BD-rate against x264 low-delay | Verdict | What happens |
|---|---|---|
| +10% or better | **GO** | write v2 amendment 2 around this budget and train it fully |
| between +10% and +40% | **GRAY** | report to the user; options are longer training, coding YUV directly, or a larger budget |
| worse than +40% | **STOP** | the 10,000 MAC/pixel budget is judged out of reach for this project now. No further v2 work until the user re-decides R7 (for example a larger budget, or a discrete GPU target) |

Why these lines: in this project's history, more training and better recipes on a fixed design
gave about 10% to 30% (stage 3: −11%; the encode tuning: −6%; the detail latent with extra
training: −31%). A pilot within +10% after a capped, short training can probably be pushed past
x264 by full training. One worse than +40% very probably cannot.

The verdict only counts if both hold: the decoder budget test passes, and the rate check is
within 1% (or the rates were rescaled).

## What the pilot does not test

- identical decoding across devices (R9);
- screen and animation content (R6: the training data is camera only);
- real-time encoding (R8);
- coding YUV directly instead of RGB;
- any claim on the test sets (UVG test, MCL-JCV, fresh Xiph).

These stay open in the requirements list.

## Order and time

1. Code:
   - the fair-evaluation tooling this pilot needs: the YUV pipeline, the anchors, the metrics,
     PCHIP BD-rate and their tests (proofs 3 and 5 of the fair-evaluation plan);
   - the pilot model, with its budget test.

   About a day and a half.
2. Download the development set (asking first) and run its anchors on the CPU. About an hour.
3. Download the Vimeo chunks (asking first). Train, at most 72 GPU hours.
4. Evaluate the pilot and the current codec on the development set, then give the verdict.
   Half a day.

That is about five days in all. The long runs of the fair evaluation (the current codec on UVG,
MCL-JCV, the ladder, animation and screen clips) come after the verdict, whatever it is.

## Amendment 1 (2026-11-28, after the code and before any data or result)

The code is written and tested. These choices were left open by the plan and are fixed here,
from the MAC budget and measured speed only:

- **Layer widths** (`configs/pilot_budget.json`): 96 feature channels at 1/8, a 128-channel
  latent at 1/16, a 64-channel hyperprior at 1/64, 8 synthesis blocks, 2 context blocks, 6
  encoder blocks of 160 channels. Counted by `tests/test_pilot_budget.py`: the decoder is
  **9,029 MAC per pixel** for an I-frame and for a P-frame (limit 10,000), the encoder 20,357
  (limit 60,000). 2.24 million parameters.
- **Rates:** 64 levels, λ log-spaced from 0.0625 down to 1/8192 in the repository's
  MSE + λ·bpp form. Our current codec spans 1e-2 to 1e-4, so the pilot reaches lower rates for
  the streaming band and about the same top quality.
- **Entropy model:** Laplace, the current codec's range coder and tables, so the rate check
  uses real code that already ships.
- **Points scored:** levels 0, 9, 18, 27, 36, 45, 54 and 63. The rate check uses CityAlley at
  levels 18 and 45.
- **Training data:** Vimeo-90K chunks 4 and 5 (8.70 and 8.86 GiB).
- **Schedule** (`configs/pilot_schedule.json`), from measured speed on random data (stage 1:
  52,600 steps per hour at batch 16; stage 2: 38,200 at batch 8):
  - chunk 4, stage 1 (I-frame and one P-frame), batch 16, 22 hours;
  - chunk 5, stage 1, batch 16, 22 hours;
  - chunk 5, stage 2 (I-frame and three P-frames on the model's own decoded features), batch 8,
    24 hours, learning rate 1e-4 cut to 1e-5 for the last 6 hours.

  That is 68 GPU hours, under the 72-hour cap. Chunk 4 is deleted after its phase, chunk 5
  after the last one.
- **Speed** on the RTX 5060 is measured for the decoder networks in fp16 at 1088x1920. The
  720p DirectML number on the Intel UHD Graphics needs the `onnxruntime-directml` package,
  which is asked for separately before that step. Its absence does not change the verdict.
