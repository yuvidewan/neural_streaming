# Plan: side information (a hyperprior) for the learned P-frame coder

Status: gate 1 NOT PASSED 2026-11-14: hyperprior vs an equally fine-tuned plain coder is -0.4% P-frame BD-rate (bar -8%); z collapsed to 8 bits/frame. Stopped. See outputs/p3_hyperprior/README.md.

## Why

`outputs/p1_pframe/README.md`: after Vimeo chunks 4-10 and an I-frame every 30 frames, the
codec is **+69.1% vs x264** with real bytes. P-frames are about 84% of the stream. Better motion is closed
(`outputs/p2_motion/README.md`: free dense flow saves at most 19.9%, under its 25% bar).

The P-coder's prior guesses each latent's mean and spread (Laplace `mean, scale`) **only
from the warped reference** (`self.prior(context)` in `scripts/p1_pframe.py`). It cannot
know where the current frame differs from the reference: new content, occlusions, motion
errors. Those are exactly where the bits go.

A hyperprior lets the encoder send a small summary `z` of the latent first. The decoder uses
`z` together with the reference to predict `mean, scale`.
- Every learned codec that reaches x264/x265 has one (Ballé 2018; the DCVC line).
- On images it is worth 15-25% over a factorized prior.
- M23 measured a hand-made version (a transmitted coarse |residual| map, no network) at -3.46
  BD-rate points on the old codec. A trained one should do clearly better.

## Design

- `h_a`: y (64 ch, stride 16) → z (32 ch, stride 64). Two stride-2 convs and LeakyReLU.
- `z` prior: factorized, one Laplace per channel with a learned scale (and mean fixed at 0).
  It is range-coded with the gate 3 scale tables, so no new coder is needed.
- `h_s`: z_hat → features at stride 16, concatenated with `prior(context)`'s features. A
  1x1 fusion conv outputs `(mean, log_scale)`.
- **Warm start:** the stage 3 coder. The fusion conv starts so that its z-path weights are
  zero, so at step 0 the model is exactly the stage 3 coder plus a few bits for `z`. Training
  can only teach it to use `z`, as `from_intra` did for the reference.
- **Rate:** R = R_y + R_z. Both count in every gate.

## Gates (stop at the first failure)

1. **Pilot (local, DAVIS TRAIN, judged on VAL-B, about a day).**
   - Two copies of the stage 3 coder get the same fine-tune: same steps, data, seed and
     learning rate (the `p2_flow_finetune.py` recipe, block16 warps). One copy has the
     hyperprior and one does not.
   - Both are scored with the gate 1 motion-oracle loop (`p2_motion_oracle.py`: GOP 10 chained
     P-frames, deployed block16 motion paid, ideal bits including `z`).
   - **Passes if** the hyperprior copy cuts P-frame BD-rate by at least **8%** vs the plain
     copy.
   - **Why 8%:** half the low end of the image literature. A short fine-tune from a warm
     start understates a fully trained hyperprior, and M23's untrained version was worth
     about 3.5 points.
2. **Vimeo training.** Re-download Vimeo chunks (about 1.5 days for all 10, each deleted
   after use) and train the hyperprior coder from the gate 1 model, as P-coder gate 2 did.
   - The plain stage 3 coder does not get this extra training. Gate 1 is therefore the
     controlled test of the hyperprior itself. Gate 3 asks the product question: is the
     codec better? Some of any gate 3 gain may be the extra epoch, and the write-up must say so.
   - Gate 2 itself is a sanity check, not a pass/fail.
3. **Real bytes (gate 3 harness).** `z` is range-coded ahead of each P-frame's `y`, and every
   stream must decode exactly.
   - **Passes if** BD-rate vs x264 beats the current best (+69.1%, GOP 30) by
     **more than 5 points**, at the same GOP.

## Cost and risks

- **Build:** gate 1 is about a day of coding plus a few hours of laptop training. Gate 2 is
  about 1.5 days of laptop training. Gate 3 is about half a day of codec work.
- **Decoder cost:** two small conv stacks per P-frame. That is negligible next to synthesis.
- **Low-rate risk:** at λ=1e-3, `z` bits can eat the gain. R_z is in every number, so the
  gates measure it.
- **Proxy risk:** the P-coder's ideal bits matched real bytes to within 0.02% in gate 3, and
  `z` is coded the same way. Gate 3's real-byte number is still the only one that counts.
