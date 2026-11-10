# Plan: better motion for the learned P-frame coder

Status: gate 1 (motion oracle) NOT PASSED 2026-11-10: free dense flow saves only 4.1% of P-frame bits (bar 25%); with the P-coder fine-tuned on dense warps (fairness check) it saves 19.9%, still under the bar. Motion line closed; block motion stays. See outputs/p2_motion/README.md. Learned-motion pilot not started.

## Why

`outputs/p1_pframe/README.md`, gate 3: the learned P-coder took the codec from +127.1% to
+91.1% vs x264, but the gain came as quality, not as smaller P-frames. P-frames still cost
0.57-0.67x an I-frame, and their residuals are still 72-81% of the stream. The P-coder sees
only `x_warp`, the previous reconstruction moved by **integer 16x16 block shifts**
(`m10h_motion_compensation.py`). Anything that rotates, zooms, deforms or moves at sub-pixel
or sub-block scale is wrong in `x_warp` and has to be paid for as residual. The learned
codecs that reach x264/x265 (DVC, FVC, DCVC) all use dense, sub-pixel, learned motion.

Motion today costs ~250 bytes per P-frame (4-10% of the stream), so cheaper motion bits are
not the prize. A better `x_warp` that shrinks the residual is.

## Gates (stop at the first failure)

1. **Motion oracle (local, no training, an afternoon).** How much would better motion
   help this P-coder at all? On VAL-B, chain GOP-10 P-frames with the gate 3 P-coders and
   score P-frame bits (motion + residual, ideal code length) and PSNR under:
   - `block16`: the deployed motion, with real motion bytes (the baseline);
   - `block8`: 8x8 integer blocks, with the motion coded by the deployed coder;
   - `dense_oracle`: dense Farneback optical flow with bilinear warping, **motion free** (not
     transmitted, so this is an upper bound for flow of that quality);
   - `zero`: no motion (the floor).

   Passes if `dense_oracle` cuts P-frame BD-rate by **at least 25%** vs `block16`. A learned
   motion coder has to pay for its flow (10-20% of the bits in the DVC line of work), so a
   free-flow gain smaller than that cannot survive transmission.
   Caveat: the P-coders were trained only on block-warped inputs. A smooth warp is a shift
   in their input, so the oracle understates what a jointly trained system could do. A
   near-miss is therefore not a clear fail.
2. **Learned motion pilot (local, DAVIS TRAIN, judged on VAL-B with the proxy).** A
   flow estimator and a flow autoencoder with its own Laplace prior, a bilinear warp and a
   small warp-refinement net, trained jointly with the P-coder on D + lambda*(R_flow +
   R_residual), warm-started from the gate 3 P-coder.
   - The flow estimator runs at the encoder only. The decoder receives the coded flow.
   - Passes if total P bits (flow + residual) drop by at least 15% at matched PSNR vs
     `block16` with the same P-coder.
   - A pretrained flow net (RAFT-small from torchvision) needs a weights download, which
     the user approves first. Otherwise the estimator trains from scratch with a
     warp-photometric loss.
3. **Vimeo training plus real bytes.** Train on Vimeo as P-coder gate 2 did, then run gate 3
   with the flow payload coded by the range coder. Every stream must decode bit-exact.
   - Bilinear warping is float, so encoder and decoder must run it identically under
     `deterministic_kernels()`. That gives same-machine exactness, the existing bar.
   - Passes if BD-rate vs x264 beats the gate 3 result (+91.1%, or the chunk 4-10
     retrained result if better) by more than 10 points.

## Cost and risks

- **Build:** gate 1 is an afternoon. Gate 2 is about 2-3 days of coding plus several hours of
  laptop training. Gate 3 is about 1.5 days of laptop training and a day of codec work.
- **Decoder cost:** decoding now runs a flow decoder and a warp per P-frame.
- **Flow bits** can eat the gain at low rate (the 1e-3 point). The rate term decides that,
  and gate 3 measures it.
- **Proxy risk:** gate 3 already showed that the proxy's P/I ratio (0.45x) did not carry over
  to real bytes (0.57-0.67x). Gate 3's real-byte BD-rate is the only number that counts.
