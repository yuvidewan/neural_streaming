# Fair evaluation: tooling and proofs (2026-11-28)

Plan: [`docs/PLAN_fair_eval.md`](../../docs/PLAN_fair_eval.md), with amendments 1 and 2. This
folder holds step 1 of amendment 2's order: the tooling and proofs 1, 2, 3 and 5. The long runs
(UVG and MCL-JCV originals, the ladder, animation, screen clips) come after the budget pilot.
No result against H.264 exists yet under this protocol.

## What was built

- [`src/nvc/evaluation/fair.py`](../../src/nvc/evaluation/fair.py): the protocol as a library.
  It covers reading and writing 8-bit 4:2:0 YUV, the two BT.709 conversions an RGB codec pays
  for, and the anchors. These are x264 and x265 on the original YUV, at preset medium with
  `-tune psnr`, written as raw `.264`/`.265` elementary streams. It also has PSNR per plane with
  the 6:1:1 weighting, PCHIP BD-rate with the old linear method as a check, the streaming band,
  and the coverage rule.
- [`scripts/fair_eval.py`](../../scripts/fair_eval.py): the steps as commands: `prepare-uvg4k`,
  `anchors` (with `--cover` for the coverage rule), `proof3`, `ceiling` (proof 4), `ours` (the
  current codec on YUV), and `report`.
- Padding in the current codec ([`scripts/p4_gate3.py`](../../scripts/p4_gate3.py)). A frame
  whose sides do not divide by 16 is padded at the right and bottom by repeating its edge
  pixels. It is coded at the padded size and cropped after decoding. The stream header records
  the original size and the padded size is derived from it. The block grid
  (`m10h_motion_compensation.TemporalStreamHeader.block_grid`) now rounds up; for sides that
  divide by 16 that is the old value. B-frame coding refuses sizes that would need padding.

## Proofs

| Proof | How | Result |
|---|---|---|
| 1. Padding changes nothing where it is not needed | UVG Beauty, 1920x1072, 96 frames, λ 1e-2 and 1e-4, encoded with `p4_gate3.py` and `m10h_motion_compensation.py` as at commit eb75a55 and as now ([`proof_padding.py`](proof_padding.py)) | **pass** at both λ: 667,755 B and 24,412,449 B, byte-identical, same reconstruction; decode in a fresh process exact at both ([`proof1_encode.json`](proof1_encode.json), [`proof1_decode.json`](proof1_decode.json)) |
| 2. Padding works | [`tests/test_padding.py`](../../tests/test_padding.py): 72x64, 70x98 and portrait 1080x1920 through the real codec (wide I-frames, A2 P-frames, bit-aware motion) | **pass**: all three decode exactly. The real 1920x1080 clip check runs on the pilot's UVG development videos (`fair_eval.py ours` checks every stream decodes exactly) |
| 3. The anchor path adds nothing | [`tests/test_fair_eval.py`](../../tests/test_fair_eval.py): x264 `-qp 0` and x265 `lossless=1` on a synthetic clip | **pass**, byte-identical. The real-video run is `fair_eval.py proof3`, first step of the pilot chain |
| 5. The metric code is right | the same test file: PSNR per plane on known errors, the 6:1:1 weighting, averaging PSNR (not MSE) over frames, PCHIP BD-rate on known cases (identical curves 0%, twice the rate +100%, half the rate −50%, a non-uniform case against a numeric integral), the band, coverage and extension rule, and the CLI end to end | **pass** (19 tests) |

One thing the tests showed: per-pixel random chroma does not survive 4:2:0 → RGB → 4:2:0.
Chroma is upsampled then averaged back, and out-of-gamut YUV clips in RGB. On a synthetic clip
with noisy, saturated colour, Y PSNR fell to 39.4 dB through the conversions alone, with no
coding at all. Real video
has smooth chroma, and proof 4 measures the real loss per video. Until then the size of our
codec's conversion handicap on real video is not known.
