# Light learned codecs: can one beat H.264 inside 10,000 multiply-adds per pixel? (2026-11-26)

An evidence review for v2 amendment 2. The question: has any published learned video codec
beaten H.264 (x264, ideally x265) on standard test sets with a whole decoder of at most about
10,000 multiply-adds (MAC) per pixel, while encoding fast enough for live streaming (so not a
codec that fits a network to each video)? And which designs come closest? The answer decides
whether `docs/REQUIREMENTS.md` R7 (decoder at most 10,000 MAC/pixel, real time on integrated
graphics) and R1 (beat H.264 under `docs/PLAN_fair_eval.md`) can be met together.

**Method.** Papers were read in full text (arXiv HTML or ar5iv) on 2026-11-26. Numbers are the
papers' own unless tagged:
- *derived*: my arithmetic on the paper's numbers, shown where it is used;
- *chained*: BD-rates combined across different anchors, test sets or papers. Indicative only,
  because BD-rates are not exactly transitive;
- *unconfirmed*: seen only in a search-engine summary, or read off a plot, not in the paper's text.

"kMAC/px" is thousands of multiply-adds per decoded pixel. A negative BD-rate means fewer bits
than the anchor. `[R]` marks this repository's own measurements. Nothing was trained,
downloaded or run for this review.

## 1. The answer

**No published codec meets all three conditions.**

- Every decoder under the budget belongs to a codec that fits a model to each video:
  - Cool-chic video, 0.9 to 1.3 kMAC/px ([[13]](https://arxiv.org/abs/2402.03179), [[14]](https://arxiv.org/abs/2501.16976));
  - C3, 4.4 kMAC/px ([[12]](https://arxiv.org/abs/2312.02753)).

  Their encoders take about 9 to 20 minutes per 1080p frame (Cool-chic, [[13]](https://arxiv.org/abs/2402.03179), [[14]](https://arxiv.org/abs/2501.16976)) or up to
  110,000 optimisation steps per video patch (C3, [[12]](https://arxiv.org/abs/2312.02753)). Live 1080p30 encoding would need
  roughly 16,000 to 36,000 times faster encoding (*derived*: 540 s and 1,200 s against 33 ms).
- Among codecs that beat x264, or an anchor that x264 trails, and that publish their cost, the
  lightest decoders are 4 to 8 times over budget:
  - KD-NVC-T: 71.7 kMAC/px ([[10]](https://arxiv.org/abs/2606.04595));
  - DCVC-UF (low-delay): about 82 (*derived*, [[9]](https://arxiv.org/abs/2606.04410));
  - Tencent's P-frame decoder: 39 or 78, depending on a resolution the paper does not state
    (*derived*, [[4]](https://arxiv.org/abs/2309.11276)).

  Tencent's I-frame decoder is heavier still: 82 or 163 kMAC/px (*derived*, [[4]](https://arxiv.org/abs/2309.11276)).
- The only real-time, generalising design near the budget is Qualcomm's MobileNVC: a 24.5
  kMAC/px P-frame decoder and a 93.4 kMAC/px I-frame decoder. Against x265 preset fast it needs
  50.8% (HEVC-B) and 56.4% (MCL-JCV) more bits in floating point, and 159.2% and 192.6% more in
  its deployed 8-bit integer form ([[2]](https://arxiv.org/abs/2310.01258)). DCVC-RT's authors describe its compression ratio as
  "even lower than x264" ([[8]](https://arxiv.org/abs/2502.20762)).

**The closest designs**, by what each misses:

| Design | What it has | What it misses |
|---|---|---|
| MobileNVC ([[2]](https://arxiv.org/abs/2310.01258)) | real-time 1080p decoding on a phone, int8, 24.5 kMAC/px P-frames | compression: loses to x265 fast by 51% to 193% |
| Cool-chic video, improved ([[14]](https://arxiv.org/abs/2501.16976)) | 1.3 kMAC/px; beats x264-medium on 9-frame clips (*chained*) | encoding: about 20 min per 1080p frame on an H100 |
| KD-NVC-T ([[10]](https://arxiv.org/abs/2606.04595)), DCVC-UF LD ([[9]](https://arxiv.org/abs/2606.04410)) | about VTM low-delay quality or better, real time on desktop GPUs | 7 to 8 times over the decoder budget |
| TinyLIC Nano ([[19]](https://arxiv.org/abs/2610.22421)), I-frames only | 10 kMAC/px; 14.8% behind VTM intra on Kodak (its standing against BPG is unconfirmed, Section 11) | images only; no video codec built on it |

**R7 and R1 together: not demonstrated by anyone. Plausible only for some designs; this is the
main research risk of v2**, as REQUIREMENTS C1 already says. Section 7 has the evidence on both
sides.

**R9, identical decoding on every device: also not demonstrated across GPU vendors.** Microsoft's
MLVC team reports that DCVC-RT, "even in its quantized form", decodes to corrupted output when
encoded on an Apple M3 NPU and decoded on an Intel Lunar Lake NPU (the paper does not say int16
for this test). They also report that integer networks are not bit-exact across
NPU vendors ([[6]](https://arxiv.org/abs/2606.28027)). DCVC-RT itself was tested only NVIDIA to NVIDIA ([[8]](https://arxiv.org/abs/2502.20762)). None of the papers
reviewed shows bit-identical decoding across NVIDIA, Intel and AMD GPUs.

## 2. Video codecs at a glance

"Beats x264?" is answered from the paper's own anchors. "Same output" is the paper's mechanism
for identical decoding on different devices, and its test.

| Codec | Venue | Type | Decoder kMAC/px (how counted) | Decode / encode speed | Headline BD-rate (anchor and conditions) | Beats x264? | Code / weights | Same output on other devices |
|---|---|---|---|---|---|---|---|---|
| MobileCodec, Qualcomm ([[1]](https://arxiv.org/abs/2207.08338)) | ACM MMSys 2022 | generalising; I + P; conv motion, no warping | P 257.1, I 130.9 (per input pixel, 2-frame sequence) [[1]](https://arxiv.org/abs/2207.08338); DeepSpeed recount at 1080p: P 259.1, I 130.9 [[2]](https://arxiv.org/abs/2310.01258) | 31 fps decode at 720p (padded to 768×1280), Snapdragon 8 phone; encode offline [[1]](https://arxiv.org/abs/2207.08338) | paper: plot only, UVG, against FFmpeg x264/x265 ultrafast and medium, zerolatency [[1]](https://arxiv.org/abs/2207.08338). Later, in AIMET simulation (not on a device): int8 +396.7% (HEVC-B), +549.0% (MCL-JCV) vs x265 fast [[2]](https://arxiv.org/abs/2310.01258) | No | none linked | 8-bit fixed point, integer entropy tables; no cross-device test reported [[1]](https://arxiv.org/abs/2207.08338), [[6]](https://arxiv.org/abs/2606.28027) |
| MobileNVC, Qualcomm ([[2]](https://arxiv.org/abs/2310.01258)) | WACV 2024 | generalising; block warping and flow extrapolation; YUV 4:2:0 | P 24.5, I 93.4 (receiver; DeepSpeed at 1080p YUV 4:2:0; warping not counted) [[2]](https://arxiv.org/abs/2310.01258) | 38.9 fps decode at 1080p, Snapdragon 8 Gen 2; encode about 3 fps, unoptimised [[2]](https://arxiv.org/abs/2310.01258) | int8: +159.2% (HEVC-B), +192.6% (MCL-JCV); fp32: +50.8%, +56.4%. Anchor: FFmpeg x265 preset fast, B-frames off, YUV 6:1:1 PSNR; codec GOP 16; points above 1.3 Mb/s dropped [[2]](https://arxiv.org/abs/2310.01258) | No ("even lower than x264", [[8]](https://arxiv.org/abs/2502.20762)) | none linked | int8 on the phone's NPU; no cross-device test [[6]](https://arxiv.org/abs/2606.28027) |
| Tencent CIT ([[4]](https://arxiv.org/abs/2309.11276)) | ACM MM 2023 | generalising; pruned DCVC-family codec | P decoder 76.3 G MAC/frame, I decoder 160.7 G [[4]](https://arxiv.org/abs/2309.11276); per pixel P 39 or 78 (*derived*; count resolution not stated) | 25.2 fps decode (RTX 2080, fp32, 1280×768); 28.1 fps encode (A100) [[4]](https://arxiv.org/abs/2309.11276) | −25.6% (same device), −24.2% (cross-device) vs FFmpeg x265 medium; UVG, 96 frames, 1920×1024 crops, codec GOP 12. In the same setup H.264 is +45.8% vs x265 [[4]](https://arxiv.org/abs/2309.11276) | Implied yes (*chained*) | none linked | floating point plus a side stream of "calibration" coordinates; 0% failures on one UVG clip over 2 CPUs and 2 NVIDIA GPUs, fp32 [[4]](https://arxiv.org/abs/2309.11276); fails in fp16 [[6]](https://arxiv.org/abs/2606.28027) |
| ELF-VC, WaveOne ([[7]](https://arxiv.org/abs/2104.14335)) | ICCV 2021 | generalising; flow and residual; motion predicted from the past | not reported; decoder 11 M parameters, encoder 38 M [[7]](https://arxiv.org/abs/2104.14335) | Titan V: 18 fps decode, 10 fps encode at 1080p; 35 and 19 at 720p [[7]](https://arxiv.org/abs/2104.14335) | UVG −44.3%, MCL-JCV −33.7% vs x264; −26.1%, −17.0% vs x265. FFmpeg preset medium, CRF, B-frames off, default tuning and GOP; scored in RGB PSNR; codec GOP 16; full sequences; one BD-rate per dataset curve [[7]](https://arxiv.org/abs/2104.14335) | Yes as reported, but scored in RGB against 4:2:0 anchors (our flaw 1) | none linked | not addressed |
| DCVC-RT, Microsoft ([[8]](https://arxiv.org/abs/2502.20762)) | CVPR 2025 | generalising; conditional coding at 1/8; no motion vectors | 174.9 decode side (*derived*: 31.5 entropy model + 34.0 decoder + 53.0 feature extractor + 56.4 reconstruction, 1080p) [[8]](https://arxiv.org/abs/2502.20762); 166.8 by fvcore [[10]](https://arxiv.org/abs/2606.04595); about 172 [R] | fp16, 1080p: 112.8 fps decode, 125.2 encode (A100); 34.1 and 39.5 (2080 Ti). int16: 38.8 and 52.3 (RTX 4090) [[8]](https://arxiv.org/abs/2502.20762). Our RTX 5060 laptop without fused kernels: 9.9 and 10.6 [R] | −21.0% vs VTM-17.0 low-delay (int16: −18.3%); YUV 4:2:0, intra period −1, all frames, real bitstreams; HM-16.25 is +42.4% in the same table [[8]](https://arxiv.org/abs/2502.20762). Our check: −47.9% (DAVIS), −65.7% (UVG) vs x264 4:4:4, RGB, pooled [R] | Yes [R], but its top quality is 0.9 to 4.1 dB below ours [R] | code and weights, MIT | int16 integer inference with int32 sums and a sigmoid table; tested A100 to 2080 Ti [[8]](https://arxiv.org/abs/2502.20762); corrupted output from an Apple M3 NPU to an Intel NPU [[6]](https://arxiv.org/abs/2606.28027) |
| DCVC-UF, Microsoft ([[9]](https://arxiv.org/abs/2606.04410)) | CVPR 2026 | DCVC-RT lineage; low-delay (LD) and 8-frame chunk (HT) variants | LD 170 G MAC/frame, about 82; HT-S 211 G, about 102; HT-L 343 G, about 165 (*derived* per pixel at 1080p) [[9]](https://arxiv.org/abs/2606.04410) | LD: 353.8 fps decode, 313.6 encode, 1080p, RTX 4090 [[9]](https://arxiv.org/abs/2606.04410) | LD −9.5%, HT-S −31.6%, HT-L −42.2% vs VTM-17.0 low-delay; YUV 4:2:0, intra period −1; HT adds up to 7 frames of delay [[9]](https://arxiv.org/abs/2606.04410) | not tested; likely (*chained*) | code in microsoft/DCVC (per its README; not checked) | not addressed in the text read |
| KD-NVC ([[10]](https://arxiv.org/abs/2606.04595)) | arXiv 2606.04595, 2026 | DCVC-RT distilled by architecture search | T: 71.7 decode, 93.7 encode. S: 93.7, 109.2 (fvcore, 1080p) [[10]](https://arxiv.org/abs/2606.04595) | T: 69.2 fps decode, 54.8 encode, 1080p, RTX 5060 with CUDA kernels; 36.0 fps decode at 480p on Apple M3 [[10]](https://arxiv.org/abs/2606.04595) | T: +0.7% (intra period −1), −1.5% (intra period 32) vs VTM-LDB-23.11. S: −11.1%, −9.0%. YUV 4:2:0 [[10]](https://arxiv.org/abs/2606.04595) | not tested; likely (*chained*) | none seen | not addressed; integer inference is future work [[10]](https://arxiv.org/abs/2606.04595) |
| MLVC / MLVC-S, Microsoft ([[6]](https://arxiv.org/abs/2606.28027)) | arXiv 2606.28027, 2026 | DCVC-RT lineage, built for laptop NPUs | about 283 and 99 kFLOP/px decode (*derived* from 587.1 and 205.7 GFLOPs per 1080p frame; FLOP-to-MAC convention not stated) [[6]](https://arxiv.org/abs/2606.28027) | MLVC-S at 1080p: about 27 fps decode, 30 encode, averaged over Apple, Intel and Qualcomm NPUs [[6]](https://arxiv.org/abs/2606.28027) | −52.0% and −36.5% (PSNR, YUV 4:2:0) vs Intel QSV hardware HEVC, preset fast, one I-frame; average of HEVC B–E and VCD 720p, same platform. x265 fast is +7.1% vs this anchor on HEVC-B [[6]](https://arxiv.org/abs/2606.28027) | not tested; likely (*chained*) | code (github.com/microsoft/mlvc) | not bit-exact by design: scale indices are sent so entropy decoding cannot lose sync; most cross-vendor pairs within 2 BD points [[6]](https://arxiv.org/abs/2606.28027) |
| UI2C, USTC ([[11]](https://arxiv.org/abs/2510.14431)) | arXiv 2510.14431, v5 2026 | DCVC-RT lineage; one model for intra and inter; two frames coded jointly | 233 decode, 157 encode (DeepSpeed) [[11]](https://arxiv.org/abs/2510.14431) | 46.1 fps decode, 65.1 encode, 1080p, RTX 3090 [[11]](https://arxiv.org/abs/2510.14431) | −12.1% vs DCVC-RT (VTM-17.0 is +23.6% vs DCVC-RT in the same table); estimated entropy, not real bitstreams; low-delay, YUV 4:2:0, intra period −1 [[11]](https://arxiv.org/abs/2510.14431) | not tested; likely (*chained*) | code (github.com/ihuixiang/UIIC) | not addressed |
| Cool-chic video, Orange ([[13]](https://arxiv.org/abs/2402.03179)) | DCC 2024 | overfitted, one tiny decoder per frame | 0.924 worst case (B-frame) [[13]](https://arxiv.org/abs/2402.03179) | encode about 9 min per 1080p frame; decode speed not reported [[13]](https://arxiv.org/abs/2402.03179) | no BD number: better than x264 at low rates, worse at high rates. CLIC24 validation, 30 videos, 33 frames; x264/x265 medium, `-tune psnr`, keyint 64; YUV 4:2:0 [[13]](https://arxiv.org/abs/2402.03179) | mixed | open source [[15]](https://orange-opensource.github.io/Cool-Chic/) | not checked |
| Cool-chic video, improved encoding, Orange ([[14]](https://arxiv.org/abs/2501.16976)) | arXiv 2501.16976, 2025 | overfitted; optical flow (RAFT) used only at the encoder | about 1.3 (intra 2.29, residue 0.77, motion 0.26 P / 0.47 B) [[14]](https://arxiv.org/abs/2501.16976) | encode about 20 min per 1080p frame on an H100 [[14]](https://arxiv.org/abs/2501.16976) | +9.3% vs HM 16.20 random access, where x264-medium is +72.5% and x265-medium +36.2%; first 9 frames of HEVC B, C, E; YUV 4:2:0 [[14]](https://arxiv.org/abs/2501.16976) | yes on 9-frame clips (*chained* through HM) | open source [[15]](https://orange-opensource.github.io/Cool-Chic/) | not checked |
| C3, Google DeepMind ([[12]](https://arxiv.org/abs/2312.02753)) | CVPR 2024 | overfitted per video patch | 4.4 (largest over rates) [[12]](https://arxiv.org/abs/2312.02753) | encode on a V100: 29 to 457 s per 1,000 steps, up to 110,000 steps per patch; about 14 h for one 75×270×320 patch at the slowest setting (*derived*) [[12]](https://arxiv.org/abs/2312.02753) | "on par with VCT"; about −20% to −25% vs HEVC medium without B-frames, RGB PSNR, UVG (read off Fig. 8, *unconfirmed*) [[12]](https://arxiv.org/abs/2312.02753) | not reported | code (per the paper) | not checked |
| NVRC-Lite, Bristol ([[16]](https://arxiv.org/abs/2512.04019)) | arXiv 2512.04019, 2025 | overfitted network per video | not in the abstract (*unconfirmed*); it says comparable light codecs are under 10 [[16]](https://arxiv.org/abs/2512.04019) | 8.4× faster encode and 2.5× faster decode than C3 [[16]](https://arxiv.org/abs/2512.04019) | up to −21.03% (PSNR) vs C3 [[16]](https://arxiv.org/abs/2512.04019) | not reported | "will be made available" [[16]](https://arxiv.org/abs/2512.04019) | not checked |

Qualcomm's 2021 demo, which came before MobileCodec, decoded 1280×704 video in real time at 30+
fps on a Snapdragon 888 phone. It used a quantized network and parallel entropy coding, and no
rate-distortion figures were given ([[3]](https://www.qualcomm.com/news/onq/2021/06/17/worlds-first-software-based-neural-video-decoder-running-hd-format-real-time)). It coded every frame as an intra frame ([[1]](https://arxiv.org/abs/2207.08338)).

## 3. Image codecs that could code the I-frames

| Codec | Venue | Decoder kMAC/px | BD-rate (anchor, metric, set) | Encoder | Notes |
|---|---|---|---|---|---|
| TinyLIC Nano ([[19]](https://arxiv.org/abs/2610.22421)) | arXiv 2610.22421, 2026 | 10 | +14.78% (Kodak), +14.68% (CLIC), +12.76% (Tecnick), +13.86% (JPEG AI set) vs VTM Intra 22.0, PSNR [[19]](https://arxiv.org/abs/2610.22421). (A "BPG +30.10%" quoted here first was a misread: 30.10 is BPG's kMAC/px column. Section 11.) | one pass | generalising; channel-autoregressive prior, 4 steps |
| TinyLIC Small ([[19]](https://arxiv.org/abs/2610.22421)) | same | 20 | +5.43%, +4.63%, +2.73%, +6.33% (same sets and anchor) [[19]](https://arxiv.org/abs/2610.22421) | one pass | |
| TinyLIC Base ([[19]](https://arxiv.org/abs/2610.22421)) | same | 50 | −0.79%, −1.76%, −4.21%, +0.91%. Whole codec in int8: +2.21%, +2.49%, −0.22%, +4.74%. Only the entropy-parameter path in int8: −0.41%, −1.45%, −3.68%, +1.52% [[19]](https://arxiv.org/abs/2610.22421) | one pass | integer cost is small at this size |
| FastNIC (Zhang et al., 2025, as tabulated in [[19]](https://arxiv.org/abs/2610.22421)) | — | 10 (encoder about 12) | +26.00% (Kodak), +34.26% (CLIC), +35.61% (Tecnick) vs VTM Intra [[19]](https://arxiv.org/abs/2610.22421) | one pass | original paper not read |
| Shallow decoder, 2 layers, with SGA ([[17]](https://arxiv.org/abs/2304.06244)) | ICCV 2023 | 20.5 (5.3 synthesis + 15.2 hyper-synthesis; entropy coding not counted) | 4.67% fewer bits than BPG on Kodak, PSNR; without SGA, 5.19% more [[17]](https://arxiv.org/abs/2304.06244). Listed at +22.22% vs VTM Intra in [[19]](https://arxiv.org/abs/2610.22421) | iterative (SGA), about 10^5 kMAC/px [[17]](https://arxiv.org/abs/2304.06244) | code public; the hyper-synthesis is most of the decoder |
| EVC Small, mask decay ([[18]](https://arxiv.org/abs/2302.05071)) | ICLR 2023 | 52 per [[19]](https://arxiv.org/abs/2610.22421); at most about 72 (*derived* from a 73.8 G decoder plus 76.6 G "others" at 1920×1088) [[18]](https://arxiv.org/abs/2302.05071) | +6.9% vs VTM (Kodak), +7.30% (HEVC B–E average), PSNR [[18]](https://arxiv.org/abs/2302.05071) | one pass | 73.2 ms decode at 1080p on a 2080 Ti [[18]](https://arxiv.org/abs/2302.05071) |
| Mean-scale hyperprior, Minnen 2018 ([[17]](https://arxiv.org/abs/2304.06244)) | NeurIPS 2018 | 109.0 | 3.30% fewer bits than BPG, Kodak [[17]](https://arxiv.org/abs/2304.06244) | one pass | reference point |
| Factorized prior, Ballé 2017 ([[17]](https://arxiv.org/abs/2304.06244)) | ICLR 2017 | 81.6 | 32.93% more bits than BPG, Kodak [[17]](https://arxiv.org/abs/2304.06244). Our anchor: CompressAI's version is −23.3% vs x264 all-intra on DAVIS 256×256, RGB, against the old handicapped x264 arm [R] | one pass | |
| JPEG AI ([[20]](https://jpeg.org/jpegai/), [[19]](https://arxiv.org/abs/2610.22421)) | ISO/IEC 6048-1:2025 | decoders from about 8 to more than 200 [[19]](https://arxiv.org/abs/2610.22421); about 23 shown on a phone [[19]](https://arxiv.org/abs/2610.22421) | per-profile figures *unconfirmed*: 13, 28 and 215 kMAC/px at −16.0%, −20.2%, −21.1% vs VVC intra, metric not known | one pass | three decoders (Main@Simple, Base, High) [[20]](https://jpeg.org/jpegai/) |
| C3, image ([[12]](https://arxiv.org/abs/2312.02753)) | CVPR 2024 | under 3 | matches VTM on CLIC2020, RGB PSNR [[12]](https://arxiv.org/abs/2312.02753) | overfitted: 22 to 48 s per 1,000 steps on a V100 for the largest CLIC image, up to 110,000 steps [[12]](https://arxiv.org/abs/2312.02753) | |
| Cool-chic v2 ([[12]](https://arxiv.org/abs/2312.02753)) | MMSP 2023 | 2.3 | better than BPG, per [[12]](https://arxiv.org/abs/2312.02753) | overfitted | |

## 4. How published BD-rates can flatter a codec

These traps appear in the papers above. They are the same kinds as in
`docs/FLAW_CHECK_2026-11-26.md`.

1. **4:2:0 anchors scored in RGB** (our flaw 1). ELF-VC scores x264 and x265 in RGB PSNR ([[7]](https://arxiv.org/abs/2104.14335)).
   C3 does the same against x265 on UVG ([[12]](https://arxiv.org/abs/2312.02753)), and the improved Cool-chic paper compares
   learned codecs in RGB on UVG ([[14]](https://arxiv.org/abs/2501.16976)). DCVC-RT's RGB track instead codes its anchors in 10-bit
   4:4:4 ([[8]](https://arxiv.org/abs/2502.20762)); that is the fair version.
2. **Visual tuning left on when scoring PSNR** (our flaw 10). MLVC measured x265 veryslow on
   HEVC-B against the same hardware anchor. With default tuning it was −3.3%. With psy-rd,
   psy-rdoq, adaptive quantization and qcomp switched off it was −17.4% ([[6]](https://arxiv.org/abs/2606.28027)). That is 14 BD
   points for x265; the effect of x264's `-tune psnr` has not been measured in these papers.
   ELF-VC ([[7]](https://arxiv.org/abs/2104.14335)) and MobileNVC ([[2]](https://arxiv.org/abs/2310.01258)) used default tuning; Cool-chic used `-tune psnr` ([[13]](https://arxiv.org/abs/2402.03179)).
3. **Different intra periods for the codec and the anchor.** MLVC's anchor table shows how much
   this moves results ([[6]](https://arxiv.org/abs/2606.28027)). MobileNVC used GOP 16 against an anchor with an unlimited GOP
   ([[2]](https://arxiv.org/abs/2310.01258)), which hurts MobileNVC. The Tencent paper gives its own GOP (12) but not the anchor's
   ([[4]](https://arxiv.org/abs/2309.11276)).
4. **Narrow rate windows.** MobileNVC drops every point above 1.3 Mb/s ([[2]](https://arxiv.org/abs/2310.01258)). DCVC-RT's wins
   in our check stop at its quality ceiling [R].
5. **Short clips.** Cool-chic was tested on 9 frames ([[14]](https://arxiv.org/abs/2501.16976)) and 33 frames ([[13]](https://arxiv.org/abs/2402.03179)). The Tencent
   paper used 96 frames ([[4]](https://arxiv.org/abs/2309.11276)), and the NYCU/UFPel case study 32 ([[5]](https://arxiv.org/abs/2410.20145)).
6. **Estimated rate instead of real bitstreams.** UI2C reports estimated entropy ([[11]](https://arxiv.org/abs/2510.14431)).
   DCVC-RT stresses that it measures real bitstreams with headers ([[8]](https://arxiv.org/abs/2502.20762)).
7. **Pooled averaging.** ELF-VC computes one BD-rate per dataset curve ([[7]](https://arxiv.org/abs/2104.14335)). Cool-chic pools
   MSE over pixels ([[13]](https://arxiv.org/abs/2402.03179)).
8. **Which encoder is "HEVC".** Hardware HEVC, x265 and HM differ by tens of points on the same
   clips ([[6]](https://arxiv.org/abs/2606.28027)).

A rough scale for reading the table (*chained*):
- HM low-delay needs 42.4% more bits than VTM low-delay ([[8]](https://arxiv.org/abs/2502.20762));
- x264-medium needs 72.5% more than HM in random access on 9-frame clips ([[14]](https://arxiv.org/abs/2501.16976)), and 45.8%
  more than x265-medium on UVG ([[4]](https://arxiv.org/abs/2309.11276)).

So "parity with VTM low-delay" means very roughly half of x264-medium's bits.

## 5. Codec by codec

### 5.1 Qualcomm's on-device codecs

**MobileCodec** ([[1]](https://arxiv.org/abs/2207.08338); ACM MMSys 2022, pp. 324–330, per [[8]](https://arxiv.org/abs/2502.20762)) was the first inter-frame neural
decoder running in real time on a phone.
- **Design:** a hyperprior image codec for I-frames. P-frames use a "flow-agnostic" convolutional
  motion network that replaces warping, then residual coding. Encoder and decoder are
  asymmetric.
- **Integers:** the whole model is 8-bit fixed point after quantization-aware training. The
  hyperprior predicts ln σ, which is quantized to an 8-bit index into precomputed integer CDF
  tables. Entropy coding is static arithmetic coding with 32-bit registers, running on 8 CPU
  threads ([[1]](https://arxiv.org/abs/2207.08338), [[2]](https://arxiv.org/abs/2310.01258)).
- **Speed:** 32 ms per frame (31 fps) at 720p on a Snapdragon 8 phone, at 40.18 dB average and
  0.30 bpp on 13 Pexels clips ([[1]](https://arxiv.org/abs/2207.08338)).
- **Rate-distortion:** reported only as a UVG plot against FFmpeg H.264 and H.265 ([[1]](https://arxiv.org/abs/2207.08338)).
  MobileNVC's authors list it at +396.7% (HEVC-B) vs x265 fast in int8, a simulated (AIMET) result ([[2]](https://arxiv.org/abs/2310.01258)).

**MobileNVC** ([[2]](https://arxiv.org/abs/2310.01258), WACV 2024) decodes 1080p YUV 4:2:0 at 38.9 fps on a Snapdragon 8 Gen 2.
- **Design:** scale-space-flow lineage. Block-based overlapped warping runs on the phone's
  warping core. A small flow extrapolator predicts the next block motion. Only the Y channel
  goes into the motion autoencoder. The pipeline overlaps NPU, GPU entropy coding (512 threads)
  and warping ([[2]](https://arxiv.org/abs/2310.01258)).
- **Ablations** ([[2]](https://arxiv.org/abs/2310.01258)):
  - dense warping is only 6.0% better than overlapped block warping, at 4.6 times the receiver
    MACs (113.6 against 24.5);
  - overlap itself is worth 19.2%;
  - a warping-free convolutional model is 51.2% worse;
  - a scale-only prior is 9.6% worse.
- **The int8 cost is large at this size:** +42.5% BD-rate against the floating-point model after
  quantization-aware training. Keeping latents and means in int16 costs +25.6% instead. Rounding
  latents and means on the plain integer grid costs +243.3% (post-training quantization) ([[2]](https://arxiv.org/abs/2310.01258)).
- Only the receiver was optimised; encoding runs at about 3 fps ([[2]](https://arxiv.org/abs/2310.01258)).

**Cross-platform and calibration work.** The calibration method usually quoted here (CIT) is
Tencent AI Lab's, not Qualcomm's (Section 5.2, [[4]](https://arxiv.org/abs/2309.11276)). Qualcomm's own route is fixed-point networks with
integer-indexed entropy tables. MobileCodec cites the low-precision entropy-parameter method of
Said, Pourreza and Le (ICIP 2022), on which it builds ([[1]](https://arxiv.org/abs/2207.08338)). MLVC's authors point out that
MobileNVC reports no cross-platform test ([[6]](https://arxiv.org/abs/2606.28027)).

### 5.2 Identical decoding across devices

- **Tencent CIT** ([[4]](https://arxiv.org/abs/2309.11276), ACM MM 2023). The encoder finds entropy parameters that sit close to a
  quantization boundary and sends their coordinates in a side stream, so the decoder rounds them
  the same way.
  - Without it, decoding on another CPU or GPU failed on 94.8% to 100% of frames. With it, 0%.
  - That test is one 96-frame UVG clip in fp32, on two CPUs and two NVIDIA GPUs ([[4]](https://arxiv.org/abs/2309.11276)).
  - MLVC found that it fails in fp16, where errors can exceed half a quantization step ([[6]](https://arxiv.org/abs/2606.28027)).
- **NYCU/UFPel case study** ([[5]](https://arxiv.org/abs/2410.20145), arXiv 2024). Quantizing only the hyperprior decoding path of
  the scale-space-flow codec to int8 stopped catastrophic failures between a Linux and a macOS
  machine.
  - It costs +9.93% (UVG) and +9.02% (HEVC-B) BD-rate.
  - Reconstructions still differ by less than 0.001 dB, because the synthesis path stays
    floating point ([[5]](https://arxiv.org/abs/2410.20145)).
- **DCVC-RT** ([[8]](https://arxiv.org/abs/2502.20762)) runs the whole model in int16 with int32 sums and a sigmoid lookup table.
  - It costs about 2.7 BD points (−21.0% to −18.3% vs VTM).
  - At 1080p it decodes 2.5 times (2080 Ti) to 5.4 times (A100) slower than fp16.
  - It was tested A100 to 2080 Ti only.
- **MLVC** ([[6]](https://arxiv.org/abs/2606.28027), Microsoft, June 2026) argues that bit-exact integer inference is not available on
  laptop NPUs:
  - NPU toolchains often reject int16 models;
  - Apple's Neural Engine emulates int8 in fp16;
  - kernel choice, operator fusion and rounding differ between vendors.

  Instead it sends scale indices through the hyperprior, so the entropy decoder cannot lose sync.
  It accepts small reconstruction drift and bounds it with periodic I-frames and long-term
  reference frames. It also restricts activations to ReLU and LeakyReLU (plus ReGLU), the only
  ones it found exact on Apple's Neural Engine.
  - Cost: from −69.6% (DCVC-RT) to −52.0% vs hardware HEVC, same platform.
  - Most encoder and decoder vendor pairs stay within 2 BD points of each other ([[6]](https://arxiv.org/abs/2606.28027)).
- **JPEG AI** says it is "designed for reproducible reconstruction across diverse hardware and
  software" ([[20]](https://jpeg.org/jpegai/)). I could not read how; the overview paper ([[21]](https://arxiv.org/abs/2510.13867)) did not parse here.
- **JVET NNVC** runs its neural tools in int16 fixed point (the SADL library). Its authors say a
  real codec needs fixed-point arithmetic to avoid drift ([[22]](https://arxiv.org/abs/2309.05846)).
- **TinyLIC** quantizes only the modules that produce entropy parameters ("Quant-Shared"). At
  50 kMAC/px this is nearly free (−0.41% against −0.94% in floating point on Kodak) ([[19]](https://arxiv.org/abs/2610.22421)).

What this means for R9:
- Integer arithmetic gives identical output only where we control the integer kernels.
- A vendor NPU or vendor-tuned operator library does not give that guarantee.
- Writing our own integer compute shaders for DirectML and WebGPU is the path still open. No
  paper here has shown it across NVIDIA, Intel and AMD.

### 5.3 ELF-VC (WaveOne)

ELF-VC ([[7]](https://arxiv.org/abs/2104.14335), ICCV 2021) is the low-delay codec most often quoted as having beaten H.264 early.
- **Results:** −44.3% (UVG) and −33.7% (MCL-JCV) vs x264, and −26.1% and −17.0% vs x265, in RGB
  PSNR ([[7]](https://arxiv.org/abs/2104.14335)).
- **Anchors:** FFmpeg x264/x265 at preset medium with CRF, B-frames off, otherwise default:
  default visual tuning and keyint 250. Scored in RGB, the setup our flaw check found
  handicapped. ELF-VC itself used a 16-frame GOP ([[7]](https://arxiv.org/abs/2104.14335)).
- **Design:** one model covers many rates. The backbone is asymmetric (38 M encoder against 11 M
  decoder parameters at 720p). An in-loop flow predictor is worth 13% ([[7]](https://arxiv.org/abs/2104.14335)).
- **Cost and speed:** no MAC count was published. It runs at 18 fps decode at 1080p on a Titan V
  in TensorFlow ([[7]](https://arxiv.org/abs/2104.14335)).
- **Animation:** it does very poorly on the four non-photorealistic MCL-JCV videos (Section 6).

No code or weights were released.

### 5.4 DCVC-RT and smaller DCVC-family codecs

**DCVC-RT** ([[8]](https://arxiv.org/abs/2502.20762)) changed the field's view of speed: memory traffic and the number of layers, not
MACs, limited speed.
- Its decode-side modules sum to 174.9 kMAC/px at 1080p (*derived*). fvcore gives 166.8 ([[10]](https://arxiv.org/abs/2606.04595))
  and this repo about 172 [R]. Table 3 of the paper quotes 385 G MAC per 1080p frame (about 186
  kMAC/px, *derived*), but its per-module table sums to 205 for all modules. The paper does not
  say what Table 3 counts.
- KD-NVC ran DCVC-RT on an RTX 5060, the GPU in this laptop, at 31.2 fps decode and 35.3 fps
  encode at 1080p with CUDA kernels ([[10]](https://arxiv.org/abs/2606.04595)). Our plain-PyTorch run measured 9.9 and 10.6 [R].

The lighter descendants:

- **DCVC-UF** ([[9]](https://arxiv.org/abs/2606.04410), CVPR 2026) codes a chunk of 8 frames into one latent.
  - The low-delay variant is −9.5% vs VTM low-delay at 170 G MAC per 1080p frame.
  - The 8-frame variant HT-S is −31.6% at 211 G (24% more MACs per frame, but 81.2 M parameters
    against 9.7 M).
  - So most of the extra gain comes from coding frames jointly, with up to 7 frames of delay,
    rather than from more MACs ([[9]](https://arxiv.org/abs/2606.04410)).
- **KD-NVC** ([[10]](https://arxiv.org/abs/2606.04595)) distils DCVC-RT with module-wise architecture search.
  - The decoder goes from 166.8 to 71.7 kMAC/px (−57%).
  - BD-rate vs VTM-LDB goes from −20.6% to +0.7% (intra period −1).
  - In other words: 57% less decoder compute for about 20 BD points (21 against DCVC-RT's published
    −20.6%, 19.7 against KD-NVC's own reproduction of DCVC-RT at −19.0%) ([[10]](https://arxiv.org/abs/2606.04595)).
- **MLVC-S** ([[6]](https://arxiv.org/abs/2606.28027)) is the smallest DCVC-RT relative with cross-device decoding.
  - About 99 kFLOP/px (*derived*).
  - −36.5% vs Intel hardware HEVC.
  - 1080p in real time on an Apple M3 Pro NPU (39.5 fps encode, 34.6 fps decode) ([[6]](https://arxiv.org/abs/2606.28027)).
- **UI2C** ([[11]](https://arxiv.org/abs/2510.14431)) is heavier (233 kMAC/px decode). It matters for R10: one model does intra and
  inter. It recovers faster after scene cuts than DCVC-RT with periodic refresh, and needs no
  refresh ([[11]](https://arxiv.org/abs/2510.14431)).

None of these was compared with x264. Their tables put them between roughly HM low-delay level
(MLVC-S) and well beyond VTM low-delay (DCVC-UF, KD-NVC-S, UI2C). All of that is far ahead of
x264 (*chained*, Section 4).

### 5.5 Codecs that fit a model to each video: C3, Cool-chic video, NVRC-Lite

These are the only video codecs under 10 kMAC/px. All of them do rate-distortion optimisation by
gradient descent at encode time.

- **C3** ([[12]](https://arxiv.org/abs/2312.02753), CVPR 2024): 4.4 kMAC/px on UVG, about level with VCT.
  - Video is split into patches of 30×180×240 to 75×270×320.
  - Each patch takes 29 to 457 s per 1,000 steps on a V100, for up to 110,000 steps ([[12]](https://arxiv.org/abs/2312.02753)).
  - The authors themselves call it impractical for real-time encoding ([[12]](https://arxiv.org/abs/2312.02753)).
- **Cool-chic video** ([[13]](https://arxiv.org/abs/2402.03179), DCC 2024):
  - 807 parameters and 924 multiplications per pixel in the worst case.
  - About x264 level: better at low rates, worse at high rates and on zooms.
  - 9 minutes per 1080p frame to encode ([[13]](https://arxiv.org/abs/2402.03179)).
- **Improved Cool-chic video** ([[14]](https://arxiv.org/abs/2501.16976), 2025) adds encoder-only optical flow and a joint
  optimisation over frames.
  - About 1,300 multiplications per pixel.
  - +9.3% vs HM 16.20 in random access on 9-frame clips, where x264-medium is +72.5%.
  - About 20 minutes per 1080p frame on an H100 ([[14]](https://arxiv.org/abs/2501.16976)).
- **NVRC-Lite** ([[16]](https://arxiv.org/abs/2512.04019)) is up to 21.03% better than C3, encoding 8.4 times faster and decoding
  2.5 times faster ([[16]](https://arxiv.org/abs/2512.04019)).

The lesson for v2: a decoder of about 1,300 MAC/px has beaten x264-medium (on 9-frame clips,
*chained*) when the encoder was allowed to search hard. Decoder capacity is not what stops these
codecs; encoding cost is.

### 5.6 Low-complexity image codecs (the I-frame part)

The I-frame decoder has to fit the same 10 kMAC/px budget (REQUIREMENTS R7).

- **TinyLIC Nano** ([[19]](https://arxiv.org/abs/2610.22421)) and **FastNIC** ([[19]](https://arxiv.org/abs/2610.22421)) are the only generalising image codecs found at
  about 10 kMAC/px. Nano is 12.8% to 14.8% behind VTM intra on four test sets. (BPG's figure in the same table
  is not confirmed; see Section 11.) FastNIC is 26% to 36% behind ([[19]](https://arxiv.org/abs/2610.22421)).
- **The shallow-decoder result** ([[17]](https://arxiv.org/abs/2304.06244)) shows how to buy decoder savings with encoder work. A
  JPEG-like or two-layer synthesis plus iterative encoding (SGA) moves the 2-layer model from
  5.19% worse than BPG to 4.67% better, at no decoder cost. The encoder then costs about
  10^5 kMAC/px ([[17]](https://arxiv.org/abs/2304.06244)).
- **EVC** ([[18]](https://arxiv.org/abs/2302.05071)) shows the encoder tolerates pruning better than the decoder. With mask decay,
  on Kodak, shrinking the encoder from Large to Small costs 2.7%, and shrinking only the decoder
  costs 5.1% ([[18]](https://arxiv.org/abs/2302.05071)).
- **A rough guide to x264** (*chained*): our own CompressAI anchor found a factorized-prior model
  needs 23.3% fewer bits than x264 all-intra [R]. Yang and Mandt count 82 kMAC/px for this
  architecture ([[17]](https://arxiv.org/abs/2304.06244)); CompressAI's channel widths may differ. That measurement used the old
  benchmark conventions (x264 4:2:0 scored in RGB, the arm flaw 1 found handicapped), so the true
  margin is unknown. The chain first written here, that TinyLIC Nano is about 60 points ahead of
  a factorized model, used a misread BPG figure and is withdrawn (Section 11). What stands:
  Nano is 14.8% behind VTM intra on Kodak, while the factorized model is 32.9% behind BPG, which
  is itself behind VTM, so Nano is well ahead of the factorized model by an unconfirmed margin. A
  10 kMAC/px I-frame coder that beats x264's I-frames looks plausible. It has not been measured
  under our protocol.

### 5.7 Neural tools inside a conventional codec (JVET NNVC)

JVET's NNVC adds neural intra prediction (7.7 kMAC/px worst case) and neural in-loop filters
(673 kMAC/px worst case) to VVC. Together they save 11.94% (random access), 9.18% (low-delay) and
10.63% (all-intra) of luma bits against VTM-11.0_nnvc ([[22]](https://arxiv.org/abs/2309.05846)). Its lower-complexity filter is
quoted at about 17 kMAC/px for about 5% savings; I saw that figure only in a search summary of
another paper's table (*unconfirmed*, [[23]](https://arxiv.org/abs/2407.10926)).

This route adds a few percent on top of a conventional core: about 5% for about 17 kMAC/px
(*unconfirmed*). Most of the compression comes from the conventional core, which costs almost no
MACs and is integer-exact by design.

## 6. Screen content, text and animation

**Animation.**
- ELF-VC does very poorly on the four non-photorealistic videos in MCL-JCV, IDs 18, 20, 24 and
  25 ([[7]](https://arxiv.org/abs/2104.14335)). The scale-space-flow codec behaves the same way ([[7]](https://arxiv.org/abs/2104.14335)).
- MCL-JCV is in our fair evaluation, and our flaw check lists SRC20 among our worst videos. It
  is worth checking whether these four videos drive part of our MCL-JCV deficit. That is a
  hypothesis, not checked.
- Qualcomm's instance-adaptive codec fine-tunes the model per video. It gains 17% to 27% and
  reports better robustness to domain shift ([[24]](https://arxiv.org/abs/2111.10302)). Fine-tuning per video is not live-stream
  speed.
- No paper found measures a generalising learned video codec on an animation test set against
  x264.

**Screen content.**
- No paper found tests a generalising learned video codec on screen content (text, slides,
  scrolling) against x264 or HEVC-SCC.
- The nearest is NeR-SC ([[25]](https://arxiv.org/abs/2605.27024), 2026). It fits a network per video, with a learned colour
  palette. It reports 40.32 dB and 41.73 dB average PSNR on the DSCVC and VCD sets, and beats
  H.264 and H.265 only at low bitrates ([[25]](https://arxiv.org/abs/2605.27024)).
- For still images, learned codecs trained for screen content claim parity with, or wins over,
  VVC-SCC (an IJCAI 2024 paper [[28]](https://ijcai.org/proceedings/2024/0134.pdf); FD-LSCIC [[27]](https://arxiv.org/abs/2502.15174)). Adapting a natural-image codec saves up to
  10% ([[26]](https://arxiv.org/abs/2402.17544)). These come from abstracts only, and their complexity was not checked.
- The tools that make HEVC-SCC and VVC strong on screen content are intra block copy, palette
  mode, adaptive colour transform and adaptive motion vector resolution ([[29]](https://arxiv.org/abs/2407.08545)). H.264 has none of
  them, so x264 is a weaker opponent on screen content than on camera video.
- JPEG AI claims visually better results "for natural photographic content" only ([[20]](https://jpeg.org/jpegai/)).
- MLVC notes that screen sharing needs text clarity, but evaluates on video calls only ([[6]](https://arxiv.org/abs/2606.28027)).

So for R6 the literature gives no answer either way. Our own amendment 1, check C, will be the
first measurement.

## 7. Verdict on R7 and R1

**Not demonstrated.**

The evidence against meeting both:
1. The only generalising, real-time codec near the budget (MobileNVC, P-frames 2.5 times over,
   I-frames 9 times over) loses to x265 fast by 51% to 193% ([[2]](https://arxiv.org/abs/2310.01258)).
2. Its int8 deployment alone cost +42.5% ([[2]](https://arxiv.org/abs/2310.01258)).
3. Integer costs vary widely between designs, and the one small video codec measured paid the
   most. TinyLIC's whole-codec int8 cost about 3 points at 50 kMAC/px ([[19]](https://arxiv.org/abs/2610.22421)), and DCVC-RT's
   int16 about 2.7 points ([[8]](https://arxiv.org/abs/2502.20762)).
4. The cheapest decoders known to be ahead of x264-class anchors are 4 to 8 times over budget:
   - Tencent's P-frame decoder, 39 or 78 kMAC/px depending on the unstated count resolution
     ([[4]](https://arxiv.org/abs/2309.11276));
   - KD-NVC-T, 71.7 ([[10]](https://arxiv.org/abs/2606.04595));
   - DCVC-UF low-delay, about 82 ([[9]](https://arxiv.org/abs/2606.04410)).

   Cutting DCVC-RT's decoder by 57% already cost about 20 BD points ([[10]](https://arxiv.org/abs/2606.04595)). Nothing published
   covers the remaining factor of 4 to 8.
5. R1 asks for a win on every test video. Published results are averages. Known losses
   cluster on cartoons ([[7]](https://arxiv.org/abs/2104.14335)) and complex motion ([[13]](https://arxiv.org/abs/2402.03179)).
6. R9 adds the integer requirement, and no paper shows identical decoding across GPU vendors
   (5.2).

The evidence that it is not hopeless:
1. A 10 kMAC/px image codec (TinyLIC Nano) is 12.8% to 14.8% behind VTM intra ([[19]](https://arxiv.org/abs/2610.22421)). I-frames that
   beat x264 look reachable (*chained*, 5.6).
2. A 1.3 kMAC/px decoder beat x264-medium on 9-frame clips when the encoder optimised each
   frame ([[14]](https://arxiv.org/abs/2501.16976), *chained*). Decoder capacity is not the binding limit once the encoder does
   more of the work.
3. DCVC-RT beats x264 4:4:4 by 48% to 66% on our own videos [R]. KD-NVC at 72 kMAC/px is still
   at VTM low-delay parity ([[10]](https://arxiv.org/abs/2606.04595)), roughly half x264's bits (*chained*). That is a large margin
   to spend on complexity.
4. MobileNVC is a 2023 residual-coding design with block motion. No conditional codec of the
   DCVC-RT kind was found anywhere near 10 to 25 kMAC/px, so the trade-off there is unmeasured,
   not known to be bad.

**Assessment:**
- R7 plus R1 is an open research bet.
- It is most likely to work if the encoder takes on work the decoder cannot afford (Directions 1
  and 2 below), and least likely as a DCVC-RT network simply shrunk 17 times.
- If it fails, note where published codecs sit:
  - the smallest decoder known to reach VTM low-delay parity is 72 kMAC/px (KD-NVC-T, [[10]](https://arxiv.org/abs/2606.04595));
  - at 25 kMAC/px the only published codec, MobileNVC, does not beat x265 fast ([[2]](https://arxiv.org/abs/2310.01258)).

  By REQUIREMENTS' estimates, 25 kMAC/px fits only Radeon-780M-class iGPUs (about 35,000 MAC/px),
  and 72 fits no integrated GPU.
- An Apple M3 Pro NPU ran MLVC-S (about 99 kFLOP/px) at 1080p in real time; Intel and Qualcomm
  NPUs decoded it at 20 to 26 fps ([[6]](https://arxiv.org/abs/2606.28027)). NPUs are outside "any PC with a GPU", and MLVC gives
  up R9's bit-exactness.
- These are inputs to the user's requirement decisions, not proposals to change them.

## 8. Three directions for v2

### Direction 1: an asymmetric conditional P-coder sized to the budget

What it is:
- A DCVC-RT-style P-coder: pixel-unshuffle to 1/8, a single-scale latent, the temporal context
  from previous decoded features, no motion bits.
- The decoder is built to 10 kMAC/px from the start. The encoder may be several times larger.
- It is distilled from a large teacher.
- One model codes intra and inter frames, like MLVC and UI2C, which also covers scene cuts (R10).
- The I-frame side is TinyLIC Nano class, or the unified model.

Evidence:
- DCVC-RT's finding that operation count, not MACs, sets speed ([[8]](https://arxiv.org/abs/2502.20762)).
- KD-NVC: distillation keeps VTM-LDB parity at 72 kMAC/px ([[10]](https://arxiv.org/abs/2606.04595)).
- Asymmetric backbones: ELF-VC (38 M encoder against 11 M decoder, [[7]](https://arxiv.org/abs/2104.14335)); EVC, where the encoder
  tolerates pruning better ([[18]](https://arxiv.org/abs/2302.05071)).
- MLVC and UI2C: one model for I and P works ([[6]](https://arxiv.org/abs/2606.28027), [[11]](https://arxiv.org/abs/2510.14431)).

Budget arithmetic (*derived*):
- At 1/8 resolution, each feature position stands for 64 pixels, so 10 kMAC/px allows 640,000
  MAC per position.
- A dense 3×3 convolution with 64 input and 64 output channels costs 36,864 per position, so
  about 17 such layers fit.
- One DCVC-RT-width pointwise pair (256 to 512 to 256 channels) costs 262,144, so about 2.4 fit.

Option: R8 allows up to 2 s of delay. Coding two to eight frames jointly bought 12 to 22 BD
points in those papers, at 24% to 40% more decoder compute:
- UI2C: −12.1% vs DCVC-RT, at 233 against 167 kMAC/px ([[11]](https://arxiv.org/abs/2510.14431));
- DCVC-UF HT-S: −31.6% against −9.5% for its low-delay variant, at 211 G against 170 G MAC per
  frame ([[9]](https://arxiv.org/abs/2606.04410)).

Our earlier B-frame oracle failed (`outputs/p9_bframes/README.md` [R]). That test reused our
one-frame-back P-coder and sent two motion fields per B-frame. The chunk designs send no motion
and are trained for joint coding.

Risk: no published conditional codec of this kind exists below about 72 kMAC/px (KD-NVC-T).
MobileNVC's result at 24.5 kMAC/px, from a different residual design, is the warning.

First test: a short pilot of a roughly 10 kMAC/px P-coder, judged against x264 with the
fair-protocol settings on development video, before any long training.

### Direction 2: spend encoder compute instead of decoder compute

What it is: keep the decoder fixed and light. At encode time, refine each frame's latents (and
optionally a few per-clip parameters) by a few gradient or search steps, within R8's real-time
budget on an RTX 3060-class GPU. It can sit on top of Direction 1.

Evidence:
- SGA alone moved a 20 kMAC/px image decoder past BPG ([[17]](https://arxiv.org/abs/2304.06244)).
- Per-frame fitting let a 1.3 kMAC/px decoder beat x264-medium on 9-frame clips ([[14]](https://arxiv.org/abs/2501.16976),
  *chained*).
- Per-video fine-tuning gains 17% to 27% and helps on content unlike the training set ([[24]](https://arxiv.org/abs/2111.10302)).
  That matters for animation and screen content (R6).

Risk: every one of these used minutes per frame or worse. How much survives at a real-time
encode budget is unmeasured; that is the first number to get.

### Direction 3, the fallback: a hybrid codec

What it is: conventional block-based prediction and an integer transform core, with small learned
parts. Candidates are a learned entropy model, a small in-loop filter, and screen-content tools
such as a palette mode. Encoder-side rate-distortion search picks modes.

Evidence:
- JVET NNVC: neural intra (7.7 kMAC/px) plus heavy filters (673 kMAC/px) add 9% to 12% over
  VVC ([[22]](https://arxiv.org/abs/2309.05846)); the low-complexity filter adds about 5% for about 17 kMAC/px (*unconfirmed*,
  [[23]](https://arxiv.org/abs/2407.10926)).
- MobileNVC's ablation: block warping costs only 6% against dense warping, at about a fifth of
  the receiver MACs ([[2]](https://arxiv.org/abs/2310.01258)).
- The conventional parts are integer-exact by construction. The learned parts would need
  fixed-point kernels, as JVET does with SADL ([[22]](https://arxiv.org/abs/2309.05846)).
- Screen-content tools are known to work ([[29]](https://arxiv.org/abs/2407.08545)).
- This project already has block motion and a C range coder.

Risk: it beats x264 only if the core is close to HEVC class. A learned layer on an H.264-like
core is unlikely to win by much, and no paper here measures that combination. It also moves the
project away from an end-to-end neural codec, which is the user's call.

### Applies to all three

- Put the entropy-parameter path in integers first; it is nearly free ([[19]](https://arxiv.org/abs/2610.22421)).
- Then test the whole decoder in int8 or int16 early. The one small video codec that published
  its int8 cost paid +42.5% ([[2]](https://arxiv.org/abs/2310.01258)).
- Test bit-exact integer convolution through DirectML on this laptop's Intel UHD against CUDA
  within the first week.
- Keep a fallback in the spirit of MLVC's scale-sending in case bit-exactness fails on some
  vendor ([[6]](https://arxiv.org/abs/2606.28027)).
- Train with screen and animation content from the start (REQUIREMENTS C2).

## 9. Not confirmed or still open

- JPEG AI's per-profile decoder cost and BD-rates. The overview paper's PDF would not parse
  here ([[21]](https://arxiv.org/abs/2510.13867)); only the range of about 8 to more than 200 kMAC/px ([[19]](https://arxiv.org/abs/2610.22421)) and the official page
  ([[20]](https://jpeg.org/jpegai/)) were checked. Also unconfirmed, from search summaries:
  - 12.5% to 27.9% gains over VVC intra at 7 to 216 kMAC/px in the draft standard;
  - the simplest decoder being 1.6 times slower than VTM intra on a CPU.
- C3's video BD-rate, read off a plot ([[12]](https://arxiv.org/abs/2312.02753)).
- NVRC-Lite's own decoder cost ([[16]](https://arxiv.org/abs/2512.04019)).
- The resolution behind Tencent's MAC counts ([[4]](https://arxiv.org/abs/2309.11276)), and whether MLVC's GFLOPs count multiply-adds
  once or twice ([[6]](https://arxiv.org/abs/2606.28027)).
- What DCVC-UF's 170 G MAC per frame includes ([[9]](https://arxiv.org/abs/2606.04410)).
- NNVC's 17 kMAC/px, 5% filter figure ([[23]](https://arxiv.org/abs/2407.10926)).
- FastNIC's original paper (numbers taken from [[19]](https://arxiv.org/abs/2610.22421)).
- The screen-content image codecs' claims ([[26]](https://arxiv.org/abs/2402.17544), [[27]](https://arxiv.org/abs/2502.15174), [[28]](https://ijcai.org/proceedings/2024/0134.pdf)): abstracts only.
- Cool-chic and C3 decode speeds and cross-device behaviour: not checked.
- No paper reviewed reports decoding on an Intel or AMD integrated GPU. The nearest are Apple M3
  ([[10]](https://arxiv.org/abs/2606.04595)) and laptop NPUs ([[6]](https://arxiv.org/abs/2606.28027)).

## 10. Sources

1. Le et al., MobileCodec: Neural Inter-frame Video Compression on Mobile Devices, ACM MMSys 2022. https://arxiv.org/abs/2207.08338 (read via https://ar5iv.labs.arxiv.org/html/2207.08338)
2. van Rozendaal et al., MobileNVC: Real-time 1080p Neural Video Compression on a Mobile Device, WACV 2024. https://arxiv.org/abs/2310.01258 (HTML v3: https://arxiv.org/html/2310.01258v3); https://openaccess.thecvf.com/content/WACV2024/html/van_Rozendaal_MobileNVC_Real-Time_1080p_Neural_Video_Compression_on_a_Mobile_Device_WACV_2024_paper.html
3. Qualcomm OnQ, World's first software-based neural video decoder running HD format in real time, June 2021. https://www.qualcomm.com/news/onq/2021/06/17/worlds-first-software-based-neural-video-decoder-running-hd-format-real-time
4. Tian et al. (Tencent AI Lab), Towards Real-Time Neural Video Codec for Cross-Platform Application Using Calibration Information, ACM MM 2023. https://arxiv.org/abs/2309.11276
5. Conceição, Porto, Peng, Agostini, Cross-Platform Neural Video Coding: A Case Study, arXiv 2024. https://arxiv.org/abs/2410.20145
6. Pärnamaa et al. (Microsoft), MLVC: A Multi-platform Learned Video Codec for Real-World Deployment, arXiv June 2026. https://arxiv.org/abs/2606.28027; code https://github.com/microsoft/mlvc
7. Rippel et al. (WaveOne), ELF-VC: Efficient Learned Flexible-Rate Video Coding, ICCV 2021. https://arxiv.org/abs/2104.14335
8. Jia et al. (Microsoft), Towards Practical Real-Time Neural Video Compression (DCVC-RT), CVPR 2025. https://arxiv.org/abs/2502.20762; code https://github.com/microsoft/DCVC
9. Li et al. (Microsoft), Ultra-Fast Neural Video Compression (DCVC-UF), CVPR 2026. https://arxiv.org/abs/2606.04410
10. Sun et al., KD-NVC: A Search-and-Distill Framework to Accelerate Neural Video Coding, arXiv June 2026. https://arxiv.org/abs/2606.04595
11. Xiang et al. (USTC), Real-Time Neural Video Compression with Unified Intra and Inter Coding (UI2C), arXiv 2510.14431, v5 2026. https://arxiv.org/abs/2510.14431; code https://github.com/ihuixiang/UIIC
12. Kim et al. (Google DeepMind), C3: High-performance and low-complexity neural compression from a single image or video, CVPR 2024. https://arxiv.org/abs/2312.02753; code https://github.com/google-deepmind/c3_neural_compression (link from the paper; not opened)
13. Leguay et al. (Orange), Cool-chic video: Learned video coding with 800 parameters, DCC 2024. https://arxiv.org/abs/2402.03179
14. Leguay et al. (Orange), Improved Encoding for Overfitted Video Codecs, arXiv 2025. https://arxiv.org/abs/2501.16976
15. Cool-chic project (open source). https://orange-opensource.github.io/Cool-Chic/
16. NVRC-Lite: Ultra-lightweight Neural Video Representation Compression, arXiv December 2025. https://arxiv.org/abs/2512.04019
17. Yang and Mandt, Computationally-Efficient Neural Image Compression with Shallow Decoders, ICCV 2023. https://arxiv.org/abs/2304.06244; code https://github.com/mandt-lab/shallow-ntc
18. Wang et al. (Microsoft), EVC: Towards Real-Time Neural Image Compression with Mask Decay, ICLR 2023. https://arxiv.org/abs/2302.05071
19. Lu, Shi, Ma (Nanjing University), End-to-End Learned Image Coding: From Rate–Distortion Optimization to Practical Deployment (TinyLIC benchmark, Table 2), arXiv October 2026. https://arxiv.org/abs/2610.22421
20. JPEG committee, Overview of JPEG AI. https://jpeg.org/jpegai/
21. Esenlik et al., An Overview of the JPEG AI Learning-Based Image Coding Standard, arXiv 2510.13867 (PDF not parsed for this review). https://arxiv.org/abs/2510.13867
22. JVET NNVC authors, Designs and Implementations in Neural Network-based Video Coding, arXiv 2309.05846. https://arxiv.org/abs/2309.05846
23. In-Loop Filtering via Trained Look-Up Tables, arXiv 2407.10926 (its table of NNVC operating points was seen only through a search summary). https://arxiv.org/abs/2407.10926
24. van Rozendaal et al. (Qualcomm), Instance-Adaptive Video Compression: Improving Neural Codecs by Training on the Test Set, TMLR 2023. https://arxiv.org/abs/2111.10302
25. NeR-SC: Adapting Neural Video Representation to Screen Content, arXiv May 2026. https://arxiv.org/abs/2605.27024
26. Adapting Learned Image Codecs to Screen Content via Adjustable Transformations, arXiv 2402.17544 (abstract via search). https://arxiv.org/abs/2402.17544
27. FD-LSCIC: Frequency Decomposition-based Learned Screen Content Image Compression, arXiv 2502.15174 (abstract via search). https://arxiv.org/abs/2502.15174
28. Superpixel-based learned screen content image compression, IJCAI 2024 (abstract via search). https://ijcai.org/proceedings/2024/0134.pdf
29. OMR-Net: A Two-Stage Octave Multi-Scale Residual Network for Screen Content Image Compression, arXiv 2407.08545 (its list of HEVC-SCC tools, via search). https://arxiv.org/abs/2407.08545

`[R]`, this repository:
- `docs/REQUIREMENTS.md`;
- `docs/FLAW_CHECK_2026-11-26.md`;
- `outputs/p14_dcvcrt_check/README.md`;
- `outputs/benchmarks/compressai_anchor/README.md`.

## 11. Evaluation of this review (2026-11-27)

This review was checked after it was written, by going back to the four papers its verdict leans
on most: MobileNVC [[2]], KD-NVC [[10]], TinyLIC [[19]] and MLVC [[6]]. Three of these appeared in
2026. All four exist, with the titles, authors and dates given. MobileNVC was read verbatim. For
the other three, the abstract and the first part of the HTML were read verbatim, and their result
tables only through a fetch summary. So the exact table figures from KD-NVC, TinyLIC and MLVC
should be checked against the PDFs before any of them decides something on its own.

**Confirmed at the source:**
- MobileNVC:
  - 24.52 kMAC/px P-frame and 93.39 I-frame receiver, DeepSpeed at 1080p YUV 4:2:0, warping not
    counted;
  - 38.9 fps decoding on HEVC-B on a Snapdragon 8 Gen 2, and about 3 fps encoding;
  - +159.2% / +192.6% (int8) and +50.8% / +56.4% (float) vs x265 fast, B-frames off, YUV 6:1:1;
  - points above 1.3 Mb/s dropped, GOP 16;
  - the +42.5% int8 cost, and the warping ablation.
- KD-NVC:
  - a DCVC-RT distillation with architecture search;
  - T decoder 71.7 and DCVC-RT 166.8 kMAC/px (fvcore, 1080p);
  - T at +0.7% / −1.5% and S at −11.1% / −9.0% vs VTM-LDB;
  - 69.2 fps decoding at 1080p on an RTX 5060, the M3 figure at 480p;
  - integer inference left as future work.
- TinyLIC:
  - Nano, Small and Base at 10, 20 and 50 kMAC/px;
  - Nano +14.78 / +14.68 / +12.76 / +13.86 vs VTM Intra (PSNR), with Small and Base as quoted;
  - FastNIC as quoted;
  - the int8 rows (named "Quant-All" and "Quant-Shared").
- MLVC:
  - Microsoft, accepted to ECCV 2026;
  - DCVC-RT gives corrupted output across Apple and Intel NPUs;
  - NPU toolchains break bit-exact integer inference;
  - scale indices are sent through the hyperprior;
  - activations limited to ReLU and LeakyReLU;
  - Tian et al.'s calibration fails in fp16;
  - the x265 tuning effect (−3.3% against −17.4%).

**Corrected in the text above:**
- "BPG +30.10% on Kodak" (TinyLIC table) was a misread: 30.10 is BPG's kMAC/px column. The chained
  claim built on it (Nano about 60 points ahead of a factorized model) is withdrawn. Nano is still
  well ahead of a factorized model, by an unconfirmed margin.
- MobileCodec's +396.7% / +549.0% are AIMET-simulated int8 results, not re-measured on a device.
- MLVC's cross-NPU corruption is for DCVC-RT "even in its quantized form". The paper does not say
  int16 for that test.
- KD-NVC's cost of shrinking is about 20 BD points: 21 against DCVC-RT's published −20.6%, 19.7
  against the paper's own reproduction at −19.0%.

**Still unconfirmed, and worth a PDF check if they matter:**
- KD-NVC-S's decoder figure (93.7 is also given as KD-NVC-T's encoder figure).
- The JPEG AI range of 8 to more than 200 kMAC/px (not found in the readable part of [[19]]).
- MLVC's per-NPU speeds, which the paper itself gives differently (a "100 FPS" headline against
  27 to 35 fps in its table for MLVC-S).
- The meaning of TinyLIC's "Quant-Shared" row.

**Weak points in the reasoning (not source errors):**
- "VTM low-delay parity is about half of x264's bits" chains a low-delay figure with a
  random-access figure from 9-frame clips on other test sets. It is very rough, and the "large
  margin to spend on complexity" argument rests on it.
- The integrated-GPU budgets it compares against (about 35,000 MAC/px for a Radeon 780M, 10,000
  for Iris Xe) are this project's own estimates in `docs/REQUIREMENTS.md`, not measurements.
- MAC count is not the whole story on integrated graphics. DCVC-RT found speed limited by memory
  traffic and layer count. Integrated GPUs share the system memory, at roughly 50 to 100 GB/s, so
  a decoder within 10,000 MAC/px can still miss real time if its feature maps are large. R7 has
  to be judged by measured frames per second on a real integrated GPU, as REQUIREMENTS already
  says, not by MAC count alone.

**Does the verdict stand?** Yes. The facts it rests on hold at the source:
- the only real-time codec near the budget (MobileNVC, 24.5 kMAC/px P-frames) loses to x265 fast
  by 51% to 193%;
- the lightest decoder shown at VTM low-delay parity (KD-NVC-T) is 71.7 kMAC/px, about 7 times
  over budget;
- the light codecs that beat x264 fit a network to each video and encode in minutes per frame;
- no paper shows identical decoding across GPU vendors.

So meeting R1, R7 and R9 together is unproven. It is a research bet, and the review's suggested
first test, a short pilot of a roughly 10 kMAC/px decoder against x264 plus a frames-per-second
test on this laptop's Intel UHD Graphics, is the cheapest way to turn it into a measurement.
