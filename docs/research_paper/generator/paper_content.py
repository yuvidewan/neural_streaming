# -*- coding: utf-8 -*-
"""Content of the paper, revised 2026-11-26 to the codec's current state.

Inline markup: *italic*, **bold**, _{subscript}, ^{superscript} (both may hold
*italic*), [@key] or [@a,@b] for citations, numbered by first appearance.
"""

TITLE = ("Gated Development of a Small Learned Video Codec: Real-Bitstream Results "
         "Against H.264 from 256×256 to 1080p")

AUTHORS = [  # used only for the camera-ready variant
    ("Aditya Aggarwal", "Amity University", "Noida, Uttar Pradesh, India", "aggarwaladitya2006@gmail.com"),
    ("Yuvraj Dewan", "Amity University", "Noida, Uttar Pradesh, India", "yuvrajdewan2005@gmail.com"),
    ("Vatsala Upadhyay", "Amity University", "Noida, Uttar Pradesh, India", "vupadhyay2@amity.edu"),
]

ABSTRACT = (
    "We report how a small learned video codec went from needing 5.5× the bits of x264 to needing "
    "fewer than x264 on 1080p video, with every change admitted or rejected by a gate fixed before "
    "its result was read and measured in real, bit-exactly decodable bytes. The codec codes every "
    "frame with one 9.1 M-parameter conditional autoencoder per rate point: a 64-channel latent at "
    "1/16 resolution plus a 32-channel detail latent at 1/8, unit-step rounding under a learned "
    "Laplace prior, and, for P-frames, a context taken from the previous frame warped by "
    "rate-constrained block motion. I-frames use the same network with a flat grey reference. "
    "Against x264 (preset medium, 4:2:0, scored in RGB PSNR) it needs 20.2% fewer bits on UVG and "
    "2.3% fewer on 30 MCL-JCV videos that no decision ever used, and 26.7% more on 256×256 DAVIS. "
    "The 1080p wins come from the high-quality end: over our three lowest rate points we need "
    "36–42% more bits than x264, and against x264 in 4:4:4 we need 18.0% and 38.7% more. Per video, "
    "we beat default x264 on 27 of 46 test videos and low-delay x264 on 35; the losses are "
    "textured scenes where the latent caps quality. An audit found no overlap between training and "
    "test frames and two biases against our codec, both fixed and re-measured. Decoding is not "
    "real time at 1080p (about 2 frames/s on a laptop GPU)."
)

KEYWORDS = ("learned video compression, conditional coding, rate–distortion training, block motion "
            "compensation, BD-rate, H.264/AVC, reproducibility.")

BODY = [
    ("h1", "Introduction"),
    ("p", "Hybrid codecs such as H.264/AVC [@wiegand2003] and H.265/HEVC [@sullivan2012] combine "
          "hand-designed motion-compensated prediction, block transforms, scalar quantization and "
          "context-adaptive arithmetic coding. Learned codecs replace these tools with networks trained "
          "for rate and distortion together [@balle2017,@balle2018,@minnen2018], and learned video "
          "codecs have gone from matching H.264 [@lu2019] to passing H.266 [@li2023,@li2024]. Those "
          "results come from large jointly trained systems and data-centre compute. We wanted to know "
          "how far a small codec gets on one 8 GB laptop GPU, and which changes earn their place."),
    ("p", "Changes to such a codec are often worth 1–5%, about what tuning on test data can produce. "
          "So we fixed the pass mark of every change before reading its result, measured it in real "
          "coded bytes with bit-exact decoding, and kept the failures. An earlier version of this work "
          "stopped at +210.7% BD-rate against x264 and pointed at the quantizer and the training "
          "objective. This paper follows that lead to the current codec. The contributions are: (i) a "
          "9.1 M-parameter conditional codec whose I- and P-frames share one architecture, with a "
          "1/8-resolution detail latent that lifts its quality ceiling; (ii) the gated path from "
          "+450.7% to +26.7% on DAVIS and to −20.2% on UVG 1080p against x264, step by step, with the "
          "four candidates that failed; (iii) results on a fresh 1080p set, per video, against four "
          "anchors and in two colour formats, which show where the codec still loses; and (iv) a "
          "leakage and reproducibility audit, whose two findings were fixed and re-measured."),

    ("h1", "Related work"),
    ("p", "Ballé et al. [@balle2017] set out the analysis, quantization and synthesis structure most "
          "learned codecs follow, with GDN [@balle2016] and additive uniform noise in place of rounding "
          "during training. The hyperprior [@balle2018] and the autoregressive context [@minnen2018] "
          "made the entropy model the main source of gains, and channel-wise contexts "
          "[@minnen2020,@he2022] made it fast. These codecs round with a fixed step and code the result "
          "under a learned density [@balle2021], which is what our codec now does. Encode-time "
          "optimization of the latent [@yang2020] refines a trained encoder's output per image; we "
          "test a simple version of it."),
    ("p", "DVC [@lu2019] coded optical flow and a pixel residual. The DCVC family [@li2021,@li2023,@li2024] "
          "replaced the residual by conditional coding, where the warped reference is an input to the "
          "encoder, decoder and prior. Our P-frame coder is conditional in this sense but keeps "
          "integer block motion. DCVC-RT [@jia2025] drops explicit motion altogether: it codes a "
          "128-channel latent at 1/16 resolution given 256-channel features of the previous frame at "
          "1/8, runs 1080p in real time on a data-centre GPU, and integerizes its network so that "
          "decoding is identical across devices. Cool-chic video [@leguay2024] shows "
          "that small learned video codecs can approach H.264 by fitting a decoder per video. Codecs "
          "are compared by BD-rate [@bjontegaard2001] on PSNR and MS-SSIM [@wang2003], usually after "
          "training on Vimeo-90K [@xue2019] and testing on UVG [@mercat2020] and MCL-JCV [@wang2016]."),

    ("h1", "The codec"),
    ("p", "Frames are coded in groups of 30: one I-frame, then 29 P-frames, each predicted from the "
          "previous decoded frame. There are no B-frames, so the decoder never waits for a future frame. "
          "Fig. 1 shows the codec. Each rate point *λ* ∈ {10^{−2}, 3×10^{−3}, 10^{−3}, 3×10^{−4}, "
          "10^{−4}} has its own pair of networks, one for I-frames and one for P-frames."),
    ("fig", "fig1_arch.png"),
    ("cap", "Fig. 1. The current codec. I- and P-frames use the same network; an I-frame's reference is "
            "a flat grey frame. *R* is a refinement network whose last layer starts at zero."),
    ("h2a", "Transform and rate training"),
    ("p", "The transform has GDN/IGDN layers and residual blocks with 192 internal channels, and maps a "
          "frame to a 64-channel latent at 1/16 resolution. It started as a distortion-only image "
          "autoencoder trained on Vimeo-90K and fine-tuned against quantization noise. That version "
          "used a per-channel quantizer that stretched each channel's range over a fixed alphabet. A "
          "channel with little energy still paid most of the full bit depth, and one more bit of depth "
          "added about one bit to every symbol. We replaced it with unit-step rounding under a learned "
          "density and trained one model per *λ* on"),
    ("eq", ["*L* = *d*(*x*, *x̂*) + *λ*·*R*(*ŷ*)"], "1"),
    ("p0", "where *d* is MSE on RGB in [0, 1] and *R* the code length in bits per pixel. During training "
           "*R* is computed on the latent plus uniform noise, and the decoder sees the rounded latent "
           "through a straight-through estimator. On P-frame residuals the unit step with a Laplace "
           "density was −21.0% BD-rate against the range-fitted grid with its histogram tables. Coded all-intra, the rate-trained transform needs 12.3% "
           "fewer bits than x264 all-intra on DAVIS TEST over 31.4–32.9 dB."),
    ("h2a", "Conditional P-frame coder"),
    ("p", "The decoder has the warped reference *x*_{w} before it reads any P-frame symbol, so the coder "
          "uses it as a condition. A context network *F* (two residual blocks, 32 channels) turns "
          "*x*_{w} into features *c*. The encoder sees *x*_{t} and *c*; a prior network predicts a mean "
          "*μ*(*c*) and a scale *b*(*c*) for every latent value; the latent is rounded to integers "
          "around *μ* and coded under Laplace(*μ*, *b*). The decoder output passes through a refinement "
          "network *R* that sees *x*_{w} and *c* again:"),
    ("eq", ["*ŷ* = *μ* + round(*y* − *μ*),  *x̂*_{*t*} = *D*(*ŷ*) + *R*(*D*(*ŷ*), *x*_{w}, *c*)"], "2"),
    ("p0", "Trained from random weights, the coder settled at the zero-rate solution (*x̂*_{*t*} ≈ "
           "*x*_{w}, 26 dB) and never left it. It works when warm-started from the intra transform, with "
           "the context's share of the first encoder layer and the last layer of *R* set to zero, so "
           "that at step 0 it is the intra codec with a conditional prior. Training on Vimeo-90K "
           "septuplets ran in three stages: one P-step from an intra-coded reference on two of ten data "
           "chunks; three unrolled P-steps on the coder's own reconstructions on a third chunk, so that "
           "it learns to stop error growth; and the remaining seven chunks."),
    ("p", "The 64-channel latent at 1/16 is 0.25 values per pixel. On detailed videos the codec could "
          "not pass about 28.5 dB at any rate. Without rounding, the intra transform reached only "
          "27.6 dB on cows and 28.6 dB on schoolgirls, so the limit was the latent's capacity and not "
          "quantization. We added a "
          "32-channel detail latent at 1/8 resolution, taken from the encoder's stride-8 features and "
          "coded with a prior that sees the context and the decoded 1/16 latent. Its synthesis branch "
          "starts at zero, so training begins from the old coder's output. With equal training on "
          "DAVIS, this cut P-frame bits by 20.6% at matched quality and raised the no-rounding ceiling "
          "on a validation video by 3.99 dB. Widening the 1/16 latent by 128 channels instead cost "
          "10.4% more bits. Each coder has 9.1 M parameters (8.75 M in the base, 0.30 M in the detail "
          "branch)."),
    ("h2a", "I-frames"),
    ("p", "The old intra transform topped out at 32.9 dB on validation frames whatever the rate. I-frames "
          "now use the P-frame architecture fed a flat grey frame as its reference, trained on single "
          "Vimeo-90K frames. Its ceiling is 38.0 dB. The P-coder of each *λ* was then trained with "
          "I-frames from the I-coder of the same *λ*."),
    ("h2a", "Rate-constrained block motion"),
    ("p", "Motion is integer-pel block matching on 16×16 blocks within ±16 pixels, so the warp is "
          "bit-exact. A field found by plain SAD search cost too much at 1080p, where a frame has 8,040 "
          "blocks: at *λ* = 10^{−2}, vectors were 51.2% of the UVG stream. The search now minimizes"),
    ("eq", ["*J*(*v*) = SAD(*v*) + *λ*_{*m*}·*r*(*v* − *v̂*)"], "3"),
    ("p0", "in raster order, where *v̂* is the H.264 median of the left, upper and upper-right vectors "
           "and *r* the code length of the difference under one of 64 Laplace tables (two bytes per "
           "P-frame pick the tables). *λ*_{*m*} was picked per *λ* on validation data, from 0.0625 to "
           "0.25. Against plain search this cut bits by 10.7% on UVG and 4.75% on DAVIS at the same "
           "quality. Vectors are now 24.8% of the UVG stream at *λ* = 10^{−2} and 2.7% at 10^{−4}."),
    ("h2a", "Bitstream and decoding"),
    ("p", "Latent values are range coded in C [@witten1987,@martin1979]; each predicted scale picks one "
          "of 64 Laplace tables, which costs 0.01% over the ideal code length. The container has a "
          "56-byte header, three 8-byte model identities and a 9-byte record per frame, and a decoder "
          "refuses a stream whose identities do not match its models. Every stream in this paper was "
          "decoded and compared with the encoder's reconstruction, and all were identical; a sample of "
          "12 also decoded identically in a fresh process. The networks run with deterministic GPU kernels, and identical decoding holds "
          "only on the same GPU: on a CPU, 0 of 10 test decodes matched."),
    ("h2a", "Gating protocol"),
    ("p", "Every candidate's pass mark was committed to the repository before its first run. DAVIS 2017 "
          "[@pont2017] is split at random into TRAIN (72 sequences), VAL-A (5), VAL-B (4) and TEST (9). "
          "Settings are chosen on VAL-A or VAL-B, and TEST is coded once per candidate. The UVG set was "
          "the primary set of two later gates (motion and MS-SSIM training), so UVG has influenced "
          "decisions. MCL-JCV was added at the end and chose nothing."),

    ("h1", "Experimental setup"),
    ("p", "*Data.* DAVIS TEST has 9 sequences (719 frames) resized and centre-cropped to 256×256. UVG "
          "[@mercat2020] has 7 videos at 120 fps and MCL-JCV [@wang2016] 30 videos at 24, 25 or 30 fps; "
          "we take the first 96 frames of each, convert them to RGB and crop to 1920×1072. That is "
          "4,271 test frames."),
    ("p", "*Anchors.* FFmpeg runs libx264 and libx265 at preset medium on rgb24 input, encodes yuv420p "
          "or yuv444p, and decodes back to RGB. The default configuration keeps the encoder's own GOP "
          "(up to 250 frames, so one I-frame per 96-frame video) and B-frames; the low-delay one has "
          "an I-frame every 30 frames, no B-frames and no scene-cut detection, as ours. CRFs run from 8 to 44 in steps of 2, plus 0 to 6 for UVG. Each video is "
          "fed at its real frame rate; DAVIS records none and is fed at 30 fps. Scoring 4:2:0 anchors "
          "in RGB charges them for the chroma they drop, so we also report 4:4:4, the stricter anchor."),
    ("p", "*Metrics.* PSNR and MS-SSIM are computed on 8-bit RGB, averaged within each video and then "
          "over videos. Rate is the whole file, container included, over all pixels. BD-rate "
          "interpolates log-rate piecewise-linearly in quality and averages over the overlap of the "
          "two curves; our five points span 28.5–37.9 dB on DAVIS, 32.0–40.9 dB on UVG and "
          "32.2–40.7 dB on MCL-JCV. Before reading any BD-rate we check that the anchor reaches our top "
          "point on every video. Negative numbers mean fewer bits than the anchor."),

    ("h1", "Results"),
    ("h2b", "From +450.7% to the current codec"),
    ("p", "Table I lists every admitted change in order, as each was measured on DAVIS TEST against "
          "default x264 4:2:0. The anchors gained CRF 8–18 at the A2 step, so that their curves cover "
          "our higher qualities; that changed the window, and both numbers are given. Rate training "
          "and the learned P-coder took the codec from +210.7% to +91.1%. More training data, a longer "
          "GOP and the detail latent took it to +39.9%. The two lower-rate models moved the number up, "
          "from +32.9% to +34.0%, because the window then reaches down to rates where the codec is "
          "weaker. On DAVIS the codec still needs 26.7% more bits than x264."),
    ("tcap", "TABLE I. Admitted Changes, DAVIS TEST, vs x264 4:2:0 (PSNR BD-rate)"),
    ("table", "table1"),
    ("h2b", "Results against H.264 and H.265"),
    ("p", "Table II and Fig. 2 give the current codec on all three sets. On UVG it needs 20.2% fewer "
          "bits than default x264 and 34.2% fewer than low-delay x264, and on MCL-JCV 2.3% and 13.1% "
          "fewer. The gap between the two 1080p sets is 17.9 points; UVG has been reused, MCL-JCV has "
          "not, so we read MCL-JCV as the honest 1080p number. Two limits apply. First, the 1080p "
          "advantage sits at high quality. Over our three lowest points (32.0–36.7 dB on UVG) we need "
          "42.3% more bits than x264 4:2:0 on UVG and 36.2% more on MCL-JCV, and Fig. 2 shows the "
          "curves crossing near 36–37 dB. Second, against x264 in 4:4:4 we lose on every set, by "
          "18.0% to 54.7%. Against x265 we need 8.6% more bits on UVG and 25.2% more on MCL-JCV."),
    ("tcap", "TABLE II. Current Codec, BD-Rate Against Each Anchor"),
    ("table", "table2"),
    ("fig", "fig2_rd.png"),
    ("cap", "Fig. 2. Rate and PSNR of our MSE models against x264 and x265 (preset medium). Low-delay "
            "means an I-frame every 30 frames and no B-frames. UVG and MCL-JCV anchors are fed at their "
            "real frame rates."),
    ("p", "A second set of ten models, fine-tuned from the MSE models on 1280/255²·(1 − MS-SSIM) + "
          "*λ*·*R*, needs 22.4% (DAVIS) and 9.7% (UVG) fewer bits than the MSE set at equal MS-SSIM. On "
          "MS-SSIM it is level with x264 4:2:0 on 1080p (+0.4% on UVG, −0.6% on MCL-JCV) and beats "
          "low-delay x264 on all three sets. It is much worse on PSNR, so the MSE set stays the PSNR "
          "codec."),
    ("h2b", "Per-video results"),
    ("p", "We beat default x264 4:2:0 on 27 of the 46 test videos (4 of 9 on DAVIS, 5 of 7 on UVG, 18 of "
          "30 on MCL-JCV), low-delay x264 on 35 and x264 4:4:4 on 11. The best videos are MCL-JCV SRC16 "
          "(−57.7%) and SRC17 (−53.7%), UVG YachtRide (−42.0%) and DAVIS gold-fish (−42.0%). The losses "
          "concentrate in a few textured scenes: DAVIS cows (+145.5%) and schoolgirls (+167.7%), and "
          "MCL-JCV SRC13 (+252.3%) and SRC20 (+336.6%). On cows we need 2.5 times x264's bits at equal "
          "PSNR, and our top point is 33.9 dB at 1.47 bpp. These are the videos where the latent caps quality, "
          "and the pooled numbers hide them. Averaged per video, the MCL-JCV result against x264 4:2:0 "
          "is +19.3%, against −2.3% pooled."),
    ("h2b", "Rejected and optional changes"),
    ("p", "Four candidates failed their gates. A motion oracle gave the P-coder a free dense flow field "
          "with no motion bits: it saved 4.1% of P-frame bits, and 19.9% with a P-coder fine-tuned on "
          "it, against a pass mark of 25%, so block motion stayed. A hyperprior for the P-coder saved "
          "0.4% against a mark of 8%; its side latent shrank to 8 bits per frame, which suggests the "
          "context already carries what it would send. The wider 1/16 latent lost to the detail latent. "
          "B-frames from the same P-coder with an interpolated reference lost to P-only coding at every "
          "*λ* on VAL-B: the far references and twice the motion cost more than the second reference "
          "saved. A GOP of 30 and the grey-reference I-coder were first picked on TEST; both were "
          "re-picked on VAL-B during the audit and came out the same."),
    ("p", "One change passed and is left out of the headline. Twenty Adam steps on the latent at encode "
          "time, against the straight-through rate and distortion, saved 6.3% of bits on DAVIS TEST "
          "with every video improved, but made encoding 2.5 to 10 times slower. Decoding is unchanged, "
          "so it is an encoder option."),
    ("h2b", "Audit"),
    ("p", "After the results above, we audited the work against its own records. We hashed 641,907 "
          "Vimeo-90K training frames against the 4,271 test frames; the 21 pairs under the threshold "
          "were all different scenes on inspection, and the DAVIS splits are disjoint. All 23 headline "
          "numbers rebuilt from the raw result files. Two findings were biased against our codec. "
          "Streams carried 1,024 bytes of unused quantizer header, up to 2.6% of the smallest DAVIS "
          "stream; removing it changed no decoded pixel and moved DAVIS from +27.6% to +26.7%. The "
          "anchors had been fed at 30 fps, and x264's rate control depends on the frame rate. At "
          "UVG's real 120 fps the same CRF lands about 1.4 dB lower, so CRF 8 no longer reached our "
          "top point and the comparison window shifted down to where we are weaker: the pooled UVG "
          "number read −1.9%. Adding CRF 0–6 restored full coverage and gave −20.2%. We now check "
          "coverage per video before reading any BD-rate."),
    ("h2b", "Speed"),
    ("p", "On a laptop RTX 5060 (8 GB), a 256×256 frame encodes in 35–41 ms and decodes in 11–13 ms. A "
          "1920×1072 frame encodes in 1.2–1.4 s and decodes in 0.42–0.55 s. At 1080p the motion search "
          "runs on the CPU in raster order, since each vector's cost depends on the vectors already "
          "chosen around it. The codec is real time at 256×256 and not at 1080p."),

    ("h1", "Conclusion"),
    ("p", "Gated one change at a time and measured in real bytes, a 9.1 M-parameter codec went from 5.5× "
          "x264's bits to fewer bits than x264 4:2:0 on two 1080p sets, 20.2% fewer on UVG and 2.3% on "
          "a set that never informed a decision. The largest steps were rate training with unit-step "
          "rounding, a conditional P-coder warm-started from the intra transform, and a detail latent "
          "at 1/8 resolution. The codec still loses at low rates, against x264 in 4:4:4, on 256×256 "
          "DAVIS and on textured scenes, and it decodes 1080p at about 2 frames/s."),
    ("p", "*Limitations.* Results use one seed and the first 96 frames of each 1080p video. DAVIS and "
          "UVG were reused across gates. BD-rates within about ±3% depend on the interpolation method. "
          "Decoding is identical only on the same GPU. The low-rate and textured-scene losses point to "
          "the transform's capacity and to motion cost at 1080p. A design along the lines of DCVC-RT "
          "[@jia2025], which carries the previous frame as decoded features in place of motion "
          "vectors and uses integer arithmetic, is aimed at both, and at real-time decoding."),
]

TABLES = {
    "table1": {
        "widths": [3450, 500, 750],
        "header": ["Change (each adds to the row above)", "GOP", "BD-rate"],
        "rows": [
            ["Initial codec: 593 k Conv/ReLU, range-fitted quantizer", "10", "+450.7%"],
            ["GDN/residual transform, quantization-aware training, refitted entropy stack", "10", "+210.7%"],
            ["Unit-step rounding, one rate-trained model per *λ* (3 *λ*)", "10", "+127.1%"],
            ["Conditional P-frame coder, 3 Vimeo-90K chunks", "10", "+91.1%"],
            ["P-coder trained on the other 7 chunks", "10", "+77.8%"],
            ["GOP of 30", "30", "+69.1%"],
            ["Detail latent at 1/8, grey-reference I-frames", "30", "+39.9%"],
            ["  same, anchors extended to CRF 8 (full overlap)", "30", "+32.9%"],
            ["Two lower-rate models (5 *λ*)", "30", "+34.0%"],
            ["Rate-constrained motion, predicted vectors", "30", "+27.6%"],
            ["Unused 1,024-byte header removed (audit)", "30", "+26.7%"],
        ],
    },
    "table2": {
        "widths": [2050, 850, 850, 850],
        "header": ["Anchor (x264 unless noted)", "DAVIS", "UVG", "MCL-JCV"],
        "rows": [
            ["*MSE models, PSNR, five rate points*"],
            ["4:2:0, default", "+26.7%", "−20.2%", "−2.3%"],
            ["4:2:0, low-delay", "+6.8%", "−34.2%", "−13.1%"],
            ["4:4:4, default", "+54.7%", "+18.0%", "+38.7%"],
            ["x265, 4:2:0, default", "+35.3%", "+8.6%", "+25.2%"],
            ["4:2:0, three lowest points", "+23.4%", "+42.3%", "+36.2%"],
            ["*MS-SSIM models, MS-SSIM*"],
            ["4:2:0, default", "+14.9%", "+0.4%", "−0.6%"],
            ["4:2:0, low-delay", "−3.2%", "−15.9%", "−12.3%"],
            ["*Videos won, MSE models, PSNR*"],
            ["4:2:0, default", "4 / 9", "5 / 7", "18 / 30"],
            ["4:2:0, low-delay", "4 / 9", "7 / 7", "24 / 30"],
            ["4:4:4, default", "3 / 9", "2 / 7", "6 / 30"],
        ],
    },
}

# IEEE style. *...* marks italic (journal / proceedings names).
REFERENCES = {
    "wiegand2003": "T. Wiegand, G. J. Sullivan, G. Bjøntegaard, and A. Luthra, “Overview of the H.264/AVC "
                   "video coding standard,” *IEEE Trans. Circuits Syst. Video Technol.*, vol. 13, no. 7, "
                   "pp. 560–576, Jul. 2003.",
    "sullivan2012": "G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency "
                    "video coding (HEVC) standard,” *IEEE Trans. Circuits Syst. Video Technol.*, vol. 22, "
                    "no. 12, pp. 1649–1668, Dec. 2012.",
    "balle2017": "J. Ballé, V. Laparra, and E. P. Simoncelli, “End-to-end optimized image compression,” in "
                 "*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.",
    "balle2018": "J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston, “Variational image "
                 "compression with a scale hyperprior,” in *Proc. ICLR*, 2018.",
    "minnen2018": "D. Minnen, J. Ballé, and G. Toderici, “Joint autoregressive and hierarchical priors for "
                  "learned image compression,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, "
                  "vol. 31, 2018.",
    "lu2019": "G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, and Z. Gao, “DVC: An end-to-end deep video "
              "compression framework,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, "
              "2019, pp. 11006–11015.",
    "li2021": "J. Li, B. Li, and Y. Lu, “Deep contextual video compression,” in *Proc. NeurIPS*, vol. 34, "
              "2021.",
    "li2023": "J. Li, B. Li, and Y. Lu, “Neural video compression with diverse contexts,” in *Proc. "
              "IEEE/CVF CVPR*, 2023, pp. 22616–22626.",
    "li2024": "J. Li, B. Li, and Y. Lu, “Neural video compression with feature modulation,” in *Proc. "
              "IEEE/CVF CVPR*, 2024, pp. 26099–26108.",
    "balle2016": "J. Ballé, V. Laparra, and E. P. Simoncelli, “Density modeling of images using a "
                 "generalized normalization transformation,” in *Proc. ICLR*, 2016.",
    "minnen2020": "D. Minnen and S. Singh, “Channel-wise autoregressive entropy models for learned image "
                  "compression,” in *Proc. IEEE Int. Conf. Image Process. (ICIP)*, 2020, pp. 3339–3343.",
    "he2022": "D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang, “ELIC: Efficient learned image "
              "compression with unevenly grouped space-channel contextual adaptive coding,” in *Proc. "
              "IEEE/CVF CVPR*, 2022, pp. 5718–5727.",
    "balle2021": "J. Ballé, P. A. Chou, D. Minnen, S. Singh, N. Johnston, E. Agustsson, S. J. Hwang, and "
                 "G. Toderici, “Nonlinear transform coding,” *IEEE J. Sel. Topics Signal Process.*, "
                 "vol. 15, no. 2, pp. 339–353, Feb. 2021.",
    "yang2020": "Y. Yang, R. Bamler, and S. Mandt, “Improving inference for neural image compression,” in "
                "*Proc. NeurIPS*, vol. 33, 2020.",
    "jia2025": "Z. Jia, B. Li, J. Li, W. Xie, L. Qi, H. Li, and Y. Lu, “Towards practical real-time neural "
               "video compression,” in *Proc. IEEE/CVF CVPR*, 2025, pp. 12543–12552.",
    "leguay2024": "T. Leguay, T. Ladune, P. Philippe, and O. Déforges, “Cool-chic video: Learned video "
                  "coding with 800 parameters,” in *Proc. Data Compression Conf. (DCC)*, 2024, "
                  "pp. 23–32.",
    "bjontegaard2001": "G. Bjøntegaard, “Calculation of average PSNR differences between RD-curves,” "
                       "ITU-T SG16/Q6 VCEG, Doc. VCEG-M33, Austin, TX, USA, Apr. 2001.",
    "wang2003": "Z. Wang, E. P. Simoncelli, and A. C. Bovik, “Multiscale structural similarity for image "
                "quality assessment,” in *Proc. 37th Asilomar Conf. Signals, Syst. Comput.*, 2003, "
                "vol. 2, pp. 1398–1402.",
    "xue2019": "T. Xue, B. Chen, J. Wu, D. Wei, and W. T. Freeman, “Video enhancement with task-oriented "
               "flow,” *Int. J. Comput. Vis.*, vol. 127, no. 8, pp. 1106–1125, Aug. 2019.",
    "mercat2020": "A. Mercat, M. Viitanen, and J. Vanne, “UVG dataset: 50/120fps 4K sequences for video "
                  "codec analysis and development,” in *Proc. ACM Multimedia Syst. Conf. (MMSys)*, 2020, "
                  "pp. 297–302.",
    "wang2016": "H. Wang, W. Gan, S. Hu, J. Y. Lin, L. Jin, L. Song, P. Wang, I. Katsavounidis, A. Aaron, "
                "and C.-C. J. Kuo, “MCL-JCV: A JND-based H.264/AVC video quality assessment dataset,” in "
                "*Proc. IEEE ICIP*, 2016, pp. 1509–1513.",
    "pont2017": "J. Pont-Tuset, F. Perazzi, S. Caelles, P. Arbeláez, A. Sorkine-Hornung, and L. Van Gool, "
                "“The 2017 DAVIS challenge on video object segmentation,” arXiv:1704.00675, 2017.",
    "witten1987": "I. H. Witten, R. M. Neal, and J. G. Cleary, “Arithmetic coding for data compression,” "
                  "*Commun. ACM*, vol. 30, no. 6, pp. 520–540, Jun. 1987.",
    "martin1979": "G. N. N. Martin, “Range encoding: An algorithm for removing redundancy from a "
                  "digitised message,” in *Proc. Video & Data Recording Conf.*, Southampton, U.K., 1979.",
}
