# Real-Time Neural Video Compression & Streaming Engine

An engine that replaces legacy pixel-based video codecs (H.264/H.265) with a
neural approach: video frames are encoded into a compact latent
representation by a deep neural network, streamed as tiny mathematical
"blueprints," and reconstructed (hallucinated) back into full-resolution
frames by a decoder network on the client.

## Core Concept: Pixels vs. Neural Math

**Traditional streaming (Netflix/YouTube):** slices a 1080p frame into
macroblocks, computes motion vectors between frames, discards
perceptually-irrelevant data, and transmits raw pixel deltas.

**This engine:** passes each frame through a Deep Encoder Network that
condenses the full `1920 x 1080 x 3` pixel array into a small, dense latent
vector. Only that vector is transmitted. On the receiving end, a Deep
Decoder Network reads the latent blueprint and reconstructs the
high-resolution frame.

## Architecture

```
[ Raw Frame Data ]
       |
       v
 +-----------+
 |  Encoder  |  (Convolutional Neural Network / Downsampling)
 +-----+-----+
       |
       v
 [ Latent Space ] --> [ Quantization & Entropy Coding ] --> (.bin / .nvc file)
       |
       v
 +-----------+
 |  Decoder  |  (Deconvolutional / Generative Network)
 +-----+-----+
       |
       v
[ Reconstructed Frame ]
```

The core model is a **Variational Autoencoder (VAE)** with temporal
conditioning between frames.

## Roadmap

### 7th Semester (Minor Project): Core Neural Codec

Goal: prove local compression works, no live streaming yet.

1. **Frame Extraction** — Load raw video and split into sequential frames
   using Python + OpenCV.
2. **The Encoder** — PyTorch CNN that downsamples spatial dimensions while
   increasing feature depth, squashing a 1080p frame into a compact latent
   feature map.
3. **Quantization & Bottleneck** — Convert floating-point latents into
   integers (quantization) to minimize storage, saved as a custom `.nvc`
   (Neural Video Codec) binary file.
4. **The Decoder** — Inverse deconvolutional/generative network that unpacks
   the binary data and reconstructs the full frame.
5. **Metrics** — Evaluate against standard video quality metrics:
   - **PSNR** (Peak Signal-to-Noise Ratio)
   - **MS-SSIM** (Multi-Scale Structural Similarity)

   Compare neural compression quality-per-file-size against standard MP4.

### 8th Semester (Major Project): Scaling & Streaming Infrastructure

Goal: deploy the trained model into a simulated real-time streaming system.

1. **Neural Interpolation (Temporal Hallucination)**
   - Transmit only 5 frames per second instead of 30.
   - A client-side temporal network (lightweight GAN / video diffusion model)
     interpolates between Frame 1 and Frame 5 to synthesize the missing
     in-between frames.
   - Result: smooth 30 FPS playback while transferring ~15% of the original
     data.

2. **Edge Super-Resolution**
   - Encode and transmit video at a low-bandwidth 360p resolution.
   - Client-side Real-Time Super-Resolution model (DLSS-style) upscales the
     360p stream to a sharp 1080p/4K output locally, without added GPU lag.

3. **Full-Stack Streaming Pipeline**
   - **Backend:** FastAPI server streaming custom neural binary packets.
   - **Frontend:** React/HTML5 Canvas web client or Python desktop dashboard
     acting as the media player.
   - **Inference:** PyTorch decoder models exported to **ONNX Runtime** for
     real-time, low-latency rendering on the client.

## Tech Stack

- **Python**, **OpenCV** — video I/O and frame extraction
- **PyTorch** — encoder/decoder (VAE), temporal interpolation, super-resolution models
- **ONNX Runtime** — optimized client-side inference
- **FastAPI** — streaming backend server
- **React / HTML5 Canvas** — web-based media player frontend

## Status

Planning stage — architecture and roadmap defined, implementation not yet
started.
