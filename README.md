# Real-Time Neural Video Compression & Streaming Engine

B.Tech AIML minor project (12-week scope). This repository builds a video
codec that represents frames as compact neural latent vectors instead of
traditional pixel/motion-vector blocks (H.264/H.265-style compression).

## Objective

Given a raw video, encode each frame into a small latent representation
using a convolutional neural encoder / Variational Autoencoder (VAE),
quantize and serialize that representation into a custom `.nvc` binary
format, then decode it back into a reconstructed frame - and measure how
that compares to conventional codecs on quality and size.

## Current Development Status

**Milestone 2 (this repository state): repository foundation + dataset preparation pipeline.**

Implemented:
- Repository/directory structure
- Python environment configuration (`requirements.txt`, `pyproject.toml`)
- A typed configuration mechanism (`src/nvc/utils/config.py`), now including
  preprocessing settings
- `scripts/check_environment.py` environment verification
- Video metadata inspection (`src/nvc/data/video_utils.py`)
- Frame extraction with configurable sampling and deterministic resize/crop
  (`src/nvc/data/frame_extraction.py`)
- Video-level train/val/test splitting, manifest generation, and dataset
  statistics (`src/nvc/data/dataset_prep.py`)
- `scripts/prepare_dataset.py` CLI tying the above together
- Package skeleton with no logic yet (`src/nvc/{models,compression,evaluation}`)

**Not implemented yet** (do not assume any of this works):
- A PyTorch `Dataset`/`DataLoader` over the extracted frames
- Any neural network (encoder, decoder, VAE)
- Quantization, entropy coding, or the `.nvc` format itself
- Reconstruction or video reassembly
- PSNR / MS-SSIM / MSE / BPP / compression-ratio / timing evaluation
- Comparison against H.264/H.265
- FastAPI serving layer
- ONNX Runtime inference
- Neural frame interpolation or super-resolution (major-project scope, later)

## Planned Architecture

```
[ Raw Frame Data ]
       |
       v
 +-----------+
 |  Encoder  |  (Convolutional Neural Network / Downsampling)
 +-----+-----+
       |
       v
 [ Latent Space ] --> [ Quantization & Entropy Coding ] --> (.nvc binary file)
       |
       v
 +-----------+
 |  Decoder  |  (Deconvolutional / Generative Network)
 +-----+-----+
       |
       v
[ Reconstructed Frame ]
```

Core model: a Variational Autoencoder (VAE) with simple temporal
conditioning between frames. None of the boxes above are implemented yet -
this diagram describes the target design that the repository structure was
built to support.

## Repository Structure

```
neural_streaming/
├── configs/                 # JSON config files (default.json)
├── data/
│   ├── raw/                 # Source video files you place here (gitignored)
│   ├── frames/              # Extracted PNG frames (gitignored)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/           # manifest.json lives here (gitignored)
├── docs/                    # Design notes, written up as work progresses
├── notebooks/                # Exploratory notebooks
├── outputs/
│   ├── checkpoints/         # Model weights (gitignored)
│   ├── compressed/          # .nvc files (gitignored)
│   ├── reconstructed/       # Decoded frames/video (gitignored)
│   ├── metrics/             # Evaluation results (gitignored)
│   └── visualizations/      # Plots/figures (gitignored)
├── scripts/
│   ├── check_environment.py
│   └── prepare_dataset.py
├── src/
│   └── nvc/
│       ├── data/            # video_utils, frame_extraction, dataset_prep (implemented)
│       ├── models/          # Encoder/decoder networks (empty package)
│       ├── compression/     # Quantization, entropy coding, .nvc I/O (empty package)
│       ├── evaluation/      # PSNR/MS-SSIM/etc. metrics (empty package)
│       └── utils/           # Config (implemented)
├── tests/
│   ├── test_project_setup.py
│   └── test_dataset_preparation.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Prerequisites

- Python 3.11 or newer
- Git
- FFmpeg (system install, used later for video I/O alongside OpenCV)
- Optional: an NVIDIA GPU with recent drivers, if you want CUDA acceleration
  later. Not required - the project runs on CPU.

## Environment Setup (Windows PowerShell)

Run these from the project root (`neural_streaming/`):

```powershell
# 1. Create the virtual environment
py -3.11 -m venv .venv

# 2. Activate it
.venv\Scripts\Activate.ps1

# 3. Upgrade pip, then install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the local `nvc` package in editable mode
#    (required so `import nvc` works from scripts/ and tests/)
pip install -e .
```

### PyTorch: CPU vs. CUDA

`pip install -r requirements.txt` installs a PyTorch build that works on
CPU-only machines out of the box. If you have an NVIDIA GPU and want CUDA
acceleration:

1. Check your GPU driver's supported CUDA version (`nvidia-smi`).
2. Visit https://pytorch.org/get-started/locally/ and copy the install
   command matching your OS, package manager, and CUDA version.
3. Run that command (it will replace the CPU build of `torch`/`torchvision`
   with a CUDA-enabled one).

You do not need to do this to develop or test the project - CUDA is
detected automatically and used only when available (see below).

### FFmpeg installation (Windows)

Check whether it's already installed:

```powershell
ffmpeg -version
```

If not found, install one of these ways:

```powershell
# Option A: winget (built into modern Windows)
winget install Gyan.FFmpeg

# Option B: Chocolatey, if you have it installed
choco install ffmpeg
```

Or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its
`bin/` folder to your PATH manually. Re-open PowerShell afterward and
re-run `ffmpeg -version` to confirm.

## Running the Environment Check

After activating `.venv` and installing dependencies:

```powershell
python scripts\check_environment.py
```

This prints the detected Python, PyTorch, OpenCV, and FFmpeg versions,
whether CUDA is available (and the GPU name if so), and whether all
required project directories exist. It reports clear `[OK]` / `[WARN]` /
`[MISSING]` lines rather than raw stack traces for expected problems (e.g.
a missing dependency or missing FFmpeg).

## CPU / CUDA Behavior

The project must run correctly on CPU-only machines. `torch.cuda.is_available()`
is used wherever device selection matters (once training/inference code
exists) so that:
- On a machine with a supported NVIDIA GPU and CUDA-enabled PyTorch, the
  GPU is used automatically.
- On any other machine, everything falls back to CPU with no code changes
  required.

`scripts/check_environment.py` reports which case applies to your machine.

## Dataset Preparation

**Status: implemented (Milestone 2).** This stage turns raw videos into a
resized, split, PNG frame dataset. It does not train anything and does not
touch compression - it only prepares data for a future PyTorch `Dataset`.

### Where raw videos go

Place source video files directly in `data/raw/` (no subfolders). This
directory is gitignored - videos are never committed.

### Supported formats

`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` (configurable via
`supported_video_extensions` in `configs/default.json`). Whether a given
file actually opens depends on your installed OpenCV/FFmpeg codec support,
not just its extension - corrupt or codec-unsupported files are detected
and skipped with a clear message rather than crashing the whole run.

### How frames are extracted

`src/nvc/data/video_utils.py` opens each video with OpenCV and validates it
(exists, readable, sane dimensions) before anything else touches it.
`src/nvc/data/frame_extraction.py` then reads frames sequentially and saves
the sampled ones - OpenCV frames are kept in their native BGR order and
written directly with `cv2.imwrite`, which expects BGR, so colors come out
correct with no extra conversion step.

### Why PNG

Frames are saved as PNG (lossless) rather than JPEG. This project measures
*neural* compression quality (PSNR/MS-SSIM, in a later milestone) against
the original video - saving training data as JPEG would bake in a second,
uncontrolled lossy compression step and make later quality comparisons
meaningless.

### Resizing / cropping behavior

By default (`preserve_aspect_ratio=True`), a frame is resized so it fully
covers the target `--width`/`--height` box, then deterministically
center-cropped down to the exact target size. This never stretches the
image. Pass `--no-preserve-aspect-ratio` to instead resize directly to the
target size, which will distort frames whose aspect ratio differs from the
target - only do this deliberately.

### Sampling

`--every-n-frames N` keeps 1 out of every N frames from each video
(`1` = every frame, the default). Output frames are numbered sequentially
in output order, e.g. `clip_000001.png`, `clip_000002.png`, ... - not by
their original position in the source video.

### Train / validation / test split (and why it's leak-free)

Splitting happens **per video**, not per frame: each source video is
assigned entirely to train, val, or test (default 80/10/10, configurable
via `--train-ratio`/`--val-ratio`/`--test-ratio`), and *all* of its
extracted frames go to that split's folder. This is what prevents data
leakage - if frames were split independently, near-duplicate frames from
the same clip could end up in both the training and test sets, making
evaluation results meaningless. The split is deterministic for a given
`--seed` (default from config), so re-running the script reproduces the
same assignment.

Two videos with the same filename stem (e.g. `clip.mp4` and `clip.mov`)
would overwrite each other's output frames, so this is rejected up front
with a clear error instead of silently corrupting the dataset - rename one
of the files.

### How to run it

```powershell
# Uses configs/default.json defaults (256x256, every frame, 80/10/10 split)
python scripts\prepare_dataset.py --input data\raw

# Common overrides
python scripts\prepare_dataset.py `
    --input data\raw `
    --width 256 `
    --height 256 `
    --every-n-frames 2 `
    --seed 42

python scripts\prepare_dataset.py --help
```

### Generated directory structure

```
data/
├── raw/
│   └── <your videos>
├── frames/
│   ├── train/<video>_000001.png ...
│   ├── val/<video>_000001.png ...
│   └── test/<video>_000001.png ...
└── processed/
    └── manifest.json
```

### Manifest

Every run writes `data/processed/manifest.json` (path configurable via
`--manifest`), containing the settings used, one record per successfully
processed video (source video, split, original resolution/FPS/frame count,
extracted frame count, target resolution, output directory, and filename
pattern), a list of any skipped videos with the reason, and a summary
(videos/frames per split, totals, approximate output size). Frame paths
are stored as a directory + filename pattern rather than one entry per
frame, since listing every frame individually would make the manifest
grow unnecessarily large. This manifest is regenerated by the script (not
committed to git, since it describes locally-present raw videos) and is
meant to be read by later training/evaluation code.

## 12-Week Roadmap (Minor Project Scope)

This roadmap covers the core neural codec only (proof of concept, local
compression). Real-time streaming, neural frame interpolation, and
super-resolution are out of scope for this phase and are deferred to a
later major project. Milestones below are a planning draft, not a fixed
contract, and each milestone requires explicit sign-off before starting.

| Week | Focus |
|------|-------|
| 1-2  | Repository foundation, environment setup, dataset selection strategy *(done - Milestone 1)* |
| 3-4  | Frame extraction pipeline (OpenCV), train/val/test dataset preparation *(done - Milestone 2)* |
| 5-6  | Encoder network: CNN downsampling architecture (VAE encoder) |
| 7    | Latent space design, quantization, and entropy coding groundwork |
| 8    | `.nvc` binary serialization format and decoder network (deconvolutional/generative) |
| 9    | End-to-end encode/decode pipeline integration and training loop |
| 10   | Evaluation: PSNR, MS-SSIM, MSE, BPP, compression ratio, encode/decode timing; baseline comparison vs. H.264/H.265 |
| 11   | Experiments, tuning, visualizations, documentation |
| 12   | Final report, demo, cleanup, presentation prep |

Development proceeds milestone by milestone with explicit approval before
moving to the next one.
