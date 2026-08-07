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

**Milestone 4 (this repository state): repository foundation + dataset ingestion + PyTorch data pipeline + a baseline convolutional autoencoder with a full training/checkpoint/reconstruction pipeline.**

This is a deterministic reconstruction baseline, not a VAE - it exists to
validate the model/training/checkpoint/inference pipeline before Milestone 5
introduces stochastic latents, quantization, and entropy coding.

Implemented:
- Repository/directory structure
- Python environment configuration (`requirements.txt`, `pyproject.toml`)
- A typed configuration mechanism (`src/nvc/utils/config.py`), now including
  preprocessing and data-loading settings
- `scripts/check_environment.py` environment verification
- Video metadata inspection (`src/nvc/data/video_utils.py`)
- Image-sequence (folder-of-frames, e.g. DAVIS-style) discovery and
  validation (`src/nvc/data/sequence_utils.py`)
- Frame extraction with configurable sampling and deterministic resize/crop,
  shared by both source types (`src/nvc/data/frame_extraction.py`)
- A small `DatasetSource` strategy abstraction (`src/nvc/data/sources.py`)
  with `VideoDatasetSource` and `ImageSequenceDatasetSource` implementations
- Item-level train/val/test splitting, unified manifest generation, and
  dataset statistics for either source type (`src/nvc/data/ingest.py`),
  plus source-type auto-detection
- The original video-only pipeline (`src/nvc/data/dataset_prep.py`) is kept
  as-is for backward compatibility
- `scripts/prepare_dataset.py` CLI supporting `--source-type video`,
  `--source-type image-sequence`, or auto-detection
- **`FrameDataset`**, a `torch.utils.data.Dataset` that reads `manifest.json`
  and lazily loads RGB frame tensors (`src/nvc/data/frame_dataset.py`)
- Train/eval transform pipelines - horizontal flip + optional crop only
  (`src/nvc/data/transforms.py`)
- `create_train_loader`/`create_val_loader`/`create_test_loader` DataLoader
  factories with Windows-safe defaults (`src/nvc/data/loaders.py`)
- Dataset/tensor validation utilities (`src/nvc/data/validation.py`)
- `get_device()` and `seed_everything()` utilities
  (`src/nvc/utils/device.py`, `src/nvc/utils/seed.py`)
- `MSE`/`PSNR` metrics (`src/nvc/evaluation/basic_metrics.py`)
- `scripts/inspect_dataset.py` - reports dataset sizes/tensor stats and can
  save a sample visualization grid
- **`BaselineAutoencoder`** - a deterministic (non-variational) convolutional
  autoencoder (`src/nvc/models/encoder.py`, `decoder.py`, `autoencoder.py`)
  with an explicit `encode()`/`decode()`/`forward()` interface
- A training engine - epoch-level train/validate loops, checkpointing, and
  resume (`src/nvc/training/trainer.py`, `checkpoint.py`)
- `scripts/train_autoencoder.py` - training CLI with a CPU smoke-test mode
  (`--max-batches`) and `--resume`
- `scripts/reconstruct.py` - loads a checkpoint, reconstructs test-split
  frames, reports MSE/PSNR, saves an Original | Reconstruction comparison
- `scripts/plot_training_history.py` - plots epoch vs. train/val MSE and
  val PSNR from the training history JSON
- Compression package skeleton with no logic yet (`src/nvc/compression`)

**Not implemented yet** (do not assume any of this works):
- Variational latents (mu/logvar, KL divergence) - the current model is a
  plain deterministic autoencoder
- Quantization, entropy coding, or the `.nvc` format itself
- Video reassembly
- Perceptual/adversarial/SSIM/rate losses - training uses plain MSE only
- MS-SSIM / BPP / compression-ratio / timing evaluation
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

Target core model: a Variational Autoencoder (VAE) with simple temporal
conditioning between frames, plus quantization and entropy coding for a real
`.nvc` bitstream. **Current state (Milestone 4):** the Encoder/Decoder boxes
are implemented as a deterministic (non-variational) baseline - see
"Baseline Autoencoder" below - trained with plain MSE. The
`[ Quantization & Entropy Coding ]` step and the `.nvc` binary format are
not implemented yet; today `encode()` returns an unquantized float32 latent
tensor.

## Repository Structure

```
neural_streaming/
├── configs/                 # JSON config files (default.json)
├── data/
│   ├── raw/                 # Video datasets: source video files go here (gitignored)
│   ├── external/            # Extracted third-party datasets, e.g. DAVIS (gitignored)
│   ├── frames/               # Extracted PNG frames (gitignored)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── processed/           # manifest.json lives here (gitignored)
├── docs/                    # Design notes, written up as work progresses
├── notebooks/                # Exploratory notebooks
├── outputs/
│   ├── checkpoints/         # latest.pt, best.pt, history.json (gitignored)
│   ├── compressed/          # .nvc files (gitignored)
│   ├── reconstructed/       # Decoded frames/video (gitignored)
│   ├── metrics/             # Evaluation results (gitignored)
│   └── visualizations/      # dataset_grid.png, reconstructions.png, training_curves.png (gitignored)
├── scripts/
│   ├── check_environment.py
│   ├── prepare_dataset.py         # video and image-sequence ingestion CLI
│   ├── inspect_dataset.py         # PyTorch data pipeline sanity-check / visualization CLI
│   ├── train_autoencoder.py       # BaselineAutoencoder training CLI (Milestone 4)
│   ├── reconstruct.py             # checkpoint -> reconstructed frames + MSE/PSNR (Milestone 4)
│   └── plot_training_history.py   # training curve plots from history.json (Milestone 4)
├── src/
│   └── nvc/
│       ├── data/
│       │   ├── video_utils.py       # video validation + metadata
│       │   ├── sequence_utils.py    # image-sequence validation + metadata
│       │   ├── frame_extraction.py  # shared sampling/resize/crop/PNG export
│       │   ├── sources.py           # DatasetSource strategy (video / image-sequence)
│       │   ├── ingest.py            # unified pipeline used by the CLI
│       │   ├── dataset_prep.py      # original Milestone 2 video-only pipeline
│       │   ├── errors.py            # shared DatasetSourceError base
│       │   ├── validation.py        # DatasetValidationError + frame tensor checks
│       │   ├── frame_dataset.py     # FrameDataset (torch.utils.data.Dataset)
│       │   ├── transforms.py        # train/eval transform pipelines
│       │   └── loaders.py           # DataLoader factories
│       ├── models/
│       │   ├── encoder.py           # Encoder: strided-conv downsampler
│       │   ├── decoder.py           # Decoder: transposed-conv upsampler
│       │   └── autoencoder.py       # BaselineAutoencoder (encode/decode/forward)
│       ├── training/
│       │   ├── trainer.py           # train_one_epoch / validate_one_epoch
│       │   └── checkpoint.py        # save_checkpoint / load_checkpoint / resume_training_state
│       ├── compression/     # Quantization, entropy coding, .nvc I/O (empty package)
│       ├── evaluation/
│       │   └── basic_metrics.py     # MSE, PSNR (MS-SSIM not yet implemented)
│       └── utils/
│           ├── config.py            # Config dataclass (implemented)
│           ├── device.py            # get_device()
│           └── seed.py              # seed_everything()
├── tests/
│   ├── helpers.py                    # shared synthetic video/sequence/manifest fixtures
│   ├── test_project_setup.py
│   ├── test_dataset_preparation.py   # video pipeline (Milestone 2)
│   ├── test_dataset_ingestion.py     # image-sequence + unified pipeline (Milestone 2.5)
│   ├── test_pytorch_pipeline.py      # FrameDataset/transforms/loaders/metrics (Milestone 3)
│   └── test_baseline_autoencoder.py  # model/training/checkpoint/reconstruction (Milestone 4)
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

## Dataset Ingestion

**Status: implemented (Milestone 2.5).** This stage turns raw source data
into a resized, split, PNG frame dataset. It does not train anything and
does not touch compression - it only prepares data for a future PyTorch
`Dataset`.

### Supported dataset types

Two source types, both producing the same kind of output:

1. **`video`** - a folder of raw video files (Milestone 2's original
   pipeline). Frames are decoded from each video with OpenCV.
2. **`image-sequence`** - a folder whose direct subfolders are each one
   already-extracted sequence of images, e.g. DAVIS's
   `JPEGImages/480p/<sequence>/00000.jpg` layout. Any dataset using that
   same "one folder per sequence" convention works - nothing is
   DAVIS-specific.

If `--source-type` is omitted, it's auto-detected from `--input`: if video
files sit directly inside it, it's treated as `video`; otherwise, if its
subfolders contain supported images, it's treated as `image-sequence`. The
CLI prints which one it picked.

### Supported layouts

```
Video:                          Image sequence:
data/raw/                       <root>/
├── clip1.mp4                   ├── bear/
├── clip2.avi                   │   ├── 00000.jpg
└── ...                         │   ├── 00001.jpg
                                 │   └── ...
                                 ├── camel/
                                 │   └── ...
                                 └── ...
```

Supported video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
(`supported_video_extensions` in `configs/default.json`). Supported image
extensions: `.jpg`, `.jpeg`, `.png` (`supported_image_extensions`), and a
sequence folder may mix them. Whether a file actually opens depends on your
installed OpenCV/FFmpeg codec support, not just its extension - corrupt,
empty, unreadable, or inconsistent-resolution items are detected and
skipped with a clear message rather than crashing the whole run.

### How to import DAVIS

1. Download the DAVIS 2017 TrainVal 480p archive and extract it anywhere,
   e.g. `D:\Datasets\DAVIS\`. Extracted, it contains
   `DAVIS\JPEGImages\480p\<sequence>\*.jpg`.
2. Point `--input` at the `480p` folder (the one whose direct children are
   the sequence folders):
   ```powershell
   python scripts\prepare_dataset.py `
       --source-type image-sequence `
       --input "D:\Datasets\DAVIS\JPEGImages\480p" `
       --output data\frames `
       --manifest data\processed\manifest.json `
       --width 256 --height 256
   ```
   (`--source-type` can be omitted - a folder of subfolders full of `.jpg`
   files auto-detects as `image-sequence`.)

### How to import a normal MP4 dataset

Unchanged from Milestone 2: drop videos directly into `data/raw/` (no
subfolders) and run:

```powershell
python scripts\prepare_dataset.py --input data\raw
```

### How frames are obtained and processed

`src/nvc/data/video_utils.py` and `src/nvc/data/sequence_utils.py` each
validate their kind of item (exists, readable, sane/consistent dimensions)
before anything else touches it, raising a clear typed error otherwise.
`src/nvc/data/frame_extraction.py` then reads frames one at a time - either
decoded from the video stream or read from each image file - and both
paths call the *same* `resize_frame()`/`save_frame_png()` code, so
resizing, cropping, sampling, and PNG output behave identically regardless
of source. OpenCV frames are kept in native BGR order and written directly
with `cv2.imwrite`, which expects BGR, so colors come out correct with no
extra conversion step.

### Why PNG

Frames are saved as PNG (lossless) rather than JPEG - including for
sequences that started out as JPEG. This project measures *neural*
compression quality (PSNR/MS-SSIM, in a later milestone) against the
original frames; re-saving as JPEG here would bake in an extra, uncontrolled
lossy step and make later quality comparisons meaningless.

### Resizing / cropping behavior

By default (`preserve_aspect_ratio=True`), a frame is resized so it fully
covers the target `--width`/`--height` box, then deterministically
center-cropped down to the exact target size. This never stretches the
image. Pass `--no-preserve-aspect-ratio` to instead resize directly to the
target size, which will distort frames whose aspect ratio differs from the
target - only do this deliberately.

### Sampling

`--every-n-frames N` keeps 1 out of every N frames from each video/sequence
(`1` = every frame, the default). Output frames are numbered sequentially
in output order, e.g. `bear_000001.png`, `bear_000002.png`, ... - not by
their original position in the source.

### Train / validation / test split (and why it's leak-free)

Splitting happens **per item** (a whole video, or a whole sequence folder),
not per frame: each item is assigned entirely to train, val, or test
(default 80/10/10, configurable via `--train-ratio`/`--val-ratio`/
`--test-ratio`), and *all* of its extracted frames go to that split's
folder. This is what prevents data leakage - if frames were split
independently, near-duplicate frames from the same clip/sequence could end
up in both the training and test sets, making evaluation meaningless. The
split is deterministic for a given `--seed`, so re-running the script
reproduces the same assignment. This logic (`assign_splits()`) is shared
by both source types, not reimplemented per type.

Two items with the same name (e.g. `clip.mp4`/`clip.mov`, or two sequence
folders both named `bear`) would overwrite each other's output frames, so
this is rejected up front with a clear error instead of silently
corrupting the dataset - rename one of them.

### How to run it

```powershell
# Video, auto-detected (uses configs/default.json defaults)
python scripts\prepare_dataset.py --input data\raw

# Video, explicit source type + overrides
python scripts\prepare_dataset.py `
    --source-type video `
    --input data\raw `
    --width 256 --height 256 --every-n-frames 2 --seed 42

# Image-sequence (e.g. DAVIS)
python scripts\prepare_dataset.py `
    --source-type image-sequence `
    --input "D:\Datasets\DAVIS\JPEGImages\480p"

python scripts\prepare_dataset.py --help
```

### Generated directory structure

```
data/
├── raw/                          # (video workflow only)
│   └── <your videos>
├── frames/
│   ├── train/<name>_000001.png ...
│   ├── val/<name>_000001.png ...
│   └── test/<name>_000001.png ...
└── processed/
    └── manifest.json
```

### Manifest

Every run writes `data/processed/manifest.json` (path configurable via
`--manifest`), with a schema that's identical regardless of source type -
downstream training/evaluation code does not need to know whether frames
came from a video or an image sequence. It contains the settings used, one
record per successfully processed item (`source_type`, `source_name`,
`split`, `frame_count`, `original_resolution`, `processed_resolution`,
`frame_directory`, plus a few extras like FPS where applicable), a list of
any skipped items with the reason, and a summary (items/frames per split,
totals, average frames per item, approximate output size). Frame paths are
stored as a directory + filename pattern rather than one entry per frame,
since listing every frame individually would make the manifest grow
unnecessarily large. This manifest is regenerated by the script (not
committed to git, since it describes locally-present raw data).

### Backward compatibility note

`src/nvc/data/dataset_prep.py` (Milestone 2's original video-only
`prepare_dataset()` function) is unchanged and still works if imported
directly - `scripts/prepare_dataset.py` now calls the newer, source-agnostic
`src/nvc/data/ingest.py` instead, which supports both source types and
produces the unified manifest described above.

## PyTorch Data Pipeline

**Status: implemented (Milestone 3).** This is the `torch.utils.data`
layer every future model (autoencoder, VAE, quantizer, ...) will consume.
It only loads and batches frames - there is no neural network, no
compression, and nothing here trains anything.

### FrameDataset

`src/nvc/data/frame_dataset.py` implements `FrameDataset(torch.utils.data.Dataset)`.
It reads `manifest.json` directly (not by globbing `data/frames/`), so it
works identically whether the underlying frames came from raw videos or an
image-sequence dataset like DAVIS - exactly the point of Milestone 2.5's
unified manifest. Construction only parses the (small) manifest JSON and
builds a flat list of file paths per split; **no image is read or decoded
until `__getitem__` is called**, so datasets of any size never get
preloaded into RAM.

Each `__getitem__` call:
1. Reads the PNG/JPEG with OpenCV (`cv2.imread`) - consistent with the
   rest of `src/nvc/data/`, which already uses OpenCV throughout.
2. Converts BGR to RGB (`cv2.cvtColor(..., cv2.COLOR_BGR2RGB)`) - OpenCV
   always loads into memory as BGR regardless of the file's actual
   (correct) colors, so this conversion is required on every read. This
   is the mirror image of `frame_extraction.py`'s write path, which
   intentionally does *not* convert, because `cv2.imwrite` expects BGR.
3. Converts to a `torch.float32` tensor, shape `[3, H, W]`, scaled to
   `[0, 1]` (dividing by 255) - **no ImageNet normalization**, since this
   is a reconstruction task, not classification; the model needs to
   reproduce actual pixel values, not classify against a pretrained
   feature distribution.
4. Validates the tensor's dtype/shape/channel-count/pixel-range
   (`src/nvc/data/validation.py`) and applies the split's transform.

Missing manifests and missing frame directories raise a clear
`DatasetValidationError` at construction time (fail fast, before any
training loop starts); a missing/corrupt individual image raises the same
error at `__getitem__` time instead of a confusing `None`/`AttributeError`
from OpenCV's silent failure.

### Transforms

`src/nvc/data/transforms.py` provides `build_train_transform(crop_size=None)`
and `build_eval_transform(crop_size=None)`. Training may apply an optional
fixed-size `RandomCrop` plus `RandomHorizontalFlip` (p=0.5) - nothing else.
Validation/test apply the deterministic counterpart (`CenterCrop` to the
same size, no flip) so eval results are reproducible run to run. No
vertical flip, color jitter, rotation, or perspective warp - those would
teach the model to reconstruct frames that don't look like real video.

### DataLoader factories

`src/nvc/data/loaders.py` provides `create_train_loader()`,
`create_val_loader()`, and `create_test_loader()`. Train shuffles (with a
seeded `torch.Generator` for reproducible epoch order); validation and
test never shuffle. All three default to `num_workers=0` - the
Windows-safe default, since worker processes on Windows are spawned (not
forked) and require the caller's entry point to be guarded by
`if __name__ == "__main__":`; pass `num_workers` > 0 explicitly once
you've set that up.

### Dataset validation

`src/nvc/data/validation.py` defines `DatasetValidationError` and
`validate_frame_tensor()`, checking dtype (`float32`), shape (`[C, H, W]`),
channel count (3), dimensions (against the manifest's recorded target
resolution, when known), and pixel range (`[0, 1]`). `FrameDataset` runs
this on every sample it loads; it's also directly unit-testable against
synthetic bad tensors.

### Visualization / inspection

```powershell
python scripts\inspect_dataset.py
python scripts\inspect_dataset.py --visualize
```

Prints total/train/val/test frame and batch counts, a sample batch's
shape/dtype/pixel range, and the selected device. `--visualize` saves a
grid of sample training frames to `outputs/visualizations/dataset_grid.png`
via `torchvision.utils.make_grid` + matplotlib - since `FrameDataset`
already converted BGR to RGB on load, no further color correction is
needed before display.

### Device utility

`src/nvc/utils/device.py` provides `get_device()`: returns
`torch.device("cuda")` and prints the GPU name + VRAM when a CUDA GPU is
available, otherwise returns `torch.device("cpu")` and prints that. Never
mandatory - everything in this project runs on CPU.

### Reproducibility

`src/nvc/utils/seed.py` provides `seed_everything(seed, deterministic=False)`,
seeding Python's `random`, NumPy, and PyTorch (CPU and any GPU) together.
`deterministic=True` additionally requests PyTorch's deterministic
algorithms - opt-in, since it can be slower and some ops don't support it.

### Metrics

`src/nvc/evaluation/basic_metrics.py` implements only `mse()` and `psnr()`
(no MS-SSIM yet). For identical inputs, `mse() == 0.0` and
`psnr() == float("inf")` - handled explicitly rather than relying on
floating-point divide-by-zero behavior.

## Baseline Autoencoder

**Status: implemented (Milestone 4).** A deterministic (non-variational)
convolutional autoencoder that establishes a reconstruction baseline and
validates the full model/training/checkpoint/inference pipeline. This is
**not** a VAE: no `mu`/`logvar`, no KL divergence, no quantization, no
entropy coding, and no rate-distortion loss - those come later, once this
baseline pipeline is trusted. Training loss is plain MSE.

### Architecture

```
RGB frame [3, 256, 256]
        |
        v
  Encoder (4x Conv2d, kernel=4 stride=2 pad=1, ReLU between)
    3 -> 32 -> 64 -> 128 -> latent_channels   (256 -> 128 -> 64 -> 32 -> 16)
        |
        v
  Latent [latent_channels, 16, 16]   (spatial, not flattened)
        |
        v
  Decoder (4x ConvTranspose2d, kernel=4 stride=2 pad=1, ReLU between, Sigmoid on output)
    latent_channels -> 128 -> 64 -> 32 -> 3   (16 -> 32 -> 64 -> 128 -> 256)
        |
        v
  Reconstructed RGB frame [3, 256, 256], values in [0, 1]
```

`kernel_size=4, stride=2, padding=1` was chosen specifically because it
halves (Encoder) or doubles (Decoder) spatial size *exactly* with no
rounding, so the reconstruction shape always matches the input shape with
no final resizing hack. Input height/width must each be divisible by 16
(four stride-2 steps); `Encoder` raises a clear `ValueError` otherwise.

`src/nvc/models/autoencoder.py` implements `BaselineAutoencoder` with an
explicit interface - `model.encode(x)`, `model.decode(z)`, and
`model(x)` (equivalent to `decode(encode(x))`) - so a later milestone can
insert a quantizer and entropy coder between `encode` and `decode` without
restructuring this class. `model.config_dict()` returns the architecture
config (`in_channels`, `latent_channels`, `base_channels`) needed to rebuild
the model from a checkpoint, and `model.num_parameters()` reports the
trainable parameter count.

With default settings (`latent_channels=64`, `base_channels=32`): **593,411
trainable parameters**, latent shape `[64, 16, 16]`.

### Training

`src/nvc/training/trainer.py` provides `train_one_epoch()` (real gradient
updates via `Adam`) and `validate_one_epoch()` (`torch.no_grad()`, no
parameter updates), both operating on the existing `FrameDataset`
DataLoaders and reusing `mse()`/`psnr()` from `src/nvc/evaluation/basic_metrics.py`
and `get_device()`/`seed_everything()` from `src/nvc/utils/`. Both accept a
`max_batches` cap, used for the CPU smoke test below.

Per-epoch history (`train_loss`, `val_loss`, `val_psnr`, `elapsed_seconds`)
is stored as plain Python dicts and written to `outputs/checkpoints/history.json`
after every epoch - machine-readable, not fabricated: only epochs that
actually ran are recorded.

Only `mse()` reconstruction loss is used - no KL divergence, perceptual
loss, adversarial loss, SSIM loss, or rate loss yet.

```powershell
# A real training run (uses configs/default.json defaults)
python scripts\train_autoencoder.py --epochs 50

# Override hyperparameters
python scripts\train_autoencoder.py --epochs 50 --batch-size 8 --learning-rate 1e-4 --latent-channels 64

python scripts\train_autoencoder.py --help
```

### Checkpoints

`src/nvc/training/checkpoint.py` provides `save_checkpoint()`/`load_checkpoint()`/
`resume_training_state()`. Every epoch, `scripts/train_autoencoder.py` writes
`outputs/checkpoints/latest.pt`, and `outputs/checkpoints/best.pt` whenever
validation MSE improves. Each checkpoint contains the model state dict,
optimizer state dict, the last completed epoch, the full metric history, and
the model's architecture config (`model.config_dict()`) - enough to resume
training exactly or rebuild the model for inference without needing the
original CLI arguments. Checkpoint binaries (`*.pt`) are gitignored, not
committed.

### Resuming training

```powershell
python scripts\train_autoencoder.py --epochs 20 --resume outputs\checkpoints\latest.pt
```

`--resume` restores model weights, optimizer state, and history, and
continues epoch numbering from where the checkpoint left off (e.g. resuming
a checkpoint at epoch 5 with `--epochs 20` runs epochs 6-25) rather than
restarting at epoch 1. A `--latent-channels` value that doesn't match the
checkpoint's architecture fails with a clear error from PyTorch's own
`load_state_dict` shape check, caught and reprinted with a hint.

### CPU smoke test

Because CUDA is unavailable on this machine, `--max-batches` caps each
epoch to a handful of batches so the entire training path - data loading,
forward pass, loss, backward pass, optimizer step, checkpointing - can be
verified in seconds with real gradient updates. **This is a pipeline check,
not a claim of a trained model:**

```powershell
python scripts\train_autoencoder.py --epochs 1 --max-batches 5
```

### Reconstruction

`scripts/reconstruct.py` loads a checkpoint (rebuilding
`BaselineAutoencoder` from the checkpoint's saved `model_config`, not from
CLI flags), encodes and decodes real test-split frames via `FrameDataset`/
`create_test_loader`, reports MSE/PSNR on that batch, and saves an
Original | Reconstruction comparison image under `outputs/visualizations/`.

```powershell
python scripts\reconstruct.py --checkpoint outputs\checkpoints\best.pt
python scripts\reconstruct.py --checkpoint outputs\checkpoints\latest.pt --num-samples 4
```

### Training curve plots

`scripts/plot_training_history.py` reads `outputs/checkpoints/history.json`
and plots epoch vs. training MSE, validation MSE, and validation PSNR to
`outputs/visualizations/training_curves.png`. It only plots epochs actually
present in the history file - a smoke-test history with one or two epochs
produces a one- or two-point plot, not a fabricated curve.

```powershell
python scripts\plot_training_history.py
```

### Raw latent dimensionality ratio - not a compression ratio

`scripts/train_autoencoder.py` reports `input_elements / latent_elements`
(e.g. `3*256*256 / (64*16*16) = 12.0`) as the **"raw latent dimensionality
ratio."** This is explicitly *not* a compression ratio: the latent tensor is
still `float32` and has not been quantized or entropy-coded, so it does not
represent an actual bitstream size. Real compression-ratio and
bits-per-pixel (BPP) numbers require quantization and entropy coding, which
are out of scope for this milestone.

## 12-Week Roadmap (Minor Project Scope)

This roadmap covers the core neural codec only (proof of concept, local
compression). Real-time streaming, neural frame interpolation, and
super-resolution are out of scope for this phase and are deferred to a
later major project. Milestones below are a planning draft, not a fixed
contract, and each milestone requires explicit sign-off before starting.

| Week | Focus |
|------|-------|
| 1-2  | Repository foundation, environment setup, dataset selection strategy *(done - Milestone 1)* |
| 3-4  | Frame extraction pipeline (OpenCV), train/val/test dataset preparation, and generalized ingestion for image-sequence datasets like DAVIS *(done - Milestones 2 & 2.5)* |
| 5-6  | PyTorch data pipeline (`FrameDataset`, transforms, DataLoaders) *(done - Milestone 3)*; baseline convolutional autoencoder (encoder/decoder CNN, MSE training loop, checkpointing, reconstruction CLI) *(done - Milestone 4)* |
| 7    | Latent space design, quantization, and entropy coding groundwork |
| 8    | `.nvc` binary serialization format and a variational encoder/decoder (mu/logvar, KL divergence) |
| 9    | End-to-end encode/decode pipeline integration and training loop |
| 10   | Evaluation: PSNR, MS-SSIM, MSE, BPP, compression ratio, encode/decode timing; baseline comparison vs. H.264/H.265 |
| 11   | Experiments, tuning, visualizations, documentation |
| 12   | Final report, demo, cleanup, presentation prep |

Development proceeds milestone by milestone with explicit approval before
moving to the next one.
