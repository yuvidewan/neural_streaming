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

**Milestone 6.5 (this repository state): a working end-to-end neural image codec (Milestone 6, unchanged) plus large-scale training data pipeline support for Vimeo-90K Septuplet.** Milestone 6.5 is architecture/pipeline only - **no retraining has happened**; the model, quantizer, entropy model, and `.nvc` format are exactly as Milestone 6 left them.

Measured on the DAVIS test split: **1.88 BPP at 27.17 dB (12.75x vs raw
uint8 RGB)** at 8-bit, or **1.38 BPP at 27.10 dB (17.41x)** at 6-bit.

The model is a deterministic autoencoder, not a VAE, and the entropy model
is a static counted table - no learned/context/hyperprior model, no
quantization-aware training, and no inter-frame prediction, so this
compresses **still frames**, not yet video.

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
- **`UniformQuantizer`** - uniform scalar affine quantization of latents,
  global and per-channel modes, arbitrary bit widths
  (`src/nvc/compression/quantization.py`)
- Raw latent storage arithmetic (`src/nvc/compression/storage_analysis.py`)
- Latent extraction and descriptive statistics
  (`src/nvc/evaluation/latent_analysis.py`)
- `scripts/analyze_latent.py` - latent statistics + distribution plots
- `scripts/quantization_experiment.py` - end-to-end quantized
  reconstruction experiment across bit widths and modes
- **Fixed quantization calibration** from the training split only
  (`src/nvc/compression/calibration.py`)
- **`EmpiricalEntropyModel`** - static per-channel symbol frequency tables
  with smoothing (`src/nvc/compression/entropy_model.py`)
- **Arithmetic entropy coder** implemented from first principles
  (`src/nvc/compression/range_coder.py`)
- **The `.nvc` binary format** with `NVCWriter`/`NVCReader` and full header
  validation (`src/nvc/compression/nvc_format.py`)
- **End-to-end codec** (`src/nvc/compression/codec.py`) plus
  `scripts/calibrate_quantizer.py`, `encode.py`, `decode.py`,
  `benchmark_codec.py`
- Real measured BPP / compression-ratio benchmarking
- **Vimeo-90K Septuplet dataset support** - discovery, official leakage-safe
  train/test splitting, deterministic subset selection, and a lazy,
  non-duplicating sequence manifest (`src/nvc/data/vimeo.py`)
- **`SequenceFrameDataset`** - generic sequence-manifest-backed Dataset the
  existing model/training code consumes with zero changes
  (`src/nvc/data/sequence_dataset.py`)
- `scripts/prepare_training_dataset.py` - Vimeo validation/statistics and
  reproducible subset-manifest builder (does not train anything)

**Not implemented yet** (do not assume any of this works):
- Variational latents (mu/logvar, KL divergence) - the current model is a
  plain deterministic autoencoder
- Learned entropy models, hyperpriors, context or autoregressive models -
  the entropy model is a static counted table
- Quantization-aware training - the model was trained on float latents only
- Inter-frame / temporal prediction, so this codes still frames, not video
- Training on Vimeo-90K - the data pipeline exists, but no training run has
  used it yet; the current checkpoint is still DAVIS-only
- Video reassembly
- Perceptual/adversarial/SSIM/rate losses - training uses plain MSE only
- MS-SSIM evaluation, and any comparison against H.264/H.265
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
"Baseline Autoencoder" below - trained with plain MSE. As of Milestone 6
**every box in this diagram is implemented for still frames**, including
quantization, entropy coding, and the `.nvc` binary file. What remains is
depth rather than coverage: the VAE formulation, a learned entropy model,
and the temporal conditioning that would make this a *video* codec rather
than an image codec applied frame by frame.

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
│   ├── metrics/             # latent_statistics.json, quantization_results.json/.csv (gitignored)
│   └── visualizations/      # dataset/reconstruction/training/latent/quantization plots (gitignored)
├── scripts/
│   ├── check_environment.py
│   ├── prepare_dataset.py         # video and image-sequence ingestion CLI
│   ├── inspect_dataset.py         # PyTorch data pipeline sanity-check / visualization CLI
│   ├── train_autoencoder.py       # BaselineAutoencoder training CLI (Milestone 4)
│   ├── reconstruct.py             # checkpoint -> reconstructed frames + MSE/PSNR (Milestone 4)
│   ├── plot_training_history.py   # training curve plots from history.json (Milestone 4)
│   ├── analyze_latent.py          # latent statistics + distribution plots (Milestone 5)
│   ├── quantization_experiment.py # quantized reconstruction experiment (Milestone 5)
│   ├── calibrate_quantizer.py     # fixed calibration + entropy model (Milestone 6)
│   ├── encode.py                  # image -> .nvc (Milestone 6)
│   ├── decode.py                  # .nvc -> image (Milestone 6)
│   ├── benchmark_codec.py         # real BPP / ratio benchmark (Milestone 6)
│   └── prepare_training_dataset.py # Vimeo-90K validate/subset CLI (Milestone 6.5)
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
│       │   ├── sequence_dataset.py  # SequenceFrameDataset (generic, sequence-manifest-backed)
│       │   ├── vimeo.py             # Vimeo-90K discovery, leakage-safe splits, subsetting
│       │   ├── transforms.py        # train/eval transform pipelines
│       │   └── loaders.py           # DataLoader factories (Frame* and Sequence*)
│       ├── models/
│       │   ├── encoder.py           # Encoder: strided-conv downsampler
│       │   ├── decoder.py           # Decoder: transposed-conv upsampler
│       │   └── autoencoder.py       # BaselineAutoencoder (encode/decode/forward)
│       ├── training/
│       │   ├── trainer.py           # train_one_epoch / validate_one_epoch
│       │   └── checkpoint.py        # save/load/resume + load_model_from_checkpoint
│       ├── compression/
│       │   ├── quantization.py      # UniformQuantizer (global / per-channel)
│       │   ├── calibration.py       # fixed params from the train split
│       │   ├── entropy_model.py     # static per-channel frequency tables
│       │   ├── range_coder.py       # arithmetic coder (from scratch)
│       │   ├── nvc_format.py        # .nvc container: header/writer/reader
│       │   ├── codec.py             # end-to-end frame <-> .nvc
│       │   └── storage_analysis.py  # raw latent storage arithmetic
│       ├── evaluation/
│       │   ├── basic_metrics.py     # MSE, PSNR (MS-SSIM not yet implemented)
│       │   └── latent_analysis.py   # latent extraction + statistics
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
│   ├── test_baseline_autoencoder.py  # model/training/checkpoint/reconstruction (Milestone 4)
│   ├── test_quantization.py          # quantizer/latent analysis/storage/integrity (Milestone 5)
│   ├── test_entropy_coding.py        # calibration/entropy model/coder/.nvc/codec (Milestone 6)
│   └── test_vimeo_dataset.py         # Vimeo discovery/leakage/subsetting/DAVIS regression (Milestone 6.5)
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

## Latent Representation & Quantization

**Status: implemented (Milestone 5).** This milestone measures the learned
latent representation and establishes a quantization layer between encoder
and decoder. It is an *analysis and design* step, not the codec:
**entropy coding is not implemented yet**, and **`.nvc` bitstream generation
is not implemented yet**. The quantizer emits integer tensors, not a
compressed file.

```
frame -> Encoder -> float latent -> Quantizer -> integer latent
      -> Dequantizer -> approximate latent -> Decoder -> reconstruction
```

### What the latent actually is

`BaselineAutoencoder.encode()` returns a `[B, 64, 16, 16]` float32 tensor:
64 channels at 1/16 the input resolution in each spatial dimension. It stays
spatial rather than flattened, which is what makes per-channel scaling (and
later, spatial entropy models) natural.

Measured over the **full DAVIS test split (719 frames, 11,780,096 latent
values)** with the epoch-50 checkpoint - see
`outputs/metrics/latent_statistics.json` for the full record:

| Statistic | Value |
|---|---|
| min / max | -16.485 / 13.713 |
| mean / median | 0.121 / 0.099 |
| standard deviation | 2.237 |
| exactly zero | 0.0000% |
| near zero (abs < 0.01) | 0.6218% |
| per-channel std | 0.553 to 3.801 |
| per-channel range width | 6.77 to 23.96 (3.5x spread) |

Two findings drive the quantizer design:

1. **The distribution is sharply peaked at zero with long tails** (visible
   in `latent_histogram.png`, especially the log-count panel). Most mass
   sits within roughly +/-5, but the observed range runs to -16.5. A single
   global scale must stretch across those rare outliers, spending most of
   its grid on values that almost never occur.
2. **Channels have genuinely different dynamic ranges** (3.5x spread), so a
   per-channel scale can fit each channel's actual range.

There are essentially no exact zeros - the encoder's final layer is linear,
with no ReLU - so sparsity is not something this representation offers.
`latent_heatmap.png` shows the channels retain visible spatial structure
(horizon lines, subject position), confirming the representation is
spatially organized rather than scrambled.

### Why quantization is necessary

The latent is float32. Storing or transmitting 32 bits per latent value
wastes most of those bits: the decoder does not need that precision, and
entropy coding (a later milestone) needs a *discrete* symbol alphabet to
assign codewords to. Quantization maps the continuous latent onto a finite
integer grid, which is the prerequisite for any real bitstream.

### The quantizer

`src/nvc/compression/quantization.py` implements `UniformQuantizer`, a
uniform scalar **affine** quantizer written out longhand (not delegated to
`torch.ao.quantization`, which would hide the arithmetic this milestone
exists to measure).

For bit width `b`, with `q_min = 0` and `q_max = 2**b - 1`, over an observed
range `[x_min, x_max]`:

```
scale      = (x_max - x_min) / (q_max - q_min)
zero_point = q_min - round(x_min / scale)

quantize:    q     = clamp(round(x / scale) + zero_point, q_min, q_max)
dequantize:  x_hat = (q - zero_point) * scale
```

Two deliberate details:

- **`zero_point` is not clamped** into `[q_min, q_max]`. It is separate
  metadata, not packed into the integer field, so clamping would be a
  self-inflicted constraint that breaks ranges not straddling zero (a
  channel centered near 5.0 needs a large negative zero_point). Leaving it
  unclamped also makes exact `0.0` exactly representable whenever 0 lies
  inside the range - worth having, since latents are roughly zero-centered.
- **Zero-width ranges are widened** to `[c - 0.5, c + 0.5]`. A constant
  tensor or channel would otherwise give `scale = 0` and divide by zero.

### Global vs. per-channel scaling

| Mode | Parameters | Behavior |
|---|---|---|
| `global` | one (scale, zero_point) for the whole tensor | Cheapest metadata. One wide-range channel stretches the grid for every other channel. |
| `per_channel` | one pair per latent channel, computed over (B, H, W) | Adapts to each channel's range, at 64x the metadata. |

Which is better was **measured, not assumed** - see below.

### Quantization experiment results

Full DAVIS test split (719 frames), epoch-50 checkpoint, seed 42. MSE is
aggregated over every pixel of the split and PSNR derived from that single
aggregate (not a mean of per-batch PSNR). Full record in
`outputs/metrics/quantization_results.json` / `.csv`.

| Configuration | Bits | PSNR (dB) | dPSNR | Image MSE | Latent MSE | Latent MAE | Max abs latent err |
|---|---|---|---|---|---|---|---|
| Float32 baseline | 32 | 27.274 | - | 0.001873 | 0 | 0 | 0 |
| Global | 8 | 27.254 | -0.02 | 0.001882 | 0.000528 | 0.019602 | 0.0591 |
| Per-channel | 8 | 27.271 | -0.00 | 0.001875 | 0.000077 | 0.007116 | 0.0470 |
| Global | 6 | 26.967 | -0.31 | 0.002010 | 0.008643 | 0.079312 | 0.2391 |
| Per-channel | 6 | 27.232 | -0.04 | 0.001891 | 0.001255 | 0.028804 | 0.1901 |
| Global | 4 | 23.746 | -3.53 | 0.004221 | 0.149804 | 0.329055 | 1.0043 |
| Per-channel | 4 | 26.597 | -0.68 | 0.002189 | 0.022086 | 0.120815 | 0.7980 |

Conclusions from the measurement:

- **8-bit is effectively free.** Per-channel 8-bit costs 0.003 dB against
  the float32 baseline - below any perceptible threshold - while using a
  quarter of the raw storage.
- **Per-channel wins at every width, and the gap widens as bits shrink**
  (0.02 dB at 8-bit, 0.27 dB at 6-bit, 2.85 dB at 4-bit). This is the
  predicted consequence of the 3.5x per-channel range spread.
- **4-bit global collapses** (-3.53 dB, with visible blocking in
  `quantization_comparison.png`), whereas 4-bit per-channel holds up
  surprisingly well at -0.68 dB.
- Latent-space error and image-space error move together but are **not the
  same quantity** and are reported separately throughout.

Note that latent MSE is far larger in magnitude than image MSE - the decoder
is partially tolerant of latent perturbation, so latent error must not be
read as a proxy for reconstruction quality.

### Raw latent storage - NOT a compression ratio

`src/nvc/compression/storage_analysis.py` computes plain "values x bits"
arithmetic. For one `[3, 256, 256]` frame (196,608 values) against one
`[64, 16, 16]` latent (16,384 values):

| Representation | Bits | Bytes | Raw size ratio vs. uint8 RGB frame |
|---|---|---|---|
| Raw uint8 RGB frame | 1,572,864 | 196,608 | 1.00x |
| float32 latent | 524,288 | 65,536 | 3.00x |
| 8-bit latent | 131,072 | 16,384 | 12.00x |
| 6-bit latent | 98,304 | 12,288 | 16.00x |
| 4-bit latent | 65,536 | 8,192 | 24.00x |

**These are theoretical raw tensor storage figures, not compression ratios
and not codec bitrates.** They exclude:

- **Entropy coding**, which is not implemented. Given how peaked the latent
  distribution is, real entropy coding should beat these figures
  substantially - a uniform `b` bits per symbol is the worst case.
- **Quantization metadata.** Scale and zero_point are not counted (64
  channels x 2 values is negligible per frame, but it is not zero).
- **Bit packing.** 6-bit and 4-bit are counted at their theoretical cost;
  nothing actually packs them yet - in memory they sit in int32 tensors.
- A fair codec baseline. The comparison is against *raw* uint8 RGB, not
  against PNG/JPEG/H.264.

### Current limitations

- Scale and zero_point are calibrated **per batch from the tensor being
  quantized** ("dynamic"). A real codec must transmit them as side
  information or freeze them from a calibration set; neither is done yet.
- The autoencoder was trained **without** quantization in the loop, so the
  decoder has never seen quantized latents during training.
  Quantization-aware training is not implemented.
- No entropy coding, no arithmetic/range coding, no `.nvc` bitstream.
- The quantizer is applied at inference only and never touches the trained
  weights - verified by the model-integrity tests.

### How to run it

```powershell
python scripts\analyze_latent.py --checkpoint outputs\checkpoints\best.pt
python scripts\quantization_experiment.py --checkpoint outputs\checkpoints\best.pt

# Quick subset runs
python scripts\analyze_latent.py --checkpoint outputs\checkpoints\best.pt --max-batches 10
python scripts\quantization_experiment.py --checkpoint outputs\checkpoints\best.pt --max-batches 10
```

Outputs: `outputs/metrics/latent_statistics.json`,
`outputs/metrics/quantization_results.json` / `.csv`, and
`latent_histogram.png`, `latent_channel_statistics.png`,
`latent_heatmap.png`, `quantization_comparison.png` under
`outputs/visualizations/`.

## Entropy Coding & the .nvc Bitstream

**Status: implemented (Milestone 6).** This is the milestone where the
project becomes an actual codec: quantized latents are entropy-coded into a
real binary file, so the numbers below are **measured bitrate**, not
theoretical tensor storage.

```
encode:  frame -> Encoder -> latent -> fixed quantizer -> symbols
               -> arithmetic coder -> .nvc

decode:  .nvc -> arithmetic decoder -> symbols -> dequantizer
              -> approximate latent -> Decoder -> frame
```

Still **not** implemented: learned/neural entropy models, hyperpriors,
context or autoregressive models, quantization-aware training, and any
inter-frame or temporal prediction. The entropy model here is a static table
counted from calibration data.

### Fixed quantization calibration

Milestone 5 derived scale/zero_point from whichever tensor was being
quantized. That cannot be a codec - the decoder receives only a bitstream
and cannot re-derive parameters it never saw. `scripts/calibrate_quantizer.py`
therefore derives them **once, from the training split only**, and writes
them to a calibration file both sides load.

Method: **per-channel percentile ranges at (0.1, 99.9)**, the documented
default. Percentiles rather than min/max because the Milestone 5 analysis
found a sharply peaked distribution with long tails (range about
[-16.5, 13.7] against a standard deviation of 2.24) - letting a few extreme
values define the grid would spend most of the quantization steps on empty
space. Setting the percentiles to (0.0, 100.0) reproduces plain min/max
calibration exactly, so that option is retained rather than removed. The
percentiles were **not** tuned against validation or test data.

Clipping is the cost of that choice and is measured, not assumed. On the
400-frame training calibration set, 0.192% of values fell outside the
8-bit range (0.170% at 6-bit, 0.110% at 4-bit); on the test split the
encoder clips 0.447% / 0.418% / 0.318%.

### Entropy model

`src/nvc/compression/entropy_model.py` counts symbol occurrences in the
calibration data to estimate `P(symbol)`, with **one independent frequency
table per latent channel**. Because symbols are coded in channel-major
order, the decoder knows which table applies at every position without any
side information.

Two safeguards make it usable as a coding model:

- **No zero probabilities.** A symbol with probability 0 is unencodable, and
  calibration cannot be assumed to have seen every symbol. Add-one (Laplace)
  smoothing plus a hard floor of `MIN_FREQUENCY = 1` guarantee every symbol
  in `[0, 2**bits)` stays encodable.
- **Integer frequencies summing to exactly 65536.** The coder's interval
  arithmetic must be bit-identical on both sides; floats would drift. The
  rounding this introduces is reported as "probability quantization" in the
  efficiency breakdown below.

Measured on the calibration set (8-bit): all 256 symbols observed, aggregate
order-0 empirical entropy **7.306 bits/symbol**, per-channel model mean
**7.179 bits/symbol**, against 8.0 bits/symbol fixed width.

### Arithmetic coder

`src/nvc/compression/range_coder.py` implements integer arithmetic coding
from first principles - no compression library is called. Arithmetic coding
was chosen over Huffman because it is not restricted to whole-bit codewords,
which matters at low bit depths, and because it is the natural base for the
learned entropy models planned later.

The interval [low, high] is held in 32-bit fixed point and renormalized as
it narrows: entirely in the lower half emits a 0, entirely in the upper half
emits a 1, and the classic **underflow** case (straddling the midpoint but
inside [1/4, 3/4)) rescales around the midpoint and records a *pending* bit
to be emitted later with the opposite polarity. The decoder mirrors every
step, keeping both sides in lockstep. `total <= 2**30` guarantees the
interval products cannot overflow.

### .nvc binary format specification

Version 1. All multi-byte integers are **little-endian** and unsigned; no
implicit padding anywhere.

```
Header (37 fixed bytes + quantization parameter block)
├── magic                4 bytes   b"NVC1"
├── format_version       uint8     currently 1
├── quantization_bits    uint8
├── quantization_mode    uint8     0 = global, 1 = per_channel
├── entropy_coder_id     uint8     1 = static_arithmetic_v1
├── image_width          uint16
├── image_height         uint16
├── image_channels       uint8
├── latent_channels      uint16
├── latent_height        uint16
├── latent_width         uint16
├── symbol_count         uint32
├── num_quant_params     uint16    1 (global) or latent_channels
├── payload_length       uint32    bytes
├── entropy_model_id     8 bytes   first 8 bytes of a SHA-256
└── quantization params  num_quant_params x { float32 scale, float32 zero_point }

Payload
└── payload_length bytes of arithmetic-coded symbols (MSB-first bit packing)
```

`NVCWriter` / `NVCReader` (`src/nvc/compression/nvc_format.py`) validate
magic bytes, version, coder id, bit depth, every dimension, the
symbol_count/latent-dimension consistency, the parameter count against the
mode, truncated parameter blocks, truncated payloads, and trailing data.
Every failure raises `NVCFormatError` with a specific message rather than
decoding silently-wrong output.

Two deliberate design choices:

- **Quantization parameters are embedded** so a .nvc is self-describing for
  dequantization. At 64 channels that is 512 bytes - real overhead on a
  single frame, reported honestly rather than hidden (see the BPP figures,
  given both payload-only and total-file). A sequence-level header shared
  across frames is the obvious fix once this codes video.
- **The entropy model is not embedded** - it is a 64x256 table, constant
  across every frame from a given calibration. The header carries
  `entropy_model_id` instead, and decoding with a mismatched calibration
  raises rather than emitting garbage.

### Encode / decode workflow

```powershell
# 1. Calibrate once from the training split (writes outputs/calibration/)
python scripts\calibrate_quantizer.py --checkpoint outputs\checkpoints\best.pt

# 2. Encode an image to .nvc
python scripts\encode.py `
    --checkpoint outputs\checkpoints\best.pt `
    --input data\frames\test\bmx-bumps_000001.png `
    --output outputs\compressed\frame.nvc

# 3. Decode it back (--reference also reports PSNR)
python scripts\decode.py `
    --checkpoint outputs\checkpoints\best.pt `
    --input outputs\compressed\frame.nvc `
    --output outputs\reconstructed\frame.png `
    --reference data\frames\test\bmx-bumps_000001.png
```

Bit depth is configurable, not a separate codec: `--bits 6` at calibration
time produces a 6-bit calibration that the same encoder/decoder consume.

### Measured results - DAVIS test split (719 frames, real .nvc files)

| Bits | Mode | PSNR (dB) | Total BPP | Payload BPP | Ratio | Mean bytes | Payload b/sym | Clipped |
|---|---|---|---|---|---|---|---|---|
| 8 | per-channel | 27.167 | 1.8844 | 1.8174 | 12.75x | 15,437 | 7.2696 | 0.447% |
| 6 | per-channel | 27.103 | 1.3819 | 1.3149 | 17.41x | 11,320 | 5.2595 | 0.418% |
| 4 | per-channel | 26.237 | 0.8692 | 0.8022 | 27.80x | 7,120 | 3.2087 | 0.318% |

Spread at 8-bit across the 719 frames: total BPP 1.7603 (min) / 1.8844
(mean) / 2.1964 (max); compression ratio 10.93x to 13.63x.

**6-bit is the better operating point than 8-bit.** Going from 6 to 8 bits
costs 36% more bits for 0.064 dB - essentially nothing. That reverses the
Milestone 5 recommendation, which could not see bitrate because nothing was
entropy-coded yet.

Header overhead is **549 bytes = 3.56%** of the mean 8-bit file (13.2% of
the smaller 4-bit file, where the same fixed cost is spread over fewer
payload bytes).

### Three representations of the same frame (8-bit operating point)

| Representation | Bytes | BPP | Ratio | PSNR |
|---|---|---|---|---|
| A. Raw uint8 RGB | 196,608 | 24.0000 | 1.00x | - (lossless source) |
| B. Fixed-width quantized latent | 16,384 | 2.0000 | 12.00x | 27.167 dB |
| C. Entropy-coded .nvc (measured) | 15,437 | 1.8844 | 12.75x | 27.167 dB |

B and C have identical PSNR because entropy coding is lossless - C is
simply B stored more efficiently, plus a header.

### Entropy coding efficiency

| Bits | Fixed width | Test entropy | Model expected | Actual payload | Coder overhead |
|---|---|---|---|---|---|
| 8 | 8.0000 | 7.3867 | 7.2694 | 7.2696 | +0.0003 |
| 6 | 6.0000 | 5.3783 | 5.2592 | 5.2595 | +0.0003 |
| 4 | 4.0000 | 3.3463 | 3.2084 | 3.2087 | +0.0003 |

(bits/symbol; "model expected" is the cross-entropy of the test symbols
under the calibrated model.)

The coder lands within **0.0003 bits/symbol** of its model's expectation -
about 0.004% overhead, essentially optimal. Reading the columns left to
right accounts for the full gap between 8.0 and 7.27 bits/symbol:

- **Fixed width -> test entropy (-0.61):** what order-0 entropy coding can
  win, given how peaked the symbol distribution is.
- **Test entropy -> model expected (-0.12):** the per-channel model beats the
  aggregate order-0 figure, because channels have genuinely different
  distributions and each gets its own table.
- **Model expected -> actual payload (+0.0003):** everything the
  implementation loses - probability quantization to integer frequencies,
  finite-sequence effects, coder termination bits, and byte padding.

Note the theoretical entropy is **not** the achievable file size: it excludes
the 549-byte header entirely. Total-file bits/symbol at 8-bit is 7.5359, not
7.2696.

### Current limitations

- The entropy model is static and order-0 per channel. It ignores spatial
  correlation between neighbouring latent positions, which a context or
  hyperprior model would exploit - the largest remaining win.
- The model was trained without quantization in the loop, and the loss had
  no rate term. Nothing has optimized the rate/distortion trade-off jointly.
- Quantization parameters are re-sent in every frame (512 of the 549 header
  bytes).
- The arithmetic coder is pure Python: about 47 ms/frame encode and 56
  ms/frame decode at 8-bit. Correct and measurable, but not real-time.
- Still image only - no inter-frame prediction, so this does not yet
  compress *video*, and there is no comparison against H.264/H.265.
- Calibration is tied to a specific checkpoint; a retrained model needs
  re-calibration (the `entropy_model_id` check makes a mismatch loud).

## Large-Scale Training Dataset

**Status: implemented (Milestone 6.5) - pipeline/architecture only. No
retraining has happened, and the model/quantizer/entropy model/`.nvc`
format are all unchanged from Milestone 6.**

DAVIS (6,208 frames) proved the pipeline end-to-end but is too small to
train a production-quality codec. This milestone adds support for
**Vimeo-90K Septuplet** (~91,701 seven-frame sequences, ~642,000 frames,
~82 GB) as a large-scale training source, without duplicating that 82 GB
onto disk a second time.

**Roles going forward:**
- **DAVIS = development/baseline dataset** - fast iteration, what every
  milestone so far has trained and benchmarked on.
- **Vimeo-90K = large-scale training dataset** - what real training runs
  will eventually use (not yet - that's eventual Milestone 7+ scope).
- **UVG / Xiph = future held-out codec evaluation** - not integrated yet;
  named here only to record the intended role.

### Why not the existing PNG-extraction pipeline

`scripts/prepare_dataset.py` (Milestones 2/2.5) resizes and re-saves every
frame as a new PNG under `data/frames/`. That's the right call for DAVIS,
but running it over Vimeo-90K would silently create a **second ~82 GB
copy** of the dataset. Instead, `scripts/prepare_training_dataset.py`
builds a small manifest that *references* the original Vimeo files by
`(sequence_id, filename)`; frames are read directly from the source
location and cropped in memory, never resized-and-rewritten to disk:

```
Original Vimeo frame -> lazy loading -> random/center 256x256 crop -> tensor
```

### Dataset-source architecture

`src/nvc/data/vimeo.py` is the **only** module that knows Vimeo-90K's
directory layout. Everything downstream - `SequenceFrameDataset`
(`src/nvc/data/sequence_dataset.py`), the `create_sequence_*_loader`
factories, and the model/training code - talks only to a generic
**sequence manifest** schema. A future sequence-oriented dataset needs only
its own discovery module producing that same schema; nothing else changes.
This mirrors the existing `DatasetSource` strategy pattern from
Milestones 2.5/6 (`sources.py`) at the architecture level, without
routing Vimeo through it, since that pipeline's `process()` step is
specifically the PNG-duplication behavior described above.

```
FrameDataset          <- data/processed/manifest.json      (DAVIS, per-frame)
SequenceFrameDataset  <- data/processed/vimeo_manifest*.json (Vimeo, per-sequence, flattened to per-frame)
```

`SequenceFrameDataset.__getitem__` returns a single `[3, H, W]` float32
tensor in `[0, 1]`, exactly like `FrameDataset` - **the existing
Milestone 4 `BaselineAutoencoder` and training engine consume Vimeo frames
with zero code changes.** (Verified: `create_sequence_train_loader` ->
`BaselineAutoencoder` -> `train_one_epoch` runs unmodified against a
synthetic Vimeo-shaped manifest - see the Milestone 6.5 test suite.)
Sequence identity is preserved alongside the flattened index -
`dataset.sequence_id_at(i)` / `dataset.frame_index_at(i)` - for future
temporal/inter-frame models; nothing in this milestone uses it yet.

### Expected directory structure

```
vimeo_septuplet/
├── sequences/
│   └── <group>/<sequence>/{im1.png, ..., im7.png}   (e.g. 00001/0001/)
├── sep_trainlist.txt      one "<group>/<sequence>" id per line
└── sep_testlist.txt       one "<group>/<sequence>" id per line
```

Nothing hardcodes the two-level `<group>/<sequence>` nesting: a sequence's
id is whatever relative-path string appears in the official list file, and
its directory is `sequences/<that string>`. `find_vimeo_root()` validates
the root and `sequences/` subdirectory exist and raises a clear,
actionable error otherwise - it does not download anything.

### Leakage prevention (critical, and tested)

`sep_trainlist.txt` and `sep_testlist.txt` are **authoritative** and are
never merged, re-split, or shuffled together. `load_official_split_ids()`
loads both and explicitly asserts they're disjoint before returning them -
a defensive check on top of trusting the official files, not a substitute
for it; a corrupted or hand-edited list raises `VimeoLeakageError` rather
than silently proceeding. Test sequences are never used for training, for
quantization calibration (Milestone 6's calibration was already
train-split-only), or for hyperparameter tuning.

`tests/test_vimeo_dataset.py::test_official_train_test_ids_are_disjoint`
and `test_overlapping_lists_are_rejected` enforce this automatically, and
`test_max_sequences_subset_cannot_reach_into_the_test_list` proves subset
selection can't cross the boundary either.

### Deterministic subset selection

```powershell
python scripts\prepare_training_dataset.py `
    --dataset vimeo90k --split train --max-sequences 10000 --seed 42
```

`select_deterministic_subset()`: sort all ids -> seed a `random.Random` ->
shuffle -> truncate. Selection uses only ids and the seed - never anything
about compression quality or model performance - so the same
`(split, max_sequences, seed)` always yields the identical subset (verified
by a reproducibility smoke test: two independent runs produced
byte-identical manifests). Omitting `--max-sequences` uses every sequence
in the official list.

### Preprocessing: crop, never stretch

Vimeo frames are 448x256; the current model expects 256x256. Reusing the
existing `transforms.py` (unchanged): `RandomCrop(256)` for training,
`CenterCrop(256)` for validation/test - exactly DAVIS's pattern, applied to
un-resized source frames instead of pre-resized ones. A crop can only
select a contiguous sub-region, so this **never stretches or distorts** the
image, unlike a resize would. (`test_crop_never_stretches_the_aspect_ratio`
verifies the cropped tensor is an exact pixel-for-pixel slice of the
source, not an interpolated resize.)

### Dataset statistics

```powershell
python scripts\prepare_training_dataset.py --dataset vimeo90k --validate-only
```

Reports sequence/frame counts **from the official list files**, not a
filesystem walk of `sequences/` (which can hold 90,000+ subdirectories) -
counts are cheap and exact without scanning. A small number of images
(default 3) are opened to report actual observed resolution, and a
20-sequence deterministic sample per split is validated (all 7 frames
present) as a structural spot-check. Pass `--full-scan` to validate every
sequence in both official lists instead - correct but potentially slow
over the full ~91,701-sequence dataset; cheap when combined with
`--max-sequences`, since cost then scales with the subset, not the whole
list.

### Storage-aware workflow

Do not assume the 82 GB *download* implies exactly 82 GB *extracted* - PNG
frames are already compressed, so extracted size is typically close to but
not identical to archive size, and this project doesn't create a second
processed copy (see above), so no separate "processed cache" size applies
here. Check actual sizes locally rather than trusting a published figure:

```powershell
# Total size of the extracted dataset
Get-ChildItem "D:\Datasets\vimeo_septuplet" -Recurse -File |
    Measure-Object -Property Length -Sum |
    ForEach-Object { "{0:N2} GB" -f ($_.Sum / 1GB) }

# Free space on the target drive before extracting
Get-PSDrive D | Select-Object Used, Free
```

### Validating a real download

```powershell
# 1. Download the official Vimeo-90K Septuplet dataset (~82 GB) yourself -
#    this project never downloads it automatically. Extract it anywhere.

# 2. Point --vimeo-root at the extracted folder and validate structure
python scripts\prepare_training_dataset.py `
    --dataset vimeo90k --vimeo-root "D:\Datasets\vimeo_septuplet" --validate-only

# 3. (Optional, slower) Validate every sequence rather than a sample
python scripts\prepare_training_dataset.py `
    --dataset vimeo90k --vimeo-root "D:\Datasets\vimeo_septuplet" `
    --validate-only --full-scan
```

Or set it once in `configs/default.json` (`"vimeo_root": "D:\\Datasets\\vimeo_septuplet"`)
instead of passing `--vimeo-root` every time. **As of this milestone, the
real dataset has not been downloaded on this machine** - only the
synthetic-fixture test suite has been run; `--validate-only` above was
executed against the (absent) default path and correctly failed with a
clear, actionable error rather than a fabricated success.

### Current limitations

- No temporal/inter-frame model - Vimeo's 7-frame temporal structure is
  preserved (sequence id + frame index) but nothing consumes it yet.
- No processed-frame cache. Lazy loading re-reads and re-crops from the
  original PNGs every epoch; a future optional cache could trade disk for
  I/O, but isn't implemented (per this milestone's explicit scope).
- No validation split for Vimeo - only the official `train`/`test` lists
  exist upstream, and none is invented here.
- `--full-scan` over the complete ~91,701-sequence dataset opens on the
  order of 640,000 file-existence checks; combine with `--max-sequences`
  for routine use.

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
| 7    | Latent space analysis and uniform scalar quantization *(done - Milestone 5)*; entropy modeling and arithmetic coding *(done - Milestone 6)* |
| 8    | `.nvc` binary serialization format *(done - Milestone 6)*; a variational encoder/decoder (mu/logvar, KL divergence) and/or a learned entropy model *(not started)* |
| 9    | End-to-end encode/decode pipeline integration *(done for still frames - Milestone 6)*; large-scale (Vimeo-90K) training data pipeline *(done - Milestone 6.5)*; temporal/inter-frame coding |
| 10   | Evaluation: PSNR, MS-SSIM, MSE, BPP, compression ratio, encode/decode timing; baseline comparison vs. H.264/H.265 |
| 11   | Experiments, tuning, visualizations, documentation |
| 12   | Final report, demo, cleanup, presentation prep |

Development proceeds milestone by milestone with explicit approval before
moving to the next one.
