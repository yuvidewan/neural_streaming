# Testing Standard

This is the reference for this project's test suite: what's covered, where,
and the rule this project follows going forward - **any change to code gets
a matching change to tests in the same commit**, not a follow-up "add tests
later" that never happens.

459 tests, ~65s on a laptop CPU. No test needs a GPU, a real Kaggle
download, or an external dataset - everything runs on synthetic data
generated in-process (`tests/helpers.py`), so the full suite runs
identically in CI, on a fresh clone, or on a machine with no internet
access at all. FFmpeg-dependent tests (`libx264`/`libx265` round trips)
skip themselves automatically when FFmpeg isn't on `PATH`, rather than
failing - see "FFmpeg-dependent tests" below.

## The rule

> **Whenever you modify `src/nvc/**` or `scripts/*.py`, update or add the
> matching tests in the same change - not as a follow-up.**

Concretely:

- **Touched a function in `src/nvc/`?** Find its test file below and update
  the existing test, or add a new one, for the specific behavior you
  changed. If the change fixes a bug, add a test that would have caught it
  (a regression test) - don't just fix the symptom.
- **Touched a `scripts/*.py` CLI** (added a flag, changed validation,
  changed what it writes)? Update the matching `tests/test_scripts_*.py`
  file. Every script's `main(argv)` must stay callable with an explicit
  `argv` list (not just `sys.argv`) - that's what makes it testable at all;
  if you write a new script, give it that signature from the start (see
  "Script contract" below).
- **Added a new script?** Give it a test file (or a section in the most
  relevant existing `tests/test_scripts_*.py`) covering: `build_arg_parser`
  defaults, at least one missing-required-file error path, and one
  happy-path `main(argv)` invocation against synthetic data that checks
  the exit code and the actual output file(s) written - not just "didn't
  crash."
- **New behavior that spans more than one script** (a checkpoint one
  script produces that another consumes, a calibration file's shape, a
  manifest schema)? Add or extend a test in `tests/test_pipelines.py`. This
  is the layer that catches a schema drift between two scripts that each
  pass their own unit tests in isolation - see "Why pipeline tests exist
  separately" below.
- **Before committing**, run the full suite (`pytest`, see "Running the
  suite") and don't commit on red. A skipped FFmpeg test is fine; a failing
  one is not.

## Test organization

Three layers, each catching a different class of regression:

| Layer | Answers | Files |
|---|---|---|
| **Unit** | Does this one function do the right thing? | `tests/test_<module_topic>.py` |
| **Script** | Does this CLI parse args, validate inputs, and write correct output? | `tests/test_scripts_*.py` |
| **Pipeline** | Do two scripts' input/output actually fit together in a real workflow? | `tests/test_pipelines.py` |

### Layer 1: Unit tests (`src/nvc/`)

One test file per topic, generally mapping to one or a few closely related
modules under `src/nvc/`:

| Test file | Covers (`src/nvc/...`) |
|---|---|
| `test_project_setup.py` | `utils/config.py` |
| `test_dataset_preparation.py` | `data/dataset_prep.py`, `data/frame_extraction.py`, `data/video_utils.py` |
| `test_dataset_ingestion.py` | `data/ingest.py`, `data/sequence_utils.py`, `data/sources.py` |
| `test_pytorch_pipeline.py` | `data/frame_dataset.py`, `data/loaders.py`, `data/transforms.py`, `data/validation.py`, `utils/device.py`, `utils/seed.py` |
| `test_vimeo_dataset.py` | `data/vimeo.py`, `data/sequence_dataset.py` (Vimeo-90K-specific) |
| `test_baseline_autoencoder.py` | `models/*`, `training/trainer.py`, `training/checkpoint.py` |
| `test_quantization.py` | `compression/quantization.py`, `compression/storage_analysis.py`, `evaluation/latent_analysis.py` |
| `test_quantization_aware_training.py` | `training/quantization_noise.py`, `compression/calibration.py` (QAT-specific) |
| `test_entropy_coding.py` | `compression/entropy_model.py`, `compression/range_coder.py`, `compression/nvc_format.py`, `compression/codec.py` |
| `test_perceptual_metrics.py` | `evaluation/basic_metrics.py`, `evaluation/perceptual_metrics.py` |
| `test_ffmpeg_utils.py` | `evaluation/ffmpeg.py`, `evaluation/codecs.py` |
| `test_rd_benchmark.py` | `evaluation/rd_benchmark.py`, `evaluation/sequences.py` |

If you add a new module under `src/nvc/` with no obvious home in this
table, either extend the closest existing file or start a new
`test_<topic>.py` and add a row here.

### Layer 2: Script tests (`scripts/*.py`)

| Test file | Covers |
|---|---|
| `test_scripts_environment_and_dataprep.py` | `check_environment.py`, `prepare_dataset.py`, `prepare_training_dataset.py`, `inspect_dataset.py` |
| `test_scripts_training.py` | `train_autoencoder.py`, `plot_training_history.py` |
| `test_scripts_codec_cli.py` | `calibrate_quantizer.py`, `encode.py`, `decode.py`, `reconstruct.py`, `analyze_latent.py`, `quantization_experiment.py`, `benchmark_codec.py` |
| `test_scripts_rd_and_range_coder_benchmarks.py` | `benchmark_rd.py`, `plot_rate_distortion.py`, `benchmark_range_coder.py` |
| `test_scripts_train_vimeo_qat_combined.py` | `train_vimeo_qat_combined.py` |
| `test_scripts_compare_m8_models.py` | `compare_m8_models.py` |

**Script contract** every `scripts/*.py` file follows, which is what makes
all of the above possible:

```python
def build_arg_parser(defaults) -> argparse.ArgumentParser: ...
def main(argv: list[str] | None = None) -> int: ...
if __name__ == "__main__":
    sys.exit(main())
```

`main` takes an explicit `argv` (defaulting to `None`, which argparse
resolves to real `sys.argv`) so tests call `mod.main(["--flag", "value"])`
directly - no subprocess, no `sys.argv` monkeypatching. Validation failures
return `1` after printing `[ERROR] ...` to stderr; argparse-level failures
(bad choices, missing required args) raise `SystemExit` on their own -
tests assert on whichever is correct for that failure. **Every script must
keep this shape.** `benchmark_range_coder.py` didn't, once - it was the
first thing this test suite caught, before a single test even ran against
it (see git history, "Migrate the arithmetic coder to C").

### Layer 3: Pipeline tests (`tests/test_pipelines.py`)

Four pipelines, each a real chain of scripts run back to back, each
consuming the previous one's actual output file:

1. `prepare_dataset.py` -> `train_autoencoder.py` -> `reconstruct.py`
2. `train_autoencoder.py` -> `calibrate_quantizer.py` -> `encode.py` -> `decode.py` (the full `.nvc` round trip, on a real frame from the manifest's own test split)
3. `train_autoencoder.py` (baseline) -> `train_autoencoder.py --qat-enabled --resume` -> `calibrate_quantizer.py` -> `benchmark_codec.py` (the QAT experiment pipeline)
4. `train_autoencoder.py` -> `calibrate_quantizer.py` at two bit depths -> `benchmark_rd.py --codecs nvc`

#### Why pipeline tests exist separately

A unit test calling `encode_symbols()` directly and a script test calling
`encode.py`'s `main()` directly can both pass while the *handoff* between
two different scripts is broken - e.g. `calibrate_quantizer.py` writes a
field under a slightly different key than `encode.py` expects to read, and
neither script's own isolated test would ever exercise the other one's
file. Pipeline tests exist specifically to catch that: they never construct
a calibration/checkpoint/manifest by hand, only by running the actual
upstream script and handing its real output to the next one. If you add a
new place where one script's output becomes another script's input, add a
pipeline test for it - that handoff is exactly the kind of regression the
other two layers cannot see.

## Shared fixtures (`tests/helpers.py`)

Never hand-write a manifest, checkpoint, or calibration file's JSON
structure directly in a test - use these, which build them through the
project's own real code paths (so they can't drift from the real schema)
and stay tiny/fast on purpose:

| Helper | Produces |
|---|---|
| `make_synthetic_video(path, ...)` | A tiny `.mp4` (for `dataset_prep.py`-family tests) |
| `make_sequence(dir, ...)` | A DAVIS-style image-sequence folder |
| `make_vimeo_dataset(root, ...)` | A synthetic Vimeo-90K Septuplet tree + split lists |
| `make_tiny_manifest(tmp_path, ...)` | A real `manifest.json`, built via `nvc.data.ingest.ingest_dataset` |
| `make_tiny_checkpoint(path, ...)` | A real, loadable `BaselineAutoencoder` checkpoint (random weights, `TINY_MODEL_KWARGS` by default - `latent_channels=4, base_channels=8`, not the production `64`/`32`) |
| `make_tiny_calibration(path, ...)` | A real, structurally valid calibration file (quantization params + entropy model), built via the project's own `calibrate_quantization_params`/`EmpiricalEntropyModel`, without needing a real encoder pass |

`TINY_MODEL_KWARGS` is the default model size for every test in this suite
that needs *a* model, not a *good* one - keep using it (or something even
smaller) for new tests; a full-size model makes tests slow for no benefit,
since correctness of the learned mapping itself is `test_baseline_autoencoder.py`'s job, not every other file's.

## FFmpeg-dependent tests

A handful of tests (H.264/H.265 encode/decode, in `test_rd_benchmark.py`
and `test_scripts_rd_and_range_coder_benchmarks.py`) need a real FFmpeg
build with `libx264`/`libx265`. They're guarded with:

```python
from nvc.evaluation.ffmpeg import has_encoders

@pytest.mark.skipif(not has_encoders(["libx264"]), reason="FFmpeg build lacks libx264")
def test_...(): ...
```

They **skip**, never fail, when the encoder isn't available - a machine
without FFmpeg still gets a fully green run, just with fewer tests
collected. If you add a new FFmpeg-dependent test, guard it the same way.

## Running the suite

```powershell
# Everything
pytest

# One layer
pytest tests/test_scripts_codec_cli.py
pytest tests/test_pipelines.py

# One test
pytest tests/test_pipelines.py::test_pipeline_train_calibrate_encode_decode_round_trip -v

# Skip the slowest handful (FFmpeg round trips) even when FFmpeg IS present
pytest -k "not h264 and not h265"
```

## Known, accepted gaps

Documented here rather than silently missing, so nobody re-discovers them
by accident:

- **`train_vimeo_qat_combined.py`'s `main()` chunk-download loop is not
  end-to-end tested** - it needs a real Kaggle download. What *is* tested,
  directly and thoroughly: every function the loop calls
  (`_extract_reconciling_collisions`, `_reset_dir_with_retry`,
  `_bootstrap_run`, `_train_one_chunk_with_early_stopping`, progress
  bookkeeping) and `main()`'s own pre-flight checks that don't need
  network access (missing `kaggle` CLI, missing calibration file). This is
  the intended shape of coverage for any future script with an external
  network/API dependency - test the logic directly, test the CLI's
  pre-flight validation, and accept that the actual network call itself
  needs a human running it for real once.
- **`colab_train_vimeo.ipynb` is not covered by this suite at all** -
  notebooks aren't pytest-collectable. Its cells are ported from (and must
  be kept in sync with, by hand) the corresponding functions in
  `scripts/train_vimeo_qat_combined.py`, which *are* covered - see that
  script's own module docstring and the comments on
  `_extract_reconciling_collisions`/`build_chunk_manifests` for the
  specific cells this applies to.
- **GPU-specific code paths** (`--device cuda`, actual mixed-precision
  execution) run on CPU in every test here, since CI/dev machines aren't
  guaranteed a GPU. `--device` itself is exercised (`"cpu"`/`"auto"`), and
  `nvc.utils.device.get_device()` has its own unit coverage in
  `test_pytorch_pipeline.py`, but a CUDA-only failure mode would not be
  caught by this suite - the honest gap, not a claim otherwise.
