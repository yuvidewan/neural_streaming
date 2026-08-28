"""Full and partial pipeline tests: chains of scripts run back to back,
each consuming the previous one's real output file - not the individual
functions in isolation (those are covered in tests/test_*.py and
tests/test_scripts_*.py), but the actual handoffs between stages that a
real workflow depends on: does prepare_dataset's manifest work as
train_autoencoder's --manifest? Does train_autoencoder's checkpoint work
as calibrate_quantizer's --checkpoint? Does THAT calibration work as
encode.py's --calibration? These are exactly the places a schema drift
between two scripts would silently break a real run without breaking
either script's own unit tests.

Everything here uses tiny synthetic data and tiny models (see
tests/helpers.py) - these are fast correctness checks, not benchmarks.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_sequence  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Pipeline 1: raw frames -> manifest -> trained checkpoint -> reconstruction ---


def test_pipeline_prepare_dataset_through_reconstruct(tmp_path):
    input_root = tmp_path / "raw"
    for i in range(6):  # enough that a 0.5/0.25/0.25 split leaves every split non-empty
        make_sequence(input_root / f"seq{i}", num_frames=6, width=32, height=32)

    prepare = _load_script("prepare_dataset")
    manifest = tmp_path / "processed" / "manifest.json"
    assert prepare.main([
        "--source-type", "image-sequence", "--input", str(input_root),
        "--output", str(tmp_path / "frames"), "--manifest", str(manifest),
        "--width", "32", "--height", "32", "--seed", "1",
        # Explicit, even split - configs/default.json's own ratios (0.8/0.1/0.1)
        # round val/test to 0 sequences at this small a dataset size.
        "--train-ratio", "0.5", "--val-ratio", "0.25", "--test-ratio", "0.25",
    ]) == 0

    train = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / "checkpoints"
    assert train.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "2",
        "--batch-size", "2", "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ]) == 0
    assert (checkpoint_dir / "best.pt").is_file()

    reconstruct = _load_script("reconstruct")
    output = tmp_path / "comparison.png"
    exit_code = reconstruct.main([
        "--checkpoint", str(checkpoint_dir / "best.pt"), "--manifest", str(manifest),
        "--num-samples", "2", "--output", str(output), "--device", "cpu",
    ])
    assert exit_code == 0
    assert output.is_file()


# --- Pipeline 2: trained checkpoint -> real calibration -> encode -> decode ---


def test_pipeline_train_calibrate_encode_decode_round_trip(tmp_path):
    from helpers import make_tiny_manifest

    manifest = make_tiny_manifest(tmp_path, width=32, height=32)

    train = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / "checkpoints"
    assert train.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "2",
        "--batch-size", "2", "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ]) == 0
    checkpoint = checkpoint_dir / "best.pt"

    calibrate = _load_script("calibrate_quantizer")
    calibration = tmp_path / "calibration.json"
    assert calibrate.main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--bits", "8", "--mode", "per_channel", "--batch-size", "2", "--max-batches", "2",
        "--output", str(calibration),
        "--metrics-dir", str(tmp_path / "metrics"), "--visualizations-dir", str(tmp_path / "viz"),
        "--device", "cpu",
    ]) == 0

    # Encode a real frame from the manifest's test split, not a hand-made image -
    # exercises the exact frame/tensor shape train_autoencoder actually saw.
    from nvc.data.loaders import create_test_loader
    test_loader = create_test_loader(manifest, batch_size=1, crop_size=32)
    frame = next(iter(test_loader))[0]
    input_image = tmp_path / "input.png"
    array = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    cv2.imwrite(str(input_image), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    encode = _load_script("encode")
    nvc_path = tmp_path / "frame.nvc"
    assert encode.main([
        "--checkpoint", str(checkpoint), "--input", str(input_image),
        "--output", str(nvc_path), "--calibration", str(calibration), "--device", "cpu",
    ]) == 0
    assert nvc_path.is_file()

    decode = _load_script("decode")
    reconstructed = tmp_path / "reconstructed.png"
    exit_code = decode.main([
        "--checkpoint", str(checkpoint), "--input", str(nvc_path),
        "--output", str(reconstructed), "--calibration", str(calibration),
        "--reference", str(input_image), "--device", "cpu",
    ])
    assert exit_code == 0
    assert reconstructed.is_file()


# --- Pipeline 3: baseline train -> QAT fine-tune (resume) -> benchmark ---


def test_pipeline_baseline_then_qat_resume_then_benchmark(tmp_path):
    from helpers import make_tiny_calibration, make_tiny_manifest

    manifest = make_tiny_manifest(tmp_path, width=32, height=32, num_sequences=4, frames_per_sequence=6)
    train = _load_script("train_autoencoder")

    baseline_dir = tmp_path / "checkpoints_baseline"
    assert train.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "2",
        "--batch-size", "2", "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(baseline_dir), "--device", "cpu",
    ]) == 0
    baseline_checkpoint = baseline_dir / "best.pt"

    # QAT calibration must come from the TRAIN split of a real checkpoint -
    # the synthetic helper stands in for scripts/calibrate_quantizer.py
    # here since QAT itself is what's under test, not calibration.
    qat_calibration = make_tiny_calibration(
        tmp_path / "qat_calib.json", checkpoint_path=baseline_checkpoint, bits=4, mode="per_channel",
    )

    qat_dir = tmp_path / "checkpoints_qat"
    exit_code = train.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(qat_dir), "--device", "cpu",
        "--resume", str(baseline_checkpoint),
        "--qat-enabled", "--qat-bits", "4", "--qat-mode", "per_channel",
        "--qat-calibration", str(qat_calibration),
    ])
    assert exit_code == 0

    # History accumulates across --resume (the baseline's own epoch-1
    # record carries over, per resume_training_state) - the newly appended
    # record is the last one, and epoch numbering continues from the
    # baseline checkpoint's epoch 1.
    history = json.loads((qat_dir / "history.json").read_text(encoding="utf-8"))
    assert history[-1]["epoch"] == 2
    assert history[-1]["qat_enabled"] is True

    from nvc.training import load_model_from_checkpoint
    qat_model, _ = load_model_from_checkpoint(qat_dir / "best.pt")
    assert qat_model.quantization_noise is None

    # And that fine-tuned checkpoint must work through the ordinary codec
    # benchmark path (calibrated separately at 8-bit) like any other model.
    calibrate = _load_script("calibrate_quantizer")
    eval_calibration = tmp_path / "eval_calib.json"
    assert calibrate.main([
        "--checkpoint", str(qat_dir / "best.pt"), "--manifest", str(manifest),
        "--bits", "8", "--batch-size", "2", "--max-batches", "2",
        "--output", str(eval_calibration),
        "--metrics-dir", str(tmp_path / "metrics"), "--visualizations-dir", str(tmp_path / "viz"),
        "--device", "cpu",
    ]) == 0

    benchmark = _load_script("benchmark_codec")
    exit_code = benchmark.main([
        "--checkpoint", str(qat_dir / "best.pt"), "--manifest", str(manifest),
        "--calibration", str(eval_calibration),
        "--max-frames", "2", "--device", "cpu",
        "--metrics-dir", str(tmp_path / "metrics"),
    ])
    assert exit_code == 0
    results = json.loads((tmp_path / "metrics" / "codec_benchmark.json").read_text(encoding="utf-8"))
    assert results["configurations"][0]["entropy_coding_lossless"] is True


# --- Pipeline 4: multi-bit-depth calibration -> RD benchmark (NVC-only) ---


def test_pipeline_multi_bit_calibration_then_rd_benchmark(tmp_path):
    from helpers import make_tiny_manifest

    manifest = make_tiny_manifest(tmp_path, width=32, height=32, num_sequences=6, frames_per_sequence=4)
    train = _load_script("train_autoencoder")
    checkpoint_dir = tmp_path / "checkpoints"
    assert train.main([
        "--manifest", str(manifest), "--epochs", "1", "--max-batches", "2", "--batch-size", "2",
        "--latent-channels", str(TINY_MODEL_KWARGS["latent_channels"]),
        "--checkpoint-dir", str(checkpoint_dir), "--device", "cpu",
    ]) == 0
    checkpoint = checkpoint_dir / "best.pt"

    calibrate = _load_script("calibrate_quantizer")
    base_calibration = tmp_path / "calib.json"  # base filename = the default/primary bit depth
    for bits, output in [(8, base_calibration), (4, tmp_path / "calib_4bit.json")]:
        assert calibrate.main([
            "--checkpoint", str(checkpoint), "--manifest", str(manifest),
            "--bits", str(bits), "--batch-size", "2", "--max-batches", "2",
            "--output", str(output),
            "--metrics-dir", str(tmp_path / "metrics"), "--visualizations-dir", str(tmp_path / "viz"),
            "--device", "cpu",
        ]) == 0

    benchmark_rd = _load_script("benchmark_rd")
    exit_code = benchmark_rd.main([
        "--manifest", str(manifest), "--split", "test", "--codecs", "nvc",
        "--checkpoint", str(checkpoint), "--calibration", str(base_calibration),
        "--nvc-bits", "8", "4",
        "--max-sequences", "2", "--max-frames-per-sequence", "2",
        "--output-dir", str(tmp_path / "benchmarks"), "--run-name", "multi_bit",
        "--device", "cpu", "--allow-calibration-mismatch",
    ])
    assert exit_code == 0
    results = json.loads((tmp_path / "benchmarks" / "multi_bit" / "results.json").read_text(encoding="utf-8"))
    nvc_configurations = {
        row["codec_configuration"] for row in results["aggregate"] if row["codec"] == "nvc"
    }
    assert nvc_configurations == {"8bit-per_channel", "4bit-per_channel"}
