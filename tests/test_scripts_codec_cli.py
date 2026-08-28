"""Script-level tests: calibrate_quantizer.py, encode.py, decode.py,
reconstruct.py, analyze_latent.py, quantization_experiment.py,
benchmark_codec.py.

These are the actual .nvc codec CLIs - the highest-value scripts to keep
covered, since encode.py/decode.py round-tripping correctly is the whole
point of the project. Each test invokes the script's own main(argv), not
just the underlying nvc.compression functions (those have their own
coverage in tests/test_entropy_coding.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from helpers import TINY_MODEL_KWARGS, make_tiny_calibration, make_tiny_checkpoint, make_tiny_manifest  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_test_image(path: Path, *, width: int = 32, height: int = 32) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), frame)
    return path


# --- calibrate_quantizer.py -----------------------------------------------


def test_calibrate_quantizer_end_to_end_writes_a_usable_calibration_file(tmp_path):
    manifest = make_tiny_manifest(tmp_path, width=32, height=32)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    mod = _load_script("calibrate_quantizer")
    output = tmp_path / "calibration.json"

    exit_code = mod.main([
        "--checkpoint", str(checkpoint),
        "--manifest", str(manifest),
        "--bits", "4", "--mode", "per_channel",
        "--batch-size", "2", "--max-batches", "2",
        "--output", str(output),
        "--metrics-dir", str(tmp_path / "metrics"),
        "--visualizations-dir", str(tmp_path / "viz"),
        "--device", "cpu",
    ])

    assert exit_code == 0
    assert output.is_file()
    doc = json.loads(output.read_text(encoding="utf-8"))
    assert doc["calibration_metadata"]["calibration_split"] == "train"
    assert doc["calibration_metadata"]["bits"] == 4
    assert doc["quantization"]["mode"] == "per_channel"


def test_calibrate_quantizer_rejects_out_of_range_bits(tmp_path):
    import pytest

    # Manifest must actually exist - the manifest-not-found check runs
    # before the bits check, and would otherwise mask it.
    manifest = make_tiny_manifest(tmp_path)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    mod = _load_script("calibrate_quantizer")
    with pytest.raises(SystemExit):
        mod.main([
            "--checkpoint", str(checkpoint),
            "--manifest", str(manifest),
            "--bits", "99",
        ])


def test_calibrate_quantizer_missing_checkpoint_fails_cleanly(tmp_path):
    mod = _load_script("calibrate_quantizer")
    exit_code = mod.main([
        "--checkpoint", str(tmp_path / "nope.pt"),
        "--manifest", str(tmp_path / "irrelevant.json"),
    ])
    assert exit_code != 0


# --- encode.py / decode.py: the core round trip ---------------------------


def test_encode_then_decode_round_trip_reports_finite_psnr(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint, bits=4)
    image = _make_test_image(tmp_path / "input.png")

    encode_mod = _load_script("encode")
    nvc_path = tmp_path / "out.nvc"
    assert encode_mod.main([
        "--checkpoint", str(checkpoint), "--input", str(image),
        "--output", str(nvc_path), "--calibration", str(calibration), "--device", "cpu",
    ]) == 0
    assert nvc_path.is_file()

    decode_mod = _load_script("decode")
    reconstructed = tmp_path / "reconstructed.png"
    exit_code = decode_mod.main([
        "--checkpoint", str(checkpoint), "--input", str(nvc_path),
        "--output", str(reconstructed), "--calibration", str(calibration),
        "--reference", str(image), "--device", "cpu",
    ])

    assert exit_code == 0
    assert reconstructed.is_file()
    # A real decoded image, not a stub - same spatial size as the input.
    decoded_array = cv2.imread(str(reconstructed))
    assert decoded_array.shape[:2] == (32, 32)


def test_decode_with_mismatched_calibration_fails_loudly(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration_a = make_tiny_calibration(
        tmp_path / "calib_a.json", checkpoint_path=checkpoint, bits=4, seed=1,
    )
    calibration_b = make_tiny_calibration(
        tmp_path / "calib_b.json", checkpoint_path=checkpoint, bits=4, seed=2,
    )
    image = _make_test_image(tmp_path / "input.png")

    encode_mod = _load_script("encode")
    nvc_path = tmp_path / "out.nvc"
    assert encode_mod.main([
        "--checkpoint", str(checkpoint), "--input", str(image),
        "--output", str(nvc_path), "--calibration", str(calibration_a), "--device", "cpu",
    ]) == 0

    decode_mod = _load_script("decode")
    # Encoded with calibration_a's entropy model, decoded with calibration_b's
    # (different random frequency tables) - the header's entropy_model_id
    # check must catch this rather than silently producing garbage.
    exit_code = decode_mod.main([
        "--checkpoint", str(checkpoint), "--input", str(nvc_path),
        "--output", str(tmp_path / "out.png"), "--calibration", str(calibration_b), "--device", "cpu",
    ])
    assert exit_code != 0


def test_encode_missing_input_image_fails_cleanly(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    mod = _load_script("encode")
    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--input", str(tmp_path / "nope.png"),
        "--output", str(tmp_path / "out.nvc"), "--calibration", str(calibration),
    ])
    assert exit_code != 0


def test_decode_of_a_corrupted_nvc_file_fails_cleanly(tmp_path):
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib.json", checkpoint_path=checkpoint)
    garbage_path = tmp_path / "garbage.nvc"
    garbage_path.write_bytes(b"not a real nvc file at all")

    mod = _load_script("decode")
    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--input", str(garbage_path),
        "--output", str(tmp_path / "out.png"), "--calibration", str(calibration),
    ])
    assert exit_code != 0


# --- reconstruct.py ---------------------------------------------------


def test_reconstruct_end_to_end(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    mod = _load_script("reconstruct")
    output = tmp_path / "comparison.png"

    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--num-samples", "2", "--output", str(output), "--device", "cpu",
    ])

    assert exit_code == 0
    assert output.is_file()


def test_reconstruct_missing_checkpoint_fails_cleanly(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    mod = _load_script("reconstruct")
    exit_code = mod.main(["--checkpoint", str(tmp_path / "nope.pt"), "--manifest", str(manifest)])
    assert exit_code != 0


# --- analyze_latent.py ---------------------------------------------------


def test_analyze_latent_end_to_end(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    mod = _load_script("analyze_latent")

    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--batch-size", "2", "--max-batches", "2", "--device", "cpu",
        "--metrics-dir", str(tmp_path / "metrics"),
        "--visualizations-dir", str(tmp_path / "viz"),
    ])

    assert exit_code == 0
    stats_path = tmp_path / "metrics" / "latent_statistics.json"
    assert stats_path.is_file()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert isinstance(stats["statistics"]["global"]["mean"], float)
    assert stats["statistics"]["num_samples"] == 4


# --- quantization_experiment.py -------------------------------------------


def test_quantization_experiment_end_to_end(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    mod = _load_script("quantization_experiment")

    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--device", "cpu",
        "--metrics-dir", str(tmp_path / "metrics"),
        "--visualizations-dir", str(tmp_path / "viz"),
    ])

    assert exit_code == 0
    results_path = tmp_path / "metrics" / "quantization_results.json"
    assert results_path.is_file()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    # One row per configuration in mod._CONFIGURATIONS (float32 baseline + 6 quantized).
    assert len(results["results"]) == len(mod._CONFIGURATIONS)


# --- benchmark_codec.py ---------------------------------------------------


def test_benchmark_codec_end_to_end_reports_lossless_entropy_coding(tmp_path):
    manifest = make_tiny_manifest(tmp_path, num_sequences=2, frames_per_sequence=4)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    calibration = make_tiny_calibration(tmp_path / "calib_8bit.json", checkpoint_path=checkpoint, bits=8)
    mod = _load_script("benchmark_codec")

    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--calibration", str(calibration),
        "--max-frames", "2", "--device", "cpu",
        "--metrics-dir", str(tmp_path / "metrics"),
    ])

    assert exit_code == 0
    results_path = tmp_path / "metrics" / "codec_benchmark.json"
    assert results_path.is_file()
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["configurations"][0]["entropy_coding_lossless"] is True
    assert results["configurations"][0]["matches_direct_quantized_path"] is True


def test_benchmark_codec_missing_calibration_file_fails_cleanly(tmp_path):
    manifest = make_tiny_manifest(tmp_path)
    checkpoint = make_tiny_checkpoint(tmp_path / "ckpt.pt")
    mod = _load_script("benchmark_codec")

    exit_code = mod.main([
        "--checkpoint", str(checkpoint), "--manifest", str(manifest),
        "--calibration", str(tmp_path / "nope.json"),
        "--max-frames", "1", "--device", "cpu",
        "--metrics-dir", str(tmp_path / "metrics"),
    ])
    assert exit_code != 0
