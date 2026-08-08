"""Tests for Milestone 6: fixed calibration, the empirical entropy model,
the arithmetic coder, the .nvc container, and end-to-end encode/decode.

The single most important property here is that entropy coding is EXACTLY
lossless: every distortion in the pipeline must be attributable to
quantization, never to the coder or the container.

All tests use tiny synthetic tensors and a tiny model so they stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvc.compression import (
    EmpiricalEntropyModel,
    NVCFormatError,
    NVCHeader,
    NVCReader,
    NVCWriter,
    QuantizationParams,
    UniformQuantizer,
    calibrate_quantization_params,
    channel_table_index,
    collect_calibration_latents,
    count_clipped,
    decode_frame,
    decode_latent,
    decode_symbols,
    empirical_entropy,
    encode_frame,
    encode_latent,
    encode_symbols,
    latent_to_symbols,
    load_calibration,
    save_calibration,
    symbols_to_latent,
)
from nvc.compression.entropy_model import (
    MIN_FREQUENCY,
    TOTAL_FREQUENCY,
    _counts_to_frequencies,
)
from nvc.compression.nvc_format import FIXED_HEADER_SIZE, MAGIC
from nvc.compression.range_coder import MAX_TOTAL_FREQUENCY
from nvc.models import BaselineAutoencoder

BIT_WIDTHS = (8, 6, 4)


# --- Helpers ---


def _uniform_model(bits: int, num_tables: int = 1) -> EmpiricalEntropyModel:
    counts = np.ones((num_tables, 2 ** bits))
    return EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=bits)


def _tiny_model() -> BaselineAutoencoder:
    torch.manual_seed(0)
    model = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8)
    model.eval()
    return model


def _tiny_setup(bits: int = 8):
    """A tiny model plus matching calibrated params and entropy model."""
    model = _tiny_model()
    torch.manual_seed(1)
    frames = torch.rand(6, 3, 32, 32)
    with torch.no_grad():
        latents = model.encode(frames)

    params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")
    symbols = np.stack([
        latent_to_symbols(latents[i : i + 1], params).reshape(latents.shape[1:])
        for i in range(latents.shape[0])
    ])
    entropy_model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=bits, num_tables=latents.shape[1]
    )
    return model, params, entropy_model


# --- Calibration ---


def test_collect_calibration_latents_returns_cpu_latents():
    model = _tiny_model()
    loader = torch.utils.data.DataLoader([torch.rand(3, 32, 32) for _ in range(6)], batch_size=2)

    latents = collect_calibration_latents(model, loader, torch.device("cpu"))

    assert latents.shape == (6, 4, 2, 2)
    assert latents.device.type == "cpu"


def test_collect_calibration_latents_respects_max_batches():
    model = _tiny_model()
    loader = torch.utils.data.DataLoader([torch.rand(3, 32, 32) for _ in range(6)], batch_size=2)

    latents = collect_calibration_latents(model, loader, torch.device("cpu"), max_batches=1)

    assert latents.shape[0] == 2


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_calibration_produces_one_parameter_per_channel(bits):
    latents = torch.randn(8, 5, 4, 4)

    params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")

    assert params.scale.shape == (1, 5, 1, 1)
    assert params.bits == bits
    assert (params.scale > 0).all()


def test_calibration_is_deterministic_for_the_same_input():
    latents = torch.randn(8, 3, 4, 4)

    first = calibrate_quantization_params(latents, bits=8, mode="per_channel")
    second = calibrate_quantization_params(latents, bits=8, mode="per_channel")

    assert torch.equal(first.scale, second.scale)
    assert torch.equal(first.zero_point, second.zero_point)


def test_full_percentile_range_reproduces_min_max_calibration():
    # (0, 100) percentiles are exactly min/max, so this documents that plain
    # min/max calibration is still reachable rather than removed.
    latents = torch.randn(8, 3, 4, 4)

    percentile = calibrate_quantization_params(
        latents, bits=8, mode="per_channel", lower_percentile=0.0, upper_percentile=100.0
    )
    reference = UniformQuantizer(8, "per_channel").compute_params(latents)

    assert torch.allclose(percentile.scale, reference.scale, atol=1e-6)


def test_percentile_calibration_narrows_the_range_versus_min_max():
    # An extreme outlier should widen min/max calibration but barely move a
    # 1st/99th percentile range - the whole reason percentiles are the default.
    latents = torch.randn(16, 1, 8, 8)
    latents[0, 0, 0, 0] = 500.0

    clipped = calibrate_quantization_params(
        latents, bits=8, mode="per_channel", lower_percentile=1.0, upper_percentile=99.0
    )
    full = calibrate_quantization_params(
        latents, bits=8, mode="per_channel", lower_percentile=0.0, upper_percentile=100.0
    )

    assert clipped.scale.item() < full.scale.item()


def test_calibration_rejects_invalid_percentiles():
    latents = torch.randn(4, 2, 4, 4)
    with pytest.raises(ValueError):
        calibrate_quantization_params(
            latents, bits=8, lower_percentile=90.0, upper_percentile=10.0
        )


def test_count_clipped_reports_values_outside_a_fixed_grid():
    latents = torch.zeros(4, 1, 4, 4)
    params = calibrate_quantization_params(
        latents + torch.randn(4, 1, 4, 4) * 0.01, bits=8, mode="per_channel"
    )
    latents[0, 0, 0, 0] = 1000.0
    latents[0, 0, 0, 1] = -1000.0

    clipping = count_clipped(latents, params)

    assert clipping["clipped_high"] >= 1
    assert clipping["clipped_low"] >= 1
    assert clipping["total_values"] == latents.numel()


def test_calibration_file_roundtrip(tmp_path):
    _, params, entropy_model = _tiny_setup()
    path = tmp_path / "calibration.json"

    save_calibration(
        path, params=params, entropy_model_data=entropy_model.to_dict(),
        metadata={"method": "test", "calibration_frames": 6},
    )
    document = load_calibration(path)

    restored_params = QuantizationParams.from_dict(document["quantization"])
    restored_model = EmpiricalEntropyModel.from_dict(document["entropy_model"])

    assert torch.allclose(restored_params.scale, params.scale)
    assert torch.allclose(restored_params.zero_point, params.zero_point)
    assert np.array_equal(restored_model.frequencies, entropy_model.frequencies)
    assert restored_model.model_id() == entropy_model.model_id()


def test_load_calibration_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "nope.json")


def test_load_calibration_rejects_incomplete_document(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"quantization": {}}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_calibration(path)


# --- Fixed quantization ---


def test_fixed_params_quantize_deterministically():
    _, params, _ = _tiny_setup()
    latent = torch.randn(1, 4, 2, 2)

    first = latent_to_symbols(latent, params)
    second = latent_to_symbols(latent, params)

    assert np.array_equal(first, second)


def test_fixed_params_do_not_recalibrate_per_input():
    # Two very different latents must map through the SAME grid; if the
    # quantizer silently recalibrated, both would span the full symbol range.
    _, params, _ = _tiny_setup()
    narrow = torch.zeros(1, 4, 2, 2)
    wide = torch.randn(1, 4, 2, 2) * 50

    narrow_symbols = latent_to_symbols(narrow, params)
    wide_symbols = latent_to_symbols(wide, params)

    assert narrow_symbols.max() - narrow_symbols.min() < wide_symbols.max() - wide_symbols.min()


def test_symbols_to_latent_inverts_latent_to_symbols():
    # Calibrated over the full range of these exact latents, so nothing
    # clips and the half-step round-trip bound is what is being measured.
    # (Percentile clipping is covered separately below.)
    model = _tiny_model()
    torch.manual_seed(1)
    with torch.no_grad():
        latents = model.encode(torch.rand(6, 3, 32, 32))
    params = calibrate_quantization_params(
        latents, bits=8, mode="per_channel",
        lower_percentile=0.0, upper_percentile=100.0,
    )
    latent = latents[:1]

    symbols = latent_to_symbols(latent, params)
    restored = symbols_to_latent(symbols, tuple(latent.shape[1:]), params)

    assert restored.shape == latent.shape
    # Dequantization lands on the nearest grid point, so error is bounded by
    # half a step - it does not recover the original float exactly.
    assert torch.allclose(restored, latent, atol=params.scale.max().item())


def test_out_of_range_values_clip_to_the_fixed_grid():
    # A fixed grid cannot represent values outside its calibrated range;
    # they must saturate at the endpoints rather than wrap or overflow.
    _, params, _ = _tiny_setup()
    latent = torch.full((1, 4, 2, 2), 1000.0)

    symbols = latent_to_symbols(latent, params)

    assert symbols.max() == 2 ** params.bits - 1
    assert symbols.min() == 2 ** params.bits - 1


def test_quantization_params_dict_roundtrip():
    _, params, _ = _tiny_setup()

    restored = QuantizationParams.from_dict(params.to_dict())

    assert torch.allclose(restored.scale, params.scale)
    assert torch.allclose(restored.zero_point, params.zero_point)
    assert restored.bits == params.bits
    assert restored.mode == params.mode


def test_quantization_params_from_dict_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        QuantizationParams.from_dict(
            {"bits": 8, "mode": "per_channel", "scale": [0.0], "zero_point": [0.0]}
        )


# --- Entropy model ---


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_frequency_tables_are_normalized_and_nonzero(bits):
    counts = np.zeros((3, 2 ** bits))
    counts[:, 0] = 10_000  # everything else never observed

    frequencies = _counts_to_frequencies(counts)

    assert frequencies.sum(axis=1).tolist() == [TOTAL_FREQUENCY] * 3
    assert (frequencies >= MIN_FREQUENCY).all()


def test_frequency_normalization_handles_extreme_skew():
    # One symbol with a million counts and 255 with none: flooring the rare
    # symbols up to MIN_FREQUENCY overshoots TOTAL_FREQUENCY, and the
    # normalizer must reclaim the excess from the dominant bin.
    counts = np.ones((1, 256))
    counts[0, 7] = 1_000_000

    frequencies = _counts_to_frequencies(counts)

    assert frequencies.sum() == TOTAL_FREQUENCY
    assert (frequencies >= MIN_FREQUENCY).all()
    assert frequencies[0, 7] == frequencies.max()


def test_unseen_symbols_keep_a_nonzero_probability():
    # A symbol absent from calibration must still be encodable, or the coder
    # would be handed a zero-width interval at encode time.
    symbols = np.zeros((4, 2, 3, 3), dtype=np.int64)
    model = EmpiricalEntropyModel.from_symbols(symbols, bits=8, num_tables=2)

    probabilities = model.probabilities()

    assert (probabilities > 0).all()
    assert probabilities[0, 200] > 0


def test_entropy_model_probabilities_sum_to_one():
    _, _, model = _tiny_setup()

    assert np.allclose(model.probabilities().sum(axis=1), 1.0)


def test_entropy_model_is_deterministic_for_the_same_symbols():
    symbols = np.random.default_rng(0).integers(0, 256, size=(5, 3, 4, 4))

    first = EmpiricalEntropyModel.from_symbols(symbols, bits=8, num_tables=3)
    second = EmpiricalEntropyModel.from_symbols(symbols, bits=8, num_tables=3)

    assert np.array_equal(first.frequencies, second.frequencies)
    assert first.model_id() == second.model_id()


def test_entropy_model_id_changes_with_content():
    base = np.zeros((4, 2, 3, 3), dtype=np.int64)
    other = base.copy()
    other[0, 0, 0, 0] = 5

    first = EmpiricalEntropyModel.from_symbols(base, bits=8, num_tables=2)
    second = EmpiricalEntropyModel.from_symbols(other, bits=8, num_tables=2)

    assert first.model_id() != second.model_id()
    assert len(first.model_id()) == 8


def test_entropy_model_dict_roundtrip():
    _, _, model = _tiny_setup()

    restored = EmpiricalEntropyModel.from_dict(model.to_dict())

    assert np.array_equal(restored.frequencies, model.frequencies)
    assert restored.bits == model.bits


def test_entropy_model_rejects_unknown_version():
    _, _, model = _tiny_setup()
    data = model.to_dict()
    data["version"] = 999

    with pytest.raises(ValueError):
        EmpiricalEntropyModel.from_dict(data)


def test_entropy_model_rejects_zero_frequency_table():
    frequencies = np.zeros((1, 256), dtype=np.int64)
    frequencies[0, 0] = TOTAL_FREQUENCY

    with pytest.raises(ValueError):
        EmpiricalEntropyModel(frequencies, bits=8)


def test_entropy_model_total_is_within_coder_limits():
    _, _, model = _tiny_setup()

    assert model.cumulative[:, -1].max() <= MAX_TOTAL_FREQUENCY


def test_uniform_distribution_entropy_equals_bit_width():
    for bits in BIT_WIDTHS:
        model = _uniform_model(bits)
        assert model.entropy_bits_per_symbol()[0] == pytest.approx(float(bits), abs=1e-6)


def test_empirical_entropy_of_a_constant_sequence_is_zero():
    assert empirical_entropy(np.zeros(100, dtype=np.int64), 256) == pytest.approx(0.0)


def test_empirical_entropy_of_a_uniform_sequence_matches_bit_width():
    symbols = np.tile(np.arange(256), 10)
    assert empirical_entropy(symbols, 256) == pytest.approx(8.0, abs=1e-9)


# --- Arithmetic coder ---


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_coder_roundtrip_is_lossless_for_random_symbols(bits):
    rng = np.random.default_rng(42)
    model = _uniform_model(bits)
    symbols = rng.integers(0, 2 ** bits, 2000)
    table_index = np.zeros(len(symbols), dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)
    decoded = decode_symbols(payload, len(symbols), model.cumulative, table_index)

    assert np.array_equal(decoded, symbols)


@pytest.mark.parametrize("length", [1, 2, 3, 7, 100])
def test_coder_roundtrip_for_short_sequences(length):
    rng = np.random.default_rng(7)
    model = _uniform_model(8)
    symbols = rng.integers(0, 256, length)
    table_index = np.zeros(length, dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)
    decoded = decode_symbols(payload, length, model.cumulative, table_index)

    assert np.array_equal(decoded, symbols)


def test_coder_roundtrip_for_a_constant_sequence():
    model = _uniform_model(8)
    symbols = np.full(500, 42, dtype=np.int64)
    table_index = np.zeros(len(symbols), dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)

    assert np.array_equal(
        decode_symbols(payload, len(symbols), model.cumulative, table_index), symbols
    )


def test_coder_roundtrip_for_a_highly_skewed_distribution():
    counts = np.ones((1, 256))
    counts[0, 7] = 1_000_000
    model = EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=8)
    symbols = np.full(2000, 7, dtype=np.int64)
    symbols[::100] = 200  # rare symbols must still round-trip
    table_index = np.zeros(len(symbols), dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)

    assert np.array_equal(
        decode_symbols(payload, len(symbols), model.cumulative, table_index), symbols
    )


def test_skewed_distribution_compresses_far_below_fixed_width():
    counts = np.ones((1, 256))
    counts[0, 7] = 1_000_000
    model = EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=8)
    symbols = np.full(4000, 7, dtype=np.int64)
    table_index = np.zeros(len(symbols), dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)

    assert len(payload) * 8 / len(symbols) < 0.5  # fixed width would be 8.0


def test_coder_roundtrip_covers_every_symbol_value():
    model = _uniform_model(8)
    symbols = np.arange(256, dtype=np.int64)
    table_index = np.zeros(256, dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)

    assert np.array_equal(decode_symbols(payload, 256, model.cumulative, table_index), symbols)


def test_coder_roundtrip_with_multiple_frequency_tables():
    rng = np.random.default_rng(3)
    skewed = np.ones((1, 256))
    skewed[0, 3] = 50_000
    cumulative = np.concatenate(
        [_uniform_model(8).cumulative,
         EmpiricalEntropyModel(_counts_to_frequencies(skewed), bits=8).cumulative]
    )
    symbols = rng.integers(0, 256, 1500)
    table_index = rng.integers(0, 2, 1500)

    payload = encode_symbols(symbols, cumulative, table_index)

    assert np.array_equal(
        decode_symbols(payload, len(symbols), cumulative, table_index), symbols
    )


def test_uniform_source_costs_about_the_fixed_width():
    rng = np.random.default_rng(11)
    model = _uniform_model(8)
    symbols = rng.integers(0, 256, 8000)
    table_index = np.zeros(len(symbols), dtype=np.int64)

    payload = encode_symbols(symbols, model.cumulative, table_index)

    # An incompressible source cannot beat 8 bits/symbol; the coder should
    # land essentially on it rather than above it.
    assert 7.99 <= len(payload) * 8 / len(symbols) <= 8.05


def test_encoding_an_empty_sequence_raises():
    model = _uniform_model(8)
    with pytest.raises(ValueError):
        encode_symbols(np.array([], dtype=np.int64), model.cumulative, np.array([], dtype=np.int64))


def test_encoding_out_of_range_symbols_raises():
    model = _uniform_model(4)
    with pytest.raises(ValueError):
        encode_symbols(np.array([99]), model.cumulative, np.zeros(1, dtype=np.int64))


def test_encoding_with_mismatched_table_index_length_raises():
    model = _uniform_model(8)
    with pytest.raises(ValueError):
        encode_symbols(np.array([1, 2, 3]), model.cumulative, np.zeros(2, dtype=np.int64))


def test_decoding_a_nonpositive_symbol_count_raises():
    model = _uniform_model(8)
    with pytest.raises(ValueError):
        decode_symbols(b"\x00", 0, model.cumulative, np.zeros(0, dtype=np.int64))


def test_coder_rejects_a_frequency_total_that_would_overflow():
    cumulative = np.array([[0, MAX_TOTAL_FREQUENCY + 1]], dtype=np.int64)
    with pytest.raises(ValueError):
        encode_symbols(np.array([0]), cumulative, np.zeros(1, dtype=np.int64))


# --- .nvc container ---


def _example_header(**overrides) -> NVCHeader:
    defaults = {
        "quantization_bits": 8,
        "quantization_mode": "per_channel",
        "image_width": 256,
        "image_height": 256,
        "image_channels": 3,
        "latent_channels": 2,
        "latent_height": 4,
        "latent_width": 4,
        "symbol_count": 32,
        "payload_length": 5,
        "entropy_model_id": b"\x01\x02\x03\x04\x05\x06\x07\x08",
        "scales": (0.5, 0.25),
        "zero_points": (10.0, -3.0),
    }
    defaults.update(overrides)
    return NVCHeader(**defaults)


def test_header_serialization_roundtrip():
    header = _example_header()

    restored = NVCHeader.unpack(header.pack())

    assert restored == header


def test_header_has_the_documented_fixed_size():
    assert FIXED_HEADER_SIZE == 37
    # 37 fixed bytes + 2 channels x (float32 scale + float32 zero_point)
    assert _example_header().header_size == 37 + 2 * 8


def test_header_starts_with_the_magic_bytes():
    assert _example_header().pack()[:4] == MAGIC


def test_corrupted_magic_is_rejected():
    data = bytearray(_example_header().pack())
    data[:4] = b"XXXX"

    with pytest.raises(NVCFormatError, match="magic"):
        NVCHeader.unpack(bytes(data))


def test_unsupported_version_is_rejected():
    data = bytearray(_example_header().pack())
    data[4] = 99

    with pytest.raises(NVCFormatError, match="version"):
        NVCHeader.unpack(bytes(data))


def test_header_shorter_than_the_fixed_size_is_rejected():
    with pytest.raises(NVCFormatError, match="too small"):
        NVCHeader.unpack(_example_header().pack()[:10])


def test_truncated_quantization_parameter_block_is_rejected():
    packed = _example_header().pack()

    with pytest.raises(NVCFormatError, match="Truncated quantization"):
        NVCHeader.unpack(packed[: FIXED_HEADER_SIZE + 4])


def test_invalid_dimensions_are_rejected():
    # Patch latent_width to 0 directly in the packed bytes (offset 17).
    data = bytearray(_example_header().pack())
    data[17:19] = (0).to_bytes(2, "little")

    with pytest.raises(NVCFormatError, match="Invalid latent_width"):
        NVCHeader.unpack(bytes(data))


def test_symbol_count_inconsistent_with_latent_dimensions_is_rejected():
    with pytest.raises(NVCFormatError, match="symbol_count"):
        NVCHeader.unpack(_example_header(symbol_count=999).pack())


def test_parameter_count_inconsistent_with_mode_is_rejected():
    # per_channel with 2 latent channels needs exactly 2 parameter pairs.
    header = _example_header(scales=(0.5,), zero_points=(0.0,))

    with pytest.raises(NVCFormatError, match="num_quantization_params"):
        NVCHeader.unpack(header.pack())


def test_nonpositive_scale_in_the_header_is_rejected():
    with pytest.raises(NVCFormatError, match="Invalid scale"):
        NVCHeader.unpack(_example_header(scales=(0.5, 0.0)).pack())


def test_writer_rejects_a_payload_length_mismatch():
    with pytest.raises(NVCFormatError, match="payload"):
        NVCWriter.to_bytes(_example_header(payload_length=10), b"abc")


def test_reader_rejects_a_truncated_payload():
    header = _example_header(payload_length=10)
    data = NVCWriter.to_bytes(header, b"0123456789")

    with pytest.raises(NVCFormatError, match="Truncated payload"):
        NVCReader.from_bytes(data[:-3])


def test_reader_rejects_trailing_data():
    header = _example_header(payload_length=5)
    data = NVCWriter.to_bytes(header, b"abcde")

    with pytest.raises(NVCFormatError, match="Trailing data"):
        NVCReader.from_bytes(data + b"junk")


def test_writer_and_reader_file_roundtrip(tmp_path):
    header = _example_header(payload_length=5)
    path = tmp_path / "frame.nvc"

    written = NVCWriter.write(path, header, b"abcde")
    restored_header, payload = NVCReader.read(path)

    assert written == header.header_size + 5
    assert restored_header == header
    assert payload == b"abcde"


def test_reader_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NVCReader.read(tmp_path / "missing.nvc")


# --- End-to-end codec ---


@pytest.mark.parametrize("bits", BIT_WIDTHS)
def test_end_to_end_encode_decode_is_symbol_lossless(bits):
    model, params, entropy_model = _tiny_setup(bits)
    frame = torch.rand(1, 3, 32, 32)

    encoded = encode_frame(model, frame, params=params, entropy_model=entropy_model)
    _, _, decoded_symbols = decode_latent(encoded.data, entropy_model=entropy_model)

    assert np.array_equal(encoded.symbols, decoded_symbols)


def test_end_to_end_reconstruction_matches_the_direct_quantized_path():
    # The .nvc route must be bit-identical to Milestone 5's
    # quantize -> dequantize -> decode path: entropy coding adds no distortion.
    model, params, entropy_model = _tiny_setup()
    frame = torch.rand(1, 3, 32, 32)

    with torch.no_grad():
        latent = model.encode(frame)
        symbols = latent_to_symbols(latent, params)
        direct = model.decode(symbols_to_latent(symbols, tuple(latent.shape[1:]), params))

    encoded = encode_frame(model, frame, params=params, entropy_model=entropy_model)
    via_nvc, _ = decode_frame(model, encoded.data, entropy_model=entropy_model)

    assert torch.equal(direct, via_nvc)


def test_decoded_frame_has_the_original_shape():
    model, params, entropy_model = _tiny_setup()
    frame = torch.rand(1, 3, 32, 32)

    encoded = encode_frame(model, frame, params=params, entropy_model=entropy_model)
    reconstruction, header = decode_frame(model, encoded.data, entropy_model=entropy_model)

    assert reconstruction.shape == frame.shape
    assert (header.image_channels, header.image_height, header.image_width) == (3, 32, 32)


def test_encode_frame_accepts_an_unbatched_frame():
    model, params, entropy_model = _tiny_setup()

    encoded = encode_frame(model, torch.rand(3, 32, 32), params=params, entropy_model=entropy_model)

    assert encoded.header.image_height == 32


def test_encode_frame_rejects_a_multi_frame_batch():
    model, params, entropy_model = _tiny_setup()

    with pytest.raises(ValueError):
        encode_frame(model, torch.rand(2, 3, 32, 32), params=params, entropy_model=entropy_model)


def test_decoding_with_a_mismatched_entropy_model_is_rejected():
    # Silently decoding with the wrong model would emit plausible-looking
    # garbage, so the id check must fail loudly instead.
    model, params, entropy_model = _tiny_setup()
    frame = torch.rand(1, 3, 32, 32)
    encoded = encode_frame(model, frame, params=params, entropy_model=entropy_model)

    other = EmpiricalEntropyModel.from_symbols(
        np.zeros((2, 4, 2, 2), dtype=np.int64), bits=8, num_tables=4
    )

    with pytest.raises(NVCFormatError, match="Entropy model mismatch"):
        decode_latent(encoded.data, entropy_model=other)


def test_encode_rejects_an_entropy_model_with_the_wrong_bit_depth():
    model, params, _ = _tiny_setup(bits=8)
    _, _, six_bit_model = _tiny_setup(bits=6)

    with pytest.raises(ValueError, match="bit"):
        encode_frame(model, torch.rand(1, 3, 32, 32), params=params, entropy_model=six_bit_model)


def test_encode_rejects_a_model_with_the_wrong_table_count():
    model, params, _ = _tiny_setup()
    wrong_tables = EmpiricalEntropyModel.from_symbols(
        np.zeros((2, 7, 2, 2), dtype=np.int64), bits=8, num_tables=7
    )

    with pytest.raises(ValueError, match="tables"):
        encode_frame(model, torch.rand(1, 3, 32, 32), params=params, entropy_model=wrong_tables)


def test_encode_result_reports_consistent_bit_accounting():
    model, params, entropy_model = _tiny_setup()

    encoded = encode_frame(model, torch.rand(1, 3, 32, 32), params=params, entropy_model=entropy_model)

    assert encoded.total_bits == encoded.payload_bits + encoded.header_bits
    assert encoded.total_bytes * 8 == encoded.total_bits
    assert encoded.bits_per_pixel(payload_only=True) < encoded.bits_per_pixel()
    assert encoded.bits_per_symbol() == pytest.approx(
        encoded.payload_bits / encoded.header.symbol_count
    )


def test_channel_table_index_follows_channel_major_order():
    index = channel_table_index(3, 2, 2)

    assert index.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_encode_latent_rejects_a_multi_frame_latent():
    _, params, entropy_model = _tiny_setup()

    with pytest.raises(ValueError):
        encode_latent(
            torch.randn(2, 4, 2, 2), params=params,
            entropy_model=entropy_model, image_shape=(3, 32, 32),
        )
