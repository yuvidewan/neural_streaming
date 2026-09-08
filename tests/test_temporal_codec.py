"""Tests for the M10G temporal codec prototype (.nvct).

The invariants here are the ones a temporal codec is worthless without, and
every one of them is a property a bug could silently violate while the pipeline
still appears to "work":

  * CAUSALITY - the encoder may use the current frame and previously
    RECONSTRUCTED frames, nothing else. No future frames, no original past
    frames the decoder cannot have.
  * SYMMETRY - the decoder must arrive at exactly the encoder's reference
    state from the bitstream alone.
  * SEQUENCE ISOLATION - reference state resets at every sequence boundary, so
    one DAVIS sequence can never predict from another.
  * CONTAINER STRICTNESS - truncated, corrupt or mistyped streams must be
    rejected, not decoded into plausible-looking garbage.

Everything runs on CPU against a tiny synthetic model, so exact-equality
assertions are meaningful (no GPU nondeterminism).
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.models.autoencoder import BaselineAutoencoder


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY = {"in_channels": 3, "latent_channels": 4, "base_channels": 8}


def _model(seed: int = 0) -> BaselineAutoencoder:
    torch.manual_seed(seed)
    model = BaselineAutoencoder(**TINY)
    model.eval()
    return model


def _frames(count: int = 12, size: int = 32, *, seed: int = 0, motion: float = 0.02):
    """A short synthetic clip: a base image drifting slightly, so consecutive
    frames are genuinely correlated the way real video is."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=generator)
    frames = []
    for index in range(count):
        shifted = torch.roll(base, shifts=int(index * motion * size) or index, dims=3)
        frames.append((shifted + 0.01 * index).clamp(0, 1))
    return torch.cat(frames, dim=0)


def _grids(temporal, model, frames, *, bits: int = 8):
    """Fit intra and residual grids the way the coder will actually use them."""
    with torch.no_grad():
        latents = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])
    intra_params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")
    intra_symbols = np.stack([
        latent_to_symbols(latents[i:i + 1], intra_params).reshape(latents.shape[1], -1)
        for i in range(latents.shape[0])])
    intra_model = EmpiricalEntropyModel.from_symbols(
        intra_symbols, bits=bits, num_tables=latents.shape[1])

    residuals = latents[1:] - latents[:-1]
    residual_params = calibrate_quantization_params(residuals, bits=bits, mode="per_channel")
    residual_symbols = np.stack([
        latent_to_symbols(residuals[i:i + 1], residual_params).reshape(residuals.shape[1], -1)
        for i in range(residuals.shape[0])])
    residual_model = EmpiricalEntropyModel.from_symbols(
        residual_symbols, bits=bits, num_tables=residuals.shape[1])
    return {
        "intra_params": intra_params, "intra_entropy_model": intra_model,
        "residual_params": residual_params, "residual_entropy_model": residual_model,
    }


def _encode(temporal, model, frames, path, grids, *, gop_size=4, reference_mode="latent"):
    return temporal.encode_sequence(
        model, frames, path,
        intra_params=grids["intra_params"],
        intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        residual_entropy_model=grids["residual_entropy_model"],
        gop_size=gop_size, reference_mode=reference_mode)


def _decode(temporal, model, path, grids, *, reference_mode="latent", return_latents=False):
    return temporal.decode_sequence_with_models(
        model, path,
        intra_entropy_model=grids["intra_entropy_model"],
        residual_entropy_model=grids["residual_entropy_model"],
        reference_mode=reference_mode, return_latents=return_latents)


# --- GOP structure and frame typing ----------------------------------------


def test_gop_pattern_is_i_then_p_and_always_starts_with_i():
    temporal = _load_script("m10g_temporal_baseline")

    types = temporal.gop_frame_types(23, 10)

    assert types[0] == temporal.FRAME_TYPE_I
    assert [i for i, t in enumerate(types) if t == temporal.FRAME_TYPE_I] == [0, 10, 20]
    assert all(t == temporal.FRAME_TYPE_P for i, t in enumerate(types) if i % 10)
    assert len(types) == 23


def test_gop_of_one_is_intra_only():
    """The benchmark's control arm is GOP=1; it must contain no P-frames at all."""
    temporal = _load_script("m10g_temporal_baseline")

    types = temporal.gop_frame_types(7, 1)

    assert types == [temporal.FRAME_TYPE_I] * 7


def test_gop_size_must_be_positive():
    temporal = _load_script("m10g_temporal_baseline")

    with pytest.raises(ValueError):
        temporal.gop_frame_types(5, 0)


def test_every_sequence_starts_with_an_i_frame_so_references_never_cross(tmp_path):
    """Sequence isolation: encoding two clips separately must produce two
    streams that each begin with an I-frame, so neither can predict from the
    other's reconstruction."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    for index, seed in enumerate((0, 1)):
        frames = _frames(6, seed=seed)
        grids = _grids(temporal, model, frames)
        path = tmp_path / f"seq{index}.nvct"
        _encode(temporal, model, frames, path, grids, gop_size=4)
        first_type, _ = next(iter(temporal.TemporalStreamReader(path)))
        assert first_type == temporal.FRAME_TYPE_I


# --- Round trip, symmetry, determinism --------------------------------------


def test_encode_decode_round_trip_produces_every_frame(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(12)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"

    encoded = _encode(temporal, model, frames, path, grids, gop_size=4)
    decoded = _decode(temporal, model, path, grids)

    assert encoded["frame_count"] == 12
    assert encoded["i_frames"] == 3 and encoded["p_frames"] == 9
    assert decoded.shape == frames.shape


def test_decoder_reaches_exactly_the_encoders_reference_state(tmp_path):
    """The symmetry invariant. If these diverge by even one ulp the P-chain
    drifts apart and the decoder silently produces a different video."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(12)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"

    encoded = _encode(temporal, model, frames, path, grids, gop_size=4)
    _, decoded_latents = _decode(temporal, model, path, grids, return_latents=True)

    assert torch.equal(encoded["encoder_latents"], decoded_latents)


def test_decoding_is_deterministic(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(10)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=5)

    first, first_latents = _decode(temporal, model, path, grids, return_latents=True)
    second, second_latents = _decode(temporal, model, path, grids, return_latents=True)

    assert torch.equal(first_latents, second_latents)
    assert torch.equal(first, second)


def test_encoding_the_same_input_twice_gives_the_same_bytes(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(8)
    grids = _grids(temporal, model, frames)

    a, b = tmp_path / "a.nvct", tmp_path / "b.nvct"
    _encode(temporal, model, frames, a, grids, gop_size=4)
    _encode(temporal, model, frames, b, grids, gop_size=4)

    assert a.read_bytes() == b.read_bytes()


def test_frames_decode_in_the_order_they_were_encoded(tmp_path):
    """Frame ordering: a permuted stream would still round-trip byte-wise, so
    the check is that decoded position i represents input i.

    Compared in LATENT space, not pixel space: the tiny test model is
    untrained, so its sigmoid reconstructions are nearly constant and cannot
    discriminate between frames at all - a pixel-space version of this test
    passes or fails on noise rather than on ordering.
    """
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(9, motion=0.06)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=3)

    _, decoded_latents = _decode(temporal, model, path, grids, return_latents=True)
    with torch.no_grad():
        source = torch.cat([model.encode(frames[i:i + 1]) for i in range(frames.shape[0])])

    for index in range(frames.shape[0]):
        own = torch.mean((decoded_latents[index] - source[index]) ** 2).item()
        others = [torch.mean((decoded_latents[index] - source[j]) ** 2).item()
                  for j in range(frames.shape[0]) if j != index]
        assert own < min(others), f"decoded frame {index} is closer to a different input"


def test_reversing_the_input_changes_the_bitstream(tmp_path):
    """Ordering is carried by the stream itself, not recovered by luck: the
    same frames in a different order must not produce the same bytes."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(8, motion=0.06)
    grids = _grids(temporal, model, frames)

    forward, backward = tmp_path / "f.nvct", tmp_path / "b.nvct"
    _encode(temporal, model, frames, forward, grids, gop_size=4)
    _encode(temporal, model, torch.flip(frames, dims=[0]), backward, grids, gop_size=4)

    assert forward.read_bytes() != backward.read_bytes()


# --- Causality ---------------------------------------------------------------


def test_a_p_frame_cannot_be_coded_without_a_reference(tmp_path):
    """Directly exercises the causality guard: a P-frame with no preceding
    decoded frame is unrepresentable, not merely undesirable."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(4)
    grids = _grids(temporal, model, frames)

    original = temporal.gop_frame_types
    try:
        # Force an illegal all-P pattern.
        temporal.gop_frame_types = lambda count, gop: [temporal.FRAME_TYPE_P] * count
        with pytest.raises((temporal.CausalityViolationError, temporal.TemporalFormatError)):
            _encode(temporal, model, frames, tmp_path / "bad.nvct", grids, gop_size=4)
    finally:
        temporal.gop_frame_types = original


def test_the_writer_refuses_a_stream_that_opens_with_a_p_frame(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    header = temporal.TemporalStreamHeader(
        gop_size=4, quantization_bits=8, quantization_mode="per_channel",
        image_width=32, image_height=32, image_channels=3,
        latent_channels=4, latent_height=2, latent_width=2, frame_count=2,
        num_intra_quantization_params=4, num_residual_quantization_params=4,
        intra_entropy_model_id=b"\x01" * 8, residual_entropy_model_id=b"\x02" * 8)
    params = calibrate_quantization_params(torch.randn(2, 4, 2, 2), bits=8, mode="per_channel")

    writer = temporal.TemporalStreamWriter(tmp_path / "bad.nvct", header, params, params)
    try:
        with pytest.raises(temporal.TemporalFormatError, match="must be an I-frame"):
            writer.append_frame(temporal.FRAME_TYPE_P, b"\x00")
    finally:
        writer._file.close()


def test_decoding_uses_only_the_bitstream_never_the_original_frames(tmp_path, monkeypatch):
    """No future-frame and no original-frame access: the decoder is handed the
    stream and the model, and nothing else. Deleting the source frames entirely
    must not change what it produces."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(8)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=4)

    before = _decode(temporal, model, path, grids)
    del frames  # the decoder has no route to these anyway; make that explicit
    after = _decode(temporal, model, path, grids)

    assert torch.equal(before, after)


def test_reference_is_the_dequantized_reconstruction_not_the_true_latent(tmp_path):
    """The reference must be a quantity the decoder can reproduce. If the
    encoder predicted from the TRUE latent, its reconstruction would be better
    than the decoder's - so equality here is what proves it does not."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(10)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"

    encoded = _encode(temporal, model, frames, path, grids, gop_size=5)
    decoded = _decode(temporal, model, path, grids)

    assert torch.equal(encoded["encoder_reconstructions"], decoded)


# --- Container strictness ----------------------------------------------------


def test_header_round_trips_and_is_the_documented_size():
    temporal = _load_script("m10g_temporal_baseline")
    header = temporal.TemporalStreamHeader(
        gop_size=10, quantization_bits=8, quantization_mode="per_channel",
        image_width=854, image_height=480, image_channels=3,
        latent_channels=64, latent_height=30, latent_width=53, frame_count=719,
        num_intra_quantization_params=64, num_residual_quantization_params=64,
        intra_entropy_model_id=b"\xaa" * 8, residual_entropy_model_id=b"\xbb" * 8)

    packed = header.pack()
    assert len(packed) == temporal.TEMPORAL_HEADER_SIZE == 44

    restored = temporal.TemporalStreamHeader.unpack(packed)
    assert restored.gop_size == 10
    assert restored.frame_count == 719
    assert restored.quantization_mode == "per_channel"
    assert restored.image_shape == (3, 480, 854)
    assert restored.latent_shape == (64, 30, 53)
    assert restored.intra_entropy_model_id == b"\xaa" * 8
    assert restored.residual_entropy_model_id == b"\xbb" * 8


def test_bad_magic_is_rejected():
    temporal = _load_script("m10g_temporal_baseline")
    header = temporal.TemporalStreamHeader(
        gop_size=4, quantization_bits=8, quantization_mode="global",
        image_width=32, image_height=32, image_channels=3,
        latent_channels=4, latent_height=2, latent_width=2, frame_count=1,
        num_intra_quantization_params=1, num_residual_quantization_params=1,
        intra_entropy_model_id=b"\x00" * 8, residual_entropy_model_id=b"\x00" * 8)
    corrupt = b"XXXX" + header.pack()[4:]

    with pytest.raises(temporal.TemporalFormatError, match="magic"):
        temporal.TemporalStreamHeader.unpack(corrupt)


def test_unknown_format_version_is_rejected():
    temporal = _load_script("m10g_temporal_baseline")
    header = temporal.TemporalStreamHeader(
        gop_size=4, quantization_bits=8, quantization_mode="global",
        image_width=32, image_height=32, image_channels=3,
        latent_channels=4, latent_height=2, latent_width=2, frame_count=1,
        num_intra_quantization_params=1, num_residual_quantization_params=1,
        intra_entropy_model_id=b"\x00" * 8, residual_entropy_model_id=b"\x00" * 8)
    packed = bytearray(header.pack())
    packed[4] = 99

    with pytest.raises(temporal.TemporalFormatError, match="version"):
        temporal.TemporalStreamHeader.unpack(bytes(packed))


def test_a_truncated_header_is_rejected():
    temporal = _load_script("m10g_temporal_baseline")

    with pytest.raises(temporal.TemporalFormatError, match="Truncated"):
        temporal.TemporalStreamHeader.unpack(b"NVCT\x01\x02")


def test_a_truncated_frame_payload_is_rejected(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(6)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=3)

    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 40])

    with pytest.raises(temporal.TemporalFormatError, match="Truncated"):
        list(temporal.TemporalStreamReader(path))


def test_trailing_data_after_the_declared_frames_is_rejected(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(6)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=3)

    path.write_bytes(path.read_bytes() + b"garbage")

    with pytest.raises(temporal.TemporalFormatError, match="Trailing"):
        list(temporal.TemporalStreamReader(path))


def test_an_unknown_frame_type_byte_is_rejected(tmp_path):
    """Corrupted per-frame metadata must fail loudly - a stray type byte would
    otherwise silently reinterpret an I-frame as a P-frame."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(6)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=3)

    data = bytearray(path.read_bytes())
    reader = temporal.TemporalStreamReader(path)
    offset = (temporal.TEMPORAL_HEADER_SIZE
              + reader.header.num_intra_quantization_params * 8
              + reader.header.num_residual_quantization_params * 8)
    data[offset] = 7  # neither 0 (I) nor 1 (P)
    path.write_bytes(bytes(data))

    with pytest.raises(temporal.TemporalFormatError, match="frame_type"):
        list(temporal.TemporalStreamReader(path))


def test_writing_fewer_frames_than_declared_is_rejected(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    header = temporal.TemporalStreamHeader(
        gop_size=4, quantization_bits=8, quantization_mode="per_channel",
        image_width=32, image_height=32, image_channels=3,
        latent_channels=4, latent_height=2, latent_width=2, frame_count=3,
        num_intra_quantization_params=4, num_residual_quantization_params=4,
        intra_entropy_model_id=b"\x01" * 8, residual_entropy_model_id=b"\x02" * 8)
    params = calibrate_quantization_params(torch.randn(2, 4, 2, 2), bits=8, mode="per_channel")

    writer = temporal.TemporalStreamWriter(tmp_path / "short.nvct", header, params, params)
    writer.append_frame(temporal.FRAME_TYPE_I, b"\x00\x01")
    with pytest.raises(temporal.TemporalFormatError, match="only 1 were written"):
        writer.close()


def test_a_mismatched_entropy_model_is_rejected(tmp_path):
    """The container stores both model ids so a decoder cannot silently pair a
    stream with the wrong calibration and emit garbage."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(6)
    grids = _grids(temporal, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(temporal, model, frames, path, grids, gop_size=3)

    other = _grids(temporal, model, _frames(6, seed=99, motion=0.2))
    with pytest.raises(temporal.TemporalFormatError, match="entropy model mismatch"):
        temporal.decode_sequence_with_models(
            model, path,
            intra_entropy_model=other["intra_entropy_model"],
            residual_entropy_model=other["residual_entropy_model"])


# --- The signed-residual design decision ------------------------------------


def test_the_decoder_cannot_emit_signed_values_which_is_why_residuals_are_latent_domain():
    """Documents the architectural constraint the design turns on: the decoder
    ends in Sigmoid, so a pixel residual in [-1, 1] is unrepresentable, while
    the latent is unconstrained and signed differences are natural."""
    model = _model()

    with torch.no_grad():
        reconstruction = model.decode(torch.randn(2, 4, 2, 2) * 50)
    assert float(reconstruction.min()) >= 0.0
    assert float(reconstruction.max()) <= 1.0

    with torch.no_grad():
        latent = model.encode(torch.rand(4, 3, 32, 32))
    residual = latent[1:] - latent[:-1]
    assert float(residual.min()) < 0.0, "latent residuals are genuinely signed"

    params = calibrate_quantization_params(residual, bits=8, mode="per_channel")
    symbols = latent_to_symbols(residual[0:1], params)
    assert symbols.min() >= 0 and symbols.max() < 256, "signed residuals map onto the code space"


def test_residual_grid_is_fitted_separately_from_the_intra_grid():
    """One shared grid would misallocate levels for both distributions."""
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(10)
    grids = _grids(temporal, model, frames)

    intra_scale = grids["intra_params"].scale
    residual_scale = grids["residual_params"].scale
    assert not torch.equal(intra_scale, residual_scale)
    assert grids["intra_entropy_model"].model_id() != grids["residual_entropy_model"].model_id()


def test_both_reference_modes_round_trip(tmp_path):
    temporal = _load_script("m10g_temporal_baseline")
    model = _model()
    frames = _frames(8)
    grids = _grids(temporal, model, frames)

    for mode in ("latent", "reencode"):
        path = tmp_path / f"{mode}.nvct"
        encoded = _encode(temporal, model, frames, path, grids, gop_size=4, reference_mode=mode)
        decoded = _decode(temporal, model, path, grids, reference_mode=mode)
        assert decoded.shape == frames.shape
        assert encoded["reference_mode"] == mode


def test_the_established_nvc_and_nvcs_formats_are_untouched():
    """M10G adds a container; it must not have altered the shipped ones."""
    from nvc.compression import nvc_format

    assert nvc_format.MAGIC == b"NVC1"
    assert nvc_format.STREAM_MAGIC == b"NVCS"
    temporal = _load_script("m10g_temporal_baseline")
    assert temporal.TEMPORAL_MAGIC == b"NVCT"
    assert temporal.TEMPORAL_MAGIC not in (nvc_format.MAGIC, nvc_format.STREAM_MAGIC)
