"""Tests for M10H motion compensation: estimation, warping, and paying for motion.

The property that makes this a compression experiment rather than a demo is that
the motion field is TRANSMITTED. Several tests here exist specifically to make
"the decoder secretly had information the bitstream did not carry" a test
failure rather than something a reader has to take on trust.

Everything runs on CPU against a tiny synthetic model, so exact-equality
assertions are meaningful. (On GPU the codec forces deterministic cuDNN
algorithms - see `deterministic_kernels` - because this design puts the network
in the reference path and `decode()` is otherwise not bit-reproducible.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.compression.calibration import calibrate_quantization_params
from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.evaluation.sequences import BenchmarkSequence
from nvc.models.autoencoder import BaselineAutoencoder

from helpers import make_sequence


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


def _moving_frames(count: int = 12, size: int = 64, *, seed: int = 0, step: int = 2):
    """A clip with real, known translation - the thing motion compensation is
    supposed to exploit."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=generator)
    return torch.cat([torch.roll(base, shifts=(index * step, index * step), dims=(2, 3))
                      for index in range(count)], dim=0)


def _grids(mc, model, frames, *, bits: int = 8, search_range: int = 8, block_size: int = 16):
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

    motion_bits = mc.motion_alphabet_bits(search_range)
    blocks = (frames.shape[2] // block_size) * (frames.shape[3] // block_size)
    motion_symbols = np.stack([np.stack([
        np.arange(blocks) % (2 * search_range + 1),
        np.arange(blocks) % (2 * search_range + 1)])])
    motion_model = EmpiricalEntropyModel.from_symbols(
        motion_symbols, bits=motion_bits, num_tables=2)
    return {
        "intra_params": intra_params, "intra_entropy_model": intra_model,
        "residual_params": residual_params, "residual_entropy_model": residual_model,
        "motion_entropy_model": motion_model,
    }


def _encode(mc, model, frames, path, grids, *, mode="mc", gop_size=4,
            block_size=16, search_range=8):
    return mc.encode_sequence(
        model, frames, path,
        intra_params=grids["intra_params"], intra_entropy_model=grids["intra_entropy_model"],
        residual_params=grids["residual_params"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"],
        mode=mode, gop_size=gop_size, block_size=block_size, search_range=search_range)


def _decode(mc, model, path, grids, *, return_latents=False):
    return mc.decode_sequence(
        model, path,
        intra_entropy_model=grids["intra_entropy_model"],
        residual_entropy_model=grids["residual_entropy_model"],
        motion_entropy_model=grids["motion_entropy_model"],
        return_latents=return_latents)


# --- motion estimation -------------------------------------------------------


def test_a_known_translation_is_recovered_exactly():
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(0)
    reference = torch.rand(1, 3, 64, 64)
    motion = torch.stack([torch.full((4, 4), 3, dtype=torch.long),
                          torch.full((4, 4), -5, dtype=torch.long)])
    current = mc.warp_blocks(reference, motion, block_size=16)

    estimated = mc.estimate_block_motion(reference, current, block_size=16, search_range=8)

    assert torch.equal(estimated, motion)
    assert torch.equal(mc.warp_blocks(reference, estimated, block_size=16), current)


def test_identical_frames_give_zero_motion():
    """The tie-break is biased toward zero, which is both correct and the
    cheapest field to code."""
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(0)
    frame = torch.rand(1, 3, 64, 64)

    motion = mc.estimate_block_motion(frame, frame, block_size=16, search_range=8)

    assert bool((motion == 0).all())


def test_motion_estimation_is_deterministic():
    mc = _load_script("m10h_motion_compensation")
    frames = _moving_frames(2)

    first = mc.estimate_block_motion(frames[0:1], frames[1:2], block_size=16, search_range=8)
    second = mc.estimate_block_motion(frames[0:1], frames[1:2], block_size=16, search_range=8)

    assert torch.equal(first, second)


def test_estimated_motion_never_exceeds_the_search_range():
    mc = _load_script("m10h_motion_compensation")
    frames = _moving_frames(2, step=30)  # motion far beyond the range

    motion = mc.estimate_block_motion(frames[0:1], frames[1:2], block_size=16, search_range=4)

    assert int(motion.abs().max()) <= 4


def test_motion_compensation_reduces_prediction_error_on_moving_content():
    mc = _load_script("m10h_motion_compensation")
    frames = _moving_frames(2, step=3)
    reference, current = frames[0:1], frames[1:2]

    motion = mc.estimate_block_motion(reference, current, block_size=16, search_range=8)
    warped = mc.warp_blocks(reference, motion, block_size=16)

    assert torch.mean((warped - current) ** 2) < torch.mean((reference - current) ** 2)


def test_warp_is_deterministic_and_shape_preserving():
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(0)
    reference = torch.rand(1, 3, 64, 64)
    motion = torch.randint(-5, 6, (2, 4, 4))

    first = mc.warp_blocks(reference, motion, block_size=16)
    second = mc.warp_blocks(reference, motion, block_size=16)

    assert torch.equal(first, second)
    assert first.shape == reference.shape


def test_zero_motion_warp_is_the_identity():
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(0)
    reference = torch.rand(1, 3, 64, 64)

    warped = mc.warp_blocks(reference, torch.zeros(2, 4, 4, dtype=torch.long), block_size=16)

    assert torch.equal(warped, reference)


def test_warp_clamps_at_the_frame_edge_rather_than_wrapping():
    """Replicate boundary handling: a block pushed off the edge must not pull
    content from the opposite side of the frame."""
    mc = _load_script("m10h_motion_compensation")
    reference = torch.zeros(1, 1, 32, 32)
    reference[:, :, :, 0] = 1.0  # a bright left column only

    warped = mc.warp_blocks(
        reference, torch.stack([torch.zeros(2, 2, dtype=torch.long),
                                torch.full((2, 2), -8, dtype=torch.long)]), block_size=16)

    assert float(warped[:, :, :, -1].max()) == 0.0, "content wrapped around the frame"


def test_a_motion_field_outside_the_range_is_refused():
    mc = _load_script("m10h_motion_compensation")
    motion = torch.stack([torch.full((2, 2), 20, dtype=torch.long),
                          torch.zeros(2, 2, dtype=torch.long)])

    with pytest.raises(ValueError, match="outside"):
        mc.motion_to_symbols(motion, search_range=8)


# --- motion is transmitted ---------------------------------------------------


def test_motion_payload_round_trips_exactly():
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(0)
    motion = torch.randint(-8, 9, (2, 4, 4))
    symbols = np.stack([mc.motion_to_symbols(motion, search_range=8).reshape(2, -1)])
    model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=mc.motion_alphabet_bits(8), num_tables=2)

    payload = mc.encode_motion_payload(motion, search_range=8, entropy_model=model)
    decoded = mc.decode_motion_payload(payload, (4, 4), search_range=8, entropy_model=model)

    assert torch.equal(decoded, motion)


def test_motion_decoding_is_deterministic():
    mc = _load_script("m10h_motion_compensation")
    torch.manual_seed(1)
    motion = torch.randint(-8, 9, (2, 4, 4))
    symbols = np.stack([mc.motion_to_symbols(motion, search_range=8).reshape(2, -1)])
    model = EmpiricalEntropyModel.from_symbols(
        symbols, bits=mc.motion_alphabet_bits(8), num_tables=2)
    payload = mc.encode_motion_payload(motion, search_range=8, entropy_model=model)

    first = mc.decode_motion_payload(payload, (4, 4), search_range=8, entropy_model=model)
    second = mc.decode_motion_payload(payload, (4, 4), search_range=8, entropy_model=model)

    assert torch.equal(first, second)


def test_p_frames_actually_carry_motion_bytes_and_i_frames_do_not(tmp_path):
    """The central honesty check: motion must cost something and must be in the
    stream. A P-frame with zero motion bytes would mean motion travelled by some
    other route."""
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(8)
    grids = _grids(mc, model, frames)

    encoded = _encode(mc, model, frames, tmp_path / "seq.nvct", grids, gop_size=4)

    assert encoded["motion_bytes"] > 0
    for record in encoded["frames"]:
        if record["frame_type"] == "I":
            assert record["motion_bytes"] == 0, "an I-frame has no temporal dependency"
        else:
            assert record["motion_bytes"] > 0, "a P-frame must transmit its motion"


def test_the_prev_mode_transmits_no_motion_at_all(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(8)
    grids = _grids(mc, model, frames)

    encoded = _encode(mc, model, frames, tmp_path / "seq.nvct", grids, mode="prev", gop_size=4)

    assert encoded["motion_bytes"] == 0


def test_total_stream_bytes_equal_the_sum_of_serialized_components(tmp_path):
    """Byte accounting must close: motion + residual + container overhead is
    exactly the file size, so no bitrate can be hidden."""
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(10)
    grids = _grids(mc, model, frames)

    encoded = _encode(mc, model, frames, tmp_path / "seq.nvct", grids, gop_size=5)

    on_disk = (tmp_path / "seq.nvct").stat().st_size
    assert encoded["container_bytes"] == on_disk
    assert (encoded["motion_bytes"] + encoded["residual_bytes"]
            + encoded["container_overhead_bytes"]) == on_disk
    assert encoded["residual_bytes"] == (encoded["i_frame_residual_bytes"]
                                         + encoded["p_frame_residual_bytes"])


def test_the_oracle_path_is_marked_as_not_rate_accounted():
    """Its dense flow is never transmitted, so its byte count is not a
    compression result and the code must say so structurally."""
    mc = _load_script("m10h_motion_compensation")

    assert mc.is_rate_accounted("prev") is True
    assert mc.is_rate_accounted("mc") is True
    assert mc.is_rate_accounted("oracle") is False


def test_an_oracle_stream_refuses_to_decode(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(8)
    grids = _grids(mc, model, frames)
    path = tmp_path / "oracle.nvct"
    _encode(mc, model, frames, path, grids, mode="oracle", gop_size=4)

    with pytest.raises(mc.TemporalFormatError, match="not transmitted"):
        _decode(mc, model, path, grids)


# --- causality ---------------------------------------------------------------


def test_encoder_and_decoder_build_the_same_motion_compensated_reference(tmp_path):
    """The invariant everything else rests on. The encoder warps with the
    DECODED motion for exactly this reason."""
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(12)
    grids = _grids(mc, model, frames)
    path = tmp_path / "seq.nvct"

    encoded = _encode(mc, model, frames, path, grids, gop_size=4)
    _, decoded_latents = _decode(mc, model, path, grids, return_latents=True)

    assert torch.equal(encoded["encoder_latents"], decoded_latents)


def test_the_decoder_reproduces_the_encoders_reconstruction_exactly(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(12)
    grids = _grids(mc, model, frames)
    path = tmp_path / "seq.nvct"

    encoded = _encode(mc, model, frames, path, grids, gop_size=4)
    decoded = _decode(mc, model, path, grids)

    assert torch.equal(encoded["encoder_reconstructions"], decoded)


def test_decoding_uses_only_the_stream_not_the_original_frames(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(8)
    grids = _grids(mc, model, frames)
    path = tmp_path / "seq.nvct"
    _encode(mc, model, frames, path, grids, gop_size=4)

    first = _decode(mc, model, path, grids)
    del frames
    second = _decode(mc, model, path, grids)

    assert torch.equal(first, second)


def test_motion_is_estimated_from_the_reconstruction_not_the_original(tmp_path, monkeypatch):
    """The encoder must estimate motion against x_hat_{t-1}, which the decoder
    has - never against the original x_{t-1}, which it does not. Recording the
    calls is the only way to see which one was actually used."""
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(8)
    grids = _grids(mc, model, frames)

    seen = []
    original = mc.estimate_block_motion

    def recording(reference, current, **kwargs):
        seen.append((reference.detach().clone(), current.detach().clone()))
        return original(reference, current, **kwargs)

    monkeypatch.setattr(mc, "estimate_block_motion", recording)
    _encode(mc, model, frames, tmp_path / "seq.nvct", grids, gop_size=4)

    assert seen, "no motion was estimated"
    for reference, current in seen:
        # The reference passed in must never be one of the ORIGINAL frames.
        for index in range(frames.shape[0]):
            assert not torch.equal(reference, frames[index:index + 1]), (
                "motion was estimated against an original frame the decoder lacks")
        # ...and the target must be an original frame the encoder legitimately has.
        assert any(torch.equal(current, frames[i:i + 1]) for i in range(frames.shape[0]))


def test_motion_estimation_never_sees_a_future_frame(tmp_path, monkeypatch):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(8)
    grids = _grids(mc, model, frames)

    order = []
    original = mc.estimate_block_motion

    def recording(reference, current, **kwargs):
        matches = [i for i in range(frames.shape[0]) if torch.equal(current, frames[i:i + 1])]
        order.append(matches[0] if matches else -1)
        return original(reference, current, **kwargs)

    monkeypatch.setattr(mc, "estimate_block_motion", recording)
    _encode(mc, model, frames, tmp_path / "seq.nvct", grids, gop_size=4)

    # Targets must arrive in strictly increasing frame order: no lookahead.
    assert order == sorted(order)
    assert len(set(order)) == len(order)


def test_a_p_frame_cannot_be_coded_without_a_reference(tmp_path, monkeypatch):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    frames = _moving_frames(4)
    grids = _grids(mc, model, frames)

    monkeypatch.setattr(mc, "gop_frame_types",
                        lambda count, gop: [mc.FRAME_TYPE_P] * count)
    with pytest.raises((mc.CausalityViolationError, mc.TemporalFormatError)):
        _encode(mc, model, frames, tmp_path / "bad.nvct", grids, gop_size=4)


def test_every_sequence_starts_with_an_i_frame_and_resets_reference_state(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    for index, seed in enumerate((0, 1)):
        frames = _moving_frames(6, seed=seed)
        grids = _grids(mc, model, frames)
        path = tmp_path / f"seq{index}.nvct"
        _encode(mc, model, frames, path, grids, gop_size=4)

        frame_type, motion, _ = next(iter(mc.TemporalStreamReader(path)))
        assert frame_type == mc.FRAME_TYPE_I
        assert motion == b"", "an I-frame must carry no temporal dependency"


def test_gop_pattern_and_intra_only_at_gop_one():
    mc = _load_script("m10h_motion_compensation")

    assert [i for i, t in enumerate(mc.gop_frame_types(23, 10)) if t == mc.FRAME_TYPE_I] \
        == [0, 10, 20]
    assert mc.gop_frame_types(7, 1) == [mc.FRAME_TYPE_I] * 7
    with pytest.raises(ValueError):
        mc.gop_frame_types(5, 0)


# --- calibrate_grids: deterministic_kernels guard -----------------------
#
# model.decode() is not bit-reproducible without deterministic_kernels()
# (its transposed convolutions can select nondeterministic cuDNN
# algorithms on GPU - see that function's own docstring). encode_sequence/
# decode_sequence already wrap their decode calls in it; calibrate_grids's
# own two decode() call sites (fitting the residual grid against a warped
# reference) did not, which is real: on GPU the residual/motion statistics
# collected here could drift slightly between processes, exactly the
# category of bug CHANGELOG.md's M10L entry measured (a 5th-significant-
# figure difference traced to this same missing guard, in a sibling
# function). These tests can only verify the GUARD is active - the actual
# numerical drift is a cuDNN/GPU-specific effect this CPU suite cannot
# reproduce - so they check torch.backends.cudnn state directly rather
# than asserting bit-exact output.


def _calibration_sequences(tmp_path, *, count=2, frames_per_sequence=4, size=64):
    sequences = []
    for index in range(count):
        directory = make_sequence(
            tmp_path / f"seq{index}", num_frames=frames_per_sequence, width=size, height=size)
        frame_paths = tuple(sorted(directory.iterdir()))
        sequences.append(BenchmarkSequence(
            dataset="synthetic", sequence_id=f"seq{index}", split="train",
            frame_paths=frame_paths, width=size, height=size))
    return sequences


def test_calibrate_grids_enables_deterministic_kernels_around_decode(tmp_path, monkeypatch):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    sequences = _calibration_sequences(tmp_path)

    observed = []
    original_decode = BaselineAutoencoder.decode

    def spying_decode(self, z):
        observed.append((torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark))
        return original_decode(self, z)

    monkeypatch.setattr(BaselineAutoencoder, "decode", spying_decode)

    mc.calibrate_grids(model, sequences, bits=4, block_size=16, search_range=4, gop_size=2)

    assert observed, "decode() was never called - this test would pass vacuously without it"
    assert all(flags == (True, False) for flags in observed), (
        "calibrate_grids called decode() without deterministic_kernels() active"
    )


def test_calibrate_grids_restores_cudnn_settings_afterward(tmp_path):
    mc = _load_script("m10h_motion_compensation")
    model = _model()
    sequences = _calibration_sequences(tmp_path)

    # Start from a deliberately non-default state, so "restored" is a real
    # assertion and not coincidentally matching torch's own defaults.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    mc.calibrate_grids(model, sequences, bits=4, block_size=16, search_range=4, gop_size=2)

    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
