"""M23 Phase B2 step 2 - the codec that actually carries the scale map.

B2 step 1 proved the gain realises in real coded bytes and that the context is
rebuildable from the bitstream. It did that outside a container. This is the
same coding, inside one: a real `.nvct` v3 stream with the map in the side
payload, written and read through the PRODUCTION container.

WHY THIS PATH EXISTS AT ALL
---------------------------
There are two `.nvct` implementations in this repository, deliberately kept
byte-compatible:

  `src/nvc/video/container.py`        production, used by `src/nvc/video/codec.py`,
                                      and where v3 lives.
  `scripts/m10h_motion_compensation`  a complete parallel copy, which is what
                                      every milestone measurement has gone
                                      through (m21 -> benchmark_parity, M22,
                                      benchmark_fullvideo_transform).

v3 was added to the production module. The measurement path uses the other one,
and M10H's script is frozen - earlier milestones' scripts are never rewritten.
So this module is M23's own encode/decode: m10h's compression primitives
(motion estimation, warping, intra coding) reused read-only, the PRODUCTION
container for I/O, and the joint (prototype, context) tables for the residual.

That also begins consolidating the duplication onto one format implementation
rather than adding v3 to a second copy of it.

THE EQUIVALENCE PROOF COMES FIRST
---------------------------------
A reimplemented I/O path could change the bytes for reasons that have nothing to
do with the scale map, and then a v3 measurement would be meaningless. So with
`side=None` this encoder takes the v2 path and must produce a stream
BYTE-IDENTICAL to `m21_refinement.encode_sequence_refined` at identity - the
deployed arm - on the same inputs. `tests/test_m23_codec.py` pins that, and the
container halves are already known to agree byte-for-byte
(`tests/test_container_v3.py`).

DECODE ORDER
------------
The side payload is written last in the frame record and read FIRST, because the
residual's entropy tables are selected by the context it carries. The decoder
derives that context from the decoded map and from nothing else; the encoder
proves the same by decoding its own map back before using it.

HARD SCOPE
----------
Frozen: the transform, the quantizer grids, M10K, the G16 network, the K=512
assignment codebook, the M13 mechanism, lambda, GOP, motion estimation and
coding, the TRAIN/VAL/TEST split. `src/nvc/` is read-only here - v3 was added in
its own commit with its own compatibility proof.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.range_coder import ResumableDecoder, decode_symbols, encode_symbols
from nvc.video.container import (
    FRAME_TYPE_I, FRAME_TYPE_P, TEMPORAL_FORMAT_VERSION, TEMPORAL_FORMAT_VERSION_V3,
    TemporalStreamHeader, TemporalStreamReader, TemporalStreamWriter,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------------------------
# The side channel's fitted parts, all from TRAIN
# --------------------------------------------------------------------------------------------


class SideSpecification:
    """Everything the side channel needs, fitted on TRAIN and frozen.

    `child_table` holds `groups * bins` joint tables indexed by
    `prototype * bins + context`, so the existing range coder coded it with no
    change: it already accepts an arbitrary per-symbol `table_index`.
    """

    def __init__(self, *, bins: int, stride: int, edges: np.ndarray,
                 map_table: EmpiricalEntropyModel, child_table: EmpiricalEntropyModel,
                 groups: int, identity: bytes) -> None:
        if bins & (bins - 1):
            raise ValueError(f"bins must be a power of two, got {bins}")
        if child_table.frequencies.shape[0] != groups * bins:
            raise ValueError(
                f"child table has {child_table.frequencies.shape[0]} rows; "
                f"{groups} prototypes x {bins} contexts needs {groups * bins}")
        self.bins = bins
        self.stride = stride
        self.edges = np.asarray(edges)
        self.map_table = map_table
        self.child_table = child_table
        self.groups = groups
        self.identity = identity

    @property
    def map_bit_width(self) -> int:
        return int(np.log2(self.bins))

    def coarse_grid(self, height: int, width: int) -> tuple[int, int]:
        """Ceil-divided, matching `m23_phaseb_gate.coarsen`'s ceil_mode pooling
        and the production header's `side_grid`."""
        return ((height + self.stride - 1) // self.stride,
                (width + self.stride - 1) // self.stride)


def quantize_scale_map(pb, residual: torch.Tensor, side: SideSpecification) -> np.ndarray:
    """The residual the decoder will reconstruct -> the coarse binned map.

    Exactly `m23_phaseb_gate`'s construction, called rather than reimplemented so
    the codec codes the map the gate measured.
    """
    coarse = pb.coarsen(pb.scale_map_signal(residual), stride=side.stride)
    return pb.to_bins(coarse, side.edges)


def context_from_map(pb, binned: np.ndarray, *, channels: int, height: int,
                     width: int) -> np.ndarray:
    """[1, h, w] binned map -> [C*H*W] context in the coder's C-major order."""
    return pb.broadcast_to_symbol_positions(
        binned, channels=channels, height=height, width=width).reshape(-1)


# --------------------------------------------------------------------------------------------
# Residual coding against the transmitted context
# --------------------------------------------------------------------------------------------


def encode_residual_with_context(ma, mk, model, assign_codebook, reference: torch.Tensor,
                                 symbols: np.ndarray, zero: torch.Tensor, *,
                                 context: np.ndarray, side: SideSpecification,
                                 timings: dict | None = None) -> tuple[bytes, float]:
    """`m13.encode_frame_recalibrated`, with the table chosen by (prototype,
    context) instead of prototype alone.

    The assignment still comes from the deployed `assign_codebook` and the
    network is the deployed G16, so the only thing that differs from the shipped
    residual path is which of the joint tables each symbol is coded against.
    """
    device = reference.device
    started = time.perf_counter()
    target = torch.from_numpy(np.asarray(symbols, dtype=np.int64))[None].to(device)
    rows = ma._rows(model.log_probabilities(reference, model.planes(target, zero.to(device))))
    if device.type == "cuda":
        torch.cuda.synchronize()
    network = time.perf_counter() - started

    started = time.perf_counter()
    _, prototype = ma._tables_for(rows, assign_codebook, mk)
    joint = np.asarray(prototype, dtype=np.int64) * side.bins + context
    tables = time.perf_counter() - started

    flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
    started = time.perf_counter()
    payload = encode_symbols(flat, side.child_table.cumulative, joint)
    coder = time.perf_counter() - started
    if timings is not None:
        for key, value in (("network", network), ("tables", tables), ("coder", coder)):
            timings[key] = timings.get(key, 0.0) + value

    cumulative = side.child_table.cumulative
    totals = cumulative[:, -1]
    lower = cumulative[joint, flat]
    upper = cumulative[joint, flat + 1]
    ideal = float(-np.log2((upper - lower) / totals[joint]).sum())
    return payload, ideal


def decode_residual_with_context(ma, mk, model, assign_codebook, payload: bytes,
                                 reference: torch.Tensor, zero: torch.Tensor, *,
                                 context: np.ndarray, side: SideSpecification,
                                 shape: tuple[int, int, int]) -> np.ndarray:
    """Inverse, group-wise through M12's `ResumableDecoder`.

    The context is already known for every position before the first group is
    decoded, because the map was read from the stream first. That is what makes a
    transmitted side channel different from a causal one: it needs no
    interleaving with the symbol decode.
    """
    channels, height, width = shape
    plane = height * width
    group = model.group_size
    device = reference.device
    zero_d = zero.to(device)
    decoded = zero_d.view(channels, 1, 1).expand(channels, height, width).clone()
    symbols = np.empty(channels * plane, dtype=np.int64)

    decoder = ResumableDecoder(payload)
    for start in range(0, channels, group):
        stop = start + group
        log_probabilities = model.log_probabilities(reference,
                                                    model.planes(decoded[None], zero_d))
        rows = ma._rows(log_probabilities[:, start:stop])
        _, prototype = ma._tables_for(rows, assign_codebook, mk)
        joint = (np.asarray(prototype, dtype=np.int64) * side.bins
                 + context[start * plane:stop * plane])
        group_symbols = decoder.decode_group(side.child_table.cumulative, joint)
        symbols[start * plane:stop * plane] = group_symbols
        decoded[start:stop] = torch.from_numpy(
            group_symbols.reshape(stop - start, height, width)).to(device)
    decoder.close()
    return symbols


# --------------------------------------------------------------------------------------------
# Whole sequences, through the production container
# --------------------------------------------------------------------------------------------


def build_header(frames: torch.Tensor, latent_shape, *, gop_size, intra_params,
                 residual_params, block_size, search_range, motion_entropy_model,
                 intra_entropy_model, residual_identity: bytes,
                 side: SideSpecification | None) -> TemporalStreamHeader:
    """The same header fields m21 writes, plus v3's three when a map is carried."""
    extra: dict[str, Any] = {}
    if side is not None:
        extra = {"format_version": TEMPORAL_FORMAT_VERSION_V3, "side_bins": side.bins,
                 "side_stride": side.stride, "side_entropy_model_id": side.identity}
    else:
        extra = {"format_version": TEMPORAL_FORMAT_VERSION}
    return TemporalStreamHeader(
        gop_size=gop_size, quantization_bits=intra_params.bits,
        quantization_mode=intra_params.mode, image_width=frames.shape[3],
        image_height=frames.shape[2], image_channels=frames.shape[1],
        latent_channels=latent_shape[0], latent_height=latent_shape[1],
        latent_width=latent_shape[2], frame_count=frames.shape[0],
        num_intra_quantization_params=intra_params.scale.numel(),
        num_residual_quantization_params=residual_params.scale.numel(),
        block_size=block_size, search_range=search_range,
        motion_bits=motion_entropy_model.bits, reference_mode="mc",
        intra_entropy_model_id=intra_entropy_model.model_id(),
        residual_entropy_model_id=residual_identity,
        motion_entropy_model_id=motion_entropy_model.model_id(), **extra)


@torch.no_grad()
def encode_sequence(mc, m13, ma, mk, pb, model, frames: torch.Tensor, spec, path, *,
                    intra_params, intra_entropy_model, residual_params,
                    motion_entropy_model, bits: int, gop_size: int, block_size: int,
                    search_range: int, side: SideSpecification | None = None
                    ) -> dict[str, Any]:
    """Closed-loop encode to a `.nvct` stream; v3 with the map when `side` is given.

    The loop is m21's deployed arm at identity refinement, with two changes and
    only two: the container is the production one, and a P-frame's residual is
    coded against the transmitted context. With `side=None` neither applies and
    the bytes must match m21's exactly - which is the equivalence test.
    """
    model.eval()
    device = next(model.parameters()).device
    frame_count = frames.shape[0]
    types = mc.gop_frame_types(frame_count, gop_size)
    latent_shape = tuple(model.encode(frames[0:1].to(device)).shape[1:])
    channels, height, width = latent_shape

    header = build_header(
        frames, latent_shape, gop_size=gop_size, intra_params=intra_params,
        residual_params=residual_params, block_size=block_size, search_range=search_range,
        motion_entropy_model=motion_entropy_model, intra_entropy_model=intra_entropy_model,
        residual_identity=spec["identity"], side=side)
    writer = TemporalStreamWriter(path, header, intra_params, residual_params)

    reconstructions, records = [], []
    previous = None
    ideal_bits = 0.0
    timings: dict[str, float] = {}
    try:
        with mc.deterministic_kernels():
            for index in range(frame_count):
                frame = frames[index:index + 1].to(device)
                latent = model.encode(frame)
                side_payload = b""
                if types[index] == FRAME_TYPE_I:
                    payload, _ = encode_latent_to_payload(
                        latent, params=intra_params, entropy_model=intra_entropy_model)
                    decoded, _ = decode_payload_to_latent(
                        payload, entropy_model=intra_entropy_model, params=intra_params,
                        shape=latent_shape)
                    reconstructed_latent = decoded.to(device)
                    writer.append_frame(types[index], b"", payload)
                    records.append({"frame_type": "I", "motion_bytes": 0,
                                    "residual_bytes": len(payload), "side_bytes": 0,
                                    "gop_position": index % gop_size})
                else:
                    motion = mc.estimate_block_motion(
                        previous, frame, block_size=block_size, search_range=search_range)
                    motion_payload = mc.encode_motion_payload(
                        motion, search_range=search_range, entropy_model=motion_entropy_model)
                    decoded_motion = mc.decode_motion_payload(
                        motion_payload, (frames.shape[2] // block_size,
                                         frames.shape[3] // block_size),
                        search_range=search_range, entropy_model=motion_entropy_model)
                    warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                    reference_latent = model.encode(warped)
                    symbols = latent_to_symbols(latent - reference_latent, residual_params)

                    if side is None:
                        payload, ideal = m13.encode_frame_recalibrated(
                            spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                            reference_latent, symbols.reshape(latent_shape), spec["zero"],
                            bits=bits, timings=timings)
                    else:
                        # The map describes the residual the decoder will hold, so
                        # it is built from the DEQUANTIZED residual.
                        residual = symbols_to_latent(symbols, latent_shape,
                                                     residual_params).to(device)
                        binned = quantize_scale_map(pb, residual, side)
                        flat_map = binned.reshape(-1)
                        map_index = np.zeros(flat_map.size, dtype=np.int64)
                        side_payload = encode_symbols(
                            flat_map, side.map_table.cumulative, map_index)
                        # Decode our own map back and use ONLY that, so the
                        # encoder cannot condition on anything the decoder lacks.
                        recovered = decode_symbols(side_payload, flat_map.size,
                                                   side.map_table.cumulative, map_index)
                        if not np.array_equal(recovered, flat_map):
                            raise RuntimeError(
                                f"frame {index}: the scale map did not survive its own "
                                "round trip; the stream would be undecodable")
                        context = context_from_map(pb, recovered.reshape(binned.shape),
                                                   channels=channels, height=height,
                                                   width=width)
                        payload, ideal = encode_residual_with_context(
                            ma, mk, spec["model"], spec["assign_codebook"], reference_latent,
                            symbols.reshape(latent_shape), spec["zero"], context=context,
                            side=side, timings=timings)
                    ideal_bits += ideal
                    writer.append_frame(types[index], motion_payload, payload, side_payload)
                    records.append({"frame_type": "P", "motion_bytes": len(motion_payload),
                                    "residual_bytes": len(payload),
                                    "side_bytes": len(side_payload),
                                    "gop_position": index % gop_size})
                    reconstructed_latent = reference_latent + symbols_to_latent(
                        symbols, latent_shape, residual_params).to(device)
                reconstruction = model.decode(reconstructed_latent)
                previous = reconstruction
                reconstructions.append(reconstruction.detach().cpu())
    finally:
        writer.close()

    return {"reconstructions": torch.cat(reconstructions, dim=0), "records": records,
            "ideal_bits": ideal_bits, "timings": timings,
            "format_version": header.format_version,
            "bytes": Path(path).stat().st_size}


@torch.no_grad()
def decode_sequence(mc, m13, ma, mk, pb, model, path, spec, *, intra_entropy_model,
                    motion_entropy_model, bits: int,
                    side: SideSpecification | None = None) -> dict[str, Any]:
    """Decode a stream using ONLY the stream, the model and the frozen tables.

    For a v3 stream the map is read from the record and the residual's context
    comes from it alone. The side specification supplies the tables, which are
    frozen artifacts exactly like the intra and motion entropy models - the
    stream carries their identity, not their contents.
    """
    model.eval()
    device = next(model.parameters()).device
    reader = TemporalStreamReader(path)
    header = reader.header
    latent_shape = header.latent_shape
    channels, height, width = latent_shape
    if header.carries_side_information and side is None:
        raise ValueError("stream declares a side channel but no SideSpecification was given")
    if side is not None and header.carries_side_information:
        if header.side_bins != side.bins or header.side_stride != side.stride:
            raise ValueError(
                f"stream declares bins={header.side_bins} stride={header.side_stride}; "
                f"the given specification has bins={side.bins} stride={side.stride}")

    reconstructions = []
    previous = None
    with mc.deterministic_kernels():
        for index, (frame_type, motion_payload, residual_payload, side_payload) in enumerate(
                reader.iter_frames_with_side()):
            if frame_type == FRAME_TYPE_I:
                decoded, _ = decode_payload_to_latent(
                    residual_payload, entropy_model=intra_entropy_model,
                    params=reader.intra_params, shape=latent_shape)
                reconstructed_latent = decoded.to(device)
            else:
                decoded_motion = mc.decode_motion_payload(
                    motion_payload, (header.image_height // header.block_size,
                                     header.image_width // header.block_size),
                    search_range=header.search_range, entropy_model=motion_entropy_model)
                warped = mc.warp_blocks(previous, decoded_motion,
                                        block_size=header.block_size)
                reference_latent = model.encode(warped)
                if header.carries_side_information:
                    grid = side.coarse_grid(height, width)
                    count = grid[0] * grid[1]
                    map_index = np.zeros(count, dtype=np.int64)
                    binned = decode_symbols(side_payload, count,
                                            side.map_table.cumulative, map_index)
                    context = context_from_map(pb, binned.reshape(1, *grid),
                                               channels=channels, height=height, width=width)
                    symbols = decode_residual_with_context(
                        ma, mk, spec["model"], spec["assign_codebook"], residual_payload,
                        reference_latent, spec["zero"], context=context, side=side,
                        shape=latent_shape)
                else:
                    symbols = m13.decode_frame_recalibrated(
                        spec["model"], spec["assign_codebook"], spec["coding_codebook"],
                        residual_payload, reference_latent, spec["zero"], bits=bits,
                        shape=latent_shape)
                reconstructed_latent = reference_latent + symbols_to_latent(
                    symbols, latent_shape, reader.residual_params).to(device)
            reconstruction = model.decode(reconstructed_latent)
            previous = reconstruction
            reconstructions.append(reconstruction.detach().cpu())

    return {"reconstructions": torch.cat(reconstructions, dim=0),
            "format_version": header.format_version}
