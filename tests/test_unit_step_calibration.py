"""Tests for `calibration.unit_step_quantization_params` - M24 Phase B's quantizer
expressed as the codec's own affine grid.

The property everything rests on: through `UniformQuantizer`, these params give
back exactly `round(x / step) * step`, the rule the Phase B models were trained
for. If that drifts, the gate measures a quantizer the models never saw.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nvc.compression.calibration import UNIT_STEP_MAX_BITS, unit_step_quantization_params
from nvc.compression.codec import (
    decode_payload_to_latent,
    encode_latent_to_payload,
    latent_to_symbols,
)
from nvc.compression.entropy_model import EmpiricalEntropyModel
from nvc.compression.quantization import UniformQuantizer, count_clipped


def _latents(seed=0, scale=3.0):
    torch.manual_seed(seed)
    return torch.randn(10, 6, 4, 4) * scale


@pytest.mark.parametrize("step", [0.5, 1.0, 1.5])
def test_it_reproduces_fixed_step_rounding_exactly(step):
    latents = _latents()
    params = unit_step_quantization_params(latents, step=step)
    quantizer = UniformQuantizer(params.bits, params.mode)

    q, _ = quantizer.quantize(latents, params)
    restored = quantizer.dequantize(q, params)

    assert torch.equal(restored, torch.round(latents / step) * step)
    assert count_clipped(latents, params)["clipped_total"] == 0


def test_the_alphabet_is_the_narrowest_that_covers_the_widest_channel():
    latents = _latents()
    params = unit_step_quantization_params(latents, step=1.0)
    symbols = torch.round(latents)
    low = symbols.amin(dim=(0, 2, 3))
    span = int((symbols.amax(dim=(0, 2, 3)) - low).max()) + 1

    assert params.bits == math.ceil(math.log2(span))
    assert 2 ** (params.bits - 1) < span <= 2 ** params.bits
    assert params.mode == "per_channel"
    assert torch.equal(params.zero_point.flatten(), -low)
    assert torch.all(params.scale == 1.0)


def test_a_constant_latent_still_gets_a_one_bit_alphabet():
    params = unit_step_quantization_params(torch.zeros(2, 3, 2, 2), step=1.0)

    assert params.bits == 1


def test_values_beyond_the_train_span_are_clamped_and_counted():
    latents = _latents()
    params = unit_step_quantization_params(latents, step=1.0)
    outlier = latents[:1].clone()
    outlier[0, 0, 0, 0] = 1e3

    assert count_clipped(outlier, params)["clipped_high"] == 1
    q, _ = UniformQuantizer(params.bits, params.mode).quantize(outlier, params)
    assert int(q.max()) <= params.q_max


def test_a_stream_coded_with_it_decodes_bit_exactly():
    """The params go through the real range coder unchanged - no format change."""
    latents = _latents()
    params = unit_step_quantization_params(latents, step=1.0)
    symbols = np.stack([latent_to_symbols(latents[i:i + 1], params).reshape(6, -1)
                        for i in range(latents.shape[0])])
    model = EmpiricalEntropyModel.from_symbols(symbols, bits=params.bits, num_tables=6)
    frame = _latents(seed=1)[:1]

    payload, sent = encode_latent_to_payload(frame, params=params, entropy_model=model)
    decoded, received = decode_payload_to_latent(payload, entropy_model=model, params=params,
                                                 shape=tuple(frame.shape[1:]))

    assert np.array_equal(sent, received)
    q, _ = UniformQuantizer(params.bits, params.mode).quantize(frame, params)
    assert torch.equal(decoded, UniformQuantizer(params.bits, params.mode).dequantize(q, params))


def test_an_alphabet_wider_than_the_limit_is_refused():
    """Every symbol holds a floor frequency, so a huge table prices nothing well."""
    with pytest.raises(ValueError, match="needs"):
        unit_step_quantization_params(_latents(scale=50.0), step=0.01)
    assert UNIT_STEP_MAX_BITS <= 12


@pytest.mark.parametrize("step", [0.0, -1.0, float("nan"), float("inf")])
def test_a_meaningless_step_is_refused(step):
    with pytest.raises(ValueError, match="step must be finite"):
        unit_step_quantization_params(_latents(), step=step)


def test_a_non_4d_tensor_is_refused():
    with pytest.raises(ValueError, match="4D"):
        unit_step_quantization_params(torch.zeros(3, 4), step=1.0)
