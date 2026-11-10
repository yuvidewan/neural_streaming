"""Gate 3's P-frame entropy coding is lossless and costs what its model says."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _gate3():
    sys.path.insert(0, str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location("p1_gate3", Path("scripts") / "p1_gate3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_table_is_a_valid_integer_distribution_with_no_zero_symbol():
    g3 = _gate3()
    tables = g3.cumulative_tables()
    assert tables.shape == (g3.SCALE_LEVELS, 2 * g3.SYMBOL_RADIUS + 2)
    assert (tables[:, 0] == 0).all() and (tables[:, -1] == g3.FREQUENCY_TOTAL).all()
    assert (np.diff(tables, axis=1) >= 1).all()


def test_scale_index_snaps_to_the_nearest_log_level_and_clamps():
    g3 = _gate3()
    levels = g3.scale_table().float()
    assert torch.equal(g3.scale_index(levels), torch.arange(g3.SCALE_LEVELS))
    edge = torch.tensor([1e-4, 1e4])
    assert g3.scale_index(edge).tolist() == [0, g3.SCALE_LEVELS - 1]


def test_a_p_frame_decodes_to_exactly_what_the_encoder_reconstructed():
    from nvc.models.residual_transform import ResidualGDNAutoencoder
    g3 = _gate3()
    torch.manual_seed(0)
    intra = ResidualGDNAutoencoder(latent_channels=8, base_channels=16).eval()
    model = g3.p1.ConditionalPFrameCoder.from_intra(intra, context_channels=8).eval()
    frame, warped = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
    tables = g3.cumulative_tables()
    with torch.no_grad():
        payload, encoded, ideal = g3.encode_p_frame(model, frame, warped, tables)
        decoded = g3.decode_p_frame(model, payload, warped, tables)
    assert torch.equal(decoded, encoded)
    # A range coder lands within a few bytes of the model's own code length.
    assert abs(len(payload) * 8 - ideal) < 64 + 0.01 * ideal
