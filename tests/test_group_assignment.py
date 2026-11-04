"""The residual encoders must assign codebook prototypes the way the decoder does.

The decoder can only assign one channel GROUP at a time; the encoders used to
assign the whole frame in one `assign_tensor` call. Those are different GEMM
shapes, and on a GPU nothing promises the same float costs from both, so a
near-tie between two prototypes could resolve differently on the two sides.
That broke a real round trip (lambda 1e-4 Phase B, 8-bit, 1 position in 1 of
144 frames). Each test here uses a codebook whose assignment deliberately
depends on the batch size, which a one-call encoder cannot survive.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from nvc.video import entropy as shipped


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHANNELS, SIDE, BITS, PROTOTYPES, GROUP = 4, 4, 4, 8, 2


class BatchSensitive:
    """A codebook whose assignment differs for a whole-frame batch: the first
    row moves to another prototype whenever more than one group arrives at once."""

    def __init__(self, codebook, group_rows: int) -> None:
        self.codebook, self.group_rows = codebook, group_rows
        self.cumulative = codebook.cumulative

    def assign_tensor(self, probabilities: torch.Tensor) -> np.ndarray:
        index = self.codebook.assign_tensor(probabilities)
        if probabilities.shape[0] > self.group_rows:
            index[0] = (index[0] + 1) % PROTOTYPES
        return index


def _frequencies(seed: int = 0) -> np.ndarray:
    mk = _load_script("m10k_learned_entropy")
    generator = np.random.default_rng(seed)
    return mk.probabilities_to_frequencies(generator.dirichlet(np.ones(2 ** BITS), PROTOTYPES))


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    reference = torch.randn(1, CHANNELS, SIDE, SIDE)
    symbols = torch.randint(0, 2 ** BITS, (CHANNELS, SIDE, SIDE)).numpy()
    zero = torch.full((CHANNELS,), 2 ** (BITS - 1), dtype=torch.long)
    return reference, symbols, zero


def _model(cls):
    torch.manual_seed(1)
    model = cls(latent_channels=CHANNELS, alphabet=2 ** BITS, hidden=8, group_size=GROUP)
    model.eval()
    return model


def test_the_batch_sensitive_codebook_does_defeat_a_one_call_encoder():
    """Guards the tests below against passing vacuously."""
    model = _model(shipped.ChannelContextEntropyModel)
    codebook = shipped.SharedCodebook(_frequencies(), bits=BITS)
    sensitive = BatchSensitive(codebook, GROUP * SIDE * SIDE)
    reference, symbols, zero = _inputs()
    with torch.no_grad():
        target = torch.from_numpy(symbols)[None]
        log_probabilities = model.log_probabilities(reference, model.planes(target, zero))
    whole = sensitive.assign_tensor(shipped._rows(log_probabilities))
    grouped = shipped.assign_by_group(sensitive, log_probabilities, GROUP)
    assert not np.array_equal(whole, grouped)


def test_shipped_encoder_round_trips_under_a_batch_sensitive_codebook():
    model = _model(shipped.ChannelContextEntropyModel)
    codebook = shipped.SharedCodebook(_frequencies(), bits=BITS)
    sensitive = BatchSensitive(codebook, GROUP * SIDE * SIDE)
    reference, symbols, zero = _inputs()

    payload = shipped.encode_residual_frame(model, sensitive, codebook, reference, symbols, zero)
    back = shipped.decode_residual_frame(model, sensitive, codebook, payload, reference, zero,
                                         shape=symbols.shape)
    assert np.array_equal(back, symbols.reshape(-1))


def test_shipped_bytes_are_unchanged_when_assignment_does_not_depend_on_the_batch():
    """Every stream the old encoder got right is reproduced byte for byte."""
    model = _model(shipped.ChannelContextEntropyModel)
    codebook = shipped.SharedCodebook(_frequencies(), bits=BITS)
    reference, symbols, zero = _inputs()
    with torch.no_grad():
        target = torch.from_numpy(symbols)[None]
        rows = shipped._rows(model.log_probabilities(reference, model.planes(target, zero)))
    old = shipped.encode_symbols(symbols.reshape(-1), codebook.cumulative,
                                 codebook.assign_tensor(rows))
    assert shipped.encode_residual_frame(model, codebook, codebook, reference, symbols,
                                         zero) == old


@pytest.mark.parametrize("coder", ["m11", "m13"])
def test_research_encoders_round_trip_under_a_batch_sensitive_codebook(coder):
    ma = _load_script("m11_ar_entropy")
    ml = _load_script("m10l_shared_codebook")
    model = _model(ma.ChannelContextEntropyModel)
    codebook = ml.SharedCodebook(_frequencies(), bits=BITS)
    sensitive = BatchSensitive(codebook, GROUP * SIDE * SIDE)
    reference, symbols, zero = _inputs()

    if coder == "m11":
        payload, _ = ma.encode_frame(model, reference, symbols, zero, bits=BITS, codebook=sensitive)
        back = ma.decode_frame(model, payload, reference, zero, bits=BITS, shape=symbols.shape,
                               codebook=sensitive)
    else:
        m13 = _load_script("m13_recalibration")
        payload, _ = m13.encode_frame_recalibrated(model, sensitive, codebook, reference, symbols,
                                                   zero, bits=BITS)
        back = m13.decode_frame_recalibrated(model, sensitive, codebook, payload, reference, zero,
                                             bits=BITS, shape=symbols.shape)
    assert np.array_equal(back, symbols.reshape(-1))
