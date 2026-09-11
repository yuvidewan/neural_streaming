"""M12 Phase A - the resumable arithmetic decoder.

M11 identified the existing coder's `decode_symbols` as stateless: decoding a
GROUP of symbols whose tables depend on groups already decoded in the same
frame meant calling `decode_symbols(payload, count, ...)` again from byte 0
with a longer prefix every time, re-walking bits already consumed. This
milestone factors the decoder's state (bit position, low/high/value) out into
`ResumableDecoder`, and `decode_symbols`'s C implementation (`rc_decode`) is
now itself a thin wrapper over the same open/decode/close primitives.

So the central claim under test is not "the two paths happen to agree" but
"the two paths are the same code, called differently" - every test here that
compares `decode_symbols` against `ResumableDecoder` is checking that this
refactor did not change behavior, not discovering a coincidence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nvc.compression.entropy_model import EmpiricalEntropyModel, TOTAL_FREQUENCY
from nvc.compression.range_coder import (
    MAX_TOTAL_FREQUENCY,
    ResumableDecoder,
    decode_symbols,
    encode_symbols,
)

BIT_DEPTHS = (5, 4, 3)  # M10H/M10K/M10L/M11's actual operating points


# --- helpers --------------------------------------------------------------


def _uniform_model(bits: int, num_tables: int = 1) -> EmpiricalEntropyModel:
    counts = np.ones((num_tables, 2 ** bits))
    from nvc.compression.entropy_model import _counts_to_frequencies
    return EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=bits)


def _peaked_model(bits: int, peak_symbol: int = 0, num_tables: int = 1) -> EmpiricalEntropyModel:
    counts = np.ones((num_tables, 2 ** bits))
    counts[:, peak_symbol] = 1_000_000
    from nvc.compression.entropy_model import _counts_to_frequencies
    return EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=bits)


def _random_frequency_model(bits: int, num_tables: int, seed: int) -> EmpiricalEntropyModel:
    rng = np.random.default_rng(seed)
    counts = rng.integers(1, 5000, size=(num_tables, 2 ** bits))
    from nvc.compression.entropy_model import _counts_to_frequencies
    return EmpiricalEntropyModel(_counts_to_frequencies(counts), bits=bits)


def _resumable_decode_in_groups(payload, cumulative, table_index, group_sizes) -> np.ndarray:
    """Decode `table_index` through ResumableDecoder split at `group_sizes`
    (which must sum to len(table_index)), and return the concatenated symbols."""
    assert sum(group_sizes) == len(table_index)
    with ResumableDecoder(payload) as decoder:
        parts = []
        position = 0
        for size in group_sizes:
            parts.append(decoder.decode_group(cumulative, table_index[position:position + size]))
            position += size
        assert decoder.symbols_decoded == len(table_index)
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


# --- 1: single-symbol decode ------------------------------------------------


@pytest.mark.parametrize("bits", BIT_DEPTHS)
def test_single_symbol_groups_match_legacy_decode(bits):
    model = _random_frequency_model(bits, num_tables=1, seed=1)
    rng = np.random.default_rng(0)
    symbols = rng.integers(0, 2 ** bits, 300)
    table_index = np.zeros(300, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    legacy = decode_symbols(payload, 300, model.cumulative, table_index)
    resumable = _resumable_decode_in_groups(payload, model.cumulative, table_index, [1] * 300)

    assert np.array_equal(legacy, symbols)
    assert np.array_equal(resumable, symbols)


# --- 2/3: one-group and multiple-group decode -------------------------------


def test_one_group_decode_matches_a_single_decode_symbols_call():
    model = _uniform_model(8)
    rng = np.random.default_rng(2)
    symbols = rng.integers(0, 256, 500)
    table_index = np.zeros(500, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    legacy = decode_symbols(payload, 500, model.cumulative, table_index)
    with ResumableDecoder(payload) as decoder:
        resumable = decoder.decode_group(model.cumulative, table_index)

    assert np.array_equal(resumable, legacy)


@pytest.mark.parametrize("group_sizes", [
    [200],
    [100, 100],
    [50, 50, 50, 50],
    [1, 199, 100],
])
def test_multiple_group_decode_matches_legacy(group_sizes):
    model = _random_frequency_model(6, num_tables=1, seed=3)
    rng = np.random.default_rng(4)
    n = sum(group_sizes)
    symbols = rng.integers(0, 64, n)
    table_index = np.zeros(n, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    legacy = decode_symbols(payload, n, model.cumulative, table_index)
    resumable = _resumable_decode_in_groups(payload, model.cumulative, table_index, group_sizes)

    assert np.array_equal(resumable, legacy)


# --- 4: pause/resume ---------------------------------------------------------


def test_decoder_can_be_paused_and_resumed_across_separate_python_statements():
    model = _uniform_model(8)
    rng = np.random.default_rng(5)
    symbols = rng.integers(0, 256, 1000)
    table_index = np.zeros(1000, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    decoder = ResumableDecoder(payload)
    first = decoder.decode_group(model.cumulative, table_index[:300])
    # ... arbitrary intervening work happens here in real use (a network
    # forward pass building the next group's tables) ...
    second = decoder.decode_group(model.cumulative, table_index[300:700])
    third = decoder.decode_group(model.cumulative, table_index[700:])
    decoder.close()

    assert np.array_equal(np.concatenate([first, second, third]), symbols)


# --- 5: arbitrary group boundaries -------------------------------------------


def test_arbitrary_group_boundaries_all_agree_with_legacy():
    model = _random_frequency_model(5, num_tables=1, seed=6)
    rng = np.random.default_rng(7)
    n = 777
    symbols = rng.integers(0, 32, n)
    table_index = np.zeros(n, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)
    legacy = decode_symbols(payload, n, model.cumulative, table_index)

    # Random, non-uniform group boundaries covering the whole stream.
    boundary_rng = np.random.default_rng(8)
    cuts = sorted(boundary_rng.choice(np.arange(1, n), size=12, replace=False).tolist())
    sizes = np.diff([0] + cuts + [n]).tolist()

    resumable = _resumable_decode_in_groups(payload, model.cumulative, table_index, sizes)
    assert np.array_equal(resumable, legacy)


# --- 6: end-of-stream ---------------------------------------------------------


def test_decoding_the_full_declared_length_terminates_cleanly_at_end_of_stream():
    # decode_group's last group ends exactly at the payload's logical end -
    # the coder's designed "read zero past the end" behavior must not need
    # to fire mid-symbol for a correctly terminated stream.
    model = _peaked_model(4, peak_symbol=3)
    symbols = np.full(2000, 3, dtype=np.int64)
    symbols[::37] = 9  # rare symbols exercise longer codewords near the end
    table_index = np.zeros(2000, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    resumable = _resumable_decode_in_groups(
        payload, model.cumulative, table_index, [500, 500, 500, 500])
    assert np.array_equal(resumable, symbols)


# --- 7: exact stream position -------------------------------------------------


def test_symbols_decoded_counter_tracks_exact_stream_position():
    model = _uniform_model(6)
    rng = np.random.default_rng(9)
    symbols = rng.integers(0, 64, 250)
    table_index = np.zeros(250, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    decoder = ResumableDecoder(payload)
    assert decoder.symbols_decoded == 0
    decoder.decode_group(model.cumulative, table_index[:40])
    assert decoder.symbols_decoded == 40
    decoder.decode_group(model.cumulative, table_index[40:250])
    assert decoder.symbols_decoded == 250
    decoder.close()


def test_two_independent_decoders_at_the_same_group_boundary_agree():
    # The state that determines "where we are in the stream" is entirely
    # inside the handle - two handles over the same payload, stopped at the
    # same boundary, must be in the identical state (same next symbol).
    model = _random_frequency_model(5, num_tables=1, seed=10)
    rng = np.random.default_rng(11)
    symbols = rng.integers(0, 32, 400)
    table_index = np.zeros(400, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    a = ResumableDecoder(payload)
    a.decode_group(model.cumulative, table_index[:150])
    b = ResumableDecoder(payload)
    b.decode_group(model.cumulative, table_index[:150])

    rest_a = a.decode_group(model.cumulative, table_index[150:])
    rest_b = b.decode_group(model.cumulative, table_index[150:])
    a.close()
    b.close()

    assert np.array_equal(rest_a, rest_b)
    assert np.array_equal(rest_a, symbols[150:])


# --- 8/9/10: random / peaked / uniform distributions --------------------------


@pytest.mark.parametrize("bits", BIT_DEPTHS)
def test_random_frequency_tables_round_trip_through_resumable_decoder(bits):
    model = _random_frequency_model(bits, num_tables=4, seed=12)
    rng = np.random.default_rng(13)
    n = 1200
    symbols = rng.integers(0, 2 ** bits, n)
    table_index = rng.integers(0, 4, n)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    resumable = _resumable_decode_in_groups(
        payload, model.cumulative, table_index, [300, 300, 300, 300])
    assert np.array_equal(resumable, symbols)


@pytest.mark.parametrize("bits", BIT_DEPTHS)
def test_peaked_distribution_round_trips_through_resumable_decoder(bits):
    model = _peaked_model(bits, peak_symbol=1)
    symbols = np.full(1500, 1, dtype=np.int64)
    symbols[::53] = 0
    table_index = np.zeros(1500, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    resumable = _resumable_decode_in_groups(
        payload, model.cumulative, table_index, [1, 2, 3, 1494])
    assert np.array_equal(resumable, symbols)


@pytest.mark.parametrize("bits", BIT_DEPTHS)
def test_uniform_distribution_round_trips_through_resumable_decoder(bits):
    model = _uniform_model(bits)
    rng = np.random.default_rng(14)
    symbols = rng.integers(0, 2 ** bits, 900)
    table_index = np.zeros(900, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    resumable = _resumable_decode_in_groups(
        payload, model.cumulative, table_index, [9] * 100)
    assert np.array_equal(resumable, symbols)


# --- 12: all existing entropy model table shapes ------------------------------


def test_resumable_decoder_matches_legacy_under_an_m10l_style_shared_codebook():
    # SharedCodebook's `cumulative`/`table_index` (M10L, and M11's codebook
    # path) is a different table SHAPE than a per-channel EmpiricalEntropyModel
    # (K prototypes vs one table per channel) - both must work unchanged.
    import importlib.util
    from pathlib import Path

    def _load_script(name):
        spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    ml = _load_script("m10l_shared_codebook")
    rng = np.random.default_rng(15)
    alphabet = 16
    k = 6
    raw = rng.random((k, alphabet)) + 0.05
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    mk = _load_script("m10k_learned_entropy")
    frequencies = mk.probabilities_to_frequencies(probabilities)
    codebook = ml.SharedCodebook(frequencies, bits=4)

    n = 2048
    table_index = rng.integers(0, k, n).astype(np.int64)
    symbols = np.array([
        rng.choice(alphabet, p=codebook.probabilities[t]) for t in table_index
    ], dtype=np.int64)
    payload = encode_symbols(symbols, codebook.cumulative, table_index)

    legacy = decode_symbols(payload, n, codebook.cumulative, table_index)
    resumable = _resumable_decode_in_groups(
        payload, codebook.cumulative, table_index, [256] * 8)

    assert np.array_equal(legacy, symbols)
    assert np.array_equal(resumable, symbols)


# --- 13/14: legacy/resumable symbol AND reconstruction equality ---------------


def test_legacy_and_resumable_paths_reconstruct_an_identical_frame():
    from nvc.compression.calibration import calibrate_quantization_params
    from nvc.compression.codec import latent_to_symbols, symbols_to_latent
    from nvc.models.autoencoder import BaselineAutoencoder

    torch.manual_seed(0)
    autoencoder = BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8).eval()
    torch.manual_seed(1)
    frames = torch.rand(6, 3, 32, 32)
    with torch.no_grad():
        latents = autoencoder.encode(frames)
    params = calibrate_quantization_params(latents, bits=8, mode="per_channel")
    latent_shape = tuple(latents.shape[1:])  # (C, H, W)
    channels, height, width = latent_shape
    plane = height * width
    flat_symbols = latent_to_symbols(latents[:1], params)  # flat, C-major

    model = EmpiricalEntropyModel.from_symbols(
        np.stack([latent_to_symbols(latents[i:i + 1], params).reshape(channels, -1)
                  for i in range(latents.shape[0])]),
        bits=8, num_tables=channels)
    table_index = np.repeat(np.arange(channels), plane)
    payload = encode_symbols(flat_symbols, model.cumulative, table_index)

    legacy_symbols = decode_symbols(payload, len(flat_symbols), model.cumulative, table_index)
    resumable_symbols = _resumable_decode_in_groups(
        payload, model.cumulative, table_index,
        [plane] * channels,  # one group per channel, like M11's G=1
    )
    assert np.array_equal(legacy_symbols, resumable_symbols)

    legacy_latent = symbols_to_latent(legacy_symbols, latent_shape, params)
    resumable_latent = symbols_to_latent(resumable_symbols, latent_shape, params)
    assert torch.equal(legacy_latent, resumable_latent)

    with torch.no_grad():
        legacy_reconstruction = autoencoder.decode(legacy_latent)
        resumable_reconstruction = autoencoder.decode(resumable_latent)
    assert torch.equal(legacy_reconstruction, resumable_reconstruction)


# --- 15: deterministic repeated decode -----------------------------------------


def test_repeated_resumable_decodes_of_the_same_payload_are_bit_identical():
    model = _random_frequency_model(4, num_tables=1, seed=16)
    rng = np.random.default_rng(17)
    symbols = rng.integers(0, 16, 600)
    table_index = np.zeros(600, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)

    runs = [
        _resumable_decode_in_groups(payload, model.cumulative, table_index, [50] * 12)
        for _ in range(3)
    ]
    for run in runs[1:]:
        assert np.array_equal(run, runs[0])


# --- 16: corrupted/truncated stream behavior ------------------------------------


def test_truncated_payload_reads_zero_bits_past_the_end_like_legacy_decode():
    # By design the coder reads 0 forever past the end of the payload (see
    # range_coder.py's module docstring) - both paths must honor that
    # identically rather than the resumable one raising or hanging.
    model = _uniform_model(8)
    rng = np.random.default_rng(18)
    symbols = rng.integers(0, 256, 200)
    table_index = np.zeros(200, dtype=np.int64)
    payload = encode_symbols(symbols, model.cumulative, table_index)
    truncated = payload[: max(1, len(payload) // 2)]

    legacy = decode_symbols(truncated, 200, model.cumulative, table_index)
    resumable = _resumable_decode_in_groups(truncated, model.cumulative, table_index, [50] * 4)

    # Not necessarily equal to the original symbols (that's the whole point
    # of truncating), but the two decode paths must still agree with each
    # other bit-for-bit, since they run the same code.
    assert np.array_equal(legacy, resumable)


def test_empty_payload_decodes_via_the_documented_read_zero_behavior():
    model = _uniform_model(8)
    table_index = np.zeros(10, dtype=np.int64)

    legacy = decode_symbols(b"", 10, model.cumulative, table_index)
    with ResumableDecoder(b"") as decoder:
        resumable = decoder.decode_group(model.cumulative, table_index)

    assert np.array_equal(legacy, resumable)


# --- API robustness ---------------------------------------------------------


def test_decode_group_on_a_closed_decoder_raises():
    model = _uniform_model(8)
    payload = encode_symbols(np.zeros(5, dtype=np.int64), model.cumulative,
                             np.zeros(5, dtype=np.int64))
    decoder = ResumableDecoder(payload)
    decoder.close()
    decoder.close()  # closing twice must be safe

    with pytest.raises(RuntimeError):
        decoder.decode_group(model.cumulative, np.zeros(1, dtype=np.int64))


def test_decode_group_rejects_an_empty_group():
    model = _uniform_model(8)
    payload = encode_symbols(np.zeros(5, dtype=np.int64), model.cumulative,
                             np.zeros(5, dtype=np.int64))
    with ResumableDecoder(payload) as decoder:
        with pytest.raises(ValueError):
            decoder.decode_group(model.cumulative, np.zeros(0, dtype=np.int64))


def test_resumable_decoder_rejects_a_frequency_total_that_would_overflow():
    payload = b"\x00\x00\x00\x00"
    cumulative = np.array([[0, MAX_TOTAL_FREQUENCY + 1]], dtype=np.int64)
    with ResumableDecoder(payload) as decoder:
        with pytest.raises(ValueError):
            decoder.decode_group(cumulative, np.zeros(1, dtype=np.int64))


def test_resumable_decoder_is_a_context_manager_that_closes_on_exception():
    model = _uniform_model(8)
    payload = encode_symbols(np.zeros(5, dtype=np.int64), model.cumulative,
                             np.zeros(5, dtype=np.int64))
    decoder_ref = None
    with pytest.raises(ZeroDivisionError):
        with ResumableDecoder(payload) as decoder:
            decoder_ref = decoder
            raise ZeroDivisionError("unrelated failure mid-decode")
    assert decoder_ref._closed is True
