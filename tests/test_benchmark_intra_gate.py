"""Tests for `scripts/benchmark_intra_gate.py` - the Stage 1 gate harness.

The load-bearing test is `test_it_defaults_to_the_same_checkpoint_as_the_parity_harness`.
The first version of this script defaulted to `vimeo_epoch17_best.pt`, which is
where the checkpoint lineage *starts* (Vimeo training, before the DAVIS
fine-tune) rather than the deployed M10F model. It calibrated a different intra
grid and coded 38% more bytes at 2.5 dB lower PSNR than the run it was supposed
to reproduce - a difference that looked like a bug in the lean coding path and
was not.

The rest covers the intra encode/decode round trip on a miniature model, and the
reference-curve plumbing. The real end-to-end check is `--verify`, which
reproduces the committed intra-only run's own totals; that needs the DAVIS frames
and the deployed checkpoint, so it lives in the run rather than here.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _script("benchmark_intra_gate")


@pytest.fixture(scope="module")
def parity():
    return _script("benchmark_parity")


# --- the mistake that cost a debugging cycle ---------------------------------------


def test_it_defaults_to_the_same_checkpoint_as_the_parity_harness(gate, parity):
    """--verify compares this run against numbers benchmark_parity measured on
    the DEPLOYED model. Pointed at any other checkpoint it reproduces nothing,
    and the mismatch reads as a bug in the coding path rather than in a default."""
    from nvc.utils.config import load_default_config

    defaults = load_default_config()
    gate_default = gate.build_arg_parser(defaults).get_default("checkpoint")
    parity_default = parity.build_arg_parser(defaults).get_default("checkpoint")

    assert gate_default == parity_default
    assert "m10f" in str(gate_default).lower()


def test_the_deployed_checkpoint_is_not_the_start_of_the_lineage(gate):
    """`vimeo_epoch17_best.pt` is the pre-fine-tune checkpoint. Naming it here
    would silently measure a different, worse model."""
    assert "vimeo_epoch17" not in str(gate.DEPLOYED_CHECKPOINT)


def test_calibration_runs_at_the_deployed_gop_not_at_one(gate):
    """calibrate_grids fits the residual grid and motion table on P-frames and
    raises on an empty collection, so it cannot run at GOP 1 - while the intra
    half it produces does not depend on the GOP at all."""
    assert gate.CALIBRATION_GOP == 10


# --- the intra round trip ---------------------------------------------------------


def _tiny_rig(gate, bits=4, latent_channels=8):
    """An intra rig assembled by hand, so the round trip can be tested without
    running calibrate_grids over a real dataset."""
    from nvc.compression.calibration import calibrate_quantization_params
    from nvc.compression.codec import latent_to_symbols
    from nvc.compression.entropy_model import EmpiricalEntropyModel

    torch.manual_seed(0)
    latents = torch.randn(6, latent_channels, 4, 4)
    params = calibrate_quantization_params(latents, bits=bits, mode="per_channel")
    # latent_to_symbols codes one frame at a time, as the real coder does.
    import numpy as np
    symbols = np.stack([latent_to_symbols(latents[i:i + 1], params)
                        for i in range(latents.shape[0])])
    model = EmpiricalEntropyModel.from_symbols(symbols.reshape(
        latents.shape[0], latent_channels, -1), bits=bits, num_tables=latent_channels)
    return {"bits": bits, "intra_params": params, "intra_entropy_model": model,
            "residual_params": params, "residual_entropy_model": model,
            "motion_entropy_model": model, "provenance": {}}


def test_encode_then_decode_is_bit_exact(gate, tmp_path):
    """The decoder must reach the encoder's reconstruction from the file alone -
    the property that makes the measured bytes a real bitstream rather than an
    estimate."""
    from nvc.models import BaselineAutoencoder

    torch.manual_seed(0)
    model = BaselineAutoencoder(latent_channels=8, base_channels=4).eval()
    rig = _tiny_rig(gate, latent_channels=8)
    frames = torch.rand(5, 3, 32, 32)
    path = tmp_path / "intra.nvct"

    encoded = gate.encode_intra_sequence(model, frames, rig, path,
                                         block_size=16, search_range=4)
    decoded = gate.decode_intra_sequence(model, path, rig)

    assert torch.equal(decoded, encoded["reconstructions"])
    assert decoded.shape == frames.shape
    assert path.stat().st_size > 0


def test_every_frame_in_the_stream_is_an_i_frame(gate, tmp_path):
    from nvc.models import BaselineAutoencoder
    from nvc.video.container import FRAME_TYPE_I, TemporalStreamReader

    torch.manual_seed(0)
    model = BaselineAutoencoder(latent_channels=8, base_channels=4).eval()
    rig = _tiny_rig(gate, latent_channels=8)
    path = tmp_path / "intra.nvct"

    gate.encode_intra_sequence(model, torch.rand(4, 3, 32, 32), rig, path,
                               block_size=16, search_range=4)

    reader = TemporalStreamReader(path)
    assert reader.header.gop_size == 1
    types = [frame_type for frame_type, _, _ in reader]
    assert types == [FRAME_TYPE_I] * 4
    assert len(types) == reader.header.frame_count


def test_a_stream_with_a_p_frame_is_refused_by_the_decoder(gate, tmp_path):
    """Negative control for the decoder's own assumption: it only handles
    I-frames, so it has to say so rather than mis-decode a P-frame's payload."""
    from nvc.models import BaselineAutoencoder
    from nvc.video.container import FRAME_TYPE_I, FRAME_TYPE_P, TemporalStreamWriter

    torch.manual_seed(0)
    model = BaselineAutoencoder(latent_channels=8, base_channels=4).eval()
    rig = _tiny_rig(gate, latent_channels=8)
    good = tmp_path / "good.nvct"
    gate.encode_intra_sequence(model, torch.rand(2, 3, 32, 32), rig, good,
                               block_size=16, search_range=4)

    from nvc.video.container import TemporalStreamHeader, TemporalStreamReader
    source = TemporalStreamReader(good)
    header = source.header
    records = list(source)
    bad = tmp_path / "bad.nvct"
    writer = TemporalStreamWriter(bad, header, rig["intra_params"], rig["residual_params"])
    writer.append_frame(FRAME_TYPE_I, b"", records[0][2])
    writer.append_frame(FRAME_TYPE_P, b"\x00", records[1][2])
    writer.close()

    with pytest.raises(ValueError, match="only I-frames"):
        gate.decode_intra_sequence(model, bad, rig)


# --- the reference curve ----------------------------------------------------------


def test_the_reference_curve_comes_from_the_committed_denominator(gate, parity, tmp_path):
    """The x264 all-intra curve is read, not re-encoded, so the gate's reference
    side is the same curve the +176.2% figure came from."""
    report = {"points": {
        f"h264_intra/crf{crf}": {
            "arm": "h264_intra", "bpp": bpp,
            "quality": {c: {"psnr": psnr, "msssim": 0.9}
                        for c in parity.CONVENTIONS},
        } for crf, bpp, psnr in [(20, 0.8, 32.0), (30, 0.3, 28.0), (40, 0.1, 24.0)]
    }}
    path = tmp_path / "denominator.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    curve = gate.reference_curve(path, "sequence_mean", "psnr", parity)

    assert len(curve) == 3
    assert gate.REFERENCE_ARM == "h264_intra"


def test_a_denominator_with_too_few_points_is_refused(gate, parity, tmp_path):
    """One point cannot define a curve, and a BD-rate computed against it would
    be a number with no meaning rather than an error."""
    path = tmp_path / "thin.json"
    path.write_text(json.dumps({"points": {"h264_intra/crf30": {
        "arm": "h264_intra", "bpp": 0.3,
        "quality": {c: {"psnr": 28.0, "msssim": 0.9} for c in parity.CONVENTIONS}}}}),
        encoding="utf-8")

    with pytest.raises(SystemExit, match="fewer than two"):
        gate.reference_curve(path, "sequence_mean", "psnr", parity)


def test_it_reuses_the_parity_harness_scorer(gate):
    """Two scorers would end the one thing that makes these numbers comparable
    across runs."""
    source = (ROOT / "scripts" / "benchmark_intra_gate.py").read_text(encoding="utf-8")

    assert 'parity = _load_script("benchmark_parity")' in source
    for helper in ("score_frames", "to_uint8_frames", "summarize_point", "bd_rate", "curve"):
        assert f"parity.{helper}(" in source
        assert f"def {helper}(" not in source


def test_it_needs_no_trained_entropy_model_or_codebook(gate):
    """The whole point: a 192-channel Stage 1 latent cannot load the 64-channel
    G16 checkpoint or the codebooks fitted to it, and an intra stream needs
    neither."""
    import ast

    path = ROOT / "scripts" / "benchmark_intra_gate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Prose, not code: the module docstring explains at length what this avoids
    # and why, so a plain substring search over the file would match its own
    # explanation. Only calls and imports count.
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = getattr(target, "attr", None) or getattr(target, "id", None)
            if name:
                called.add(name)
            for argument in node.args:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    called.add(argument.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            called.add(getattr(node, "module", "") or "")
            called.update(alias.name for alias in node.names)

    for absent in ("m11_ar_entropy", "m10k_learned_entropy", "m10l_shared_codebook",
                   "prepare_rate_point", "check_provenance", "SharedCodebook"):
        assert absent not in called, f"{absent} would reintroduce the 64-channel dependency"
    # and the one research script it DOES load, for the calibration recipe only
    assert "m10h_motion_compensation" in called
