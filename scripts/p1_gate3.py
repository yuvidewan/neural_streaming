"""P1 gate 3: the learned P-frame coder inside the real video codec, with real bytes.

THE QUESTION
------------
Gates 1 and 2 scored ONE P-step from an intra-quality reference, with an ideal code length.
Here each P-frame comes from the previous P-frame's reconstruction, along a GOP of 10, with
real arithmetic-coded bytes in a real file that is decoded again. Is the resulting video
codec better than the deployed one, measured against x264?

    deployed Phase B codec: +127.1% BD-rate vs x264 (outputs/stage1_refit/phase_b_lambda3e-4)
    gate 3 passes if the learned-P codec beats that by more than 10 points (< +117.1%)

WHAT IS CODED, PER RATE POINT (one P-coder per lambda, paired with its intra checkpoint)
---------------------------------------------------------------------------------------
    I-frames  the deployed rig's intra path, unchanged: same checkpoint, same quantizer and
              entropy model, same payload (`m21.prepare_rate_point`)
    motion    the deployed integer 16x16 block search on the previous RECONSTRUCTION, coded
              with the deployed motion entropy model, round-tripped before warping
    P-frames  the learned coder: c = F(x_warp), y = E_p(x_t, c), (mu, b) = P(c),
              s = round(y - mu), x_hat = S(mu + s) + R(.), clamped to [0, 1]

The P-frame symbols are coded with the project's own range coder against a Laplace(b) whose
scale is snapped to a fixed table of SCALE_LEVELS log-spaced values. Encoder and decoder
build the identical integer tables and pick the identical table per symbol from (mu, b),
which both compute from x_warp. The file is a `.nvct` v2 container written by the deployed
writer, so framing costs what it costs the deployed codec. Every stream is decoded from the
file and its reconstruction must equal the encoder's bit for bit.

Scoring, BD-rate and the x264/x265 curves come from `benchmark_parity` and its stored
Phase B runs, under the same primary convention. Nothing here is tuned on DAVIS TEST: the
P-coders are the gate 2 models as trained, scored once.

Usage:
    p1_gate3.py [--models-dir outputs/p1_pframe/gate2_local/stage2] [--lambdas 1e-3 3e-4 1e-4]
                [--max-sequences N --max-frames N]   (smoke runs only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_parity as bp  # noqa: E402
import p1_pframe as p1  # noqa: E402

from nvc.compression.range_coder import decode_symbols, encode_symbols  # noqa: E402
from nvc.evaluation.sequences import discover_sequences  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402
from nvc.utils.device import get_device  # noqa: E402
from nvc.utils.seed import seed_everything  # noqa: E402

BITS = {"1e-3": 7, "3e-4": 8, "1e-4": 8}          # the Phase B parity runs' bit depths
REFIT_DIR = Path("outputs/stage1_refit")
PARITY_DIR = Path("outputs/benchmarks")
DEPLOYED_BD_RATE = 127.1                           # Phase B vs x264, seq-mean PSNR
GATE_MARGIN = 10.0
SCALE_LEVELS = 64
SCALE_MIN, SCALE_MAX = p1.SCALE_FLOOR, 64.0
SYMBOL_RADIUS = 255                                # symbols coded on [-255, 255]
FREQUENCY_TOTAL = 1 << 24


# --------------------------------------------------------------------------------------------
# Entropy coding of the learned P-frame latent
# --------------------------------------------------------------------------------------------


def scale_table() -> torch.Tensor:
    return torch.exp(torch.linspace(np.log(SCALE_MIN), np.log(SCALE_MAX), SCALE_LEVELS,
                                    dtype=torch.float64))


def scale_index(scale: torch.Tensor) -> torch.Tensor:
    """Nearest table entry in log space. Encoder and decoder both call this on the same
    float32 scale tensor, so they pick the same table."""
    step = (np.log(SCALE_MAX) - np.log(SCALE_MIN)) / (SCALE_LEVELS - 1)
    position = (torch.log(scale.double()) - np.log(SCALE_MIN)) / step
    return position.round().clamp(0, SCALE_LEVELS - 1).long()


def cumulative_tables() -> np.ndarray:
    """[SCALE_LEVELS, 2R+2] integer cumulative frequencies of a zero-mean Laplace(scale)
    discretized on unit bins over [-R, R]. Every symbol keeps a frequency of at least 1."""
    values = torch.arange(-SYMBOL_RADIUS, SYMBOL_RADIUS + 1, dtype=torch.float64)
    tables = []
    for scale in scale_table():
        bits = p1.laplace_bits(values, scale)            # the training loss's own mass
        mass = torch.pow(2.0, -bits)
        mass = mass / mass.sum()
        free = FREQUENCY_TOTAL - len(values)
        frequency = 1 + torch.floor(mass * free).long()
        frequency[SYMBOL_RADIUS] += FREQUENCY_TOTAL - int(frequency.sum())   # rest to zero
        tables.append(np.concatenate([[0], np.cumsum(frequency.numpy())]))
    return np.stack(tables).astype(np.int64)


def p_frame_side(model: p1.ConditionalPFrameCoder, warped: torch.Tensor):
    """What the decoder can compute before reading a symbol: context, mean, table index."""
    context = model.context(warped)
    mean, log_scale = model.prior(context).chunk(2, dim=1)
    scale = log_scale.exp().clamp_min(p1.SCALE_FLOOR)
    return context, mean, scale_index(scale)


def p_frame_reconstruct(model, symbols, mean, warped, context) -> torch.Tensor:
    base = model.synthesis(mean + symbols)
    return (base + model.refine(torch.cat([base, warped, context], dim=1))).clamp(0, 1)


def encode_p_frame(model, frame, warped, tables) -> tuple[bytes, torch.Tensor, float]:
    context, mean, index = p_frame_side(model, warped)
    y = model.encoder(torch.cat([frame, context], dim=1))
    symbols = torch.round(y - mean)
    peak = int(symbols.abs().max())
    if peak > SYMBOL_RADIUS:
        raise ValueError(f"P-frame symbol {peak} is outside the coded range +-{SYMBOL_RADIUS}")
    flat_symbols = (symbols.flatten().long() + SYMBOL_RADIUS).cpu().numpy()
    flat_index = index.flatten().cpu().numpy()
    payload = encode_symbols(flat_symbols, tables, flat_index)
    ideal = float(p1.laplace_bits(symbols, scale_table().to(symbols.device)[index].float()).sum())
    return payload, p_frame_reconstruct(model, symbols, mean, warped, context), ideal


def decode_p_frame(model, payload, warped, tables) -> torch.Tensor:
    context, mean, index = p_frame_side(model, warped)
    flat = decode_symbols(payload, index.numel(), tables, index.flatten().cpu().numpy())
    symbols = (torch.from_numpy(np.asarray(flat, dtype=np.int64)) - SYMBOL_RADIUS)
    symbols = symbols.reshape(mean.shape).to(mean.device, mean.dtype)
    return p_frame_reconstruct(model, symbols, mean, warped, context)


def model_identity(model: torch.nn.Module) -> bytes:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.digest()[:8]


# --------------------------------------------------------------------------------------------
# One sequence: encode to a .nvct file, decode it back
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def encode_sequence(mc, m21, intra, pcoder, frames, path, rig, tables, *, gop, block_size,
                    search_range, identity) -> dict[str, Any]:
    device = next(intra.parameters()).device
    count = frames.shape[0]
    types = mc.gop_frame_types(count, gop)
    intra_params = rig["intra_params"]
    latent_shape = tuple(intra.encode(frames[0:1].to(device)).shape[1:])
    header = mc.TemporalStreamHeader(
        gop_size=gop, quantization_bits=intra_params.bits,
        quantization_mode=intra_params.mode, image_width=frames.shape[3],
        image_height=frames.shape[2], image_channels=frames.shape[1],
        latent_channels=latent_shape[0], latent_height=latent_shape[1],
        latent_width=latent_shape[2], frame_count=count,
        num_intra_quantization_params=intra_params.scale.numel(),
        num_residual_quantization_params=intra_params.scale.numel(),
        block_size=block_size, search_range=search_range,
        motion_bits=rig["motion_entropy_model"].bits, reference_mode="mc",
        intra_entropy_model_id=rig["intra_entropy_model"].model_id(),
        residual_entropy_model_id=identity,
        motion_entropy_model_id=rig["motion_entropy_model"].model_id())
    writer = mc.TemporalStreamWriter(path, header, intra_params, intra_params)
    grid = (frames.shape[2] // block_size, frames.shape[3] // block_size)
    reconstructions, records, previous = [], [], None
    with mc.deterministic_kernels():
        for t in range(count):
            frame = frames[t:t + 1].to(device)
            if types[t] == mc.FRAME_TYPE_I:
                payload, _ = m21.encode_latent_to_payload(
                    intra.encode(frame), params=intra_params,
                    entropy_model=rig["intra_entropy_model"])
                latent, _ = m21.decode_payload_to_latent(
                    payload, entropy_model=rig["intra_entropy_model"], params=intra_params,
                    shape=latent_shape)
                reconstruction = intra.decode(latent.to(device))
                writer.append_frame(types[t], b"", payload)
                records.append({"type": "I", "motion": 0, "residual": len(payload),
                                "ideal_bits": 0.0})
            else:
                motion = mc.estimate_block_motion(previous, frame, block_size=block_size,
                                                  search_range=search_range)
                motion_payload = mc.encode_motion_payload(
                    motion, search_range=search_range,
                    entropy_model=rig["motion_entropy_model"])
                decoded_motion = mc.decode_motion_payload(
                    motion_payload, grid, search_range=search_range,
                    entropy_model=rig["motion_entropy_model"])
                warped = mc.warp_blocks(previous, decoded_motion, block_size=block_size)
                payload, reconstruction, ideal = encode_p_frame(pcoder, frame, warped, tables)
                writer.append_frame(types[t], motion_payload, payload)
                records.append({"type": "P", "motion": len(motion_payload),
                                "residual": len(payload), "ideal_bits": ideal})
            previous = reconstruction
            reconstructions.append(reconstruction.cpu())
    writer.close()
    return {"reconstructions": torch.cat(reconstructions), "records": records}


@torch.no_grad()
def decode_sequence(mc, m21, intra, pcoder, path, rig, tables, *, identity) -> torch.Tensor:
    device = next(intra.parameters()).device
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    for label, declared, supplied in (
            ("P-frame", header.residual_entropy_model_id, identity),
            ("intra", header.intra_entropy_model_id, rig["intra_entropy_model"].model_id()),
            ("motion", header.motion_entropy_model_id, rig["motion_entropy_model"].model_id())):
        if declared[:8] != supplied[:8]:
            raise mc.TemporalFormatError(f"{label} model mismatch: stream declares "
                                         f"{declared.hex()}, supplied is {supplied.hex()}")
    reconstructions, previous = [], None
    with mc.deterministic_kernels():
        for frame_type, motion_payload, payload in reader:
            if frame_type == mc.FRAME_TYPE_I:
                latent, _ = m21.decode_payload_to_latent(
                    payload, entropy_model=rig["intra_entropy_model"],
                    params=reader.intra_params, shape=header.latent_shape)
                reconstruction = intra.decode(latent.to(device))
            else:
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=rig["motion_entropy_model"])
                warped = mc.warp_blocks(previous, motion, block_size=header.block_size)
                reconstruction = decode_p_frame(pcoder, payload, warped, tables)
            previous = reconstruction
            reconstructions.append(reconstruction.cpu())
    return torch.cat(reconstructions)


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def load_pcoder(models_dir: Path, lam: str, device) -> p1.ConditionalPFrameCoder:
    saved = torch.load(models_dir / f"lambda{lam}" / "model.pt", map_location="cpu",
                       weights_only=False)
    model = p1.ConditionalPFrameCoder(**saved["config"])
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval()


def build_rig(args, m21, md, intra, lam: str, bits: int, device, train_full, motion_train):
    checkpoint = Path(p1.CHECKPOINT.format(lam))
    tag = f"phase_b_lambda{lam}"
    return m21.prepare_rate_point(
        intra, bits=bits, manifest=args.manifest, checkpoint=checkpoint,
        m11_dir=REFIT_DIR / f"m11_{tag}", m10k_dir=REFIT_DIR / f"m10k_{tag}", device=device,
        cache_dir=args.cache_dir or md.DEFAULT_CACHE_DIR, train_full=train_full,
        motion_train=motion_train, calibration_frames=args.calibration_frames,
        # The rig's I-frame and motion models were calibrated at the deployed GOP; their
        # provenance check includes it. Coding at another --gop reuses them unchanged.
        gop_size=bp.DEPLOYED_GOP, block_size=args.block_size, search_range=args.search_range,
        motion_cache=REFIT_DIR / f"m14_{tag}" / "motion_table.json", motion_identity=None,
        motion_cache_key=md._file_digest(checkpoint))


def stored_points() -> dict[str, list[dict[str, Any]]]:
    """The Phase B parity runs' classical curves and deployed NVC points, by arm."""
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for lam in p1.LAMBDAS:
        report = json.loads((PARITY_DIR / f"parity_phase_b_lambda{lam}"
                             / f"parity_phase_b_lambda{lam}.json").read_text(encoding="utf-8"))
        for label, point in report["points"].items():
            if point["arm"].startswith("nvc_") or lam == p1.LAMBDAS[0]:
                by_arm.setdefault(point["arm"], []).append(point)
    return by_arm


def comparisons(learned: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = stored_points()
    out: dict[str, Any] = {}
    for test_name, test_points in (("learned_p", learned),
                                   ("nvc_deployed", by_arm["nvc_deployed"])):
        for base_name in ("h264", "h265", "h264_lowdelay", "h265_lowdelay"):
            if base_name not in by_arm:
                continue
            entry = {}
            for metric in ("psnr", "msssim"):
                for convention in bp.CONVENTIONS:
                    base = bp.curve(by_arm[base_name], convention, metric)
                    test = bp.curve(test_points, convention, metric)
                    entry[f"{convention}/{metric}"] = bp.bd_rate(base, test)
            out[f"{test_name}_vs_{base_name}"] = entry
    deployed = bp.curve(by_arm["nvc_deployed"], bp.PRIMARY_CONVENTION, "psnr")
    out["learned_p_vs_nvc_deployed"] = {
        metric: bp.bd_rate(bp.curve(by_arm["nvc_deployed"], bp.PRIMARY_CONVENTION, metric),
                           bp.curve(learned, bp.PRIMARY_CONVENTION, metric))
        for metric in ("psnr", "msssim")}
    out["deployed_curve"] = deployed
    return out


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = bp.build_arg_parser(defaults)
    parser.set_defaults(output_dir=p1.OUTPUT_DIR, output_name="gate3.json")
    parser.add_argument("--models-dir", type=Path,
                        default=p1.OUTPUT_DIR / "gate2_local" / "stage2")
    parser.add_argument("--lambdas", nargs="+", default=list(p1.LAMBDAS))
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)

    mc = bp._load_script("m10h_motion_compensation")
    m21 = bp._load_script("m21_refinement")
    md = bp._load_script("m11_data")
    from nvc.training.checkpoint import load_model_from_checkpoint

    sequences = discover_sequences(args.manifest, split="test",
                                   max_sequences=args.max_sequences,
                                   max_frames_per_sequence=args.max_frames)
    frame_cache = {s.sequence_id: s.load_frames() for s in sequences}
    smoke = args.max_sequences is not None or args.max_frames is not None
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    tables = cumulative_tables()
    stream_dir = Path(tempfile.mkdtemp(prefix="p1_gate3_"))
    report_path = args.output_dir / args.output_name
    report: dict[str, Any] = {
        "models_dir": str(args.models_dir), "gop": args.gop, "smoke": smoke,
        "scale_levels": SCALE_LEVELS, "symbol_radius": SYMBOL_RADIUS,
        "frequency_total": FREQUENCY_TOTAL,
        "dataset": {"sequences": [s.sequence_id for s in sequences],
                    "frames": sum(s.frame_count for s in sequences)},
        "points": {}}
    print(f"gate 3: {len(sequences)} sequences, {report['dataset']['frames']} frames, "
          f"models from {args.models_dir}", flush=True)

    for lam in args.lambdas:
        bits = BITS[lam]
        intra, _ = load_model_from_checkpoint(Path(p1.CHECKPOINT.format(lam)), device=device)
        intra.eval()
        pcoder = load_pcoder(args.models_dir, lam, device)
        identity = model_identity(pcoder)
        rig = build_rig(args, m21, md, intra, lam, bits, device, train_full, motion_train)
        rows, exact = [], True
        tally = {"i_bytes": 0, "p_motion_bytes": 0, "p_residual_bytes": 0, "framing_bytes": 0,
                 "i_frames": 0, "p_frames": 0, "p_ideal_bits": 0.0,
                 "i_mse": [], "p_mse": []}
        for sequence in sequences:
            frames = frame_cache[sequence.sequence_id]
            path = stream_dir / f"{lam}_{sequence.sequence_id}.nvct"
            started = time.perf_counter()
            encoded = encode_sequence(mc, m21, intra, pcoder, frames, path, rig, tables,
                                      gop=args.gop, block_size=args.block_size,
                                      search_range=args.search_range, identity=identity)
            encode_seconds = time.perf_counter() - started
            size = path.stat().st_size
            started = time.perf_counter()
            decoded = decode_sequence(mc, m21, intra, pcoder, path, rig, tables,
                                      identity=identity)
            decode_seconds = time.perf_counter() - started
            path.unlink()
            same = torch.equal(decoded, encoded["reconstructions"])
            exact &= same
            scores = bp.score_frames(bp.to_uint8_frames(decoded), frames, device=device)
            rows.append({"sequence": sequence.sequence_id, "bytes": size,
                         "pixels": sequence.total_pixels, "encode_seconds": encode_seconds,
                         "decode_seconds": decode_seconds, **scores})
            body = 0
            for record, mse in zip(encoded["records"], scores["mse"]):
                key = "i" if record["type"] == "I" else "p"
                tally[f"{key}_frames"] += 1
                tally[f"{key}_mse"].append(mse)
                if key == "i":
                    tally["i_bytes"] += record["residual"]
                else:
                    tally["p_motion_bytes"] += record["motion"]
                    tally["p_residual_bytes"] += record["residual"]
                    tally["p_ideal_bits"] += record["ideal_bits"]
                body += record["motion"] + record["residual"]
            tally["framing_bytes"] += size - body
            print(f"  [{lam}] {sequence.sequence_id:14s} {size:>9,} B  "
                  f"PSNR {np.mean(scores['psnr']):.3f}  decode-exact={same}", flush=True)

        def psnr_of(values):
            return float(10 * np.log10(1 / np.mean(values))) if values else None

        split = {
            "i_frames": tally["i_frames"], "p_frames": tally["p_frames"],
            "bytes_per_i_frame": tally["i_bytes"] / max(tally["i_frames"], 1),
            "bytes_per_p_frame": (tally["p_motion_bytes"] + tally["p_residual_bytes"])
            / max(tally["p_frames"], 1),
            "p_residual_bytes_per_frame": tally["p_residual_bytes"] / max(tally["p_frames"], 1),
            "p_motion_bytes_per_frame": tally["p_motion_bytes"] / max(tally["p_frames"], 1),
            "p_residual_overhead_vs_ideal": tally["p_residual_bytes"] * 8
            / max(tally["p_ideal_bits"], 1.0) - 1,
            "i_bytes": tally["i_bytes"], "p_motion_bytes": tally["p_motion_bytes"],
            "p_residual_bytes": tally["p_residual_bytes"],
            "framing_bytes": tally["framing_bytes"],
            "pooled_psnr_i": psnr_of(tally["i_mse"]), "pooled_psnr_p": psnr_of(tally["p_mse"]),
        }
        split["p_over_i"] = split["bytes_per_p_frame"] / max(split["bytes_per_i_frame"], 1)
        point = bp.summarize_point(f"learned_p/{lam}", "learned_p", lam, rows, extra={
            "bits": bits, "lambda": lam, "decode_reconstruction_exact": exact,
            "pcoder_identity": identity.hex(), "split": split})
        report["points"][lam] = point
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        q = point["quality"][bp.PRIMARY_CONVENTION]
        print(f"  learned_p/{lam}: {point['total_bytes']:,} B  bpp {point['bpp']:.4f}  "
              f"PSNR {q['psnr']:.3f}  MS-SSIM {q['msssim']:.4f}  P/I {split['p_over_i']:.3f}  "
              f"coder overhead {100 * split['p_residual_overhead_vs_ideal']:+.2f}%  "
              f"decode-exact={exact}", flush=True)

    report["valid"] = all(p["decode_reconstruction_exact"] for p in report["points"].values())
    if len(report["points"]) >= 2 and not smoke:
        report["comparisons"] = comparisons(list(report["points"].values()))
        learned_bd = report["comparisons"]["learned_p_vs_h264"][f"{bp.PRIMARY_CONVENTION}/psnr"]
        deployed_bd = report["comparisons"]["nvc_deployed_vs_h264"][
            f"{bp.PRIMARY_CONVENTION}/psnr"]
        report["gate"] = {"learned_vs_x264": learned_bd, "deployed_vs_x264": deployed_bd,
                          "threshold": deployed_bd - GATE_MARGIN,
                          "passed": (learned_bd is not None and report["valid"]
                                     and learned_bd < deployed_bd - GATE_MARGIN)}
        print("\nBD-rate vs x264 (seq-mean PSNR): learned-P "
              f"{learned_bd:+.1f}%, deployed {deployed_bd:+.1f}%", flush=True)
        for key, entry in report["comparisons"].items():
            if key.endswith(("h264", "h265", "lowdelay")):
                print(f"  {key:32s} PSNR {entry['sequence_mean/psnr']:+.1f}%  "
                      f"MS-SSIM {entry['sequence_mean/msssim']:+.1f}%", flush=True)
        print(f"GATE 3 ({'PASSED' if report['gate']['passed'] else 'NOT PASSED'}): "
              f"BD-rate vs x264 < {deployed_bd - GATE_MARGIN:+.1f}%", flush=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"run valid (decode-exact everywhere): {report['valid']}\nReport: {report_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
