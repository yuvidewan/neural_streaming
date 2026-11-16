"""Plan A gate 3: the wide-latent (A2) codec with real bytes, against x264.

`docs/PLAN_wider_latent.md`, "Gate 2 details". The `p1_gate3.py` harness, unchanged in
everything but the coders:

    P-frames  the A2 P-coder from gate 2: the 64-channel 1/16 latent y, then the 32-channel
              1/8 detail latent x, coded given the decoded y
    I-frames  --i-frames old:  the deployed rig's intra path, exactly as in p1_gate3
              --i-frames wide: the wide I-coder from gate 2 (the A2 architecture fed a flat
                               gray reference), coded the same way as a P-frame

Both latents use p1_gate3's Laplace scale tables and range coder. A frame's payload is
[4-byte length of y's stream][y's stream][x's stream]: the decoder must decode y before it
can compute x's tables. The 4 bytes count. Every stream must decode bit for bit.

Gate (fixed before any gate 2 result): passes if either arm is at or below +61.1% vs x264
(more than 8 points better than +69.1%, GOP 30). Both arms are always reported.

Usage:
    p4_gate3.py --i-frames old|wide [--gop 30] [--models-dir outputs/p4_wide_latent/vimeo]
    p4_gate3.py --verdict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_parity as bp  # noqa: E402
import p1_gate3 as g3  # noqa: E402
import p1_pframe as p1  # noqa: E402
import p4_wide_latent as p4  # noqa: E402

from nvc.compression.range_coder import decode_symbols, encode_symbols  # noqa: E402
from nvc.evaluation.sequences import discover_sequences  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402
from nvc.utils.device import get_device  # noqa: E402
from nvc.utils.seed import seed_everything  # noqa: E402

BASELINE = 69.1            # stage 3 codec vs x264, GOP 30 (outputs/p1_pframe/gate3_stage3_gop30.json)
GATE_MARGIN = 8.0
OUTPUT_DIR = p4.OUTPUT_DIR


# --------------------------------------------------------------------------------------------
# Entropy coding of the two latents
# --------------------------------------------------------------------------------------------


def _code(symbols: torch.Tensor, scale: torch.Tensor, tables) -> tuple[bytes, float]:
    peak = int(symbols.abs().max())
    if peak > g3.SYMBOL_RADIUS:
        raise ValueError(f"symbol {peak} is outside the coded range +-{g3.SYMBOL_RADIUS}")
    index = g3.scale_index(scale)
    payload = encode_symbols((symbols.flatten().long() + g3.SYMBOL_RADIUS).cpu().numpy(),
                             tables, index.flatten().cpu().numpy())
    ideal = float(p1.laplace_bits(symbols, g3.scale_table().to(symbols.device)[index].float()).sum())
    return payload, ideal


def _decode(payload: bytes, scale: torch.Tensor, like: torch.Tensor, tables) -> torch.Tensor:
    index = g3.scale_index(scale)
    flat = decode_symbols(payload, index.numel(), tables, index.flatten().cpu().numpy())
    symbols = torch.from_numpy(np.asarray(flat, dtype=np.int64)) - g3.SYMBOL_RADIUS
    return symbols.reshape(like.shape).to(like.device, like.dtype)


def _y_side(model: p4.WideLatentPFrameCoder, warped: torch.Tensor):
    base = model.base
    context = base.context(warped)
    features = base.prior[:-1](context)
    mean, log_scale = base.prior[-1](features).chunk(2, dim=1)
    return context, features, mean, log_scale.exp().clamp_min(p1.SCALE_FLOOR)


def _x_side(model: p4.WideLatentPFrameCoder, features, decoded_y):
    coarse = F.interpolate(decoded_y, scale_factor=2, mode="nearest")
    x_mean, x_log_scale = model.extra_prior(
        torch.cat([model.extra_ctx(features), coarse], dim=1)).chunk(2, dim=1)
    return x_mean, x_log_scale.exp().clamp_min(p1.SCALE_FLOOR)


def _reconstruct(model, decoded_y, decoded_x, warped, context) -> torch.Tensor:
    syn = model.base.synthesis.net
    s = syn[2](syn[1](syn[0](decoded_y)))
    synthesized = syn[3:](s + model.extra_syn(decoded_x))
    out = synthesized + model.base.refine(torch.cat([synthesized, warped, context], 1))
    return out.clamp(0, 1)


def tune_latents(model, frame, warped, lam: float, steps: int, lr: float):
    """Encode-time latent tuning (`docs/PLAN_encode_tuning.md`).

    Adam on the latents y and x for `steps` steps, minimizing the frame's coded cost
    mse + lam * bits/pixel. Rounding is straight-through for both the decoder and the rate, so
    the forward value at every step IS the coded cost with rounding (ideal bits). Returns the
    best iterate seen; step 0 (the encoder's own latents) is a candidate, so the coded cost
    never goes up. The decoder is untouched: it decodes whatever symbols it is sent.
    """
    with torch.no_grad():
        context, features, mean, scale = _y_side(model, warped)
        h = model.base.encoder.net[:-1](torch.cat([frame, context], dim=1))
        y0, x0 = model.base.encoder.net[-1](h), model.extra_enc(h)
    if steps <= 0:
        return y0, x0, None
    pixels = frame.shape[-1] * frame.shape[-2]
    y = y0.clone().requires_grad_(True)
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([y, x], lr=lr)
    best, best_cost, first_cost = (y0, x0), float("inf"), None
    with torch.enable_grad():
        for step in range(steps + 1):
            centred_y = y - mean
            decoded_y = mean + centred_y + (torch.round(centred_y) - centred_y).detach()
            x_mean, x_scale = _x_side(model, features, decoded_y)
            centred_x = x - x_mean
            symbols_x = centred_x + (torch.round(centred_x) - centred_x).detach()
            bits = (p1.laplace_bits(decoded_y - mean, scale).sum()
                    + p1.laplace_bits(symbols_x, x_scale).sum())
            reconstruction = _reconstruct(model, decoded_y, x_mean + symbols_x, warped, context)
            cost = F.mse_loss(reconstruction, frame) + lam * bits / pixels
            value = float(cost.detach())
            if first_cost is None:
                first_cost = value
            if value < best_cost:
                best_cost, best = value, (y.detach().clone(), x.detach().clone())
            if step == steps:
                break
            optimizer.zero_grad(set_to_none=True)
            cost.backward()
            optimizer.step()
    return best[0], best[1], {"first_cost": first_cost, "best_cost": best_cost}


def encode_frame(model, frame, warped, tables, tune=None) -> tuple[bytes, torch.Tensor, float]:
    """Payload, reconstruction and ideal bits of one frame under an A2 model.

    `tune` = (lam, steps, lr) turns on encode-time latent tuning; None codes the encoder's
    own latents exactly as gate 3 did."""
    if model.kind != "a2":
        raise ValueError(f"p4_gate3 codes A2 models only, got {model.kind!r}")
    context, features, mean, scale = _y_side(model, warped)
    if tune is None:
        h = model.base.encoder.net[:-1](torch.cat([frame, context], dim=1))
        y, x = model.base.encoder.net[-1](h), model.extra_enc(h)
    else:
        y, x, _ = tune_latents(model, frame, warped, *tune)
    y_symbols = torch.round(y - mean)
    decoded_y = mean + y_symbols
    x_mean, x_scale = _x_side(model, features, decoded_y)
    x_symbols = torch.round(x - x_mean)
    y_payload, y_ideal = _code(y_symbols, scale, tables)
    x_payload, x_ideal = _code(x_symbols, x_scale, tables)
    payload = struct.pack("<I", len(y_payload)) + y_payload + x_payload
    reconstruction = _reconstruct(model, decoded_y, x_mean + x_symbols, warped, context)
    return payload, reconstruction, y_ideal + x_ideal


def decode_frame(model, payload: bytes, warped, tables) -> torch.Tensor:
    (y_length,) = struct.unpack("<I", payload[:4])
    context, features, mean, scale = _y_side(model, warped)
    decoded_y = mean + _decode(payload[4:4 + y_length], scale, mean, tables)
    x_mean, x_scale = _x_side(model, features, decoded_y)
    decoded_x = x_mean + _decode(payload[4 + y_length:], x_scale, x_mean, tables)
    return _reconstruct(model, decoded_y, decoded_x, warped, context)


def gray(frame: torch.Tensor) -> torch.Tensor:
    return torch.full_like(frame, p4.GRAY)


# --------------------------------------------------------------------------------------------
# One sequence
# --------------------------------------------------------------------------------------------


@torch.no_grad()
def encode_sequence(mc, m21, intra, icoder, pcoder, frames, path, rig, tables, *, gop,
                    block_size, search_range, identity, tune=None) -> dict[str, Any]:
    """`p1_gate3.encode_sequence` with A2 P-frames, and wide I-frames when `icoder` is set."""
    device = next(pcoder.parameters()).device
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
                if icoder is None:
                    payload, _ = m21.encode_latent_to_payload(
                        intra.encode(frame), params=intra_params,
                        entropy_model=rig["intra_entropy_model"])
                    latent, _ = m21.decode_payload_to_latent(
                        payload, entropy_model=rig["intra_entropy_model"], params=intra_params,
                        shape=latent_shape)
                    reconstruction, ideal = intra.decode(latent.to(device)), 0.0
                else:
                    payload, reconstruction, ideal = encode_frame(icoder, frame, gray(frame),
                                                                  tables, tune)
                writer.append_frame(types[t], b"", payload)
                records.append({"type": "I", "motion": 0, "residual": len(payload),
                                "ideal_bits": ideal})
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
                payload, reconstruction, ideal = encode_frame(pcoder, frame, warped, tables,
                                                              tune)
                writer.append_frame(types[t], motion_payload, payload)
                records.append({"type": "P", "motion": len(motion_payload),
                                "residual": len(payload), "ideal_bits": ideal})
            previous = reconstruction
            reconstructions.append(reconstruction.cpu())
    writer.close()
    return {"reconstructions": torch.cat(reconstructions), "records": records}


@torch.no_grad()
def decode_sequence(mc, m21, intra, icoder, pcoder, path, rig, tables, *, identity):
    device = next(pcoder.parameters()).device
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    for label, declared, supplied in (
            ("coder", header.residual_entropy_model_id, identity),
            ("intra", header.intra_entropy_model_id, rig["intra_entropy_model"].model_id()),
            ("motion", header.motion_entropy_model_id, rig["motion_entropy_model"].model_id())):
        if declared[:8] != supplied[:8]:
            raise mc.TemporalFormatError(f"{label} model mismatch: stream declares "
                                         f"{declared.hex()}, supplied is {supplied.hex()}")
    shape = (1, header.image_channels, header.image_height, header.image_width)
    reconstructions, previous = [], None
    with mc.deterministic_kernels():
        for frame_type, motion_payload, payload in reader:
            if frame_type == mc.FRAME_TYPE_I:
                if icoder is None:
                    latent, _ = m21.decode_payload_to_latent(
                        payload, entropy_model=rig["intra_entropy_model"],
                        params=reader.intra_params, shape=header.latent_shape)
                    reconstruction = intra.decode(latent.to(device))
                else:
                    flat = torch.full(shape, p4.GRAY, device=device)
                    reconstruction = decode_frame(icoder, payload, flat, tables)
            else:
                motion = mc.decode_motion_payload(
                    motion_payload, header.block_grid, search_range=header.search_range,
                    entropy_model=rig["motion_entropy_model"])
                warped = mc.warp_blocks(previous, motion, block_size=header.block_size)
                reconstruction = decode_frame(pcoder, payload, warped, tables)
            previous = reconstruction
            reconstructions.append(reconstruction.cpu())
    return torch.cat(reconstructions)


def identity_of(*models) -> bytes:
    digest = hashlib.sha256()
    for model in models:
        digest.update(g3.model_identity(model))
    return digest.digest()[:8]


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def report_name(i_frames: str, gop: int) -> str:
    return f"gate3_a2_{i_frames}_gop{gop}.json"


def command_verdict(gop: int) -> int:
    threshold = BASELINE - GATE_MARGIN
    arms = {}
    for i_frames in ("old", "wide"):
        path = OUTPUT_DIR / report_name(i_frames, gop)
        if path.is_file():
            report = json.loads(path.read_text(encoding="utf-8"))
            if "comparisons" in report:
                arms[i_frames] = {
                    "vs_x264": report["comparisons"]["learned_p_vs_h264"][
                        f"{bp.PRIMARY_CONVENTION}/psnr"],
                    "vs_x264_lowdelay": report["comparisons"].get(
                        "learned_p_vs_h264_lowdelay", {}).get(f"{bp.PRIMARY_CONVENTION}/psnr"),
                    "valid": report["valid"]}
    passed = any(a["valid"] and a["vs_x264"] is not None and a["vs_x264"] <= threshold
                 for a in arms.values())
    verdict = {"gop": gop, "baseline": BASELINE, "threshold": threshold, "arms": arms,
               "passed": passed}
    (OUTPUT_DIR / f"gate3_verdict_gop{gop}.json").write_text(json.dumps(verdict, indent=2),
                                                             encoding="utf-8")
    for name, a in arms.items():
        print(f"  A2 P + {name} I-frames: {a['vs_x264']:+.1f}% vs x264  valid={a['valid']}")
    print(f"GATE 3 ({'PASSED' if passed else 'NOT PASSED'}): at or below {threshold:+.1f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    parser = bp.build_arg_parser(defaults)
    parser.set_defaults(output_dir=OUTPUT_DIR, gop=30)
    parser.add_argument("--models-dir", type=Path, default=p4.VIMEO_DIR)
    parser.add_argument("--i-frames", choices=("old", "wide"), default="wide")
    parser.add_argument("--lambdas", nargs="+", default=list(p1.LAMBDAS))
    parser.add_argument("--verdict", action="store_true")
    parser.add_argument("--tune-steps", type=int, default=0,
                        help="encode-time latent tuning steps per frame (0 = off, as gate 3)")
    parser.add_argument("--tune-lr", type=float, default=0.03)
    parser.add_argument("--eval-manifest", type=Path, default=None,
                        help="code the test split of another manifest (e.g. UVG from "
                             "prepare_uvg.py); the coding tables still come from --manifest")
    parser.add_argument("--val-b", action="store_true",
                        help="code the 4 VAL-B videos instead of TEST (for choosing settings)")
    args = parser.parse_args(argv)
    if args.verdict:
        return command_verdict(args.gop)
    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    split_tag = "valb" if args.val_b else "test"
    external = args.eval_manifest is not None
    if external and args.output_name == "parity.json":
        parser.error("--eval-manifest needs an explicit --output-name")
    if args.output_name != "parity.json":
        output_name = args.output_name
    elif args.tune_steps > 0 or args.val_b:
        output_name = (f"tune_{split_tag}_s{args.tune_steps}_lr{args.tune_lr:g}_"
                       f"{args.i_frames}_gop{args.gop}.json")
    else:
        output_name = report_name(args.i_frames, args.gop)

    mc = bp._load_script("m10h_motion_compensation")
    m21 = bp._load_script("m21_refinement")
    md = bp._load_script("m11_data")
    from nvc.training.checkpoint import load_model_from_checkpoint

    if external:
        sequences = discover_sequences(args.eval_manifest, split="test",
                                       max_sequences=args.max_sequences,
                                       max_frames_per_sequence=args.max_frames)
    elif args.val_b:
        sequences = m21.val_b_sequences(args.manifest, count=4, max_frames=args.max_frames)
    else:
        sequences = discover_sequences(args.manifest, split="test",
                                       max_sequences=args.max_sequences,
                                       max_frames_per_sequence=args.max_frames)
    smoke = args.max_sequences is not None or args.max_frames is not None
    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    tables = g3.cumulative_tables()
    stream_dir = Path(tempfile.mkdtemp(prefix="p4_gate3_"))
    report_path = args.output_dir / output_name
    report: dict[str, Any] = {
        "models_dir": str(args.models_dir), "i_frames": args.i_frames, "gop": args.gop,
        "smoke": smoke, "split": split_tag,
        "eval_manifest": str(args.eval_manifest) if external else None, "tune_steps": args.tune_steps,
        "tune_lr": args.tune_lr if args.tune_steps > 0 else None, "dataset": {"sequences": [s.sequence_id for s in sequences],
                                    "frames": sum(s.frame_count for s in sequences)},
        "points": {}}
    print(f"p4 gate 3: {len(sequences)} sequences, GOP {args.gop}, {args.i_frames} I-frames, "
          f"models from {args.models_dir}", flush=True)

    for lam in args.lambdas:
        bits = g3.BITS[lam]
        intra, _ = load_model_from_checkpoint(Path(p1.CHECKPOINT.format(lam)), device=device)
        intra.eval()
        pcoder = p4.load_vimeo("p", lam, device, args.models_dir)
        icoder = p4.load_vimeo("i", lam, device, args.models_dir) if args.i_frames == "wide" else None
        identity = identity_of(pcoder, *([icoder] if icoder is not None else []))
        rig = g3.build_rig(args, m21, md, intra, lam, bits, device, train_full, motion_train)
        rows, exact = [], True
        tally = {"i_bytes": 0, "p_bytes": 0, "framing_bytes": 0, "i_frames": 0, "p_frames": 0,
                 "ideal_bits": 0.0, "coded_bytes": 0, "i_mse": [], "p_mse": []}
        for sequence in sequences:
            frames = sequence.load_frames()      # one video at a time: 1080p does not fit
            path = stream_dir / f"{lam}_{sequence.sequence_id}.nvct"
            started = time.perf_counter()
            encoded = encode_sequence(mc, m21, intra, icoder, pcoder, frames, path, rig, tables,
                                      gop=args.gop, block_size=args.block_size,
                                      search_range=args.search_range, identity=identity,
                                      tune=((float(lam), args.tune_steps, args.tune_lr)
                                            if args.tune_steps > 0 else None))
            encode_seconds = time.perf_counter() - started
            size = path.stat().st_size
            started = time.perf_counter()
            decoded = decode_sequence(mc, m21, intra, icoder, pcoder, path, rig, tables,
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
                tally[f"{key}_bytes"] += record["motion"] + record["residual"]
                if record["ideal_bits"]:
                    tally["ideal_bits"] += record["ideal_bits"]
                    tally["coded_bytes"] += record["residual"]
                body += record["motion"] + record["residual"]
            tally["framing_bytes"] += size - body
            print(f"  [{lam}] {sequence.sequence_id:14s} {size:>9,} B  "
                  f"PSNR {np.mean(scores['psnr']):.3f}  decode-exact={same}", flush=True)

        def psnr_of(values):
            return float(10 * np.log10(1 / np.mean(values))) if values else None

        split = {
            "i_frames": tally["i_frames"], "p_frames": tally["p_frames"],
            "bytes_per_i_frame": tally["i_bytes"] / max(tally["i_frames"], 1),
            "bytes_per_p_frame": tally["p_bytes"] / max(tally["p_frames"], 1),
            "i_bytes": tally["i_bytes"], "p_bytes": tally["p_bytes"],
            "framing_bytes": tally["framing_bytes"],
            "coder_overhead_vs_ideal": tally["coded_bytes"] * 8 / max(tally["ideal_bits"], 1.0) - 1,
            "pooled_psnr_i": psnr_of(tally["i_mse"]), "pooled_psnr_p": psnr_of(tally["p_mse"]),
        }
        point = bp.summarize_point(f"learned_p/{lam}", "learned_p", lam, rows, extra={
            "bits": bits, "lambda": lam, "decode_reconstruction_exact": exact,
            "coder_identity": identity.hex(), "split": split})
        report["points"][lam] = point
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        q = point["quality"][bp.PRIMARY_CONVENTION]
        print(f"  A2/{args.i_frames}/{lam}: {point['total_bytes']:,} B  bpp {point['bpp']:.4f}  "
              f"PSNR {q['psnr']:.3f}  MS-SSIM {q['msssim']:.4f}  I PSNR {split['pooled_psnr_i']:.2f}  "
              f"P PSNR {split['pooled_psnr_p']:.2f}  coder overhead "
              f"{100 * split['coder_overhead_vs_ideal']:+.2f}%  decode-exact={exact}", flush=True)

    report["valid"] = all(p["decode_reconstruction_exact"] for p in report["points"].values())
    if len(report["points"]) >= 2 and not smoke and not args.val_b and not external:
        report["comparisons"] = g3.comparisons(list(report["points"].values()))
        for key, entry in report["comparisons"].items():
            if key.startswith("learned_p_vs_h26"):
                print(f"  {key:32s} PSNR {entry['sequence_mean/psnr']:+.1f}%  "
                      f"MS-SSIM {entry['sequence_mean/msssim']:+.1f}%", flush=True)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"run valid (decode-exact everywhere): {report['valid']}\nReport: {report_path}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
