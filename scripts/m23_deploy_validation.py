"""M23 Phase B2 step 2 - the v3 codec measured on real containers, then TEST.

Three things in order, and the order matters:

  1 EQUIVALENCE. With the side channel off, `m23_codec.encode_sequence` must
    produce a stream BYTE-IDENTICAL to `m21_refinement.encode_sequence_refined`
    at identity - the deployed arm. A reimplemented I/O path could change the
    bytes for reasons unrelated to the scale map, and then everything below
    would be measuring the wrong thing. If this fails the run stops.
  2 VAL-B. Both arms coded to real containers, round-tripped, total stream bytes
    and quality compared. This is where the configuration is confirmed before
    the one read of TEST.
  3 TEST, once. Same measurement on the held-out split, plus BD-rate against the
    classical curves already measured on this exact split and convention in
    `outputs/benchmarks/parity_refit/parity_64ch_qat.json`, so the headline is
    comparable to the +214.17% on record rather than a new scale.

THE STRIDE WAS SELECTED ON VAL-A, AND VAL-B DISAGREES
-----------------------------------------------------
B1b swept the map's resolution. On VAL-A - the tuning split, which is where a
hyperparameter is chosen - stride 1 wins at all three rate points (+2.1758 /
+2.2347 / +1.9922%). On VAL-B it wins only at 5-bit; stride 2 is better at 4- and
3-bit (+1.7691 / +1.7889% against +1.7043 / +1.3463%).

Stride 1 is used here, because selecting it on VAL-B would be choosing a
hyperparameter on the split that reports it - the exact thing the
TRAIN/VAL-A/VAL-B/TEST structure exists to prevent, and the way an optimistic
number gets manufactured. The disagreement is a variance warning worth stating
plainly: the three strides are close enough at 4- and 3-bit that TEST may land
anywhere in that range, and if it comes in below VAL-B's figure this is the
reason rather than a surprise.

Run:
  ./.venv/Scripts/python.exe scripts/m23_deploy_validation.py --split val
  ./.venv/Scripts/python.exe scripts/m23_deploy_validation.py --split test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

# Pre-registered: VAL-A's choice, not VAL-B's. See the module docstring.
SELECTED_STRIDE = 1
SELECTED_BINS = 8


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    m23 = _load_script("m23_hyperprior")
    parser = argparse.ArgumentParser(
        description="M23 B2 step 2: the v3 codec on real containers, then TEST.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=m23.DEFAULT_CHECKPOINT)
    parser.add_argument("--m11-dir", type=Path, default=m23.DEFAULT_M11_DIR)
    parser.add_argument("--m10k-dir", type=Path, default=m23.DEFAULT_M10K_DIR)
    parser.add_argument("--m22-dir", type=Path,
                        default=Path("outputs/stage1_refit/m22_64ch_qat"))
    parser.add_argument("--motion-cache", type=Path,
                        default=Path("outputs/stage1_refit/m14_64ch_qat/motion_table.json"))
    parser.add_argument("--parity-report", type=Path, default=Path(
        "outputs/benchmarks/parity_refit/parity_64ch_qat.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m23_hyperprior"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(m23.RATE_POINTS))
    parser.add_argument("--context-bins", type=int, default=SELECTED_BINS)
    parser.add_argument("--context-stride", type=int, default=SELECTED_STRIDE)
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--train-frames-per-sequence", type=int, default=8)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=None,
                        help="smoke runs only")
    parser.add_argument("--max-frames", type=int, default=None, help="smoke runs only")
    parser.add_argument("--skip-equivalence", action="store_true",
                        help="only for a smoke run; the real run must prove it")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


# --------------------------------------------------------------------------------------------
# Fitting the side channel on TRAIN
# --------------------------------------------------------------------------------------------


def fit_side_specification(cv, pb, cx, md, m12, m10k, ma, ml, model, spec, args, bits, data,
                           device, log) -> Any:
    """The map's bin edges, its coding table, and the joint (prototype, context)
    tables - all from TRAIN, with the smoothing strength chosen on VAL-A.

    The parent is the DEPLOYED `coding_codebook`, so the baseline arm below is
    the shipped codec rather than a re-recalibration of it, and the candidate
    differs from it in exactly one way.
    """
    alphabet = 2 ** bits
    train_symbols = data["train_symbols"].astype(np.int64)
    val_symbols = data["val_symbols"].astype(np.int64)
    select_mask, _ = md.split_validation(data)
    channels, height, width = train_symbols.shape[1:]
    residual_params = data["calibration"]["residual_params"]
    zero = spec["zero"]

    train_k = m12.m11_g16_prototype_indices(spec["model"], spec["assign_codebook"],
                                            train_symbols, data["train_references"], zero,
                                            device=device)
    val_k = m12.m11_g16_prototype_indices(spec["model"], spec["assign_codebook"],
                                          val_symbols, data["val_references"], zero,
                                          device=device)

    m23h = _load_script("m23_hyperprior")
    train_residual = m23h.dequantized_residual(train_symbols, residual_params,
                                               shape=(channels, height, width))
    val_residual = m23h.dequantized_residual(val_symbols, residual_params,
                                             shape=(channels, height, width))
    train_coarse = pb.coarsen(pb.scale_map_signal(train_residual), stride=args.context_stride)
    val_coarse = pb.coarsen(pb.scale_map_signal(val_residual), stride=args.context_stride)
    edges = pb.train_bin_edges(train_coarse, bins=args.context_bins)
    train_binned = pb.to_bins(train_coarse, edges)
    val_binned = pb.to_bins(val_coarse, edges)
    map_table = cv.map_coder_table(train_binned, bins=args.context_bins)

    flat = lambda a: a.reshape(-1)
    parent = np.asarray(spec["coding_codebook"].probabilities, dtype=np.float64)
    groups = spec["assign_codebook"].size
    t_ctx = flat(pb.broadcast_to_symbol_positions(train_binned, channels=channels,
                                                  height=height, width=width))
    v_ctx = pb.broadcast_to_symbol_positions(val_binned, channels=channels, height=height,
                                             width=width)
    child_counts = cx.fit_child(flat(train_symbols), flat(train_k), t_ctx,
                                args.context_bins, groups, alphabet)
    a_sym, a_k = flat(val_symbols[select_mask]), flat(val_k[select_mask])
    a_ctx = flat(v_ctx[select_mask])
    side_a = pb.context_map_bits(train_binned, val_binned[select_mask],
                                 bins=args.context_bins) / a_sym.size
    strength, _ = cx.select_strength(lambda s: cx.child_bits(
        child_counts, parent, a_sym, a_k, a_ctx, args.context_bins, s) / a_sym.size + side_a)
    child = cv.smoothed_child_probabilities(child_counts, parent, card=args.context_bins,
                                            strength=strength)
    child_table = cv.coder_tables(m10k, child, bits=bits)
    log(f"      side channel: {args.context_bins} bins at stride {args.context_stride}, "
        f"child {child_table.frequencies.shape} (strength {strength:g})")

    codec = _load_script("m23_codec")
    identity = ma.model_identity(spec["model"], m10k_identity=bytes(8),
                                 calibration_signature="m23_side", bits=bits,
                                 codebook=spec["coding_codebook"])[:8]
    return codec.SideSpecification(
        bins=args.context_bins, stride=args.context_stride, edges=edges,
        map_table=map_table, child_table=child_table, groups=groups,
        identity=identity), {"strength": float(strength), "map_table_id":
                             map_table.model_id().hex(),
                             "child_tables": int(child_table.frequencies.shape[0])}


# --------------------------------------------------------------------------------------------
# The equivalence proof
# --------------------------------------------------------------------------------------------


def prove_equivalence(codec, mc, m13, ma, mk, pb, m21, model, sequence, spec, args, bits,
                      rig, device, log) -> dict[str, Any]:
    """With the side channel off, M23's encoder must byte-match m21's deployed arm.

    Without this the measurement below cannot distinguish the scale map from an
    accident of reimplemented I/O.
    """
    frames = sequence.load_frames()
    with tempfile.TemporaryDirectory(prefix="m23_equiv_") as directory:
        mine = Path(directory) / "m23.nvct"
        theirs = Path(directory) / "m21.nvct"
        codec.encode_sequence(
            mc, m13, ma, mk, pb, model, frames, spec, mine,
            intra_params=rig["intra_params"], intra_entropy_model=rig["intra_entropy_model"],
            residual_params=rig["residual_params"],
            motion_entropy_model=rig["motion_entropy_model"], bits=bits,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range,
            side=None)
        m21.encode_sequence_refined(
            mc, m13, model, frames, spec, theirs, m21.IDENTITY,
            intra_params=rig["intra_params"], intra_entropy_model=rig["intra_entropy_model"],
            residual_params=rig["residual_params"],
            motion_entropy_model=rig["motion_entropy_model"], bits=bits,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
        mine_bytes, theirs_bytes = mine.read_bytes(), theirs.read_bytes()
    identical = mine_bytes == theirs_bytes
    log(f"    equivalence on {sequence.sequence_id}: m23 {len(mine_bytes):,} B vs "
        f"m21 {len(theirs_bytes):,} B -> {'BYTE-IDENTICAL' if identical else 'DIFFERENT'}")
    return {"sequence": sequence.sequence_id, "m23_bytes": len(mine_bytes),
            "m21_bytes": len(theirs_bytes), "byte_identical": identical}


# --------------------------------------------------------------------------------------------
# One arm on one split
# --------------------------------------------------------------------------------------------


def measure_arm(codec, mc, m13, ma, mk, pb, bp, model, sequences, spec, args, bits, rig,
                side, device, stream_dir, log) -> dict[str, Any]:
    """Encode, decode and score every sequence of a split through real containers."""
    rows, exact = [], True
    for sequence in sequences:
        frames = sequence.load_frames()
        path = stream_dir / f"{'v3' if side else 'v2'}_{bits}bit_{sequence.sequence_id}.nvct"
        started = time.perf_counter()
        result = codec.encode_sequence(
            mc, m13, ma, mk, pb, model, frames, spec, path,
            intra_params=rig["intra_params"], intra_entropy_model=rig["intra_entropy_model"],
            residual_params=rig["residual_params"],
            motion_entropy_model=rig["motion_entropy_model"], bits=bits, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range, side=side)
        encode_seconds = time.perf_counter() - started
        size = path.stat().st_size
        started = time.perf_counter()
        decoded = codec.decode_sequence(
            mc, m13, ma, mk, pb, model, path, spec,
            intra_entropy_model=rig["intra_entropy_model"],
            motion_entropy_model=rig["motion_entropy_model"], bits=bits, side=side)
        decode_seconds = time.perf_counter() - started
        frame_exact = torch.equal(decoded["reconstructions"],
                                  result["reconstructions"])
        exact &= frame_exact
        path.unlink()
        scores = bp.score_frames(bp.to_uint8_frames(decoded["reconstructions"]), frames,
                                 device=device)
        side_bytes = sum(r["side_bytes"] for r in result["records"])
        rows.append({"sequence": sequence.sequence_id, "bytes": size,
                     "pixels": sequence.total_pixels, "side_bytes": side_bytes,
                     "motion_bytes": sum(r["motion_bytes"] for r in result["records"]),
                     "residual_bytes": sum(r["residual_bytes"] for r in result["records"]),
                     "encode_seconds": encode_seconds, "decode_seconds": decode_seconds,
                     "round_trip_exact": frame_exact,
                     "format_version": result["format_version"], **scores})
        log(f"      {sequence.sequence_id:<16} {size:>9,} B  side {side_bytes:>6,}  "
            f"exact={frame_exact}")
    point = bp.summarize_point(f"{'v3' if side else 'v2'}/{bits}bit",
                               "nvc_m23_side" if side else "nvc_m23_baseline",
                               f"{bits}bit", rows,
                               extra={"bits": bits, "round_trip_exact": exact,
                                      "total_side_bytes": sum(r["side_bytes"]
                                                              for r in rows)})
    return point


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    codec = _load_script("m23_codec")
    mc = _load_script("m10h_motion_compensation")
    m13 = _load_script("m13_recalibration")
    ma = _load_script("m11_ar_entropy")
    mk = _load_script("m10k_learned_entropy")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    cx = _load_script("m11_causal_context")
    m12 = _load_script("m12_spatial_offline_gate")
    m21 = _load_script("m21_refinement")
    cv = _load_script("m23_coded_validation")
    pb = _load_script("m23_phaseb_gate")
    bp = _load_script("benchmark_parity")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    if args.split == "test":
        sequences = discover_sequences(args.manifest, split="test",
                                       max_sequences=args.max_sequences,
                                       max_frames_per_sequence=args.max_frames)
    else:
        sequences = m21.val_b_sequences(args.manifest, count=4,
                                        max_frames=args.max_frames)
        if args.max_sequences:
            sequences = sequences[:args.max_sequences]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"m23_deploy_{args.split}.json"
    report: dict[str, Any] = {
        "phase": f"M23 Phase B2 step 2 - the v3 codec on real containers ({args.split})",
        "split": args.split,
        "sequences": [s.sequence_id for s in sequences],
        "frames": sum(s.frame_count for s in sequences),
        "pre_registered": {
            "context_bins": args.context_bins, "context_stride": args.context_stride,
            "stride_selected_on": "VAL-A",
            "stride_val_a_choice": "stride 1 at all three rate points",
            "stride_val_b_disagreement": ("VAL-B prefers stride 2 at 4- and 3-bit; "
                                          "selecting on VAL-B would be choosing a "
                                          "hyperparameter on the reporting split"),
            "equivalence_required": not args.skip_equivalence,
            "round_trip_required": True,
        },
        "configuration": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "device": str(device), "platform": platform.platform()},
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "rate_points": [],
    }

    def persist() -> None:
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    train_full = discover_sequences(args.manifest, split="train")
    motion_train = m21.broad_train_sequences(
        args.manifest, frames_per_sequence=args.train_frames_per_sequence)
    stream_dir = Path(tempfile.mkdtemp(prefix="m23_streams_"))

    print("=" * 112)
    print(f"M23 B2 STEP 2 - THE v3 CODEC ON REAL CONTAINERS, SPLIT = {args.split.upper()}")
    print(f"  {len(sequences)} sequences, {report['frames']} frames, GOP {args.gop}")
    print(f"  side channel: {args.context_bins} bins at stride {args.context_stride} "
          f"(selected on VAL-A)")
    print("=" * 112, flush=True)

    for bits in args.rate_points:
        print(f"\n  ---- {bits}-bit ----", flush=True)
        rig = m21.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device,
            cache_dir=cache_dir, train_full=train_full, motion_train=motion_train,
            calibration_frames=args.calibration_frames, gop_size=args.gop,
            block_size=args.block_size, search_range=args.search_range,
            motion_cache=args.motion_cache,
            motion_identity=None, motion_cache_key=md._file_digest(args.checkpoint))
        spec = rig["spec"]
        data = md.load_or_collect(
            model, checkpoint=args.checkpoint, manifest=args.manifest, bits=bits,
            device=device, calibration_frames=args.calibration_frames, cache_dir=cache_dir,
            log=lambda m: print("      " + m, flush=True))
        side, side_meta = fit_side_specification(
            cv, pb, cx, md, m12, mk, ma, ml, model, spec, args, bits, data, device,
            log=lambda m: print(m, flush=True))

        entry: dict[str, Any] = {"bits": bits, "side": side_meta}
        if not args.skip_equivalence:
            proof = prove_equivalence(codec, mc, m13, ma, mk, pb, m21, model,
                                      sequences[0], spec, args, bits, rig, device,
                                      log=lambda m: print(m, flush=True))
            entry["equivalence"] = proof
            if not proof["byte_identical"]:
                print("[ERROR] M23's v2 path does not byte-match m21's deployed arm; "
                      "refusing to measure.", file=sys.stderr)
                report["rate_points"].append(entry)
                persist()
                return 1

        print("    baseline (v2, deployed stack):", flush=True)
        entry["baseline"] = measure_arm(codec, mc, m13, ma, mk, pb, bp, model, sequences,
                                        spec, args, bits, rig, None, device, stream_dir,
                                        log=lambda m: print(m, flush=True))
        print("    candidate (v3, + transmitted scale map):", flush=True)
        entry["candidate"] = measure_arm(codec, mc, m13, ma, mk, pb, bp, model, sequences,
                                         spec, args, bits, rig, side, device, stream_dir,
                                         log=lambda m: print(m, flush=True))

        base, cand = entry["baseline"], entry["candidate"]
        entry["bytes_saved_percent"] = 100.0 * (base["total_bytes"] - cand["total_bytes"]) \
            / base["total_bytes"]
        entry["side_share_percent"] = 100.0 * cand["total_side_bytes"] / cand["total_bytes"]
        q_base = base["quality"][bp.PRIMARY_CONVENTION]
        q_cand = cand["quality"][bp.PRIMARY_CONVENTION]
        entry["delta_psnr_db"] = q_cand["psnr"] - q_base["psnr"]
        entry["delta_msssim"] = q_cand["msssim"] - q_base["msssim"]
        print(f"    -> {base['total_bytes']:,} B -> {cand['total_bytes']:,} B   "
              f"{entry['bytes_saved_percent']:+.4f}% bytes   "
              f"side {entry['side_share_percent']:.2f}%   "
              f"dPSNR {entry['delta_psnr_db']:+.4f} dB   "
              f"exact={base['round_trip_exact'] and cand['round_trip_exact']}", flush=True)
        report["rate_points"].append(entry)
        persist()

    # ---- BD-rate against the classical curves already measured on this split ----
    if args.split == "test" and args.parity_report.is_file() and len(args.rate_points) > 1:
        parity = json.loads(args.parity_report.read_text(encoding="utf-8"))
        by_arm: dict[str, list] = {}
        for point in parity["points"].values():
            by_arm.setdefault(point["arm"], []).append(point)
        comparisons: dict[str, Any] = {}
        for label in ("baseline", "candidate"):
            arm_points = [entry[label] for entry in report["rate_points"]]
            for classical in sorted(n for n in by_arm if not n.startswith("nvc_")):
                base_curve = bp.curve(by_arm[classical], bp.PRIMARY_CONVENTION, "psnr")
                test_curve = bp.curve(arm_points, bp.PRIMARY_CONVENTION, "psnr")
                comparisons[f"{label}_vs_{classical}"] = {
                    "bd_rate_percent": bp.bd_rate(base_curve, test_curve),
                    "quality_overlap": bp.overlap(base_curve, test_curve),
                }
        report["comparisons"] = comparisons
        report["comparison_note"] = (
            "classical curves taken from the parity_refit report, measured on this exact "
            "split with this exact convention, so these BD-rates are directly comparable "
            "to the +214.17% on record")
        persist()

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["all_round_trips_exact"] = all(
        entry[label]["round_trip_exact"] for entry in report["rate_points"]
        for label in ("baseline", "candidate"))
    persist()

    print("\n" + "=" * 112)
    print(f"SUMMARY - {args.split.upper()}, real containers, side payload included")
    print("=" * 112)
    print(f"{'bits':>5} {'baseline B':>13} {'candidate B':>13} {'side B':>9} "
          f"{'bytes':>10} {'dPSNR':>9} {'exact':>6}")
    for entry in sorted(report["rate_points"], key=lambda e: -e["bits"]):
        print(f"{entry['bits']:>5} {entry['baseline']['total_bytes']:>13,} "
              f"{entry['candidate']['total_bytes']:>13,} "
              f"{entry['candidate']['total_side_bytes']:>9,} "
              f"{entry['bytes_saved_percent']:>+9.4f}% {entry['delta_psnr_db']:>+8.4f} "
              f"{str(entry['baseline']['round_trip_exact'] and entry['candidate']['round_trip_exact']):>6}")
    if "comparisons" in report:
        print("\nBD-rate (PSNR, sequence_mean) against the classical arms on this split:")
        for key, value in report["comparisons"].items():
            bd = value["bd_rate_percent"]
            print(f"  {key:34s} {'n/a' if bd is None else f'{bd:+.2f}%'}")
    print(f"\nall round trips exact: {report['all_round_trips_exact']}")
    print(f"Report: {report_path}")
    return 0 if report["all_round_trips_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
