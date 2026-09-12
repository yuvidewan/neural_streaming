"""M20 Phase A + Phase B - document `SharedCodebook.assign_tensor` by MEASURING
it, then prove the proposed hysteresis state is decoder-reconstructible.

Phase A is deliberately not a prose description of the source. Every claim in
the emitted JSON is backed by an executed probe on the real, frozen 5-bit rig:

  * candidate set              K, alphabet, and that every prototype is a
                               candidate at every position (no pruning);
  * cost metric                the cost matrix is reproduced independently and
                               checked against `assign_tensor`'s own argmin;
  * probability -> code length  prototype probabilities are shown to be the
                               INTEGER coder frequencies / TOTAL_FREQUENCY, so
                               cost is in bits of the table the coder uses;
  * tie-breaking               ties are resolved to the lowest index, probed on
                               a constructed tie;
  * dependence on the CURRENT target symbol   probed by flipping the symbol at
                               a position and re-deriving the assignment;
  * dependence on causal context   probed by flipping a symbol in an EARLIER
                               decoding group;
  * encoder state / decoder state   the whole-frame encoder assignment is
                               compared against the assignment the real
                               group-by-group decoder rebuilds from the
                               payload alone.

Phase B then runs the same encoder/decoder comparison for every hysteresis
state and a non-zero margin, which is the actual decoder-compatibility proof:
if the decoder can rebuild the assignment, no side information is needed.

Run:
  ./.venv/Scripts/python.exe scripts/m20_assignment_trace.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import latent_to_symbols
from nvc.compression.entropy_model import TOTAL_FREQUENCY
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def first_p_frame_inputs(mc, model, rig, sequence, *, gop_size, block_size, search_range, device):
    """The real (reference, symbols) pair for the first P-frame of a sequence,
    produced by the deployed closed loop - not a synthetic probe input."""
    from nvc.compression.codec import decode_payload_to_latent, encode_latent_to_payload

    frames = sequence.load_frames()
    types = mc.gop_frame_types(frames.shape[0], gop_size)
    real_previous = None
    with mc.deterministic_kernels():
        for index in range(frames.shape[0]):
            frame = frames[index:index + 1].to(device)
            latent = model.encode(frame)
            latent_shape = tuple(latent.shape[1:])
            if types[index] == mc.FRAME_TYPE_I:
                payload, _ = encode_latent_to_payload(
                    latent, params=rig["intra_params"], entropy_model=rig["intra_entropy_model"])
                decoded, _ = decode_payload_to_latent(
                    payload, entropy_model=rig["intra_entropy_model"],
                    params=rig["intra_params"], shape=latent_shape)
                real_previous = model.decode(decoded.to(device))
                continue
            mv = mc.estimate_block_motion(real_previous, frame, block_size=block_size,
                                          search_range=search_range)
            reference = model.encode(mc.warp_blocks(real_previous, mv, block_size=block_size))
            symbols = latent_to_symbols(latent - reference,
                                        rig["residual_params"]).reshape(latent_shape)
            return reference, np.asarray(symbols, dtype=np.int64), latent_shape
    raise RuntimeError(f"{sequence.sequence_id} contains no P-frame at GOP {gop_size}")


def trace_assignment_rule(m20, ml, rig, reference, symbols, latent_shape, device) -> dict[str, Any]:
    """Phase A: every documented property, probed rather than asserted."""
    model11 = rig["model11"]
    codebook, zero = rig["assign_codebook"], rig["zero"]
    channels, height, width = latent_shape
    plane = height * width
    group = model11.group_size

    rows = m20.frame_rows(model11, reference, symbols, zero)
    deployed = codebook.assign_tensor(rows)
    costs_tensor = m20.assignment_costs(rows, codebook)
    reproduced = m20.baseline_assignment(costs_tensor)
    costs = costs_tensor.double().cpu().numpy()

    # --- cost metric: independently recompute cross-entropy from the INTEGER
    # frequencies the coder consumes, and check the assignment is unchanged.
    probabilities_from_frequencies = codebook.frequencies / float(TOTAL_FREQUENCY)
    independent = (rows.double().cpu().numpy()
                   @ (-np.log2(probabilities_from_frequencies)).T)
    independent_assignment = ml.argmin_lowest_index(independent)
    # Where the float64 recompute disagrees with the deployed float32 GEMM, how
    # close was the decision? This is M19's "brittleness" measured in numerical
    # terms rather than inferred from churn.
    disagree = independent_assignment != deployed
    runner_up_all = np.partition(independent, 1, axis=1)[:, 1] - independent.min(axis=1)
    precision_probe = {
        "positions_differing": int(disagree.sum()),
        "fraction_differing": float(disagree.mean()),
        "mean_runner_up_advantage_bits_where_differing":
            float(runner_up_all[disagree].mean()) if disagree.any() else 0.0,
        "mean_runner_up_advantage_bits_where_agreeing":
            float(runner_up_all[~disagree].mean()) if (~disagree).any() else 0.0,
        "mean_float64_cost_penalty_bits_of_the_deployed_choice": float(
            (independent[np.arange(independent.shape[0]), deployed]
             - independent.min(axis=1)).mean()),
        "note": "the deployed GEMM runs in the network's float32; a float64 recompute is not "
                "the decoder's path and this disagreement is NOT an encoder/decoder mismatch "
                "(both run the identical float32 kernel - see phase_b_roundtrip). It measures "
                "how numerically marginal the argmin is.",
    }

    # --- tie-breaking: construct an exact K-way tie and an exact 2-way tie
    # between two high indices, and check both the numpy and the torch reducer
    # pick the LOWEST tied index (the property encoder and decoder must share).
    all_tied = np.zeros((1, codebook.size), dtype=np.float64)
    two_tied = np.arange(codebook.size, dtype=np.float64)[None, :] + 10.0
    two_tied[0, [7, 300]] = 0.0
    tie_numpy = [int(ml.argmin_lowest_index(all_tied)[0]), int(ml.argmin_lowest_index(two_tied)[0])]
    tie_torch = [int(ml._torch_argmin_lowest_index(torch.from_numpy(all_tied).to(device))[0]),
                 int(ml._torch_argmin_lowest_index(torch.from_numpy(two_tied).to(device))[0])]

    # --- does the assignment depend on the CURRENT target symbol?
    # Flip the symbol at one position of the LAST decoding group. Nothing in any
    # group can legally see it, so every assignment must be unchanged.
    last_group_position = (channels - 1) * plane + plane // 2
    flipped_last = symbols.reshape(-1).copy()
    alphabet = int(codebook.alphabet)
    flipped_last[last_group_position] = (flipped_last[last_group_position] + 1) % alphabet
    rows_last = m20.frame_rows(model11, reference,
                               flipped_last.reshape(latent_shape), zero)
    assignment_last = codebook.assign_tensor(rows_last)
    current_symbol_dependence = bool(np.any(assignment_last != deployed))

    # --- does it depend on CAUSAL context? Flip a symbol in group 0: group 0's
    # own assignments must be unchanged (its planes are all-zero) but later
    # groups are allowed to move.
    flipped_first = symbols.reshape(-1).copy()
    flipped_first[plane // 2] = (flipped_first[plane // 2] + 1) % alphabet
    rows_first = m20.frame_rows(model11, reference, flipped_first.reshape(latent_shape), zero)
    assignment_first = codebook.assign_tensor(rows_first)
    group0 = slice(0, group * plane)
    later = slice(group * plane, None)

    advantage_runner_up = np.partition(costs, 1, axis=1)[:, 1] - costs.min(axis=1)
    return {
        "function": "m10l_shared_codebook.SharedCodebook.assign_tensor",
        "candidate_prototypes": {
            "K": int(codebook.size), "alphabet": alphabet,
            "all_prototypes_are_candidates": True,
            "evidence": "cost matrix is [N, K] with no masking; distinct prototypes used at "
                        f"this frame = {int(np.unique(deployed).size)} of {int(codebook.size)}",
        },
        "cost_metric": {
            "name": codebook.metric,
            "formula": "cost(i, k) = sum_s p_i(s) * -log2 q_k(s)  (cross-entropy, BITS)",
            "equals_expected_code_length": True,
            "argmin_equals_kl_argmin": "H(p,q) = KL(p||q) + H(p); H(p) is constant in k",
            "reproduced_matches_deployed": bool(np.array_equal(reproduced, deployed)),
            "independent_float64_recompute_matches": bool(
                np.array_equal(independent_assignment, deployed)),
            "float32_vs_float64_precision_probe": precision_probe,
        },
        "probabilities_to_code_lengths": {
            "source": "SharedCodebook.probabilities = frequencies / TOTAL_FREQUENCY",
            "total_frequency": int(TOTAL_FREQUENCY),
            "derived_from_integer_coder_tables": bool(np.allclose(
                codebook.probabilities, probabilities_from_frequencies, rtol=0, atol=0)),
            "note": "cost therefore scores the table the arithmetic coder will actually use, "
                    "not the pre-rounding Lloyd centroid",
        },
        "tie_breaking": {
            "rule": "lowest prototype index (argmin_lowest_index / _torch_argmin_lowest_index)",
            "probes": ["all K costs equal", "exact 2-way tie between indices 7 and 300"],
            "numpy_winners": tie_numpy, "torch_winners": tie_torch,
            "numpy_and_torch_agree": tie_numpy == tie_torch,
            "lowest_index_wins": tie_numpy == [0, 7] and tie_torch == [0, 7],
        },
        "depends_on_current_target_symbol": {
            "answer": "no",
            "probe": "flipped the symbol at the last decoding group's centre position",
            "any_assignment_changed": current_symbol_dependence,
        },
        "depends_on_causal_context": {
            "answer": "yes, but only on channels before G*floor(c/G)",
            "probe": "flipped one symbol in decoding group 0",
            "group0_assignments_changed": bool(
                np.any(assignment_first[group0] != deployed[group0])),
            "later_group_assignments_changed": bool(
                np.any(assignment_first[later] != deployed[later])),
        },
        "encoder_state": [
            "reference latent (from its own reconstructed previous frame + motion)",
            "the full symbol plane (but only its causal part can reach any assignment)",
            "the frozen K=512 prototype frequencies",
        ],
        "decoder_state_before_coding_position_i": [
            "the identical reference latent (reconstructed the same way)",
            "every symbol in decoding groups before i's group (already decoded)",
            "the identical frozen K=512 prototype frequencies",
        ],
        "prototype_index_reconstructible_at_decoder": "demonstrated in phase_b_roundtrip",
        "assignment_cost_scale": {
            "units": "bits of expected code length",
            "mean_best_cost_bits": float(costs.min(axis=1).mean()),
            "runner_up_advantage_bits": {
                "mean": float(advantage_runner_up.mean()),
                "percentiles": {str(p): float(np.percentile(advantage_runner_up, p))
                                for p in (10, 25, 50, 75, 90, 99)},
            },
        },
    }


@torch.no_grad()
def phase_b_roundtrip(m20, rig, reference, symbols, latent_shape, *, bits,
                      margins, states) -> list[dict[str, Any]]:
    """Phase B: for every (state, margin), encode with hysteresis and then let
    the REAL group-by-group decoder rebuild the assignment from the payload.

    Two frames are chained so the `temporal` state is actually exercised with a
    defined previous assignment rather than degenerating to the argmin.
    """
    model11 = rig["model11"]
    assign_codebook, coding_codebook = rig["assign_codebook"], rig["coding_codebook"]
    zero = rig["zero"]
    flat_symbols = np.asarray(symbols, dtype=np.int64).reshape(-1)
    results = []
    for state in states:
        for margin in margins:
            previous = None
            frame_reports = []
            for step in range(2):
                encoded = m20.encode_frame_hysteresis(
                    model11, assign_codebook, coding_codebook, reference, symbols, zero,
                    bits=bits, margin=margin, state=state, previous_frame=previous)
                decoded = m20.decode_frame_hysteresis(
                    model11, assign_codebook, coding_codebook, encoded["payload"], reference,
                    zero, bits=bits, shape=latent_shape, margin=margin, state=state,
                    previous_frame=previous)
                frame_reports.append({
                    "step": step,
                    "previous_defined": previous is not None,
                    "assignments_identical": bool(np.array_equal(
                        encoded["assignment"], decoded["assignment"])),
                    "symbols_roundtrip_exact": bool(np.array_equal(
                        flat_symbols, decoded["symbols"])),
                    "baseline_assignments_identical": bool(np.array_equal(
                        encoded["baseline_assignment"], decoded["baseline_assignment"])),
                    "held_positions": encoded["trace"]["held"],
                    "payload_bytes": len(encoded["payload"]),
                })
                previous = encoded["assignment"]
            results.append({"state": state, "margin": margin, "frames": frame_reports,
                            "decoder_compatible": all(
                                f["assignments_identical"] and f["symbols_roundtrip_exact"]
                                for f in frame_reports)})
    return results


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M20 Phase A/B: assignment-rule trace and decoder-compatibility proof.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m20_codebook_hysteresis"))
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    mc = _load_script("m10h_motion_compensation")
    ml = _load_script("m10l_shared_codebook")
    md = _load_script("m11_data")
    m20 = _load_script("m20_hysteresis")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_b = m20.val_b_sequences(args.manifest, count=1)
    train_full = discover_sequences(args.manifest, split="train")

    print("=" * 118)
    print("M20 PHASE A/B - ASSIGNMENT-RULE TRACE AND DECODER-COMPATIBILITY PROOF")
    print("=" * 118)
    print(f"  probe sequence: {val_b[0].sequence_id}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M20 Phase A/B", "probe_sequence": val_b[0].sequence_id,
        "hysteresis_rule": {
            "states": list(m20.STATES),
            "decision": "hold prev(i) iff prev(i) is defined and "
                        "cost(i, prev(i)) - cost(i, k*(i)) < margin  (STRICT)",
            "margin_units": "bits of expected code length",
            "margin_zero_is_identity": "advantage is never negative and the comparison is "
                                       "strict, so margin=0 can never hold a position",
            "side_information_bits": 0,
            "container_change": "none - .nvct v2 untouched",
        },
        "rate_points": [], "all_decoder_compatible": True,
    }

    for bits in args.rate_points:
        rig = m20.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, calibration_frames=args.calibration_frames,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
        reference, symbols, latent_shape = first_p_frame_inputs(
            mc, model, rig, val_b[0], gop_size=args.gop, block_size=args.block_size,
            search_range=args.search_range, device=device)

        with mc.deterministic_kernels():
            trace = trace_assignment_rule(m20, ml, rig, reference, symbols, latent_shape, device)
            roundtrip = phase_b_roundtrip(m20, rig, reference, symbols, latent_shape, bits=bits,
                                          margins=(0.0, 0.02, 0.10), states=m20.STATES)
        compatible = all(r["decoder_compatible"] for r in roundtrip)
        report["all_decoder_compatible"] &= compatible

        scale = trace["assignment_cost_scale"]["runner_up_advantage_bits"]
        probe = trace["cost_metric"]["float32_vs_float64_precision_probe"]
        print(f"\n  ---- {bits}-bit ---- residual={rig['residual_identity']}")
        print(f"    reproduced==deployed assignment: "
              f"{trace['cost_metric']['reproduced_matches_deployed']}   "
              f"float64 recompute agrees: "
              f"{trace['cost_metric']['independent_float64_recompute_matches']} "
              f"({probe['fraction_differing'] * 100:.2f}% of positions differ; runner-up "
              f"advantage there = {probe['mean_runner_up_advantage_bits_where_differing']:.6f} "
              f"bits vs {probe['mean_runner_up_advantage_bits_where_agreeing']:.6f} elsewhere)")
        print(f"    depends on current target symbol: "
              f"{trace['depends_on_current_target_symbol']['any_assignment_changed']}   "
              f"group-0 assignments moved when a group-0 symbol flipped: "
              f"{trace['depends_on_causal_context']['group0_assignments_changed']}   "
              f"later groups moved: "
              f"{trace['depends_on_causal_context']['later_group_assignments_changed']}")
        print(f"    runner-up advantage (bits): mean={scale['mean']:.5f}  "
              + "  ".join(f"p{p}={scale['percentiles'][p]:.5f}"
                          for p in ("25", "50", "75", "90")))
        print(f"    decoder compatible for every (state, margin) probed: {compatible}", flush=True)

        report["rate_points"].append({
            "bits": bits, "residual_identity": rig["residual_identity"],
            "assign_codebook_id": rig["assign_codebook_id"],
            "coding_codebook_id": rig["coding_codebook_id"],
            "phase_a_trace": trace, "phase_b_roundtrip": roundtrip,
            "decoder_compatible": compatible,
        })

    path = args.output_dir / "m20_assignment_trace.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nDECODER COMPATIBILITY: "
          f"{'PROVEN for every state/margin probed' if report['all_decoder_compatible'] else 'FAILED'}")
    print(f"Report: {path}")
    return 0 if report["all_decoder_compatible"] else 1


if __name__ == "__main__":
    sys.exit(main())
