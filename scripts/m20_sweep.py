"""M20 Phases C-G - the margin sweep, on VAL-B, at 5/4/3 bits.

WHAT IS SWEPT, AND WHY IT IS DECLARED UP FRONT
-----------------------------------------------
`m20_hysteresis.MARGIN_SWEEP` and `m20_hysteresis.STATES` are frozen in that
module, not chosen here and not chosen after looking at a result. Phase C also
requires the sweep to be on the right SCALE, so this script measures the
empirical distribution of the quantity the margin is compared against - the
"advantage" cost(i, prev(i)) - cost(i, k*(i)), in bits - under the deployed
(margin = 0) chain, and reports its percentiles alongside the declared grid.
If the declared grid did not span that distribution the sweep would have to be
renormalized BEFORE any byte result was read; the percentiles are emitted so
that judgement is auditable rather than asserted.

WHAT IS MEASURED (Phase D)
--------------------------
For every (bit depth, state, margin): real and oracle residual payload BYTES
from the real arithmetic coder, ideal bits, routing churn against the oracle
arm, the symbol-change x assignment-change 2x2 with its excess-bit shares, the
codebook assignment distribution, mean code length, how many positions the rule
HELD, and - the thing that decides the milestone - whether holding them made
the actual code length better or worse (Phase E).

The primary metric is actual residual payload bytes. Churn reduction is
reported but is explicitly NOT the success criterion.

ONE PASS, MANY MARGINS
----------------------
The expensive per-frame work (motion search, the autoencoder, G16, the [N, K]
cost matrix) does not depend on the margin, and neither does the residual
symbol plane - hysteresis only re-routes which table codes an unchanged symbol.
So every configuration is evaluated inside a single closed-loop pass, and the
REAL chain advances on the deployed (margin = 0) symbols exactly as
`encode_multi` does. The oracle arm stays a read-only side channel, as in
M16-M19.

Run:
  ./.venv/Scripts/python.exe scripts/m20_sweep.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvc.compression.codec import (
    decode_payload_to_latent, encode_latent_to_payload, latent_to_symbols, symbols_to_latent,
)
from nvc.evaluation.sequences import discover_sequences
from nvc.utils.config import load_default_config
from nvc.utils.device import get_device
from nvc.utils.seed import seed_everything

DEFAULT_M14_DAVIS = Path("outputs/m14_entropy_audit/m14_davis_benchmark.json")
ADVANTAGE_PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(states, margins, diagnostics=()) -> list[tuple[str, float]]:
    """(state, margin) pairs. margin = 0 is the identity control and is
    evaluated ONCE - it is the same assignment for every state by
    construction, so running it three times would only invite a spurious
    "the states differ at margin 0" reading.

    `diagnostics` are appended with margin 0 and are NOT hysteresis candidates;
    they are the reference points described in `m20_hysteresis.DIAGNOSTIC_RULES`
    and are reported in their own table.
    """
    out = [("baseline", 0.0)]
    out.extend((state, float(margin)) for state in states for margin in margins if margin > 0)
    out.extend((name, 0.0) for name in diagnostics)
    return out


class Accumulator:
    """Running totals for one (bit depth, state, margin), over all VAL-B
    P-frames. Everything is a sum or a count so the result is order-independent
    and identical across processes."""

    def __init__(self, codebook_size: int) -> None:
        self.real_bytes = 0
        self.oracle_bytes = 0
        self.real_ideal_bits = 0.0
        self.real_code_len_bits = 0.0
        self.positions = 0
        self.held = 0
        self.held_better = 0
        self.held_worse = 0
        self.held_equal = 0
        self.held_code_len_delta_bits = 0.0
        self.assignment_changed = 0
        self.symbol_changed = 0
        self.cells = {(s, a): [0, 0.0] for s in (False, True) for a in (False, True)}
        self.histogram = np.zeros(codebook_size, dtype=np.int64)
        self.gop = {"boundary": {"real_bytes": 0, "held": 0, "routing_only": 0, "positions": 0,
                                 "frames": 0, "assignment_changed": 0},
                    "ordinary": {"real_bytes": 0, "held": 0, "routing_only": 0, "positions": 0,
                                 "frames": 0, "assignment_changed": 0}}

    def add_frame(self, *, real_bytes, oracle_bytes, ideal_bits, code_len_real, code_len_oracle,
                  assignment_real, assignment_oracle, baseline_real, symbols_real, symbols_oracle,
                  code_len_baseline, is_boundary) -> None:
        self.real_bytes += real_bytes
        self.oracle_bytes += oracle_bytes
        self.real_ideal_bits += ideal_bits
        self.real_code_len_bits += float(code_len_real.sum())
        self.positions += assignment_real.size
        self.histogram += np.bincount(assignment_real, minlength=self.histogram.size)

        held_mask = assignment_real != baseline_real
        held = int(held_mask.sum())
        self.held += held
        if held:
            delta = code_len_real[held_mask] - code_len_baseline[held_mask]
            self.held_code_len_delta_bits += float(delta.sum())
            self.held_worse += int((delta > 0).sum())
            self.held_better += int((delta < 0).sum())
            self.held_equal += int((delta == 0).sum())

        assignment_changed = assignment_real != assignment_oracle
        symbol_changed = symbols_real != symbols_oracle
        self.assignment_changed += int(assignment_changed.sum())
        self.symbol_changed += int(symbol_changed.sum())
        delta_excess = code_len_real - code_len_oracle
        for sym in (False, True):
            for assign in (False, True):
                mask = (symbol_changed == sym) & (assignment_changed == assign)
                cell = self.cells[(sym, assign)]
                cell[0] += int(mask.sum())
                cell[1] += float(delta_excess[mask].sum())

        bucket = self.gop["boundary" if is_boundary else "ordinary"]
        bucket["frames"] += 1
        bucket["real_bytes"] += real_bytes
        bucket["held"] += held
        bucket["positions"] += assignment_real.size
        bucket["assignment_changed"] += int(assignment_changed.sum())
        bucket["routing_only"] += int((~symbol_changed & assignment_changed).sum())

    def to_dict(self) -> dict[str, Any]:
        total_excess = sum(cell[1] for cell in self.cells.values())
        cells = {}
        for (sym, assign), (count, bits) in self.cells.items():
            cells[f"symbol_changed={sym}_assignment_changed={assign}"] = {
                "positions": count,
                "fraction_of_positions": count / self.positions if self.positions else 0.0,
                "sum_delta_code_len_bits": bits,
                "mean_delta_code_len_bits": bits / count if count else 0.0,
                "share_of_total_excess_bits": bits / total_excess if total_excess else 0.0,
            }
        used = int((self.histogram > 0).sum())
        probability = self.histogram / max(self.histogram.sum(), 1)
        nonzero = probability[probability > 0]
        return {
            "real_residual_bytes": self.real_bytes,
            "oracle_residual_bytes": self.oracle_bytes,
            "real_ideal_bits": self.real_ideal_bits,
            "mean_code_length_bits": (self.real_code_len_bits / self.positions
                                      if self.positions else 0.0),
            "positions": self.positions,
            "held_positions": self.held,
            "held_fraction": self.held / self.positions if self.positions else 0.0,
            "held_made_code_length_worse": self.held_worse,
            "held_made_code_length_better": self.held_better,
            "held_made_code_length_equal": self.held_equal,
            "held_code_length_delta_bits": self.held_code_len_delta_bits,
            "routing_churn_vs_oracle": (self.assignment_changed / self.positions
                                        if self.positions else 0.0),
            "symbol_change_rate_vs_oracle": (self.symbol_changed / self.positions
                                             if self.positions else 0.0),
            "total_excess_bits_vs_oracle": total_excess,
            "decomposition_2x2": cells,
            "routing_only_share_of_excess_bits":
                cells["symbol_changed=False_assignment_changed=True"]["share_of_total_excess_bits"],
            "assignment_distribution": {
                "prototypes_used": used,
                "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
                "top1_share": float(probability.max()),
                "top10_share": float(np.sort(probability)[-10:].sum()),
            },
            "gop_position": {name: dict(bucket) for name, bucket in self.gop.items()},
        }


@torch.no_grad()
def sweep_sequence(mc, m13, m20, model, rig, sequence, configs, *, bits, gop_size, block_size,
                   search_range, device, accumulators, advantage_samples, roundtrip_every,
                   roundtrip_records, identity_records, rng) -> int:
    """One VAL-B sequence, all configurations, one closed-loop pass."""
    model11 = rig["model11"]
    assign_codebook, coding_codebook = rig["assign_codebook"], rig["coding_codebook"]
    residual_params, zero = rig["residual_params"], rig["zero"]
    coding_probabilities = coding_codebook.probabilities
    frames = sequence.load_frames()
    types = mc.gop_frame_types(frames.shape[0], gop_size)

    real_previous = None
    oracle_previous_raw = None
    # One previous-assignment chain per (config, arm); reset at every I-frame,
    # exactly as a decoder would reset it.
    chains: dict[tuple[tuple[str, float], str], np.ndarray | None] = {
        (config, arm): None for config in configs for arm in ("real", "oracle")}
    p_frames = 0

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
                oracle_previous_raw = frame
                for key in chains:
                    chains[key] = None
                continue

            is_boundary = (index - 1) % gop_size == 0
            oracle_previous = model.decode(model.encode(oracle_previous_raw))
            mv_real = mc.estimate_block_motion(real_previous, frame, block_size=block_size,
                                               search_range=search_range)
            mv_oracle = mc.estimate_block_motion(oracle_previous, frame, block_size=block_size,
                                                 search_range=search_range)
            reference_real = model.encode(mc.warp_blocks(real_previous, mv_real,
                                                         block_size=block_size))
            reference_oracle = model.encode(mc.warp_blocks(oracle_previous, mv_oracle,
                                                           block_size=block_size))
            symbols_real = latent_to_symbols(latent - reference_real,
                                             residual_params).reshape(latent_shape)
            symbols_oracle = latent_to_symbols(latent - reference_oracle,
                                               residual_params).reshape(latent_shape)

            per_arm: dict[str, dict[str, Any]] = {}
            snapshot: dict[tuple[tuple[str, float], str], np.ndarray | None] = {}
            for arm, reference, symbols in (("real", reference_real, symbols_real),
                                            ("oracle", reference_oracle, symbols_oracle)):
                rows = m20.frame_rows(model11, reference, symbols, zero)
                cost_tensor = m20.assignment_costs(rows, assign_codebook)
                best = m20.baseline_assignment(cost_tensor)
                costs = cost_tensor.double().cpu().numpy()
                flat = np.asarray(symbols, dtype=np.int64).reshape(-1)
                diagnostic_assignments = {
                    "coding_metric": m20.coding_metric_assignment(rows, coding_codebook),
                    "oracle_table": m20.oracle_table_assignment(coding_codebook, flat)}
                del rows, cost_tensor
                arm_data: dict[str, Any] = {"best": best, "flat": flat, "costs": costs,
                                            "reference": reference, "symbols": symbols,
                                            "assignments": {}, "payloads": {}, "code_len": {}}
                arm_data["baseline_code_len"] = -np.log2(np.maximum(
                    coding_probabilities[best, flat], 1e-300))
                for config in configs:
                    state, margin = config
                    previous = chains[(config, arm)]
                    snapshot[(config, arm)] = previous
                    if state in diagnostic_assignments:
                        assignment = diagnostic_assignments[state]
                    elif margin <= 0.0:
                        assignment = best.copy()
                    else:
                        assignment, _ = m20.apply_hysteresis(
                            costs, best, margin=margin, state=state, previous_frame=previous,
                            shape=latent_shape, group_size=model11.group_size)
                    payload, ideal = m20.code_with_assignment(coding_codebook, flat, assignment)
                    arm_data["assignments"][config] = assignment
                    arm_data["payloads"][config] = (payload, ideal)
                    arm_data["code_len"][config] = -np.log2(np.maximum(
                        coding_probabilities[assignment, flat], 1e-300))
                    chains[(config, arm)] = assignment
                per_arm[arm] = arm_data

            # Phase C scale probe: the advantage the margin is compared against,
            # measured under the DEPLOYED (margin = 0) chain for each state.
            if advantage_samples is not None:
                _sample_advantages(per_arm["real"], snapshot[(("baseline", 0.0), "real")],
                                   latent_shape, model11.group_size, advantage_samples, rng)

            for config in configs:
                real = per_arm["real"]
                oracle = per_arm["oracle"]
                accumulators[config].add_frame(
                    real_bytes=len(real["payloads"][config][0]),
                    oracle_bytes=len(oracle["payloads"][config][0]),
                    ideal_bits=real["payloads"][config][1],
                    code_len_real=real["code_len"][config],
                    code_len_oracle=oracle["code_len"][config],
                    assignment_real=real["assignments"][config],
                    assignment_oracle=oracle["assignments"][config],
                    baseline_real=real["best"],
                    symbols_real=real["flat"], symbols_oracle=oracle["flat"],
                    code_len_baseline=real["baseline_code_len"],
                    is_boundary=is_boundary)

            # Phase D's decoder round-trip, on a deterministic subsample. The
            # margin=0 config additionally checks byte-identity against the
            # DEPLOYED `m13.encode_frame_recalibrated` - the identity control
            # that makes every delta in this sweep a delta from production.
            if roundtrip_every and p_frames % roundtrip_every == 0:
                deployed_payload, deployed_ideal = m13.encode_frame_recalibrated(
                    model11, assign_codebook, coding_codebook, reference_real, symbols_real,
                    zero, bits=bits)
                base = per_arm["real"]["payloads"][("baseline", 0.0)]
                identity_records.append({
                    "sequence": sequence.sequence_id, "index": index,
                    "deployed_bytes": len(deployed_payload), "m20_bytes": len(base[0]),
                    "payload_identical": deployed_payload == base[0],
                    "ideal_bits_identical": deployed_ideal == base[1],
                })
                for config in configs:
                    state, margin = config
                    if state in m20.DIAGNOSTIC_RULES:
                        continue        # reference points, not M20 candidates
                    decoded = m20.decode_frame_hysteresis(
                        model11, assign_codebook, coding_codebook,
                        per_arm["real"]["payloads"][config][0], reference_real, zero, bits=bits,
                        shape=latent_shape, margin=margin,
                        state="temporal" if state == "baseline" else state,
                        previous_frame=snapshot[(config, "real")])
                    roundtrip_records.append({
                        "sequence": sequence.sequence_id, "index": index,
                        "state": state, "margin": margin,
                        "symbols_exact": bool(np.array_equal(decoded["symbols"],
                                                             per_arm["real"]["flat"])),
                        "assignment_exact": bool(np.array_equal(
                            decoded["assignment"], per_arm["real"]["assignments"][config])),
                    })

            reconstructed_latent = reference_real + symbols_to_latent(
                per_arm["real"]["flat"], latent_shape, residual_params).to(device)
            real_previous = model.decode(reconstructed_latent)
            oracle_previous_raw = frame
            p_frames += 1
    return p_frames


def _sample_advantages(arm_data, previous_baseline, latent_shape, group_size, samples,
                       rng) -> None:
    """Phase C's scale probe: how big is cost(i, prev(i)) - cost(i, k*(i)) for
    each state's previous candidate, measured under the DEPLOYED (margin = 0)
    chain? This is the distribution the declared margin grid has to span.

    `previous_baseline` is the previous P-frame's deployed assignment (None at
    GOP position 1, where the temporal state has no candidate at all).
    """
    costs, best = arm_data["costs"], arm_data["best"]
    channels, _, width = latent_shape
    plane = costs.shape[0] // channels
    del width
    best_cost = costs[np.arange(costs.shape[0]), best]
    pick = rng.choice(costs.shape[0], size=min(1024, costs.shape[0]), replace=False)

    if previous_baseline is not None:
        samples["temporal"].append(costs[pick, previous_baseline[pick]] - best_cost[pick])
    # Group g's candidate is group g-1's assignment at the same (h, w); group 0
    # has none, which the identity prefix below represents as a zero advantage.
    offset = group_size * plane
    group_source = np.concatenate([best[:offset], best[:-offset]])
    samples["channel_group"].append(costs[pick, group_source[pick]] - best_cost[pick])
    raster_source = np.concatenate([best[:1], best[:-1]])
    samples["raster"].append(costs[pick, raster_source[pick]] - best_cost[pick])


def build_arg_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M20 Phases C-G: the pre-declared margin sweep on VAL-B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=defaults.processed_data_dir / "manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/m10f_lambda_boundary/lambda_3.0e-04_seed42/best.pt"))
    parser.add_argument("--m11-dir", type=Path, default=Path("outputs/m11_autoregressive_entropy"))
    parser.add_argument("--m10k-dir", type=Path, default=Path("outputs/m10k_learned_entropy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m20_codebook_hysteresis"))
    parser.add_argument("--m14-davis", type=Path, default=DEFAULT_M14_DAVIS)
    parser.add_argument("--output-name", type=str, default="m20_sweep.json")
    parser.add_argument("--rate-points", type=int, nargs="+", default=[5, 4, 3])
    parser.add_argument("--calibration-frames", type=int, default=400)
    parser.add_argument("--val-sequences", type=int, default=4)
    parser.add_argument("--val-frames-per-sequence", type=int, default=None)
    parser.add_argument("--gop", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--search-range", type=int, default=16)
    parser.add_argument("--roundtrip-every", type=int, default=20,
                        help="Decode-verify every Nth P-frame (0 disables).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    defaults = load_default_config()
    args = build_arg_parser(defaults).parse_args(argv)

    mc = _load_script("m10h_motion_compensation")
    m13 = _load_script("m13_recalibration")
    md = _load_script("m11_data")
    m20 = _load_script("m20_hysteresis")

    seed_everything(args.seed)
    device = get_device() if args.device == "auto" else torch.device(args.device)
    cache_dir = args.cache_dir or md.DEFAULT_CACHE_DIR

    from nvc.training.checkpoint import load_model_from_checkpoint
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    val_b = m20.val_b_sequences(args.manifest, count=args.val_sequences,
                               max_frames=args.val_frames_per_sequence)
    train_full = discover_sequences(args.manifest, split="train")
    configs = configurations(m20.STATES, m20.MARGIN_SWEEP, m20.DIAGNOSTIC_RULES)

    davis = json.loads(args.m14_davis.read_text(encoding="utf-8"))
    share = {}
    for name, arm in davis["arms"].items():
        if name.startswith("motion@"):
            share[arm["bits"]] = (arm["total_p_frame_residual_bytes"]
                                  / arm["total_container_bytes"])

    print("=" * 126)
    print("M20 PHASES C-G - PRE-DECLARED MARGIN SWEEP (VAL-B, held out; never TEST)")
    print("=" * 126)
    print(f"  VAL-B: {[s.sequence_id for s in val_b]}")
    print(f"  declared states : {list(m20.STATES)}")
    print(f"  declared margins: {list(m20.MARGIN_SWEEP)} (bits of expected code length)")
    print(f"  configurations  : {len(configs)}  (margin=0 shared as the identity control)")
    print(f"  deployed P-residual share of total stream: "
          + "  ".join(f"{b}-bit={share[b] * 100:.2f}%" for b in sorted(share, reverse=True)),
          flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "phase": "M20 Phases C-G", "val_b_sequence_ids": [s.sequence_id for s in val_b],
        "declared_states": list(m20.STATES), "declared_margins": list(m20.MARGIN_SWEEP),
        "p_residual_share_of_total_stream": {str(k): v for k, v in share.items()},
        "rate_points": [],
    }

    for bits in args.rate_points:
        started = time.perf_counter()
        rig = m20.prepare_rate_point(
            model, bits=bits, manifest=args.manifest, checkpoint=args.checkpoint,
            m11_dir=args.m11_dir, m10k_dir=args.m10k_dir, device=device, cache_dir=cache_dir,
            train_full=train_full, calibration_frames=args.calibration_frames,
            gop_size=args.gop, block_size=args.block_size, search_range=args.search_range)
        size = rig["assign_codebook"].size
        accumulators = {config: Accumulator(size) for config in configs}
        advantage_samples = {state: [] for state in m20.STATES}
        roundtrip_records: list[dict[str, Any]] = []
        identity_records: list[dict[str, Any]] = []
        rng = np.random.default_rng(args.seed + bits)
        p_frames = 0
        for sequence in val_b:
            p_frames += sweep_sequence(
                mc, m13, m20, model, rig, sequence, configs, bits=bits, gop_size=args.gop,
                block_size=args.block_size, search_range=args.search_range, device=device,
                accumulators=accumulators, advantage_samples=advantage_samples,
                roundtrip_every=args.roundtrip_every, roundtrip_records=roundtrip_records,
                identity_records=identity_records, rng=rng)

        baseline = accumulators[("baseline", 0.0)].to_dict()
        rows = []
        for config in configs:
            state, margin = config
            summary = accumulators[config].to_dict()
            delta_bytes = summary["real_residual_bytes"] - baseline["real_residual_bytes"]
            residual_percent = (-delta_bytes / baseline["real_residual_bytes"] * 100
                                if baseline["real_residual_bytes"] else 0.0)
            total_stream_percent = residual_percent * share.get(bits, 0.0)
            rows.append({
                "state": state, "margin": margin,
                "is_hysteresis_candidate": state not in m20.DIAGNOSTIC_RULES,
                "is_decoder_compatible": state != "oracle_table",
                **summary,
                "delta_residual_bytes_vs_margin0": delta_bytes,
                "residual_gain_percent": residual_percent,
                "total_stream_gain_percent": total_stream_percent,
                "verdict": m20.verdict(total_stream_percent),
                "churn_reduction_absolute": (baseline["routing_churn_vs_oracle"]
                                             - summary["routing_churn_vs_oracle"]),
            })

        scale = {}
        for state, chunks in advantage_samples.items():
            if not chunks:
                continue
            values = np.concatenate(chunks)
            scale[state] = {
                "samples": int(values.size), "mean_bits": float(values.mean()),
                "fraction_exactly_zero": float((values == 0).mean()),
                "percentiles_bits": {str(p): float(np.percentile(values, p))
                                     for p in ADVANTAGE_PERCENTILES},
            }

        print(f"\n  ---- {bits}-bit ----  {p_frames} P-frames, "
              f"{time.perf_counter() - started:.1f}s, residual={rig['residual_identity']}")
        for state in m20.STATES:
            if state in scale:
                pct = scale[state]["percentiles_bits"]
                print(f"    advantage scale [{state:>13}]: mean={scale[state]['mean_bits']:.5f}  "
                      + " ".join(f"p{p}={pct[p]:.5f}" for p in ("25", "50", "75", "90"))
                      + f"  zero={scale[state]['fraction_exactly_zero'] * 100:.1f}%")
        header = (f"    {'state':>13} {'margin':>7} {'residual bytes':>15} {'delta':>10} "
                  f"{'resid %':>9} {'stream %':>9} {'churn':>8} {'held':>7} {'worse':>7}")

        def _line(row):
            return (f"    {row['state']:>13} {row['margin']:>7.3f} "
                    f"{row['real_residual_bytes']:>15,} "
                    f"{row['delta_residual_bytes_vs_margin0']:>+10,} "
                    f"{row['residual_gain_percent']:>+9.4f} "
                    f"{row['total_stream_gain_percent']:>+9.4f} "
                    f"{row['routing_churn_vs_oracle'] * 100:>7.2f}% "
                    f"{row['held_fraction'] * 100:>6.2f}% "
                    f"{(row['held_made_code_length_worse'] / max(row['held_positions'], 1)) * 100:>6.2f}%")

        print(header)
        for row in rows:
            if row["is_hysteresis_candidate"]:
                print(_line(row))
        print("    -- reference points (NOT M20 candidates) --")
        for row in rows:
            if not row["is_hysteresis_candidate"]:
                print(_line(row) + ("  [decoder-available]" if row["is_decoder_compatible"]
                                    else "  [NON-CAUSAL upper bound]"))
        exact = all(r["symbols_exact"] and r["assignment_exact"] for r in roundtrip_records)
        identical = all(r["payload_identical"] and r["ideal_bits_identical"]
                        for r in identity_records)
        print(f"    decoder round trips: {len(roundtrip_records)} checked, all exact = {exact}")
        print(f"    margin=0 vs deployed m13.encode_frame_recalibrated: "
              f"{len(identity_records)} frames, byte-identical = {identical}", flush=True)

        report["rate_points"].append({
            "bits": bits, "p_frames": p_frames,
            "residual_identity": rig["residual_identity"],
            "assign_codebook_id": rig["assign_codebook_id"],
            "coding_codebook_id": rig["coding_codebook_id"],
            "advantage_scale": scale, "configurations": rows,
            "roundtrip_checked": len(roundtrip_records), "roundtrip_all_exact": exact,
            "roundtrip_records": roundtrip_records,
            "margin0_identical_to_deployed": identical,
            "margin0_identity_records": identity_records,
        })

    path = args.output_dir / args.output_name
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
