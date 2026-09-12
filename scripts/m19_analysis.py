"""M19 - analysis pass over `m19_reference_error_diagnostic.py`'s raw
per-frame output (`m19_raw_{bits}bit.json`). Pure numpy/stats, no GPU, no
model - every number here is a deterministic reduction of already-collected
data, never a fresh diagnostic pass (keeps the expensive GPU sweep to
exactly one run per bit depth).

Run:
  ./.venv/Scripts/python.exe scripts/m19_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DIR = Path("outputs/m19_reference_error_audit")
RATE_POINTS = (5, 4, 3)


def _load(bits: int, directory: Path) -> list[dict[str, Any]]:
    return json.loads((directory / f"m19_raw_{bits}bit.json").read_text(encoding="utf-8"))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _channel_concentration(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    stacked = np.stack([np.asarray(r[field]) for r in rows])  # [N, channels]
    per_channel = stacked.mean(axis=0)
    order = np.argsort(-per_channel)
    total = per_channel.sum()
    channels = len(per_channel)
    top10 = max(1, round(channels * 0.10))
    top25 = max(1, round(channels * 0.25))
    return {
        "per_channel_mean": per_channel.tolist(), "ranked_channel_indices": order.tolist(),
        "top10pct_share_percent": float(per_channel[order[:top10]].sum() / total * 100) if total else 0.0,
        "top25pct_share_percent": float(per_channel[order[:top25]].sum() / total * 100) if total else 0.0,
    }


def _codebook_routing_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    magnitudes = np.concatenate([np.asarray(r["error_magnitude_per_symbol"]) for r in rows])
    changed = np.concatenate([np.asarray(r["assignment_changed_per_symbol"]) for r in rows])
    delta_len = np.concatenate([np.asarray(r["delta_code_len_per_symbol"]) for r in rows])
    symbol_changed = np.concatenate([np.asarray(r["symbol_changed_per_symbol"]) for r in rows])

    deciles = np.quantile(magnitudes, np.linspace(0, 1, 11))
    bucket_index = np.clip(np.digitize(magnitudes, deciles[1:-1]), 0, 9)
    p_changed_by_bucket = [float(changed[bucket_index == b].mean()) if (bucket_index == b).any() else None
                          for b in range(10)]
    mean_bucket_magnitude = [float(magnitudes[bucket_index == b].mean()) if (bucket_index == b).any() else None
                             for b in range(10)]

    mean_delta_when_changed = float(delta_len[changed.astype(bool)].mean()) if changed.any() else 0.0
    mean_delta_when_unchanged = float(delta_len[~changed.astype(bool)].mean()) if (~changed.astype(bool)).any() else 0.0

    # Phase E 2x2 decomposition: symbol changed x assignment changed.
    cells = {}
    for sym_changed in (0, 1):
        for assign_changed in (0, 1):
            mask = (symbol_changed == sym_changed) & (changed == assign_changed)
            cells[f"symbol_changed={bool(sym_changed)}_assignment_changed={bool(assign_changed)}"] = {
                "count": int(mask.sum()), "fraction": float(mask.mean()),
                "mean_delta_code_len": float(delta_len[mask].mean()) if mask.any() else 0.0,
            }

    return {
        "n_positions": int(magnitudes.size),
        "mean_delta_code_len_bits_when_assignment_changed": mean_delta_when_changed,
        "mean_delta_code_len_bits_when_assignment_unchanged": mean_delta_when_unchanged,
        "p_assignment_changed_by_error_magnitude_decile": p_changed_by_bucket,
        "mean_error_magnitude_by_decile": mean_bucket_magnitude,
        "overall_fraction_assignment_changed": float(changed.mean()),
        "overall_fraction_symbol_changed": float(symbol_changed.mean()),
        "decomposition_2x2": cells,
        "total_excess_bits_from_assignment_changes": float(delta_len[changed.astype(bool)].sum()),
        "total_excess_bits_from_unchanged_assignment": float(delta_len[~changed.astype(bool)].sum()),
    }


def _spatial_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _mean(field):
        return float(np.mean([r[field] for r in rows]))

    region_means = {name: float(np.mean([r["spatial_region_error"][name] for r in rows]))
                    for name in ("flat", "texture", "edge")}
    return {
        "mean_autocorr_h": _mean("autocorr_h"), "mean_autocorr_v": _mean("autocorr_v"),
        "mean_low_freq_energy": _mean("low_freq_energy"), "mean_high_freq_energy": _mean("high_freq_energy"),
        "low_high_freq_ratio": _mean("low_freq_energy") / max(_mean("high_freq_energy"), 1e-12),
        "region_mean_pixel_error": region_means,
        "edge_over_flat_error_ratio": region_means["edge"] / max(region_means["flat"], 1e-12),
        "mean_boundary_error": _mean("boundary_error"), "mean_interior_error": _mean("interior_error"),
        "boundary_over_interior_ratio": _mean("boundary_error") / max(_mean("interior_error"), 1e-12),
    }


def _motion_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sad = np.array([r["sad_reference"] for r in rows])
    motion_mag = np.array([r["motion_magnitude"] for r in rows])
    pixel_err = np.array([r["pixel_error"]["rms"] for r in rows])
    churn = np.array([r["assignment_changed_fraction"] for r in rows])
    byte_gap = np.array([r["real_bytes"] - r["oracle_bytes"] for r in rows])
    return {
        "corr_sad_vs_pixel_error": _corr(sad, pixel_err),
        "corr_sad_vs_churn": _corr(sad, churn),
        "corr_motion_magnitude_vs_pixel_error": _corr(motion_mag, pixel_err),
        "corr_motion_magnitude_vs_byte_gap": _corr(motion_mag, byte_gap),
        "corr_pixel_error_vs_churn": _corr(pixel_err, churn),
        "corr_pixel_error_vs_byte_gap": _corr(pixel_err, byte_gap),
    }


def _control_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = sum(r["real_bytes"] for r in rows)
    oracle = sum(r["oracle_bytes"] for r in rows)
    shuffled = sum(r["shuffled_bytes"] for r in rows)
    gap = real - oracle
    recovered = (real - shuffled) / gap * 100 if gap else 0.0
    return {
        "total_real_bytes": real, "total_oracle_bytes": oracle, "total_shuffled_bytes": shuffled,
        "real_to_oracle_gap_bytes": gap,
        "shuffle_fraction_of_gap_recovered_percent": recovered,
        "interpretation": (
            "shuffle recovers most of the gap -> MAGNITUDE dominates, shape secondary"
            if recovered > 60 else
            "shuffle recovers little/none of the gap -> STRUCTURE matters, not just magnitude"
            if recovered < 25 else
            "shuffle recovers a moderate fraction -> both magnitude and structure contribute"),
    }


def _gop_position_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    boundary = [r for r in rows if r["is_boundary"]]
    ordinary = [r for r in rows if not r["is_boundary"]]
    out = {}
    for name, subset in (("boundary", boundary), ("ordinary", ordinary)):
        if not subset:
            out[name] = {"count": 0}
            continue
        real = sum(r["real_bytes"] for r in subset)
        oracle = sum(r["oracle_bytes"] for r in subset)
        out[name] = {
            "count": len(subset), "real_bytes": real, "oracle_bytes": oracle,
            "gain_percent": (real - oracle) / real * 100 if real else 0.0,
            "mean_assignment_churn": float(np.mean([r["assignment_changed_fraction"] for r in subset])),
            "mean_pixel_error_rms": float(np.mean([r["pixel_error"]["rms"] for r in subset])),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M19 analysis pass over raw diagnostic data.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--rate-points", type=int, nargs="+", default=list(RATE_POINTS))
    args = parser.parse_args(argv)

    report: dict[str, Any] = {"phase": "M19 analysis", "rate_points": []}
    for bits in args.rate_points:
        rows = _load(bits, args.input_dir)
        print(f"\n{'=' * 110}\n{bits}-BIT ({len(rows)} P-frames)\n{'=' * 110}")

        spatial = _spatial_analysis(rows)
        print(f"  SPATIAL: autocorr h/v={spatial['mean_autocorr_h']:.4f}/{spatial['mean_autocorr_v']:.4f}  "
             f"low/high freq ratio={spatial['low_high_freq_ratio']:.3f}  "
             f"edge/flat error ratio={spatial['edge_over_flat_error_ratio']:.3f}  "
             f"boundary/interior ratio={spatial['boundary_over_interior_ratio']:.3f}")

        channel_mae = _channel_concentration(rows, "channel_mae")
        channel_churn = _channel_concentration(rows, "channel_churn")
        print(f"  CHANNEL: top10%% share of MAE={channel_mae['top10pct_share_percent']:.2f}%%  "
             f"top25%%={channel_mae['top25pct_share_percent']:.2f}%%   "
             f"top10%% share of churn={channel_churn['top10pct_share_percent']:.2f}%%  "
             f"top25%%={channel_churn['top25pct_share_percent']:.2f}%%")

        routing = _codebook_routing_analysis(rows)
        print(f"  ROUTING: P(assignment changed) by error decile: "
             f"{[round(p, 3) if p is not None else None for p in routing['p_assignment_changed_by_error_magnitude_decile']]}")
        print(f"           mean Delta code-length | changed={routing['mean_delta_code_len_bits_when_assignment_changed']:.4f} bits  "
             f"| unchanged={routing['mean_delta_code_len_bits_when_assignment_unchanged']:.4f} bits")
        for cell, values in routing["decomposition_2x2"].items():
            print(f"           {cell}: fraction={values['fraction']:.4f}  "
                 f"mean_delta_bits={values['mean_delta_code_len']:.4f}")

        motion = _motion_interaction(rows)
        print(f"  MOTION: corr(SAD, pixel_err)={motion['corr_sad_vs_pixel_error']:.3f}  "
             f"corr(SAD, churn)={motion['corr_sad_vs_churn']:.3f}  "
             f"corr(pixel_err, byte_gap)={motion['corr_pixel_error_vs_byte_gap']:.3f}")

        control = _control_analysis(rows)
        print(f"  CONTROL: real={control['total_real_bytes']:,} oracle={control['total_oracle_bytes']:,} "
             f"shuffled={control['total_shuffled_bytes']:,}  "
             f"shuffle recovers {control['shuffle_fraction_of_gap_recovered_percent']:.2f}% of gap")
        print(f"           -> {control['interpretation']}")

        gop = _gop_position_split(rows)
        print(f"  GOP POSITION: boundary gain={gop['boundary']['gain_percent']:.4f}% "
             f"(churn={gop['boundary']['mean_assignment_churn']*100:.2f}%)  "
             f"ordinary gain={gop['ordinary']['gain_percent']:.4f}% "
             f"(churn={gop['ordinary']['mean_assignment_churn']*100:.2f}%)")

        report["rate_points"].append({
            "bits": bits, "n_frames": len(rows), "spatial": spatial,
            "channel_mae_concentration": channel_mae, "channel_churn_concentration": channel_churn,
            "codebook_routing": routing, "motion_interaction": motion, "control": control,
            "gop_position": gop,
        })

    path = args.output_dir / "m19_analysis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
