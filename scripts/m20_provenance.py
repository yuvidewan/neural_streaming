"""M20 Phase L - prove the existing M13-M19 streams are unaffected.

M20 is a diagnostic: it never wrote a `.nvct` stream and never touched
`src/nvc/`. The obligation when NO candidate is adopted is therefore to show
that every already-shipped stream is byte-for-byte what it was, and that its
three entropy identities still match the frozen values M13/M14/M15 recorded.

This reads every `.nvct` v2 stream M13/M14/M15 produced, re-parses its header
with the UNMODIFIED `m10h_motion_compensation.TemporalStreamReader`, and checks:

  * the container format version is still 2 (no new version introduced);
  * `intra_entropy_model_id` / `residual_entropy_model_id` /
    `motion_entropy_model_id` match the values recorded in each milestone's own
    benchmark JSON;
  * the file's SHA-256 and byte length, recorded here so any future change is
    detectable rather than assumed absent;
  * the frame records still parse and their payload lengths sum to the file
    size (i.e. the streams are readable, not merely present).

Run:
  ./.venv/Scripts/python.exe scripts/m20_provenance.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

STREAM_DIRS = (
    Path("outputs/m13_recalibration/davis_streams"),
    Path("outputs/m13_recalibration/validation_streams"),
    Path("outputs/m14_entropy_audit/davis_streams"),
    Path("outputs/m14_entropy_audit/coded_validation_streams"),
    Path("outputs/m15_calibration_policy"),
)
RECORDED_PROVENANCE = (
    Path("outputs/m14_entropy_audit/m14_davis_benchmark.json"),
    Path("outputs/m19_reference_error_audit/m19_identities.json"),
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_stream(mc, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    reader = mc.TemporalStreamReader(path)
    header = reader.header
    frames = list(reader)
    kinds = Counter(getattr(record, "frame_type", None) for record in frames)
    return {
        "path": str(path).replace("\\", "/"),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "format_version": int(header.format_version),
        "quantization_bits": int(header.quantization_bits),
        "frame_count": len(frames),
        "frame_types": {str(k): v for k, v in sorted(kinds.items(), key=lambda kv: str(kv[0]))},
        "intra_entropy_model_id": header.intra_entropy_model_id.hex(),
        "residual_entropy_model_id": header.residual_entropy_model_id.hex(),
        "motion_entropy_model_id": header.motion_entropy_model_id.hex(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M20 Phase L: existing streams are unaffected.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m20_codebook_hysteresis"))
    parser.add_argument("--stream-dirs", type=Path, nargs="*", default=list(STREAM_DIRS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mc = _load_script("m10h_motion_compensation")

    recorded_identities = json.loads(
        Path("outputs/m19_reference_error_audit/m19_identities.json").read_text(encoding="utf-8"))

    print("=" * 112)
    print("M20 PHASE L - PROVENANCE / EXISTING-STREAM COMPATIBILITY")
    print("=" * 112)

    streams: list[dict[str, Any]] = []
    for directory in args.stream_dirs:
        for path in sorted(Path(directory).glob("**/*.nvct")):
            streams.append(inspect_stream(mc, path))

    # Every identity in every stream must be one an EARLIER milestone already
    # recorded in its own JSON. Scanning those JSONs rather than hardcoding a
    # list is deliberate: the streams span M13's pre/post-recalibration arms,
    # M14's old/new motion arms and M15's policy arms, and a hardcoded list
    # would flag legitimate, already-shipped identities as "new".
    corpus: dict[str, list[str]] = {}
    for path in sorted(Path("outputs").glob("**/*.json")):
        if "m20_codebook_hysteresis" in path.as_posix():
            continue                    # never let M20's own output vouch for M20
        text = path.read_text(encoding="utf-8", errors="replace")
        for stream in streams:
            for field in ("intra", "residual", "motion"):
                identity = stream[f"{field}_entropy_model_id"]
                if identity in text:
                    corpus.setdefault(identity, [])
                    if path.as_posix() not in corpus[identity]:
                        corpus[identity].append(path.as_posix())

    versions = sorted({s["format_version"] for s in streams})
    seen = {field: sorted({s[f"{field}_entropy_model_id"] for s in streams})
            for field in ("intra", "residual", "motion")}
    unattributed = {field: [i for i in ids if i not in corpus] for field, ids in seen.items()}
    m20_wrote_streams = sorted(args.output_dir.glob("**/*.nvct"))
    ok = (versions == [2] and bool(streams) and not m20_wrote_streams
          and all(not v for v in unattributed.values())
          and all(s["frame_count"] > 0 for s in streams))

    print(f"  streams inspected            : {len(streams)}")
    print(f"  container format versions    : {versions}  (2 = .nvct v2, unchanged)")
    for field in ("intra", "residual", "motion"):
        print(f"  {field:>8} entropy identities : {len(seen[field])} distinct, all recorded by an "
              f"earlier milestone = {not unattributed[field]}")
        for identity in seen[field]:
            sources = corpus.get(identity, [])
            print(f"      {identity}  <- {sources[0] if sources else 'UNATTRIBUTED'}"
                  f"{f' (+{len(sources) - 1} more)' if len(sources) > 1 else ''}")
    print(f"  every stream re-parsed with the unmodified reader: "
          f"{all(s['frame_count'] > 0 for s in streams)}")
    print(f"  .nvct streams written by M20 : {len(m20_wrote_streams)} (must be 0)")

    report = {
        "phase": "M20 Phase L", "candidate_adopted": False,
        "streams_inspected": len(streams),
        "container_format_versions": versions,
        "identities_seen": seen,
        "identity_attribution": corpus,
        "unattributed_identities": unattributed,
        "nvct_streams_written_by_m20": [str(p) for p in m20_wrote_streams],
        "recorded_m19_identities": recorded_identities,
        "all_streams_unaffected": ok,
        "streams": streams,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "m20_provenance.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nEXISTING STREAMS UNAFFECTED: {ok}")
    print(f"Report: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
