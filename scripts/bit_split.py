"""Where the bits go: the full-video benchmark's NVC streams, split by frame type.

Runs `benchmark_parity`'s NVC arm unchanged (same rig, same encoder, same
decode-exact check) and reads every `.nvct` it writes before it is deleted,
tallying container bytes into header, I-frame, P-frame motion, P-frame residual
and side information, plus PSNR by frame type. The classical arms are skipped:
this answers a question about our own stream, not a comparison.

Usage: scripts/bit_split.py <benchmark_parity arguments> --split-output out.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_parity as bp  # noqa: E402

from nvc.video.container import FRAME_TYPE_I, TemporalStreamReader  # noqa: E402

TALLY: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))


def _psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a.float() - b.float()) ** 2).flatten(1).mean(1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-12))


def _wrap_decoder(m21, frames_by_length):
    decode = m21.decode_sequence_refined

    def decode_and_tally(mc, m13, model, path, spec, identity, *, bits, **kwargs):
        data = Path(path).read_bytes()
        reader = TemporalStreamReader(data)
        tally = TALLY[f"{bits}bit"]
        types = []
        body = 0
        for frame_type, motion, residual, side in reader.iter_frames_with_side():
            types.append(frame_type)
            if frame_type == FRAME_TYPE_I:
                tally["i_frame_bytes"] += len(motion) + len(residual) + len(side)
                tally["i_frames"] += 1
            else:
                tally["p_motion_bytes"] += len(motion)
                tally["p_residual_bytes"] += len(residual)
                tally["p_side_bytes"] += len(side)
                tally["p_frames"] += 1
            body += len(motion) + len(residual) + len(side)
        tally["total_bytes"] += len(data)
        tally["framing_bytes"] += len(data) - body

        decoded = decode(mc, m13, model, path, spec, identity, bits=bits, **kwargs)
        reconstruction = decoded["reconstructions"].cpu().clamp(0, 1)
        reference = frames_by_length[reconstruction.shape[0]].pop(0)
        psnr = _psnr((reconstruction * 255).round() / 255, reference)
        kinds = torch.tensor([t == FRAME_TYPE_I for t in types])
        tally["i_psnr_sum"] += psnr[kinds].sum().item()
        tally["p_psnr_sum"] += psnr[~kinds].sum().item()
        return decoded

    m21.decode_sequence_refined = decode_and_tally


def main(argv: list[str]) -> int:
    output = Path(argv[argv.index("--split-output") + 1])
    argv = [a for i, a in enumerate(argv)
            if a != "--split-output" and (i == 0 or argv[i - 1] != "--split-output")]

    frames_by_length: dict[int, list[torch.Tensor]] = defaultdict(list)
    load = bp._load_script

    def load_and_wrap(name):
        module = load(name)
        if name == "m21_refinement":
            _wrap_decoder(module, frames_by_length)
        return module

    def run_nvc_only(args, device, sequences, frame_cache, report, persist):
        # The decoder wrapper sees streams in sequence order; queue the
        # references the same way, once per rate point and arm.
        for _ in args.rate_points:
            for _ in args.arms:
                for sequence in sequences:
                    frames = frame_cache[sequence.sequence_id]
                    frames_by_length[frames.shape[0]].append(frames)
        return original_run_nvc(args, device, sequences, frame_cache, report, persist)

    original_run_nvc = bp.run_nvc
    bp._load_script = load_and_wrap
    bp.run_classical = lambda *a, **k: None
    bp.run_nvc = run_nvc_only
    code = bp.main(argv + ["--no-resume"])

    result = {}
    for point, tally in TALLY.items():
        total = tally["total_bytes"]
        result[point] = {
            **{k: v for k, v in tally.items() if not k.endswith("_sum")},
            "share": {k.removesuffix("_bytes"): tally[k] / total for k in
                      ("i_frame_bytes", "p_motion_bytes", "p_residual_bytes", "p_side_bytes",
                       "framing_bytes")},
            "bytes_per_i_frame": tally["i_frame_bytes"] / tally["i_frames"],
            "bytes_per_p_frame": (tally["p_motion_bytes"] + tally["p_residual_bytes"]
                                  + tally["p_side_bytes"]) / tally["p_frames"],
            "i_frame_psnr": tally["i_psnr_sum"] / tally["i_frames"],
            "p_frame_psnr": tally["p_psnr_sum"] / tally["p_frames"],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
