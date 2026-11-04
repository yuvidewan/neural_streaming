"""The page's "where we stand against H.264" panel, read from the committed reports.

Nothing here measures anything. Every number comes from a benchmark report already
in the repo, so the page can only show what a full DAVIS TEST run produced:

- the x264 all-intra curve and the starting codec, from the committed intra-only run
  (`benchmark_parity.py --gop 1`);
- each Stage 1 model, from the gate report `benchmark_intra_gate.py` wrote next to it.

All of it is intra only (every frame coded on its own), DAVIS TEST, quality as
`sequence_mean` PSNR - the convention every intra BD-rate in the project is quoted in.
A report that is missing (a fresh clone without one of them) is simply left out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONVENTION = "sequence_mean"
INTRA_REPORT = Path("outputs/benchmarks/parity_intra/parity_intra.json")

# Oldest first: the order the models were built in, which is the story the page tells.
STAGES = (
    {"id": "deployed", "name": "Starting codec", "source": "intra"},
    {"id": "run2", "name": "64-channel redesign",
     "source": Path("outputs/stage1_vimeo/colab_run2_64ch/gate_stage1_64ch")},
    {"id": "run4", "name": "+ quantization-aware training",
     "source": Path("outputs/stage1_vimeo/colab_run4_64ch_qat/gate_qat_epoch23")},
    {"id": "run5", "name": "+ rate-aware training (Phase B)",
     "source": Path("outputs/stage1_vimeo/colab_run5_phase_b/gate_phase_b_unitstep.json")},
)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _curve(points: list[dict[str, Any]]) -> list[dict[str, float]]:
    return sorted(({"bpp": p["bpp"], "psnr": p["quality"][CONVENTION]["psnr"]} for p in points),
                  key=lambda p: p["bpp"])


def load_scoreboard(repo_root: str | Path) -> dict[str, Any] | None:
    """The reference curve and every stage that has a report, or None without the reference."""
    root = Path(repo_root)
    intra = _read(root / INTRA_REPORT)
    if intra is None:
        return None
    intra_points = list(intra["points"].values())
    stages = []
    for stage in STAGES:
        if stage["source"] == "intra":
            points = [p for p in intra_points if p["arm"] == "nvc_deployed"]
            bd = intra["comparisons"]["nvc_deployed_vs_h264_intra"][f"{CONVENTION}/psnr"]
            report = str(INTRA_REPORT)
        else:
            gate = _read(root / stage["source"])
            if gate is None:
                continue
            points = list(gate["points"].values())
            bd = gate["comparisons"][f"{CONVENTION}/psnr"]
            report = str(stage["source"])
        stages.append({"id": stage["id"], "name": stage["name"], "report": report.replace("\\", "/"),
                       "bd_rate_percent": bd["bd_rate_percent"], "curve": _curve(points)})
    return {
        "convention": CONVENTION,
        "dataset": "DAVIS TEST, every frame coded on its own (intra)",
        "reference": {"name": "H.264 (x264, all-intra)",
                      "curve": _curve([p for p in intra_points if p["arm"] == "h264_intra"])},
        "stages": stages,
    }
