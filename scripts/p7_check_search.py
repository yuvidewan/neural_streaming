"""docs/PLAN_speed.md gate checks 1 and 2: the batched motion search changes nothing.

Runs `p4_gate3.py` with the motion module patched so every call of `estimate_block_motion`
(the batched search) also runs `_estimate_block_motion_loop` (the original) on the same real
reference and frame, and counts any field that differs. Any extra arguments go to p4_gate3.

    p7_check_search.py --lambdas 1e-3 --output-dir outputs/p7_speed --output-name check_davis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_parity as bp  # noqa: E402
import p4_gate3 as g4  # noqa: E402

STATS = {"fields": 0, "different": 0}
_original_loader = bp._load_script


def _patched_loader(name: str):
    module = _original_loader(name)
    if name == "m10h_motion_compensation":
        fast = module.estimate_block_motion

        def checked(reference, current, **kwargs):
            new = fast(reference, current, **kwargs)
            old = module._estimate_block_motion_loop(reference, current, **kwargs)
            STATS["fields"] += 1
            if not torch.equal(new, old):
                STATS["different"] += 1
                print(f"  [check] field {STATS['fields']} DIFFERS", flush=True)
            return new

        module.estimate_block_motion = checked
    return module


def main() -> int:
    bp._load_script = _patched_loader
    code = g4.main(sys.argv[1:])
    print(f"[check] motion fields compared: {STATS['fields']}, different: {STATS['different']}")
    argv = sys.argv[1:]
    out = Path(argv[argv.index("--output-dir") + 1]) / argv[argv.index("--output-name") + 1]
    out.with_suffix(".search_check.json").write_text(json.dumps(STATS), encoding="utf-8")
    return code or (1 if STATS["different"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
