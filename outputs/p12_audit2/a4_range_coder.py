"""docs/PLAN_audit2.md, check A4: range coder edge cases, the C coder against the retired
Python reference.

The Python reference in src/nvc/compression/range_coder.py is commented out. It is
reinstated here, for this test only, by un-commenting those lines in memory. The source file
is not changed. Uses the production tables (p1_gate3.cumulative_tables: 64 Laplace scales,
symbols on [-255, 255]).

    .venv/Scripts/python outputs/p12_audit2/a4_range_coder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import p1_gate3 as g3  # noqa: E402
import p4_gate3 as pg  # noqa: E402
from nvc.compression import range_coder as rc  # noqa: E402

O = Path("outputs/p12_audit2")


def python_reference():
    lines = Path(rc.__file__).read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "# from bisect import bisect_right")
    end = next(i for i, l in enumerate(lines) if "end of retired Python reference" in l)
    code = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines[start:end])
    namespace = dict(vars(rc))
    exec(code, namespace)
    return namespace["_encode_symbols_python"], namespace["_decode_symbols_python"]


def main() -> int:
    enc_py, dec_py = python_reference()
    tables = g3.cumulative_tables()
    R, L = g3.SYMBOL_RADIUS, g3.SCALE_LEVELS
    out = {"cases": {}}

    def round_trip(label, symbols, index, compare_python=True):
        symbols = np.asarray(symbols, dtype=np.int64) + R
        index = np.asarray(index, dtype=np.int64)
        try:
            payload = rc.encode_symbols(symbols, tables, index)
            back = rc.decode_symbols(payload, len(symbols), tables, index)
            row = {"ok": bool(np.array_equal(back, symbols)), "bytes": len(payload)}
            if compare_python:
                row["python_same_bytes"] = enc_py(symbols, tables, index) == payload
                row["python_decodes"] = bool(np.array_equal(
                    dec_py(payload, len(symbols), tables, index), symbols))
        except Exception as error:
            row = {"ok": False, "raised": f"{type(error).__name__}: {error}"}
        out["cases"][label] = row
        print(f"[a4] {label}: {row}", flush=True)
        return row

    n = 4096
    round_trip("symbols at +R and -R, smallest scale", [R, -R] * (n // 2), [0] * n)
    round_trip("symbols at +R and -R, largest scale", [R, -R] * (n // 2), [L - 1] * n)
    round_trip("all zero, smallest scale", [0] * n, [0] * n)
    round_trip("all zero, largest scale", [0] * n, [L - 1] * n)
    round_trip("one symbol", [0], [5])
    round_trip("one symbol at +R, smallest scale", [R], [0])
    rng = np.random.default_rng(20261123)
    round_trip("every scale, alternating extremes", np.tile([R, -R, 0, 1, -1], L * 20)[:L * 100],
               np.repeat(np.arange(L), 100))
    # Beyond the coded range: must raise, never wrap.
    try:
        pg._code(torch.tensor([[R + 1.0]]), torch.tensor([[1.0]]), tables)
        out["cases"]["symbol R+1 through p4_gate3._code"] = {"raised": None}
    except ValueError as error:
        out["cases"]["symbol R+1 through p4_gate3._code"] = {"raised": f"ValueError: {error}"}
    # An empty payload: p4_gate3._decode of an x latent always has symbols, so the coder must
    # refuse an empty request rather than return nothing silently.
    try:
        rc.decode_symbols(b"", 0, tables, np.zeros(0, dtype=np.int64))
        out["cases"]["decode 0 symbols"] = {"raised": None}
    except ValueError as error:
        out["cases"]["decode 0 symbols"] = {"raised": f"ValueError: {error}"}
    try:
        rc.encode_symbols(np.zeros(0, dtype=np.int64), tables, np.zeros(0, dtype=np.int64))
        out["cases"]["encode 0 symbols"] = {"raised": None}
    except ValueError as error:
        out["cases"]["encode 0 symbols"] = {"raised": f"ValueError: {error}"}
    # Scales outside the table snap to its ends (index clamped): smaller than SCALE_FLOOR and
    # far above SCALE_MAX.
    idx = g3.scale_index(torch.tensor([1e-6, g3.SCALE_MIN, g3.SCALE_MAX, 1e6]))
    out["cases"]["scale_index at and beyond the table ends"] = {"indices": idx.tolist()}
    # Random: 1,000 arrays, Laplace-like symbols with random scales, C vs Python.
    random_ok = random_same = 0
    for _ in range(1000):
        size = int(rng.integers(1, 2000))
        index = rng.integers(0, L, size)
        scale = g3.scale_table().numpy()[index]
        symbols = np.clip(np.round(rng.laplace(0, scale)), -R, R).astype(np.int64)
        if rng.random() < 0.05:
            symbols[rng.integers(0, size)] = R * (1 if rng.random() < 0.5 else -1)
        s = symbols + R
        payload = rc.encode_symbols(s, tables, index)
        random_ok += np.array_equal(rc.decode_symbols(payload, size, tables, index), s)
        random_same += enc_py(s, tables, index) == payload
    out["random_1000"] = {"round_trip_ok": int(random_ok), "c_equals_python_bytes": int(random_same)}
    print(f"[a4] random: {out['random_1000']}", flush=True)
    out["min_frequency"] = int(np.diff(tables, axis=1).min())
    out["all_ok"] = (all(c.get("ok", True) for c in out["cases"].values()
                         if "ok" in c) and random_ok == 1000 and random_same == 1000
                     and all(c.get("python_same_bytes", True) for c in out["cases"].values()))
    (O / "a4_range_coder.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"all ok: {out['all_ok']}; min frequency {out['min_frequency']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
