"""docs/PLAN_fair_eval.md proof 1: padding changes nothing where it is not needed.

UVG Beauty (1920x1072, 96 frames) at lambda 1e-2 and 1e-4 is encoded twice: with p4_gate3.py and
m10h_motion_compensation.py as they were at commit eb75a55 (before padding), and with the
current files. The streams must be byte-identical, and each new stream must decode in a fresh
process to the encoder's own reconstruction.

    proof_padding.py encode     both versions, both lambdas; writes proof1_encode.json
    proof_padding.py decode     fresh process: decode the new streams with the current code
    proof_padding.py check      exit 0 only if everything matched
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "outputs/p12_audit2")
import bitstream as bs  # noqa: E402  (current scripts; builds coders exactly as p4_gate3.main)

O = Path("outputs/p15_fair_eval")
SCRATCH = Path("C:/fair_scratch/proof1")
BEFORE = "eb75a55"
LAMBDAS = ("1e-2", "1e-4")


def load_old(name: str, alias: str):
    """A script as it was at BEFORE, imported under another module name."""
    source = subprocess.run(["git", "show", f"{BEFORE}:scripts/{name}.py"], capture_output=True,
                            check=True).stdout
    path = SCRATCH / "old_scripts" / f"{alias}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmd_encode(device) -> int:
    old_pg = load_old("p4_gate3", "old_p4_gate3")
    old_mc = load_old("m10h_motion_compensation", "old_m10h")
    frames = bs.sequence("uvg", "Beauty").load_frames()
    out = {}
    for lam in LAMBDAS:
        c = bs.Coders(lam, device)
        hashes = {}
        for tag, pg, mc in (("old", old_pg, old_mc), ("new", bs.pg, bs.mc)):
            path = SCRATCH / f"{tag}_Beauty_{lam}.nvct"
            started = time.time()
            enc = pg.encode_sequence(mc, bs.m21, c.intra, c.icoder, c.pcoder, frames, path,
                                     c.rig, c.tables, gop=30, block_size=c.args.block_size,
                                     search_range=c.args.search_range, identity=c.identity,
                                     motion_lambda=bs.MOTION_LAMBDA[lam])
            hashes[tag] = bs.frames_hash(enc["reconstructions"])
            print(f"[encode] {tag} {lam}: {path.stat().st_size:,} B in {time.time() - started:.0f} s",
                  flush=True)
        old, new = (SCRATCH / f"{t}_Beauty_{lam}.nvct" for t in ("old", "new"))
        out[lam] = {"old_bytes": old.stat().st_size, "new_bytes": new.stat().st_size,
                    "byte_identical": old.read_bytes() == new.read_bytes(),
                    "same_reconstruction": hashes["old"] == hashes["new"],
                    "encoder_hash": hashes["new"]}
        print(f"[encode] {lam}: {out[lam]}", flush=True)
        (O / "proof1_encode.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        del c
        torch.cuda.empty_cache()
    return 0


def cmd_decode(device) -> int:
    enc = json.loads((O / "proof1_encode.json").read_text(encoding="utf-8"))
    out = {}
    for lam in LAMBDAS:
        c = bs.Coders(lam, device)
        decoded = c.decode(SCRATCH / f"new_Beauty_{lam}.nvct").cpu()
        out[lam] = {"decode_exact": bs.frames_hash(decoded) == enc[lam]["encoder_hash"]}
        print(f"[decode] {lam}: {out[lam]}", flush=True)
        del c
        torch.cuda.empty_cache()
    (O / "proof1_decode.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def cmd_check() -> int:
    enc = json.loads((O / "proof1_encode.json").read_text(encoding="utf-8"))
    dec = json.loads((O / "proof1_decode.json").read_text(encoding="utf-8"))
    ok = (set(enc) == set(dec) == set(LAMBDAS)
          and all(e["byte_identical"] and e["same_reconstruction"] for e in enc.values())
          and all(d["decode_exact"] for d in dec.values()))
    print(f"PROOF 1: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "check":
        sys.exit(cmd_check())
    device = torch.device("cuda")
    sys.exit(cmd_encode(device) if command == "encode" else cmd_decode(device))
