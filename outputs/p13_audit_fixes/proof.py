"""docs/PLAN_audit2_fixes.md, fix 1, proof 2: re-encode the second audit's 12 kept streams with
the block-less container and compare them byte for byte with the old files.

    proof.py encode    code the 12 streams again (new code), compare with the kept old files
    proof.py decode    decode the new streams in THIS (fresh) process; compare with the encoder
    proof.py check     exit 0 only if every stream passed both (the chain stops otherwise)

Expected new file = old file with the two quantization-parameter counts (offsets 24-27) set to
0 and the 1,024 bytes of blocks after the 56-byte header removed. Nothing else may change.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "outputs/p12_audit2")
import bitstream as bs  # noqa: E402

O = Path("outputs/p13_audit_fixes")
OLD = Path("C:/audit2_scratch/streams")
NEW = Path("C:/audit2_scratch/streams_noq")
HEADER = bs.mc.TEMPORAL_HEADER_SIZE


def expected_from(old: bytes) -> bytes:
    return old[:24] + b"\0\0\0\0" + old[28:HEADER] + old[HEADER + 1024:]


def cmd_encode(device) -> int:
    NEW.mkdir(parents=True, exist_ok=True)
    audit = json.loads(Path("outputs/p12_audit2/a_encode.json").read_text(encoding="utf-8"))
    out, by_lam = {}, {}
    for tag, name, lam in bs.SAMPLE:
        by_lam.setdefault(lam, []).append((tag, name))
    for lam, items in by_lam.items():
        c = bs.Coders(lam, device)
        for tag, name in items:
            fname = f"{tag}_{name}_{lam}.nvct"
            frames = bs.sequence(tag, name).load_frames()
            enc = bs.pg.encode_sequence(bs.mc, bs.m21, c.intra, c.icoder, c.pcoder, frames,
                                        NEW / fname, c.rig, c.tables, gop=30,
                                        block_size=c.args.block_size,
                                        search_range=c.args.search_range, identity=c.identity,
                                        motion_lambda=bs.MOTION_LAMBDA[lam])
            old, new = (OLD / fname).read_bytes(), (NEW / fname).read_bytes()
            encoder_hash = bs.frames_hash(enc["reconstructions"])
            out[fname] = {"lambda": lam, "old_bytes": len(old), "new_bytes": len(new),
                          "saved": len(old) - len(new),
                          "equals_old_minus_blocks": new == expected_from(old),
                          "same_reconstruction_as_audit": encoder_hash
                          == audit[fname]["encoder_hash"],
                          "encoder_hash": encoder_hash}
            print(f"[encode] {fname}: {out[fname]}", flush=True)
            (O / "proof_encode.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        del c
        torch.cuda.empty_cache()
    return 0


def cmd_decode(device) -> int:
    enc = json.loads((O / "proof_encode.json").read_text(encoding="utf-8"))
    out, by_lam = {}, {}
    for fname, e in enc.items():
        by_lam.setdefault(e["lambda"], []).append(fname)
    for lam, names in by_lam.items():
        c = bs.Coders(lam, device)
        for fname in names:
            started = time.time()
            try:
                same = bs.frames_hash(c.decode(NEW / fname).cpu()) == enc[fname]["encoder_hash"]
                out[fname] = {"same": same, "decode_s": round(time.time() - started, 1)}
            except Exception as error:
                out[fname] = {"same": False, "error": f"{type(error).__name__}: {error}"}
            print(f"[decode] {fname}: {out[fname]}", flush=True)
            (O / "proof_decode.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        del c
    return 0


def cmd_check() -> int:
    enc = json.loads((O / "proof_encode.json").read_text(encoding="utf-8"))
    dec = json.loads((O / "proof_decode.json").read_text(encoding="utf-8"))
    ok = (len(enc) == len(bs.SAMPLE) == len(dec)
          and all(e["equals_old_minus_blocks"] and e["saved"] == 1024
                  and e["same_reconstruction_as_audit"] for e in enc.values())
          and all(d["same"] for d in dec.values()))
    print(f"PROOF 2: {'PASS' if ok else 'FAIL'} ({len(enc)} streams)")
    return 0 if ok else 1


def main() -> int:
    command = sys.argv[1]
    device = torch.device("cuda")
    if command == "encode":
        return cmd_encode(device)
    if command == "decode":
        return cmd_decode(device)
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
