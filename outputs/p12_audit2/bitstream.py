"""docs/PLAN_audit2.md, part A: the entropy coder and the bitstream.

    bitstream.py encode        A1/A3: code the sample, keep the .nvct files, write the hash of
                               every reconstruction and check the byte accounting
    bitstream.py decode        A1: decode every kept stream in THIS (new) process from the file
                               and the model files only; compare hashes
    bitstream.py decode --device cpu --only-davis
                               A6: the same on the CPU (expected to differ: a known limit)
    bitstream.py wrong         A2: decode with the wrong models; each must raise
    bitstream.py damage        A5: 20 truncated / bit-flipped streams, each in its own process
                               with a timeout

The sample, fixed in the plan: the first two DAVIS TEST videos (bmx-bumps, car-turn) at all
5 lambdas, and the first UVG video (Beauty) at 1e-2 and 1e-4. Production settings: wide
I-frames, GOP 30, bit-aware motion with the production motion lambdas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402
import p1_gate3 as g3  # noqa: E402
import p1_pframe as p1  # noqa: E402
import p4_gate3 as pg  # noqa: E402
import p4_wide_latent as p4  # noqa: E402
from nvc.evaluation.sequences import discover_sequences  # noqa: E402
from nvc.training.checkpoint import load_model_from_checkpoint  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402

O = Path("outputs/p12_audit2")
STREAMS = Path("C:/audit2_scratch/streams")
MANIFEST_UVG = Path("C:/uvg_scratch/uvg/manifest.json")
LOW_RATE_DIR = Path("outputs/p6_low_rate/vimeo")
MOTION_LAMBDA = {"1e-2": 0.25, "3e-3": 0.25, "1e-3": 0.125, "3e-4": 0.125, "1e-4": 0.0625}
SAMPLE = ([("davis", "bmx-bumps", lam) for lam in MOTION_LAMBDA]
          + [("davis", "car-turn", lam) for lam in MOTION_LAMBDA]
          + [("uvg", "Beauty", "1e-2"), ("uvg", "Beauty", "1e-4")])

mc = bp._load_script("m10h_motion_compensation")
m21 = bp._load_script("m21_refinement")
md = bp._load_script("m11_data")


def frames_hash(frames: torch.Tensor) -> str:
    return hashlib.sha256(frames.float().contiguous().numpy().tobytes()).hexdigest()


class Coders:
    """Everything p4_gate3.main builds for one lambda, built the same way."""

    def __init__(self, lam: str, device, *, i_lam: str | None = None, p_lam: str | None = None):
        defaults = load_default_config()
        self.args = bp.build_arg_parser(defaults).parse_args([])
        self.device = device
        rig_lam = lam if lam in g3.BITS else "1e-3"

        def models_dir(l):
            return p4.VIMEO_DIR if l in p1.LAMBDAS else LOW_RATE_DIR

        self.intra, _ = load_model_from_checkpoint(Path(p1.CHECKPOINT.format(rig_lam)), device=device)
        self.intra.eval()
        p_lam, i_lam = p_lam or lam, i_lam or lam
        self.pcoder = p4.load_vimeo("p", p_lam, device, models_dir(p_lam))
        self.icoder = p4.load_vimeo("i", i_lam, device, models_dir(i_lam))
        # The identity the ENCODER would declare for these coders.
        self.identity = pg.identity_of(self.pcoder, self.icoder)
        manifest = self.args.manifest
        train_full = discover_sequences(manifest, split="train")
        motion_train = m21.broad_train_sequences(
            manifest, frames_per_sequence=self.args.train_frames_per_sequence)
        self.rig = g3.build_rig(self.args, m21, md, self.intra, rig_lam, g3.BITS[rig_lam], device,
                                train_full, motion_train)
        self.tables = g3.cumulative_tables()

    def decode(self, path: Path, identity: bytes | None = None) -> torch.Tensor:
        return pg.decode_sequence(mc, m21, self.intra, self.icoder, self.pcoder, path, self.rig,
                                  self.tables, identity=identity or self.identity)


def sequence(tag: str, name: str):
    manifest = load_default_config().processed_data_dir / "manifest.json" if tag == "davis" \
        else MANIFEST_UVG
    for s in discover_sequences(manifest, split="test"):
        if s.sequence_id == name:
            return s
    raise KeyError(name)


def byte_accounting(path: Path) -> dict:
    data = path.read_bytes()
    reader = mc.TemporalStreamReader(path)
    h = reader.header
    quant = (h.num_intra_quantization_params + h.num_residual_quantization_params) * 8
    records = list(reader)
    motion = sum(len(m) for _, m, _ in records)
    residual = sum(len(r) for _, _, r in records)
    framing = mc.TEMPORAL_HEADER_SIZE + quant + 9 * len(records)
    return {"file": len(data), "header": mc.TEMPORAL_HEADER_SIZE, "quant_blocks": quant,
            "frame_records": 9 * len(records), "motion": motion, "residual": residual,
            "adds_up": framing + motion + residual == len(data),
            "unused_by_wide_i_frames": quant}


def cmd_encode(device) -> int:
    STREAMS.mkdir(parents=True, exist_ok=True)
    out = {}
    by_lam = {}
    for tag, name, lam in SAMPLE:
        by_lam.setdefault(lam, []).append((tag, name))
    for lam, items in by_lam.items():
        c = Coders(lam, device)
        for tag, name in items:
            seq = sequence(tag, name)
            frames = seq.load_frames()
            path = STREAMS / f"{tag}_{name}_{lam}.nvct"
            started = time.time()
            enc = pg.encode_sequence(mc, m21, c.intra, c.icoder, c.pcoder, frames, path, c.rig,
                                     c.tables, gop=30, block_size=c.args.block_size,
                                     search_range=c.args.search_range, identity=c.identity,
                                     motion_lambda=MOTION_LAMBDA[lam])
            records = enc["records"]
            acc = byte_accounting(path)
            acc["records_match_payloads"] = (
                sum(r["motion"] for r in records) == acc["motion"]
                and sum(r["residual"] for r in records) == acc["residual"])
            out[path.name] = {"tag": tag, "video": name, "lambda": lam, "frames": len(records),
                              "encoder_hash": frames_hash(enc["reconstructions"]),
                              "identity": c.identity.hex(), "bytes": acc,
                              "encode_s": round(time.time() - started, 1)}
            print(f"[encode] {path.name}: {acc['file']:,} B adds_up={acc['adds_up']} "
                  f"records_match={acc['records_match_payloads']}", flush=True)
            (O / "a_encode.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        del c
        torch.cuda.empty_cache()
    return 0


def cmd_decode(device, only_davis: bool, tag_out: str) -> int:
    enc = json.loads((O / "a_encode.json").read_text(encoding="utf-8"))
    out = {}
    by_lam = {}
    for fname, e in enc.items():
        if only_davis and e["tag"] != "davis":
            continue
        by_lam.setdefault(e["lambda"], []).append(fname)
    for lam, names in by_lam.items():
        c = Coders(lam, device)
        for fname in names:
            started = time.time()
            try:
                got = frames_hash(c.decode(STREAMS / fname).cpu())
                same = got == enc[fname]["encoder_hash"]
                out[fname] = {"same": same, "decode_s": round(time.time() - started, 1)}
            except Exception as error:   # report, don't stop
                out[fname] = {"same": False, "error": f"{type(error).__name__}: {error}"}
            print(f"[decode {device}] {fname}: {out[fname]}", flush=True)
            (O / f"{tag_out}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        del c
    print(f"all identical: {all(v['same'] for v in out.values())} ({len(out)} streams)")
    return 0


def cmd_wrong(device) -> int:
    """A2: one stream (DAVIS bmx-bumps, 1e-3), decoded with coders it was not made with."""
    path = STREAMS / "davis_bmx-bumps_1e-3.nvct"
    cases = {"right coders (control)": dict(),
             "P- and I-coder of 3e-4": dict(p_lam="3e-4", i_lam="3e-4"),
             "P-coder of 3e-4 only": dict(p_lam="3e-4"),
             "I-coder of 3e-4 only": dict(i_lam="3e-4"),
             "low-rate 3e-3 coders": dict(p_lam="3e-3", i_lam="3e-3")}
    out = {}
    for label, kw in cases.items():
        c = Coders("1e-3", device, **kw)
        try:
            frames = c.decode(path)
            out[label] = {"raised": None, "returned_frames": tuple(frames.shape)}
        except Exception as error:
            out[label] = {"raised": type(error).__name__, "message": str(error)[:200]}
        print(f"[wrong] {label}: {out[label]}", flush=True)
    # The rig (intra/motion entropy models) of another lambda, right learned coders.
    c = Coders("3e-4", device, p_lam="1e-3", i_lam="1e-3")
    try:
        c.decode(path)
        out["rig of 3e-4, right coders"] = {"raised": None}
    except Exception as error:
        out["rig of 3e-4, right coders"] = {"raised": type(error).__name__,
                                            "message": str(error)[:200]}
    print(f"[wrong] rig of 3e-4: {out['rig of 3e-4, right coders']}", flush=True)
    out["identity_covers"] = ("sha256 over every tensor of the P-coder and I-coder state_dicts, "
                              "first 8 bytes; intra and motion entropy models by their own ids")
    (O / "a_wrong.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def damage_cases() -> list[dict]:
    rng = random.Random(20261123)
    davis = sorted(p.name for p in STREAMS.glob("davis_*.nvct"))
    cases = []
    for i in range(20):
        name = davis[i % len(davis)]
        size = (STREAMS / name).stat().st_size
        if i < 10:
            cases.append({"id": i, "stream": name, "kind": "truncate",
                          "at": rng.randrange(1, size)})
        else:
            cases.append({"id": i, "stream": name, "kind": "bitflip",
                          "at": rng.randrange(mc.TEMPORAL_HEADER_SIZE, size),
                          "bit": rng.randrange(8)})
    return cases


def cmd_damage_one(device, case_id: int) -> int:
    case = damage_cases()[case_id]
    enc = json.loads((O / "a_encode.json").read_text(encoding="utf-8"))[case["stream"]]
    data = bytearray((STREAMS / case["stream"]).read_bytes())
    if case["kind"] == "truncate":
        data = data[:case["at"]]
    else:
        data[case["at"]] ^= 1 << case["bit"]
    path = STREAMS / f"damaged_{case_id}.nvct"
    path.write_bytes(bytes(data))
    c = Coders(enc["lambda"], device)
    try:
        frames = c.decode(path).cpu()
        same = frames_hash(frames) == enc["encoder_hash"]
        finite = bool(torch.isfinite(frames).all())
        result = {"outcome": "returned frames", "identical_to_original": same, "finite": finite}
    except Exception as error:
        result = {"outcome": "raised", "error": f"{type(error).__name__}: {str(error)[:160]}"}
    path.unlink(missing_ok=True)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


def cmd_damage(device) -> int:
    out = {}
    for case in damage_cases():
        started = time.time()
        try:
            run = subprocess.run([sys.executable, "-u", __file__, "damage-one", "--case",
                                  str(case["id"]), "--device", str(device)],
                                 capture_output=True, text=True, timeout=900)
            line = [l for l in run.stdout.splitlines() if l.startswith("RESULT ")]
            result = json.loads(line[-1][7:]) if line else {
                "outcome": "process died", "returncode": run.returncode,
                "stderr": run.stderr[-300:]}
        except subprocess.TimeoutExpired:
            result = {"outcome": "hang (killed after 900 s)"}
        result.update(case, seconds=round(time.time() - started))
        out[str(case["id"])] = result
        print(f"[damage] {result}", flush=True)
        (O / "a_damage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["encode", "decode", "wrong", "damage", "damage-one"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-davis", action="store_true")
    parser.add_argument("--case", type=int)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    if args.command == "encode":
        return cmd_encode(device)
    if args.command == "decode":
        return cmd_decode(device, args.only_davis, args.out or f"a_decode_{args.device}")
    if args.command == "wrong":
        return cmd_wrong(device)
    if args.command == "damage":
        return cmd_damage(device)
    return cmd_damage_one(device, args.case)


if __name__ == "__main__":
    sys.exit(main())
