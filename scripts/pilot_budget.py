"""The 10,000-MAC/pixel decoder pilot (docs/PLAN_budget_pilot.md).

    pilot_budget.py macs
        Decoder and encoder MAC per pixel at 1920x1088, I and P (the budget test's numbers).
    pilot_budget.py bench [--stage 1|2] [--batch N]
        Training steps per second on random data (to size the schedule, before any data).
    pilot_budget.py train --out DIR --chunks A B [--scratch-dir DIR]
        The fixed schedule in configs/pilot_schedule.json: phases of (chunk, stage, GPU hours),
        72 GPU hours at most. Resumes from DIR/latest.pt. Each Vimeo chunk is downloaded when
        its first phase starts and deleted after its last.
    pilot_budget.py eval --checkpoint C --manifest M --out R.json
        The development set under the fair protocol: GOP 30, P-only, RGB in and out, scored in
        YUV against the original; rate = ideal code length.
    pilot_budget.py ratecheck --checkpoint C --manifest M --out K.json
        The first video at two levels through the real range coder: real size vs ideal.
    pilot_budget.py speed --checkpoint C
        Decoder frames per second at 1080p on the GPU (fp16, decoder networks only).
    pilot_budget.py verdict --manifest M --anchors S.json --results R.json --ratecheck K.json
        The plan's fixed verdict: mean per-video BD-rate (YUV PSNR, PCHIP) against x264
        low-delay -tune psnr. GO <= +10%, GRAY +10..+40%, STOP > +40%. Counts only if the
        budget test passes and the rate check holds (or the rates are rescaled).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nvc.evaluation import fair  # noqa: E402
from nvc.pilot.model import (PilotCodec, decoder_macs_per_pixel,  # noqa: E402
                             encoder_macs_per_pixel, load_config, pad_to)

SCHEDULE_PATH = Path(__file__).resolve().parents[1] / "configs" / "pilot_schedule.json"
EVAL_LEVELS = (0, 9, 18, 27, 36, 45, 54, 63)     # fixed before any result
RATECHECK_LEVELS = (18, 45)
GOP = 30
CAP_HOURS = 72.0


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def load_schedule(path: Path = SCHEDULE_PATH) -> dict[str, Any]:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    total = sum(p["hours"] for p in schedule["phases"])
    if total > CAP_HOURS:
        raise SystemExit(f"schedule asks for {total} GPU hours; the plan's cap is {CAP_HOURS}")
    return schedule


# --------------------------------------------------------------------------------------------
# macs, bench
# --------------------------------------------------------------------------------------------


def cmd_macs(args) -> int:
    model = PilotCodec().eval()
    out = {"parameters": sum(p.numel() for p in model.parameters()),
           "config": model.config}
    for kind, intra in (("i", True), ("p", False)):
        out[f"decoder_{kind}"] = decoder_macs_per_pixel(model, intra=intra)
        out[f"encoder_{kind}"] = encoder_macs_per_pixel(model, intra=intra)
    print(json.dumps(out, indent=1))
    if args.out:
        write_json(args.out, out)
    return 0


def training_step(model, optimizer, clip: torch.Tensor, scaler_dtype=torch.bfloat16
                  ) -> dict[str, float]:
    """One step on clip [N, T, 3, H, W] in [0, 1]: frame 0 an I-frame (empty context), the
    rest P-frames on the model's own decoded features; a random level per sample. Loss =
    mean over frames of MSE + lambda * bpp."""
    n, frames, _, h, w = clip.shape
    levels = torch.randint(0, model.config["levels"], (n,), device=clip.device)
    lam = model.lambdas[levels]
    optimizer.zero_grad(set_to_none=True)
    total, mses, bpps = 0.0, [], []
    with torch.autocast("cuda", dtype=scaler_dtype):
        features = None
        for t in range(frames):
            x_hat, features, bits = model(clip[:, t], features, levels)
            mse = ((x_hat.float() - clip[:, t]) ** 2).mean((1, 2, 3))
            bpp = bits / (h * w)
            total = total + (mse + lam * bpp).mean()
            mses.append(mse.detach())
            bpps.append(bpp.detach())
    loss = total / frames
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    mse = torch.stack(mses).mean(0)
    return {"loss": float(loss), "mse": float(mse.mean()), "bpp": float(torch.stack(bpps).mean()),
            "psnr_low": float((10 * torch.log10(1 / mse[levels < 16])).mean())
            if (levels < 16).any() else float("nan"),
            "psnr_high": float((10 * torch.log10(1 / mse[levels >= 48])).mean())
            if (levels >= 48).any() else float("nan")}


def cmd_bench(args) -> int:
    device = torch.device("cuda")
    model = PilotCodec().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    frames = 2 if args.stage == 1 else 4
    clip = torch.rand(args.batch, frames, 3, 256, 256, device=device)
    for _ in range(5):
        training_step(model, optimizer, clip)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        training_step(model, optimizer, clip)
    torch.cuda.synchronize()
    rate = args.steps / (time.perf_counter() - started)
    print(f"stage {args.stage}, batch {args.batch}: {rate:.2f} steps/s, "
          f"{rate * 3600:.0f} steps/h, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0


# --------------------------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------------------------


def cmd_train(args) -> int:
    import p1_pframe as p1
    chunks_mod = p1._load_script("train_vimeo_qat_combined")
    schedule = load_schedule()
    device = torch.device("cuda")
    torch.manual_seed(schedule.get("seed", 0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"
    model = PilotCodec().to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=schedule["lr"])
    state = {"phase": 0, "phase_seconds": 0.0, "step": 0, "history": []}
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        print(f"[resume] phase {state['phase']}, {state['phase_seconds'] / 3600:.2f} h into it, "
              f"step {state['step']}", flush=True)
    write_json(out / "schedule_used.json", {"schedule": schedule, "chunks": args.chunks,
                                            "config": model.config})
    phases = schedule["phases"]
    last_use = {p["chunk_slot"]: i for i, p in enumerate(phases)}

    def checkpoint() -> None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state, "config": model.config}, out / "latest.tmp")
        for attempt in range(30):   # OneDrive or antivirus can briefly lock the file
            try:
                (out / "latest.tmp").replace(latest)
                return
            except PermissionError:
                time.sleep(2)
        raise RuntimeError("could not replace latest.pt")

    while state["phase"] < len(phases):
        phase = phases[state["phase"]]
        chunk = args.chunks[phase["chunk_slot"]]
        scratch = Path(args.scratch_dir) / f"chunk_{chunk}"
        root = next((p.parent.parent.parent for p in scratch.rglob("im1.png")), None) \
            if scratch.is_dir() else None
        if root is None:
            root = chunks_mod.download_and_extract_chunk(
                chunk, scratch, dataset_owner=chunks_mod.KAGGLE_DATASET_OWNER,
                dataset_prefix=chunks_mod.KAGGLE_DATASET_PREFIX)
        clip_dirs = p1.complete_clip_dirs(root)
        frames = 2 if phase["stage"] == 1 else 4
        loader = torch.utils.data.DataLoader(
            p1.VimeoClips(clip_dirs, frames, schedule["crop"]), batch_size=phase["batch"],
            shuffle=True, num_workers=args.num_workers, drop_last=True,
            persistent_workers=args.num_workers > 0)
        budget = phase["hours"] * 3600
        print(f"[phase {state['phase']}] chunk {chunk} ({len(clip_dirs)} clips), stage "
              f"{phase['stage']}, batch {phase['batch']}, {phase['hours']} h", flush=True)
        last_save, window, seen = time.time(), {}, 0
        tick = time.time()
        while state["phase_seconds"] < budget:
            for clip in loader:
                lr = schedule["lr"] * (schedule["lr_final_factor"]
                                       if state["phase_seconds"] >= phase.get("decay_after_hours",
                                                                             1e9) * 3600 else 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                clip = clip.to(device, non_blocking=True).float() / 255
                stats = training_step(model, optimizer, clip)
                if not math.isfinite(stats["loss"]):
                    raise RuntimeError(f"non-finite loss at step {state['step']}")
                now = time.time()
                state["phase_seconds"] += now - tick
                tick = now
                state["step"] += 1
                seen += 1
                for k, v in stats.items():
                    if math.isfinite(v):
                        window[k] = window.get(k, 0.0) + v
                        window[k + "_n"] = window.get(k + "_n", 0) + 1
                if state["step"] % args.log_every == 0:
                    entry = {k: window[k] / window[k + "_n"] for k in stats if k + "_n" in window}
                    entry.update(step=state["step"], phase=state["phase"], lr=lr,
                                 hours=state["phase_seconds"] / 3600)
                    state["history"].append(entry)
                    print(f"  step {state['step']:7d}  loss {entry['loss']:.5f}  "
                          f"bpp {entry['bpp']:.4f}  PSNR low {entry.get('psnr_low', 0):.2f} / "
                          f"high {entry.get('psnr_high', 0):.2f}  {entry['hours']:.2f} h",
                          flush=True)
                    window = {}
                if now - last_save > schedule["checkpoint_minutes"] * 60:
                    checkpoint()
                    last_save = time.time()
                    tick = time.time()       # saving time is not training time
                if state["phase_seconds"] >= budget:
                    break
        state["phase"] += 1
        state["phase_seconds"] = 0.0
        checkpoint()
        torch.save({"model": model.state_dict(), "config": model.config,
                    "phase_done": state["phase"], "step": state["step"]},
                   out / f"phase{state['phase'] - 1}.pt")
        if last_use[phase["chunk_slot"]] == state["phase"] - 1:
            print(f"[chunk {chunk}] last phase done; deleting {scratch}", flush=True)
            shutil.rmtree(scratch, ignore_errors=True)
    torch.save({"model": model.state_dict(), "config": model.config, "step": state["step"]},
               out / "model.pt")
    print(f"training complete: {state['step']} steps", flush=True)
    return 0


# --------------------------------------------------------------------------------------------
# eval, ratecheck, speed
# --------------------------------------------------------------------------------------------


def load_model(path: Path, device) -> PilotCodec:
    saved = torch.load(path, map_location=device, weights_only=False)
    model = PilotCodec(saved["config"]).to(device).eval()
    model.load_state_dict(saved["model"])
    return model


def video_frames(video: fair.YuvVideo, raw: np.ndarray) -> torch.Tensor:
    rgb = fair.yuv_to_rgb(raw, video.width, video.height)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float() / 255.0


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    return (torch.round(frames.clamp(0, 1) * 255).to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous().cpu().numpy())


@torch.no_grad()
def code_video(model, frames: torch.Tensor, level: int, device, coder=None):
    """GOP 30, P-only. Returns (reconstruction [N,3,H,W] cropped, ideal bits, real bytes or
    None, per-frame records)."""
    h, w = frames.shape[-2:]
    recon, records, bits, real = [], [], 0.0, 0
    features = None
    for t in range(frames.shape[0]):
        if t % GOP == 0:
            features = None
        frame = pad_to(frames[t:t + 1].to(device))
        coded = model.encode(frame, features, level)
        ideal = model.ideal_bits(coded)
        record = {"type": "I" if t % GOP == 0 else "P", "ideal_bits": ideal}
        if coder is not None:
            payload = coder.pack(coded)
            record["bytes"] = len(payload)
            real += len(payload)
            decoded, _ = model.decode(coder.reader(payload), features, level, frame.shape)
            record["decode_exact"] = bool(torch.equal(decoded, coded["x_hat"]))
        features = coded["features"]
        bits += ideal
        records.append(record)
        recon.append(coded["x_hat"][..., :h, :w].cpu())
    return torch.cat(recon), bits, (real if coder is not None else None), records


def cmd_eval(args) -> int:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    results = (json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file()
               else {"codec": f"pilot ({args.checkpoint})", "rate": "ideal code length",
                     "videos": {}})
    for video in fair.load_manifest(args.manifest):
        raw = fair.read_yuv(video)
        frames = video_frames(video, raw)
        for level in args.levels:
            label = f"level{level}"
            if any(p["label"] == label for p in results["videos"].get(video.name, [])):
                continue
            started = time.perf_counter()
            recon, bits, _, records = code_video(model, frames, level, device)
            yuv = fair.rgb_to_yuv420(to_uint8(recon))
            point = fair.rate_point(bits / 8, video)
            point.update(fair.score_yuv(raw, yuv, video.width, video.height))
            point.update({"label": label, "level": level, "ideal_bits": bits,
                          "lambda": float(model.lambdas[level]),
                          "i_bits": sum(r["ideal_bits"] for r in records if r["type"] == "I"),
                          "seconds": time.perf_counter() - started})
            results["videos"].setdefault(video.name, []).append(point)
            results["videos"][video.name].sort(key=lambda p: p["level"])
            write_json(args.out, results)
            print(f"[{video.name}] level {level:2d}: {point['bpp']:.4f} bpp  "
                  f"PSNR_YUV {point['psnr_yuv']:.3f}  Y {point['psnr_y']:.3f}", flush=True)
    return 0


class RangeCoder:
    """The current codec's range coder and Laplace tables (p1_gate3 / p4_gate3), for the
    rate check. A frame's payload: [4 B z length][z][4 B anchor length][anchors][rest]."""

    def __init__(self) -> None:
        import p1_gate3 as g3
        import p4_gate3 as pg
        self.g3, self.pg, self.tables = g3, pg, g3.cumulative_tables()

    def pack(self, coded) -> bytes:
        anchor = coded["anchor"].expand_as(coded["sym_a"])
        z, _ = self.pg._code(coded["z"], coded["z_scale"], self.tables)
        a, _ = self.pg._code(coded["sym_a"][anchor], coded["scale_a"][anchor], self.tables)
        b, _ = self.pg._code(coded["sym_b"][~anchor], coded["scale_b"][~anchor], self.tables)
        return struct.pack("<I", len(z)) + z + struct.pack("<I", len(a)) + a + b

    def reader(self, payload: bytes):
        (zl,) = struct.unpack_from("<I", payload, 0)
        z = payload[4:4 + zl]
        (al,) = struct.unpack_from("<I", payload, 4 + zl)
        a = payload[8 + zl:8 + zl + al]
        b = payload[8 + zl + al:]
        parts = {"z": z, "a": a, "b": b}

        def read(kind, scale, mask):
            if mask is None:
                return self.pg._decode(parts[kind], scale, scale, self.tables)
            m = mask.expand_as(scale)
            out = torch.zeros_like(scale)
            out[m] = self.pg._decode(parts[kind], scale[m], scale[m], self.tables)
            return out
        return read


def cmd_ratecheck(args) -> int:
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    video = fair.load_manifest(args.manifest)[0]
    frames = video_frames(video, fair.read_yuv(video))
    coder, out = RangeCoder(), {"video": video.name, "levels": {}}
    for level in RATECHECK_LEVELS:
        _, bits, real, records = code_video(model, frames, level, device, coder)
        ratio = real * 8 / bits
        out["levels"][str(level)] = {"ideal_bits": bits, "real_bytes": real, "ratio": ratio,
                                     "decode_exact": all(r["decode_exact"] for r in records)}
        print(f"[ratecheck] {video.name} level {level}: real/ideal = {ratio:.4f}  "
              f"decode-exact={out['levels'][str(level)]['decode_exact']}", flush=True)
    ratios = [v["ratio"] for v in out["levels"].values()]
    out["within_1_percent"] = all(abs(r - 1) <= 0.01 for r in ratios)
    out["scale_factor"] = 1.0 if out["within_1_percent"] else float(np.mean(ratios))
    write_json(args.out, out)
    print(f"within 1%: {out['within_1_percent']}  (scale factor {out['scale_factor']:.4f})")
    return 0


@torch.no_grad()
def cmd_speed(args) -> int:
    device = torch.device("cuda")
    model = load_model(args.checkpoint, device).half()
    h, w = 1088, 1920
    features = torch.zeros(1, model.channels, h // 8, w // 8, device=device, dtype=torch.half)

    def read(kind, scale, mask):
        return torch.zeros_like(scale)

    for _ in range(5):
        model.decode(read, features, 32, (1, 3, h, w))
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.frames):
        model.decode(read, features, 32, (1, 3, h, w))
    torch.cuda.synchronize()
    fps = args.frames / (time.perf_counter() - started)
    out = {"gpu": torch.cuda.get_device_name(0), "size": [w, h], "precision": "fp16",
           "decoder_networks_fps": fps}
    print(json.dumps(out, indent=1))
    if args.out:
        write_json(args.out, out)
    return 0


def cmd_verdict(args) -> int:
    import subprocess

    import fair_eval as fe
    budget = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_pilot_budget.py",
                             "-k", "budget"], capture_output=True, text=True)
    check = json.loads(args.ratecheck.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    factor = check["scale_factor"]
    if factor != 1.0:          # the plan: rescale every pilot rate by the measured ratio
        for points in results["videos"].values():
            for point in points:
                point["bpp"] *= factor
    videos = fair.load_manifest(args.manifest)
    store = fair.AnchorStore.open(args.anchors)
    report = fe.build_report(videos, store, results, list(fair.HEADLINE_ARMS))
    score = report["arms"]["x264_lowdelay"]["full_range"]["mean"]
    if score is None:
        verdict = "NO SCORE"
    elif score <= 10.0:
        verdict = "GO"
    elif score <= 40.0:
        verdict = "GRAY"
    else:
        verdict = "STOP"
    out = {"verdict": verdict, "score_vs_x264_lowdelay": score,
           "budget_test_passed": budget.returncode == 0, "rate_check": check,
           "rates_rescaled_by": factor,
           "coverage_ok": report["arms"]["x264_lowdelay"]["coverage_ok"],
           "valid": budget.returncode == 0, "report": report}
    write_json(args.out, out)
    print(f"VERDICT: {verdict}  (mean per-video BD-rate vs x264 low-delay: "
          f"{'n/a' if score is None else f'{score:+.1f}%'}; budget test "
          f"{'passed' if out['valid'] else 'FAILED - result not valid'}; rate check "
          f"{'within 1%' if check['within_1_percent'] else f'rescaled x{factor:.4f}'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("macs")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_macs)
    p = sub.add_parser("bench")
    p.add_argument("--stage", type=int, default=1, choices=(1, 2))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--steps", type=int, default=30)
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("train")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chunks", type=int, nargs="+", required=True)
    p.add_argument("--scratch-dir", type=Path, default=Path("C:/vimeo_scratch"))
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=500)
    p.set_defaults(func=cmd_train)
    for name, func in (("eval", cmd_eval), ("ratecheck", cmd_ratecheck)):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cuda")
        if name == "eval":
            p.add_argument("--levels", type=int, nargs="+", default=list(EVAL_LEVELS))
        p.set_defaults(func=func)
    p = sub.add_parser("verdict")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--ratecheck", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_verdict)
    p = sub.add_parser("speed")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frames", type=int, default=50)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_speed)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
