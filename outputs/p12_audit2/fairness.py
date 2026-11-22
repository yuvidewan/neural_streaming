"""docs/PLAN_audit2.md, part B (B1, B3, B4): is the x264/x265 comparison fair?

    fairness.py b1     lossless x264 (-qp 0) through the SAME rgb24 -> YUV -> rgb24 pipe the
                       anchors use, 4:2:0 and 4:4:4, on DAVIS TEST, UVG and MCL-JCV: the
                       quality ceiling of the anchor pipe in our RGB metric
    fairness.py b3     (a) the real frame structure of every anchor arm (ffprobe); (b) the
                       fixed 30 fps against the real fps, on one UVG and one MCL-JCV video
    fairness.py b4     the anchors and our codec are scored on the same frames

B1 rule (fixed in the plan): material if the 4:4:4 lossless PSNR is below 50 dB on any set.
B3 rule: the fps effect is material only if a BD-rate moves by 1 point or more.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "scripts")
import benchmark_parity as bp  # noqa: E402
from nvc.evaluation.sequences import discover_sequences  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402

O = Path("outputs/p12_audit2")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFESTS = {"davis": load_default_config().processed_data_dir / "manifest.json",
             "uvg": Path("C:/uvg_scratch/uvg/manifest.json"),
             "mcljcv": Path("C:/mcljcv_scratch/mcljcv/manifest.json")}
UVG_FPS = {"Beauty": 120}   # UVG (Mercat et al., MMSys 2020): the 1080p clips are 120 fps


class LosslessArm(bp.ClassicalArm):
    __slots__ = ()

    def encoder_arguments(self, crf: int) -> list[str]:
        return ["-c:v", "libx264", "-qp", "0", "-preset", "medium", "-pix_fmt", self.pix_fmt]


def sequences(tag: str):
    return discover_sequences(MANIFESTS[tag], split="test")


def score(source: np.ndarray, frames: torch.Tensor, decoded: np.ndarray) -> dict:
    s = bp.score_frames(bp.uint8_to_frames(decoded).to(DEVICE), frames.to(DEVICE), device=DEVICE)
    return {"psnr": float(np.mean(s["psnr"])), "msssim": float(np.mean(s["msssim"])),
            "min_frame_psnr": float(np.min(s["psnr"]))}


def cmd_b1() -> int:
    out_path = O / "b1_lossless.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {}
    work = Path(tempfile.mkdtemp(prefix="audit2_b1_"))
    for tag in MANIFESTS:
        for seq in sequences(tag):
            key = f"{tag}/{seq.sequence_id}"
            if key in out:
                continue
            frames = seq.load_frames()
            source = bp.frames_to_uint8(frames)
            row = {}
            for fmt in ("yuv420p", "yuv444p"):
                result = bp.ffmpeg_round_trip(source, LosslessArm("lossless", "libx264",
                                                                  lowdelay=False, gop=30,
                                                                  pix_fmt=fmt), 0, work)
                row[fmt] = score(source, frames, result["decoded"])
                row[fmt]["bpp"] = result["bytes"] * 8 / source[..., 0].size
            out[key] = row
            print(f"[b1] {key}: 420 {row['yuv420p']['psnr']:.2f} dB, 444 {row['yuv444p']['psnr']:.2f} dB",
                  flush=True)
            out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    summary = {}
    for tag in MANIFESTS:
        rows = [v for k, v in out.items() if k.startswith(tag + "/") and k != "summary"]
        summary[tag] = {fmt: {"sequence_mean_psnr": float(np.mean([r[fmt]["psnr"] for r in rows])),
                              "lowest_video_psnr": float(np.min([r[fmt]["psnr"] for r in rows])),
                              "sequence_mean_msssim": float(np.mean([r[fmt]["msssim"] for r in rows]))}
                        for fmt in ("yuv420p", "yuv444p")}
    out["summary"] = summary
    out["material"] = any(s["yuv444p"]["lowest_video_psnr"] < 50 for s in summary.values())
    out["conversion"] = probe_conversion(work)
    out_path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("summary", "material", "conversion")}, indent=1))
    return 0


def probe_conversion(work: Path) -> dict:
    """What ffmpeg writes into the anchor streams about the colour matrix and range."""
    src = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    video = work / "probe.mp4"
    subprocess.run([bp.find_ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                    "rgb24", "-s", "64x64", "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(video)], input=src.tobytes(), check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=pix_fmt,color_range,color_space,color_transfer,color_primaries",
                            "-of", "json", str(video)], capture_output=True, text=True, check=True)
    return json.loads(probe.stdout)["streams"][0]


def frame_types(source: np.ndarray, arm: bp.ClassicalArm, crf: int, fps: int, work: Path) -> list[str]:
    count, h, w, _ = source.shape
    video = work / f"{arm.name}.mp4"
    subprocess.run([bp.find_ffmpeg(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                    "rgb24", "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "pipe:0",
                    *arm.encoder_arguments(crf), str(video)], input=source.tobytes(), check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "frame=pict_type", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, check=True)
    return [line.strip().strip(",") for line in probe.stdout.split() if line.strip()]


def curve(source, frames, arm, crfs, fps, work) -> list[tuple[float, float]]:
    old = bp.FRAMERATE
    bp.FRAMERATE = fps
    try:
        points = []
        for crf in crfs:
            r = bp.ffmpeg_round_trip(source, arm, crf, work)
            s = score(source, frames, r["decoded"])
            points.append((r["bytes"] * 8 / source[..., 0].size, s["psnr"]))
        return points
    finally:
        bp.FRAMERATE = old


def cmd_b3() -> int:
    work = Path(tempfile.mkdtemp(prefix="audit2_b3_"))
    out = {"structure": {}, "fps": {}}
    seq = sequences("davis")[0]
    source = bp.frames_to_uint8(seq.load_frames())
    for fmt in ("yuv420p", "yuv444p"):
        for arm in bp.classical_arms(30, fmt):
            types = frame_types(source, arm, 28, 30, work)
            out["structure"][arm.name] = {
                "frames": len(types), "i_positions": [i for i, t in enumerate(types) if t == "I"],
                "b_frames": types.count("B"), "lowdelay": arm.lowdelay,
                "described_as": arm.describe()["structure"]}
            print(f"[b3] {arm.name}: {out['structure'][arm.name]}", flush=True)
    mcl = json.loads(MANIFESTS["mcljcv"].read_text(encoding="utf-8"))["items"]
    mcl_pick = sorted((i for i in mcl if i["fps"] != 30), key=lambda i: i["source_name"])[0]
    picks = [("uvg", "Beauty", UVG_FPS["Beauty"]), ("mcljcv", mcl_pick["source_name"], mcl_pick["fps"])]
    for tag, name, fps in picks:
        seq = next(s for s in sequences(tag) if s.sequence_id == name)
        frames = seq.load_frames()
        source = bp.frames_to_uint8(frames)
        for arm in (bp.classical_arms(30, "yuv420p")[0], bp.classical_arms(30, "yuv444p")[0]):
            fixed = curve(source, frames, arm, (22, 30, 38), 30, work)
            real = curve(source, frames, arm, (22, 30, 38), fps, work)
            bd = bp.bd_rate(fixed, real)
            out["fps"][f"{tag}/{name}/{arm.name}"] = {
                "real_fps": fps, "points_30fps": fixed, "points_real_fps": real,
                "bd_rate_real_vs_30": bd, "material": bd is not None and abs(bd) >= 1.0}
            print(f"[b3] {tag}/{name} {arm.name} at {fps} fps: BD {bd}", flush=True)
    (O / "b3_anchor_settings.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


def cmd_b4() -> int:
    out = {}
    pairs = {"davis": ("outputs/benchmarks/parity_classical_444_gop30/parity_classical_444_gop30.json",
                       "outputs/p8_motion/test_motion.json"),
             "uvg": ("outputs/benchmarks/uvg_classical_444_gop30/parity.json",
                     "outputs/p8_motion/uvg_motion.json"),
             "mcljcv": ("outputs/benchmarks/mcljcv_classical_444_gop30/parity.json",
                        "outputs/p11_mcljcv/mcljcv_mse.json")}
    for tag, (anchor_path, codec_path) in pairs.items():
        anchor = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
        codec = json.loads(Path(codec_path).read_text(encoding="utf-8"))

        def frames_of(report):
            p = next(iter(v for v in report["points"].values()
                          if v.get("arm", "").startswith(("h264", "learned"))))
            # pixels per video, derived from the stored bytes and bpp (no pixel field is stored)
            return {s["sequence"]: round(s["bytes"] * 8 / s["bpp"]) for s in p["per_sequence"]}

        a, c = frames_of(anchor), frames_of(codec)
        seq = sequences(tag)[0]
        frames = seq.load_frames()
        via_loader = hashlib.sha256(bp.frames_to_uint8(frames).tobytes()).hexdigest()
        direct = hashlib.sha256(np.stack([np.asarray(Image.open(p).convert("RGB"))
                                          for p in seq.frame_paths]).tobytes()).hexdigest()
        out[tag] = {"same_videos": sorted(a) == sorted(c),
                    "same_pixel_counts": a == c,
                    "loader_equals_png_on_disk": via_loader == direct,
                    "checked_video": seq.sequence_id,
                    "anchor_report": anchor_path, "codec_report": codec_path}
        print(f"[b4] {tag}: {out[tag]}", flush=True)
    out["scoring"] = ("both paths call benchmark_parity.score_frames with the frames from "
                      "Sequence.load_frames() as the reference (p4_gate3.main and "
                      "run_classical_by_sequence / the per-point path)")
    (O / "b4_same_frames.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit({"b1": cmd_b1, "b3": cmd_b3, "b4": cmd_b4}[sys.argv[1]]())
