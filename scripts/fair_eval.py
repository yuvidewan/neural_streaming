"""The fair evaluation of docs/PLAN_fair_eval.md (protocol in `nvc.evaluation.fair`).

Every step is resumable and writes JSON. One heavy job at a time.

    fair_eval.py prepare-uvg4k --archives DIR --output DIR [--names ...]
        The budget pilot's development set (docs/PLAN_budget_pilot.md): from each UVG 4K
        archive, the first 96 frames, downscaled 3840x2160 -> 1920x1080 in 4:2:0 with lanczos,
        kept as YUV. Writes DIR/manifest.json. The extracted 4K YUV is deleted; the archive too
        unless --keep-archives.
    fair_eval.py anchors --manifest M --store S.json [--arms ...] [--crf ...] [--cover R.json]
        x264/x265 on the original YUV, raw elementary streams, YUV scoring. --cover adds CRFs
        in steps of 2 until every arm covers the test codec's quality range (coverage rule).
    fair_eval.py proof3 --manifest M --video NAME
        Lossless x264 (-qp 0) and x265 (lossless=1) must decode to the source byte for byte.
    fair_eval.py ceiling --manifest M --out C.json
        Proof 4: YUV -> RGB -> YUV with no coding, per video and plane (an RGB codec's ceiling).
    fair_eval.py ours --manifest M --out R.json [--lambdas ...]
        The current codec as shipped (MSE set, GOP 30, wide I-frames, bit-aware motion), fed
        RGB converted from the YUV, its output converted back and scored against the YUV.
    fair_eval.py report --manifest M --store S.json --test R.json [--arms ...] [--out F.json]
        Per-video PCHIP BD-rate (with the linear check), full range and streaming band, and
        the mean / median / wins per arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nvc.evaluation import fair  # noqa: E402

UVG4K_DEV = ("CityAlley", "FlowerKids", "RaceNight", "SunBath")   # fixed in the pilot plan
DOWNSCALE_FILTER = "scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int"
MOTION_LAMBDA = {"1e-2": 0.25, "3e-3": 0.25, "1e-3": 0.125, "3e-4": 0.125, "1e-4": 0.0625}
OUR_LAMBDAS = ("1e-2", "3e-3", "1e-3", "3e-4", "1e-4")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def environment() -> dict[str, Any]:
    return {"ffmpeg": fair.ffmpeg_version(), "cpu_threads": os.cpu_count(),
            "machine": platform.processor(), "python": platform.python_version()}


# --------------------------------------------------------------------------------------------
# prepare-uvg4k
# --------------------------------------------------------------------------------------------


def cmd_prepare_uvg4k(args) -> int:
    import py7zr
    args.output.mkdir(parents=True, exist_ok=True)
    videos = []
    for name in args.names:
        target = args.output / f"{name}_1920x1080_{args.fps}fps_{args.frames}f.yuv"
        need = 1920 * 1080 * 3 // 2 * args.frames
        if not (target.is_file() and target.stat().st_size == need):
            found = sorted(args.archives.glob(f"{name}_*8bit*.7z")) or sorted(
                args.archives.glob(f"{name}.7z"))
            if not found:
                raise SystemExit(f"{name}: no 8-bit archive in {args.archives}")
            archive = found[0]
            work = args.output / "_4k" / name
            work.mkdir(parents=True, exist_ok=True)
            print(f"[{name}] extracting {archive}", flush=True)
            with py7zr.SevenZipFile(archive) as handle:
                handle.extractall(work)
            candidates = [p for p in work.rglob("*.yuv") if "10bit" not in p.name.lower()]
            if not candidates:
                raise SystemExit(f"{name}: no 8-bit .yuv in {archive}")
            source = max(candidates, key=lambda p: p.stat().st_size)
            print(f"[{name}] {source.name} -> {target.name}", flush=True)
            subprocess.run(
                [fair.ffmpeg_binary(), "-hide_banner", "-v", "error", "-y", "-f", "rawvideo",
                 "-pix_fmt", "yuv420p", "-s", "3840x2160", "-r", str(args.fps), "-i", str(source),
                 "-frames:v", str(args.frames), "-vf", DOWNSCALE_FILTER, "-pix_fmt", "yuv420p",
                 "-f", "rawvideo", str(target)], check=True)
            shutil.rmtree(work)
            if not args.keep_archives:
                archive.unlink()
        if target.stat().st_size != need:
            raise SystemExit(f"{name}: {target} has {target.stat().st_size} bytes, need {need}")
        videos.append(fair.YuvVideo(name, Path(target.name), 1920, 1080, float(args.fps),
                                    args.frames).to_json())
        print(f"[{name}] ready", flush=True)
    shutil.rmtree(args.output / "_4k", ignore_errors=True)
    write_json(args.output / "manifest.json", {
        "set": "uvg4k_dev", "plan": "docs/PLAN_budget_pilot.md",
        "preparation": f"first {args.frames} frames at {args.fps} fps, 3840x2160 -> 1920x1080 "
                       f"4:2:0 with '{DOWNSCALE_FILTER}'", "videos": videos})
    print(f"manifest: {args.output / 'manifest.json'}")
    return 0


# --------------------------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------------------------


def load_test(path: Path | None) -> dict[str, list[tuple[float, float]]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {video: fair.curve_of(points) for video, points in data["videos"].items()}


def cmd_anchors(args) -> int:
    videos = fair.load_manifest(args.manifest)
    store = fair.AnchorStore.open(args.store)
    store.data["environment"] = environment()
    store.save()
    tests = load_test(args.cover)
    streams = args.streams or args.store.parent / "anchor_streams"
    for video in videos:
        for arm_name in args.arms:
            arm = fair.ARMS[arm_name]
            pending = [c for c in args.crf if not store.has(video.name, arm_name, c)]
            rounds = 0
            while True:
                for crf in pending:
                    started = time.perf_counter()
                    point = fair.encode_anchor(
                        video, arm, crf, streams / video.name / f"{arm_name}_crf{crf}{arm.suffix}")
                    point["seconds"] = time.perf_counter() - started
                    store.add(video.name, arm_name, point)
                    print(f"[{video.name}] {arm_name:22s} crf {crf:2d}  {point['bpp']:.4f} bpp  "
                          f"PSNR_YUV {point['psnr_yuv']:.3f}  ({point['seconds']:.0f} s)",
                          flush=True)
                pending = []
                if video.name in tests:
                    pending = [c for c in fair.extra_crfs(store.points(video.name, arm_name),
                                                          tests[video.name])
                               if not store.has(video.name, arm_name, c)]
                    rounds += 1
                    if pending:
                        print(f"[{video.name}] {arm_name}: coverage short, adding CRF {pending}",
                              flush=True)
                    if rounds > 30:
                        raise SystemExit(f"{video.name} {arm_name}: coverage never reached")
                if not pending:
                    break
    return 0


# --------------------------------------------------------------------------------------------
# proof 3, proof 4
# --------------------------------------------------------------------------------------------


def cmd_proof3(args) -> int:
    video = next(v for v in fair.load_manifest(args.manifest) if v.name == args.video)
    out, ok = {}, True
    for arm_name in ("x264_default", "x265_default"):
        arm = fair.ARMS[arm_name]
        path = args.scratch / f"proof3_{video.name}{arm.suffix}"
        point = fair.encode_anchor(video, arm, None, path, lossless=True)
        out[arm_name] = {k: point[k] for k in ("bytes", "identical_to_source", "decoded_sha256")}
        ok &= point["identical_to_source"]
        path.unlink()
        print(f"[proof 3] {video.name} {arm_name} lossless: identical={point['identical_to_source']}",
              flush=True)
    out["source_sha256"] = hashlib.sha256(fair.read_yuv(video).tobytes()).hexdigest()
    out["pass"] = ok
    write_json(args.out, out)
    print(f"PROOF 3: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_ceiling(args) -> int:
    out = {}
    for video in fair.load_manifest(args.manifest):
        raw = fair.read_yuv(video)
        back = fair.rgb_to_yuv420(fair.yuv_to_rgb(raw, video.width, video.height))
        score = fair.score_yuv(raw, back, video.width, video.height)
        score.pop("per_frame_y")
        out[video.name] = score
        print(f"[ceiling] {video.name}: Y {score['psnr_y']:.2f}  U {score['psnr_u']:.2f}  "
              f"V {score['psnr_v']:.2f}  YUV {score['psnr_yuv']:.2f}", flush=True)
    write_json(args.out, out)
    return 0


# --------------------------------------------------------------------------------------------
# ours
# --------------------------------------------------------------------------------------------


class OurCodec:
    """Everything p4_gate3.main builds for one lambda, built the same way (MSE set)."""

    def __init__(self, lam: str, device) -> None:
        import torch  # noqa: F401
        import benchmark_parity as bp
        import p1_gate3 as g3
        import p1_pframe as p1
        import p4_gate3 as pg
        import p4_wide_latent as p4
        from nvc.evaluation.sequences import discover_sequences
        from nvc.training.checkpoint import load_model_from_checkpoint
        from nvc.utils.config import load_default_config

        self.pg, self.bp = pg, bp
        self.mc = bp._load_script("m10h_motion_compensation")
        self.m21 = bp._load_script("m21_refinement")
        md = bp._load_script("m11_data")
        self.args = bp.build_arg_parser(load_default_config()).parse_args([])
        rig_lam = lam if lam in g3.BITS else "1e-3"
        models_dir = p4.VIMEO_DIR if lam in p1.LAMBDAS else Path("outputs/p6_low_rate/vimeo")
        self.intra, _ = load_model_from_checkpoint(Path(p1.CHECKPOINT.format(rig_lam)),
                                                   device=device)
        self.intra.eval()
        self.pcoder = p4.load_vimeo("p", lam, device, models_dir)
        self.icoder = p4.load_vimeo("i", lam, device, models_dir)
        self.identity = pg.identity_of(self.pcoder, self.icoder)
        manifest = self.args.manifest
        self.rig = g3.build_rig(self.args, self.m21, md, self.intra, rig_lam, g3.BITS[rig_lam],
                                device, discover_sequences(manifest, split="train"),
                                self.m21.broad_train_sequences(
                                    manifest,
                                    frames_per_sequence=self.args.train_frames_per_sequence))
        self.tables = g3.cumulative_tables()
        self.motion_lambda = MOTION_LAMBDA[lam]

    def encode(self, frames, path: Path):
        return self.pg.encode_sequence(
            self.mc, self.m21, self.intra, self.icoder, self.pcoder, frames, path, self.rig,
            self.tables, gop=30, block_size=self.args.block_size,
            search_range=self.args.search_range, identity=self.identity,
            motion_lambda=self.motion_lambda)

    def decode(self, path: Path):
        return self.pg.decode_sequence(self.mc, self.m21, self.intra, self.icoder, self.pcoder,
                                       path, self.rig, self.tables, identity=self.identity)


def rgb_tensor(rgb: np.ndarray):
    import torch
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).float() / 255.0


def rgb_uint8(frames) -> np.ndarray:
    import torch
    return (torch.round(frames.clamp(0, 1) * 255).to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous().numpy())


def cmd_ours(args) -> int:
    import torch
    device = torch.device(args.device)
    videos = fair.load_manifest(args.manifest)
    results = (json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file()
               else {"codec": "current (MSE set, GOP 30, wide I, bit-aware motion)",
                     "videos": {}})
    streams = args.streams or args.out.parent / "our_streams"
    for lam in args.lambdas:
        todo = [v for v in videos
                if not any(p["label"] == lam for p in results["videos"].get(v.name, []))]
        if not todo:
            continue
        codec = OurCodec(lam, device)
        for video in todo:
            raw = fair.read_yuv(video)
            frames = rgb_tensor(fair.yuv_to_rgb(raw, video.width, video.height))
            path = streams / f"{video.name}_{lam}.nvct"
            path.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            encoded = codec.encode(frames, path)
            encode_s = time.perf_counter() - started
            started = time.perf_counter()
            decoded = codec.decode(path)
            decode_s = time.perf_counter() - started
            exact = torch.equal(decoded.cpu(), encoded["reconstructions"])
            yuv = fair.rgb_to_yuv420(rgb_uint8(decoded.cpu()))
            point = fair.rate_point(path.stat().st_size, video)
            point.update(fair.score_yuv(raw, yuv, video.width, video.height))
            point.update({"label": lam, "decode_exact": exact, "encode_seconds": encode_s,
                          "decode_seconds": decode_s,
                          "recon_sha256": hashlib.sha256(
                              rgb_uint8(encoded["reconstructions"]).tobytes()).hexdigest()})
            results["videos"].setdefault(video.name, []).append(point)
            write_json(args.out, results)
            print(f"[{lam}] {video.name}: {point['bpp']:.4f} bpp  PSNR_YUV {point['psnr_yuv']:.3f}"
                  f"  Y {point['psnr_y']:.3f}  decode-exact={exact}", flush=True)
            del frames, encoded, decoded
        del codec
        torch.cuda.empty_cache()
    return 0


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------


def build_report(videos: list[fair.YuvVideo], store: fair.AnchorStore, test: dict[str, Any],
                 arms: list[str], metric: str = "psnr_yuv") -> dict[str, Any]:
    per_arm: dict[str, Any] = {}
    for arm in arms:
        rows, full, band = {}, {}, {}
        for video in videos:
            points = test["videos"].get(video.name)
            anchor = store.points(video.name, arm)
            if not points or not anchor:
                full[video.name] = band[video.name] = None
                continue
            curve, anchor_curve = fair.curve_of(points, metric), fair.curve_of(anchor, metric)
            stream_band = fair.streaming_band(curve)
            row = fair.compare(anchor_curve, curve, band=stream_band)
            row["streaming_band"] = stream_band
            rows[video.name] = row
            full[video.name] = row["bd_rate"]
            band[video.name] = row.get("band_bd_rate") if stream_band else None
        per_arm[arm] = {"videos": rows, "full_range": fair.summarize(full),
                        "streaming_band": fair.summarize(band),
                        "coverage_ok": all(r["coverage"]["top"] and r["coverage"]["bottom"]
                                           for r in rows.values()),
                        "method_flags": [v for v, r in rows.items() if r["method_gap_flag"]]}
    return {"metric": metric, "arms": per_arm}


def cmd_report(args) -> int:
    videos = fair.load_manifest(args.manifest)
    store = fair.AnchorStore.open(args.store)
    test = json.loads(args.test.read_text(encoding="utf-8"))
    report = build_report(videos, store, test, args.arms)
    report["test"] = str(args.test)
    for arm, r in report["arms"].items():
        f, b = r["full_range"], r["streaming_band"]
        fmt = lambda x: "  n/a  " if x is None else f"{x:+7.1f}%"   # noqa: E731
        print(f"{arm:24s} full: mean {fmt(f['mean'])} median {fmt(f['median'])} "
              f"wins {f['wins']}/{f['scored']}   band: mean {fmt(b['mean'])} "
              f"wins {b['wins']}/{b['scored']}   coverage ok: {r['coverage_ok']}")
        for video, row in r["videos"].items():
            print(f"    {video:14s} {fmt(row['bd_rate'])}  (linear {fmt(row['linear'])})"
                  f"  band {fmt(row.get('band_bd_rate'))}")
    if args.out:
        write_json(args.out, report)
    return 0


# --------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-uvg4k")
    p.add_argument("--archives", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", nargs="+", default=list(UVG4K_DEV))
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--keep-archives", action="store_true")
    p.set_defaults(func=cmd_prepare_uvg4k)

    p = sub.add_parser("anchors")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=list(fair.HEADLINE_ARMS),
                   choices=sorted(fair.ARMS))
    p.add_argument("--crf", nargs="+", type=int, default=list(fair.CRF_LADDER))
    p.add_argument("--cover", type=Path, default=None,
                   help="test-codec results: extend CRFs until every arm covers them")
    p.add_argument("--streams", type=Path, default=None)
    p.set_defaults(func=cmd_anchors)

    p = sub.add_parser("proof3")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--scratch", type=Path, default=Path("C:/fair_scratch"))
    p.add_argument("--out", type=Path, default=Path("outputs/p15_fair_eval/proof3.json"))
    p.set_defaults(func=cmd_proof3)

    p = sub.add_parser("ceiling")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_ceiling)

    p = sub.add_parser("ours")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lambdas", nargs="+", default=list(OUR_LAMBDAS))
    p.add_argument("--streams", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=cmd_ours)

    p = sub.add_parser("report")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--store", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--arms", nargs="+", default=list(fair.HEADLINE_ARMS))
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
