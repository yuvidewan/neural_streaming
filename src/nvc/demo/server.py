"""The demo's web server: precomputed results, live upload, and the page.

    from nvc.demo.server import create_app
    app = create_app(repo_root)            # then serve with uvicorn

Everything heavy runs on ONE background worker thread, one job at a time, so a
second upload queues behind the first instead of fighting it for the CPU. The
page polls `/api/jobs/<id>` for progress.

The server is meant for 127.0.0.1. It writes only under `<repo>/outputs/demo/`,
generates every clip id itself, and serves media through `StaticFiles`, which
refuses paths that leave its directory.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nvc.demo import clips as clip_store
from nvc.demo import reference
from nvc.demo.bundles import (
    TRAINED_KIND,
    BundleInfo,
    bundle_files,
    default_bundle,
    describe,
    sort_bundles,
)
from nvc.demo.pipeline import process_clip
from nvc.video.bundle import BundleError
from nvc.video.codec import VideoCodec

DEMO_DIR = Path("outputs/demo")
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
CODEC_CACHE_SIZE = 3
_CHUNK = 1024 * 1024


@dataclass
class Job:
    id: str
    kind: str                      # "upload" or "encode"
    status: str = "queued"         # queued | running | done | error
    stage: str = "queued"
    done: int = 0
    total: int = 0
    clip_id: str | None = None
    bundle_id: str | None = None
    error: str | None = None


class DemoState:
    """Paths, the bundle list, loaded codecs and jobs. One per app."""

    def __init__(self, repo_root: Path, demo_dir: Path, *, max_frames: int, device: str | None) -> None:
        self.repo_root = repo_root
        self.demo_dir = demo_dir
        self.clips_dir = demo_dir / "clips"
        self.uploads_dir = demo_dir / "uploads"
        self.max_frames = max_frames
        self.device = device
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._codecs: OrderedDict[str, VideoCodec] = OrderedDict()
        self._bundles: dict[tuple[str, int, int], BundleInfo] = {}
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvc-demo")

    # --- bundles and codecs --------------------------------------------------------------

    def bundles(self) -> list[BundleInfo]:
        """The bundles on disk now. Verifying one costs a model build, so results are
        cached against the file's size and modification time - which also means a
        bundle dropped into the folder while the server runs is picked up."""
        fresh = {}
        for path, kind in bundle_files(self.repo_root):
            stat = path.stat()
            key = (str(path), stat.st_size, stat.st_mtime_ns)
            info = self._bundles.get(key)
            if info is None:
                try:
                    info = describe(path, kind)
                except (BundleError, OSError):
                    continue
            fresh[key] = info
        self._bundles = fresh
        return sort_bundles(list(fresh.values()))

    def bundle(self, bundle_id: str) -> BundleInfo:
        for info in self.bundles():
            if info.id == bundle_id:
                return info
        raise HTTPException(status_code=404, detail=f"No codec bundle named '{bundle_id}'.")

    def codec(self, info: BundleInfo) -> VideoCodec:
        with self._lock:
            codec = self._codecs.get(info.id)
            if codec is not None:
                self._codecs.move_to_end(info.id)
                return codec
        codec = VideoCodec.from_file(info.path, device=self.device)
        with self._lock:
            self._codecs[info.id] = codec
            while len(self._codecs) > CODEC_CACHE_SIZE:
                self._codecs.popitem(last=False)
        return codec

    # --- jobs ----------------------------------------------------------------------------

    def new_job(self, kind: str, **fields: Any) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, **fields)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for name, value in fields.items():
                setattr(job, name, value)

    def job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="No such job.")
            return asdict(job)

    def submit(self, job: Job, work) -> None:
        def run() -> None:
            self.update(job, status="running")
            try:
                work()
            except Exception as exc:        # reported to the page, never swallowed
                self.update(job, status="error", error=str(exc) or type(exc).__name__)
            else:
                self.update(job, status="done", stage="done")
        self._worker.submit(run)

    def run_clip(self, job: Job, clip_id: str, info: BundleInfo) -> None:
        codec = self.codec(info)
        process_clip(self.clips_dir / clip_id, info, codec,
                     progress=lambda stage, done, total: self.update(
                         job, stage=stage, done=done, total=total))


def create_app(repo_root: str | Path, *, demo_dir: str | Path | None = None,
               max_frames: int = clip_store.DEFAULT_MAX_FRAMES,
               device: str | None = None) -> FastAPI:
    """Build the demo app. `demo_dir` defaults to `<repo_root>/outputs/demo`."""
    repo_root = Path(repo_root).resolve()
    state = DemoState(repo_root, Path(demo_dir) if demo_dir else repo_root / DEMO_DIR,
                      max_frames=max_frames, device=device)
    app = FastAPI(title="Neural video codec demo", docs_url=None, redoc_url=None)
    app.state.demo = state

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        bundles = state.bundles()
        chosen = default_bundle(bundles)
        return {
            "bundles": [b.to_json() for b in bundles],
            "default_bundle": chosen.id if chosen else None,
            "standin_only": bool(bundles) and all(b.kind != TRAINED_KIND for b in bundles),
            "h264_available": reference.h264_available(),
            "max_frames": state.max_frames,
            "clip_size": clip_store.CLIP_SIZE,
            "video_suffixes": list(clip_store.VIDEO_SUFFIXES),
        }

    @app.get("/api/clips")
    def clips() -> list[dict[str, Any]]:
        return clip_store.list_clips(state.clips_dir)

    def clip_info(clip_id: str) -> dict[str, Any]:
        try:
            return clip_store.load_clip(state.clips_dir, clip_id)
        except clip_store.ClipError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/clips/{clip_id}/results/{bundle_id}")
    def result(clip_id: str, bundle_id: str):
        clip_info(clip_id)
        info = state.bundle(bundle_id)
        path = state.clips_dir / clip_id / info.id / "result.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="This clip has not been encoded with this codec yet.")
        return FileResponse(path, media_type="application/json",
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/clips/{clip_id}/results/{bundle_id}")
    def encode(clip_id: str, bundle_id: str) -> dict[str, Any]:
        clip_info(clip_id)
        info = state.bundle(bundle_id)
        job = state.new_job("encode", clip_id=clip_id, bundle_id=info.id)
        state.submit(job, lambda: state.run_clip(job, clip_id, info))
        return {"job_id": job.id}

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...), bundle_id: str = Form("")) -> dict[str, Any]:
        bundles = state.bundles()
        if not bundles:
            raise HTTPException(status_code=409, detail="No codec bundle is available. Run scripts/build_demo.py first.")
        info = state.bundle(bundle_id) if bundle_id else default_bundle(bundles)
        original_name = Path(file.filename or "clip").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in clip_store.VIDEO_SUFFIXES:
            raise HTTPException(status_code=415, detail=(
                f"'{suffix or original_name}' is not a supported video type. Use one of: "
                + ", ".join(clip_store.VIDEO_SUFFIXES)))
        staged = state.uploads_dir / f"{uuid.uuid4().hex}{suffix}"
        written = 0
        try:
            with staged.open("wb") as target:
                while chunk := file.file.read(_CHUNK):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail=(
                            f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."))
                    target.write(chunk)
        except HTTPException:
            staged.unlink(missing_ok=True)
            raise
        if written == 0:
            staged.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        job = state.new_job("upload", bundle_id=info.id)
        name = Path(original_name).stem

        def work() -> None:
            try:
                state.update(job, stage="ingest", done=0, total=1)
                try:
                    frames, fps = clip_store.frames_from_video(staged, max_frames=state.max_frames)
                except clip_store.ClipError as exc:
                    # The staged file has a random name; report the one the user chose.
                    raise clip_store.ClipError(
                        f"'{original_name}' could not be read as a video.") from exc
                clip = clip_store.create_clip(state.clips_dir, frames, name=name, source="upload", fps=fps)
                state.update(job, clip_id=clip["id"])
            finally:
                staged.unlink(missing_ok=True)
            state.run_clip(job, clip["id"], info)

        state.submit(job, work)
        return {"job_id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> JSONResponse:
        return JSONResponse(state.job(job_id), headers={"Cache-Control": "no-store"})

    @app.delete("/api/clips/{clip_id}")
    def delete_clip(clip_id: str) -> dict[str, Any]:
        info = clip_info(clip_id)
        if info.get("source") != "upload":
            raise HTTPException(status_code=403, detail="Only uploaded clips can be removed.")
        shutil.rmtree(state.clips_dir / clip_id, ignore_errors=True)
        return {"deleted": clip_id}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.mount("/media", StaticFiles(directory=state.clips_dir), name="media")
    return app
