"""The web demo (`nvc.demo`): the traced encoder, clips, stand-in bundles, the
per-clip pipeline, H.264 matching and the FastAPI server.

The central contract is that the demo shows the REAL codec: `trace_encode` must
write the same bytes as `VideoCodec.encode`, and every stream it writes must
decode, with the real decoder, to the frames the encoder reconstructed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from helpers import make_synthetic_video

from nvc.demo import bundles as demo_bundles
from nvc.demo import clips as demo_clips
from nvc.demo import pipeline, reference
from nvc.demo.trace import trace_encode
from nvc.models import BaselineAutoencoder
from nvc.video import CodecBundle, VideoCodec
from nvc.video.container import FRAME_TYPE_I, FRAME_TYPE_P
from nvc.video.motion import warp_blocks

SIZE = 64                         # 64x64 frames -> 4x4 latent, 4x4 motion blocks
BITS, GOP, BLOCK, RANGE = 4, 4, 16, 4

needs_h264 = pytest.mark.skipif(not reference.h264_available(),
                                reason="FFmpeg with libx264 is not on PATH")


def _frames(count: int = 9, seed: int = 0, size: int = SIZE) -> np.ndarray:
    """[N, size, size, 3] uint8: a smooth pattern drifting a few pixels a frame."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.nn.functional.interpolate(torch.rand(1, 3, 8, 8, generator=generator),
                                           size=(size * 2, size * 2), mode="bilinear",
                                           align_corners=False)[0]
    frames = [base[:, 2 * t % size:2 * t % size + size, t % size:t % size + size]
              for t in range(count)]
    return pipeline.to_uint8(torch.stack(frames).clamp(0, 1))


def _autoencoder() -> BaselineAutoencoder:
    torch.manual_seed(0)
    return BaselineAutoencoder(in_channels=3, latent_channels=4, base_channels=8).eval()


def _standin(bits: int = BITS) -> CodecBundle:
    sequences = [pipeline.to_float(_frames(5, seed)) for seed in (1, 2)]
    return demo_bundles.build_standin_bundle(_autoencoder(), sequences, bits=bits, gop_size=GOP,
                                             block_size=BLOCK, search_range=RANGE)


@pytest.fixture(scope="module")
def bundle() -> CodecBundle:
    return _standin()


@pytest.fixture(scope="module")
def codec(bundle) -> VideoCodec:
    return VideoCodec(bundle, device="cpu")


@pytest.fixture()
def repo(tmp_path, bundle):
    """A miniature repository root holding one stand-in bundle."""
    bundle.save(tmp_path / demo_bundles.STANDIN_DIR / "standin_4bit.pt")
    return tmp_path


# --- trace ---------------------------------------------------------------------------------


def test_trace_writes_the_same_bytes_as_the_codec(codec):
    frames = pipeline.to_float(_frames())
    encoded, reconstructions = codec.encode(frames, return_reconstructions=True)
    trace = trace_encode(codec, frames)
    assert trace.encoded.data == encoded.data
    assert trace.encoded.motion_bytes == encoded.motion_bytes
    assert trace.encoded.residual_bytes == encoded.residual_bytes
    assert torch.equal(trace.reconstructions, reconstructions)
    assert torch.equal(codec.decode(trace.encoded.data), trace.reconstructions)


def test_trace_keeps_each_frames_intermediates(codec):
    frames = pipeline.to_float(_frames())
    trace = trace_encode(codec, frames)
    assert [f.frame_type for f in trace.frames] == [
        FRAME_TYPE_I if i % GOP == 0 else FRAME_TYPE_P for i in range(len(frames))]
    for previous, frame in zip((None,) + trace.frames[:-1], trace.frames):
        assert frame.motion_bytes == trace.encoded.motion_bytes[frame.index]
        assert frame.residual_bytes == trace.encoded.residual_bytes[frame.index]
        if frame.frame_type == FRAME_TYPE_I:
            assert frame.motion is None and frame.prediction is None and frame.activity is None
            assert frame.motion_bytes == 0
            continue
        assert frame.motion.shape == (2, SIZE // BLOCK, SIZE // BLOCK)
        assert np.abs(frame.motion).max() <= RANGE
        # The prediction is exactly the previous RECONSTRUCTION warped by the coded vectors.
        expected = warp_blocks(previous.reconstruction[None], torch.from_numpy(frame.motion),
                               block_size=BLOCK)[0]
        assert torch.equal(frame.prediction, expected)
        assert frame.activity.shape == (SIZE // 16, SIZE // 16)
        assert (frame.activity >= 0).all()


def test_trace_finds_the_motion_in_a_moving_clip(codec):
    trace = trace_encode(codec, pipeline.to_float(_frames()))
    assert any(f.motion is not None and np.abs(f.motion).sum() > 0 for f in trace.frames)


def test_trace_reports_progress_once_per_frame(codec):
    calls = []
    trace_encode(codec, pipeline.to_float(_frames(5)), progress=lambda d, t: calls.append((d, t)))
    assert calls == [(i, 5) for i in range(1, 6)]


def test_trace_rejects_unusable_frames(codec):
    with pytest.raises(ValueError, match="divisible"):
        trace_encode(codec, torch.zeros(2, 3, 40, 64))
    with pytest.raises(ValueError, match="empty"):
        trace_encode(codec, torch.zeros(0, 3, SIZE, SIZE))


# --- stand-in bundles ----------------------------------------------------------------------


def test_standin_bundle_is_a_valid_labelled_bundle(bundle, tmp_path):
    bundle.verify()
    assert bundle.provenance["kind"] == demo_bundles.STANDIN_KIND
    # One shared residual table: nothing is predicted from context.
    tables = bundle.coding_codebook_frequencies
    assert (tables == tables[0]).all()
    path = tmp_path / "standin.pt"
    bundle.save(path)
    assert CodecBundle.load(path).computed_identities() == bundle.computed_identities()


def test_standin_bundle_rebuilds_identically(bundle):
    again = _standin()
    assert again.computed_identities() == bundle.computed_identities()
    assert torch.equal(again.intra_frequencies, bundle.intra_frequencies)


def test_standin_build_does_not_disturb_the_global_rng():
    autoencoder = _autoencoder()             # seeds the RNG itself, so build it first
    sequences = [pipeline.to_float(_frames(3, 1))]
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    demo_bundles.build_standin_bundle(autoencoder, sequences, bits=BITS, search_range=RANGE)
    assert torch.equal(torch.rand(3), expected)


def test_standin_needs_sequences_of_at_least_two_frames():
    with pytest.raises(ValueError, match="two or more"):
        demo_bundles.build_standin_bundle(_autoencoder(), [pipeline.to_float(_frames(1))], bits=4)
    with pytest.raises(ValueError, match="two or more"):
        demo_bundles.build_standin_bundle(_autoencoder(), [], bits=4)


def test_discovery_prefers_trained_bundles_and_skips_broken_files(repo, bundle):
    _standin(3).save(repo / demo_bundles.STANDIN_DIR / "standin_3bit.pt")
    bundle.save(repo / demo_bundles.TRAINED_DIR / "nvc_m22_4bit.pt")
    (repo / demo_bundles.TRAINED_DIR / "half_copied.pt").write_bytes(b"not a bundle")
    found = demo_bundles.discover_bundles(repo)
    assert [(b.id, b.kind) for b in found] == [
        ("nvc_m22_4bit", "trained"), ("standin_3bit", "standin"), ("standin_4bit", "standin")]
    assert found[0].label == "M22, 4-bit" and found[0].note == ""
    assert found[1].label == "Stand-in, 3-bit" and "not a project result" in found[1].note
    assert demo_bundles.default_bundle(found).id == "nvc_m22_4bit"
    assert demo_bundles.default_bundle(found[1:]).id == "standin_4bit"
    assert demo_bundles.default_bundle([]) is None
    assert "path" not in found[0].to_json()


# --- clips ---------------------------------------------------------------------------------


def test_fit_frame_centre_crops_then_resizes():
    frame = np.zeros((40, 100, 3), dtype=np.uint8)
    frame[:, 30:70] = 255                      # the centre 40 columns
    fitted = demo_clips.fit_frame(frame, 32)
    assert fitted.shape == (32, 32, 3)
    assert fitted.min() == 255                 # only the centre square survived
    assert demo_clips.fit_frame(np.zeros((32, 32, 3), dtype=np.uint8), 32).shape == (32, 32, 3)
    with pytest.raises(demo_clips.ClipError):
        demo_clips.fit_frame(np.zeros((32, 32), dtype=np.uint8), 32)


def test_frames_from_video_reads_and_limits_frames(tmp_path):
    video = make_synthetic_video(tmp_path / "clip.mp4", num_frames=10, width=64, height=48, fps=10.0)
    frames, fps = demo_clips.frames_from_video(video, max_frames=6, size=32)
    assert frames.shape == (6, 32, 32, 3) and frames.dtype == np.uint8
    assert fps == pytest.approx(10.0)
    assert len(demo_clips.frames_from_video(video, max_frames=99, size=32)[0]) == 10


def test_frames_from_video_refuses_what_is_not_a_video(tmp_path):
    with pytest.raises(demo_clips.ClipError, match="not found"):
        demo_clips.frames_from_video(tmp_path / "missing.mp4")
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"\x00" * 4096)
    with pytest.raises(demo_clips.ClipError):
        demo_clips.frames_from_video(garbage)
    with pytest.raises(demo_clips.ClipError, match="max_frames"):
        demo_clips.frames_from_video(garbage, max_frames=0)


def test_frames_round_trip_losslessly_through_png(tmp_path):
    frames = _frames(4)
    demo_clips.write_frames(tmp_path / "f", frames)
    assert np.array_equal(demo_clips.read_frames(tmp_path / "f", 4), frames)
    assert np.array_equal(
        demo_clips.frames_from_images(sorted((tmp_path / "f").glob("*.png")), size=SIZE), frames)
    with pytest.raises(demo_clips.ClipError, match="missing frame"):
        demo_clips.read_frames(tmp_path / "f", 5)


def test_create_load_and_list_clips(tmp_path):
    first = demo_clips.create_clip(tmp_path, _frames(3), name="B clip", source="precomputed",
                                   clip_id="b-clip")
    demo_clips.create_clip(tmp_path, _frames(3), name="A clip", source="precomputed", clip_id="a-clip")
    upload = demo_clips.create_clip(tmp_path, _frames(2), name="My Video!", source="upload", fps=30)
    assert first["frames"] == 3 and first["width"] == SIZE
    assert upload["id"].startswith("my-video-") and demo_clips.is_clip_id(upload["id"])
    assert demo_clips.load_clip(tmp_path, "b-clip") == first
    (tmp_path / "b-clip" / "standin_4bit").mkdir()
    (tmp_path / "b-clip" / "standin_4bit" / "result.json").write_text("{}")
    listed = demo_clips.list_clips(tmp_path)
    assert [c["name"] for c in listed] == ["A clip", "B clip", "My Video!"]
    assert listed[1]["results"] == ["standin_4bit"] and listed[0]["results"] == []
    assert demo_clips.list_clips(tmp_path / "nowhere") == []


@pytest.mark.parametrize("bad", ["../x", "a/b", "a\\b", "..", "", "C:", "a.b", "A", "x" * 65])
def test_clip_ids_cannot_name_a_path(tmp_path, bad):
    assert not demo_clips.is_clip_id(bad)
    with pytest.raises(demo_clips.ClipError):
        demo_clips.load_clip(tmp_path, bad)
    with pytest.raises(demo_clips.ClipError):
        demo_clips.create_clip(tmp_path, _frames(2), name="n", source="upload", clip_id=bad)


# --- pipeline ------------------------------------------------------------------------------


def test_score_matches_a_hand_computed_psnr():
    original = np.full((2, SIZE, SIZE, 3), 100, dtype=np.uint8)
    decoded = original.copy()
    decoded[1] += 10                           # MSE = (10/255)^2 on frame 1 only
    scored = pipeline.score(decoded, original)
    assert scored["psnr_per_frame"][0] is None           # identical frame: infinite PSNR
    assert scored["psnr_per_frame"][1] == pytest.approx(20 * np.log10(255 / 10), abs=1e-3)
    assert scored["psnr"] == pytest.approx(scored["psnr_per_frame"][1])
    assert scored["msssim"] is None                      # 64 px is too small for MS-SSIM
    json.dumps(scored)                                   # no infinities, no numpy scalars


def test_uint8_conversion_rounds_like_a_display():
    frames = torch.tensor([0.0, 0.5 / 255, 0.49 / 255, 1.0, 1.2, -0.1]).reshape(1, 1, 1, 6)
    assert pipeline.to_uint8(frames.repeat(1, 3, 1, 1))[0, 0, :, 0].tolist() == [0, 0, 0, 255, 255, 0]
    frame = _frames(1)
    assert np.array_equal(pipeline.to_uint8(pipeline.to_float(frame)), frame)


def test_process_clip_writes_a_complete_decodable_result(repo, codec):
    info = demo_bundles.describe(repo / demo_bundles.STANDIN_DIR / "standin_4bit.pt", "standin")
    frames = _frames()
    clip = demo_clips.create_clip(repo / "clips", frames, name="t", source="precomputed", clip_id="t")
    stages = []
    result = pipeline.process_clip(repo / "clips" / "t", info, codec, with_h264=False,
                                   progress=lambda stage, done, total: stages.append(stage))
    out = repo / "clips" / "t" / "standin_4bit"
    assert json.loads((out / "result.json").read_text()) == result
    assert list(dict.fromkeys(stages)) == ["encode", "decode", "h264", "score"]

    nvc = result["nvc"]
    count = clip["frames"]
    assert result["frames"] == count and result["raw_bytes"] == count * SIZE * SIZE * 3
    assert nvc["decoder_exact"] is True
    assert nvc["frame_types"] == ["I" if i % GOP == 0 else "P" for i in range(count)]
    # Byte accounting closes: payloads plus container overhead is the file on disk.
    stream = (out / "stream.nvct").read_bytes()
    assert nvc["total_bytes"] == len(stream)
    assert sum(nvc["motion_bytes"]) + sum(nvc["residual_bytes"]) + nvc["container_bytes"] == len(stream)
    assert nvc["container_bytes"] > 0
    assert nvc["bpp"] == pytest.approx(len(stream) * 8 / (count * SIZE * SIZE))
    # The frames on disk are what the real decoder produces from that stream.
    assert np.array_equal(demo_clips.read_frames(out / "nvc", count),
                          pipeline.to_uint8(codec.decode(stream)))
    for index, kind in enumerate(nvc["frame_types"]):
        has_prediction = (out / "pred" / demo_clips.frame_name(index)).is_file()
        assert has_prediction == (kind == "P")
        assert (nvc["motion"][index] is None) == (kind == "I")
        assert (nvc["activity"][index] is None) == (kind == "I")
    assert len(nvc["psnr_per_frame"]) == count and nvc["psnr"] > 0
    assert result["h264"] == {"available": False, "reason": "H.264 comparison was not requested."}
    assert result["bundle"]["kind"] == "standin" and "path" not in result["bundle"]


def test_process_clip_never_leaves_a_stale_result_behind(repo, codec, monkeypatch):
    info = demo_bundles.describe(repo / demo_bundles.STANDIN_DIR / "standin_4bit.pt", "standin")
    demo_clips.create_clip(repo / "clips", _frames(5), name="t", source="precomputed", clip_id="t")
    pipeline.process_clip(repo / "clips" / "t", info, codec, with_h264=False)
    marker = repo / "clips" / "t" / "standin_4bit" / "result.json"
    assert marker.is_file()

    def explode(*args, **kwargs):
        raise RuntimeError("encoder failed")
    monkeypatch.setattr(pipeline, "trace_encode", explode)
    with pytest.raises(RuntimeError):
        pipeline.process_clip(repo / "clips" / "t", info, codec, with_h264=False)
    assert not marker.exists()               # a failed re-run must not be served as a result


def test_process_clip_reports_a_missing_ffmpeg_instead_of_failing(repo, codec, monkeypatch):
    info = demo_bundles.describe(repo / demo_bundles.STANDIN_DIR / "standin_4bit.pt", "standin")
    demo_clips.create_clip(repo / "clips", _frames(5), name="t", source="precomputed", clip_id="t")
    monkeypatch.setattr(reference, "h264_available", lambda: False)
    result = pipeline.process_clip(repo / "clips" / "t", info, codec)
    assert result["h264"]["available"] is False and "FFmpeg" in result["h264"]["reason"]
    assert result["nvc"]["decoder_exact"] is True


# --- H.264 at a matched bitrate ------------------------------------------------------------


@needs_h264
def test_h264_match_is_the_best_quality_that_fits(tmp_path):
    frames = _frames(8)
    demo_clips.write_frames(tmp_path / "original", frames)
    pattern = tmp_path / "original" / "%03d.png"
    loose = reference.encode_h264(pattern, tmp_path / "probe.mp4", crf=20, fps=25)
    match = reference.match_bitrate(pattern, tmp_path / "out.mp4", target_bytes=loose, fps=25)
    assert match.within_target and match.total_bytes <= loose
    assert match.total_bytes == (tmp_path / "out.mp4").stat().st_size
    assert match.crf <= 20                      # crf 20 fits, so the pick is at least that good
    if match.crf > reference.CRF_BEST:          # ...and one step better would not fit
        assert reference.encode_h264(pattern, tmp_path / "probe.mp4", crf=match.crf - 1, fps=25) > loose
    assert not list(tmp_path.glob("*.search.*"))
    assert reference.decode_to_frames(tmp_path / "out.mp4", tmp_path / "decoded") == 8


@needs_h264
def test_h264_match_flags_a_target_it_cannot_reach(tmp_path):
    demo_clips.write_frames(tmp_path / "original", _frames(4))
    match = reference.match_bitrate(tmp_path / "original" / "%03d.png", tmp_path / "out.mp4",
                                    target_bytes=10, fps=25)
    assert match.crf == reference.CRF_WORST and not match.within_target
    with pytest.raises(ValueError):
        reference.match_bitrate(tmp_path / "original" / "%03d.png", tmp_path / "out.mp4",
                                target_bytes=0, fps=25)


@needs_h264
def test_process_clip_includes_h264_at_no_more_than_our_bytes(repo, codec):
    info = demo_bundles.describe(repo / demo_bundles.STANDIN_DIR / "standin_4bit.pt", "standin")
    demo_clips.create_clip(repo / "clips", _frames(), name="t", source="precomputed", clip_id="t")
    result = pipeline.process_clip(repo / "clips" / "t", info, codec)
    h264, out = result["h264"], repo / "clips" / "t" / "standin_4bit"
    assert h264["available"] is True
    assert h264["total_bytes"] == (out / "h264.mp4").stat().st_size
    assert h264["within_target"] == (h264["total_bytes"] <= result["nvc"]["total_bytes"])
    assert len(h264["psnr_per_frame"]) == result["frames"]
    assert len(list((out / "h264").glob("*.png"))) == result["frames"]


# --- the server ----------------------------------------------------------------------------


@pytest.fixture()
def client(repo, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from nvc.demo.server import create_app

    monkeypatch.setattr(reference, "h264_available", lambda: False)     # keep these tests fast
    demo_clips.create_clip(repo / "outputs" / "demo" / "clips", _frames(6), name="example",
                           source="precomputed", clip_id="example")
    with TestClient(create_app(repo, max_frames=5, device="cpu")) as test_client:
        yield test_client


def _wait(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def _video_bytes(tmp_path: Path, frames: int = 8) -> bytes:
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (96, 64))
    assert writer.isOpened()
    for frame in _frames(frames, size=96):
        writer.write(np.ascontiguousarray(frame[:64, :, ::-1]))
    writer.release()
    return path.read_bytes()


def test_server_lists_bundles_and_clips(client):
    status = client.get("/api/status").json()
    assert [b["id"] for b in status["bundles"]] == ["standin_4bit"]
    assert status["default_bundle"] == "standin_4bit" and status["standin_only"] is True
    assert status["max_frames"] == 5 and status["h264_available"] is False
    clips = client.get("/api/clips").json()
    assert [(c["id"], c["source"], c["results"]) for c in clips] == [("example", "precomputed", [])]


def test_server_serves_the_page_and_its_assets(client):
    page = client.get("/")
    assert page.status_code == 200 and "Neural video codec" in page.text
    for asset in ("app.js", "style.css"):
        assert client.get(f"/static/{asset}").status_code == 200
    assert client.get("/media/example/original/000.png").headers["content-type"] == "image/png"


def test_server_encodes_an_existing_clip_on_request(client):
    assert client.get("/api/clips/example/results/standin_4bit").status_code == 404
    started = client.post("/api/clips/example/results/standin_4bit").json()
    job = _wait(client, started["job_id"])
    assert job["status"] == "done" and job["clip_id"] == "example" and job["error"] is None
    result = client.get("/api/clips/example/results/standin_4bit").json()
    assert result["nvc"]["decoder_exact"] is True and result["frames"] == 6
    assert client.get("/media/example/standin_4bit/nvc/005.png").status_code == 200
    assert client.get("/api/clips").json()[0]["results"] == ["standin_4bit"]


def test_server_upload_becomes_a_clip_limited_to_max_frames(client, tmp_path, repo):
    response = client.post("/api/upload", data={"bundle_id": "standin_4bit"},
                           files={"file": ("Holiday Clip.mp4", _video_bytes(tmp_path), "video/mp4")})
    assert response.status_code == 200
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "done", job
    clip = next(c for c in client.get("/api/clips").json() if c["source"] == "upload")
    assert clip["id"] == job["clip_id"] and clip["name"] == "Holiday Clip"
    assert clip["frames"] == 5 and clip["width"] == demo_clips.CLIP_SIZE      # max_frames=5
    result = client.get(f"/api/clips/{clip['id']}/results/standin_4bit").json()
    assert result["nvc"]["decoder_exact"] is True
    assert not list((repo / "outputs" / "demo" / "uploads").iterdir())       # staging is cleaned up

    assert client.delete(f"/api/clips/{clip['id']}").json() == {"deleted": clip["id"]}
    assert [c["id"] for c in client.get("/api/clips").json()] == ["example"]


def test_server_refuses_unusable_uploads(client, repo):
    wrong_type = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert wrong_type.status_code == 415 and ".txt" in wrong_type.json()["detail"]
    empty = client.post("/api/upload", files={"file": ("empty.mp4", b"", "video/mp4")})
    assert empty.status_code == 400
    unknown = client.post("/api/upload", data={"bundle_id": "nope"},
                          files={"file": ("a.mp4", b"x", "video/mp4")})
    assert unknown.status_code == 404

    garbage = client.post("/api/upload", files={"file": ("broken.mp4", b"\x00" * 4096, "video/mp4")})
    job = _wait(client, garbage.json()["job_id"])
    assert job["status"] == "error" and "broken.mp4" in job["error"]
    assert not list((repo / "outputs" / "demo" / "uploads").iterdir())
    assert [c["id"] for c in client.get("/api/clips").json()] == ["example"]


def test_server_caps_the_upload_size(client, monkeypatch, repo):
    from nvc.demo import server
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    response = client.post("/api/upload", files={"file": ("big.mp4", b"\x01" * 5000, "video/mp4")})
    assert response.status_code == 413
    assert not list((repo / "outputs" / "demo" / "uploads").iterdir())


def test_server_rejects_bad_ids_and_protects_precomputed_clips(client):
    assert client.get("/api/clips/nope/results/standin_4bit").status_code == 404
    assert client.get("/api/clips/example/results/nope").status_code == 404
    assert client.post("/api/clips/..%2F..%2Fx/results/standin_4bit").status_code == 404
    assert client.get("/api/jobs/unknown").status_code == 404
    assert client.delete("/api/clips/example").status_code == 403
    assert client.get("/media/../../pyproject.toml").status_code == 404
    assert client.get("/media/%2e%2e/bundles/standin_4bit.pt").status_code == 404


def test_server_without_any_bundle_says_so(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from nvc.demo.server import create_app

    with TestClient(create_app(tmp_path, device="cpu")) as bare:
        status = bare.get("/api/status").json()
        assert status["bundles"] == [] and status["default_bundle"] is None
        assert status["standin_only"] is False
        refused = bare.post("/api/upload", files={"file": ("a.mp4", b"x", "video/mp4")})
        assert refused.status_code == 409 and "build_demo" in refused.json()["detail"]


def test_server_picks_up_a_bundle_added_while_running(client, repo):
    _standin(3).save(repo / demo_bundles.STANDIN_DIR / "standin_3bit.pt")
    assert [b["id"] for b in client.get("/api/status").json()["bundles"]] == [
        "standin_3bit", "standin_4bit"]
