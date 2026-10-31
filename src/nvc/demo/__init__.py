"""A local web demo of the video codec.

    python scripts/build_demo.py      # precompute a few DAVIS clips
    python scripts/run_demo.py        # serve the page at http://127.0.0.1:8000

The page plays the original, this codec's reconstruction and H.264 at the same
bitrate side by side, shows where each frame's bytes went, and steps through
one frame's motion vectors, prediction, residual and error.

Modules:
- trace:     `trace_encode`, the encoder loop with its intermediates kept
- reference: H.264 at a matched bitrate, through FFmpeg
- clips:     turning a video file or a folder of frames into a demo clip
- pipeline:  `process_clip`, one clip through one codec bundle, written to disk
- bundles:   finding codec bundles, and building a stand-in when none is present
- server:    the FastAPI app (precomputed results, live upload, static page)

Nothing here changes a coded byte: `trace_encode` is tested to write the same
stream as `VideoCodec.encode`. FastAPI and uvicorn are an optional extra
(`pip install -e .[demo]`); importing `nvc` or `nvc.video` never needs them.
"""
