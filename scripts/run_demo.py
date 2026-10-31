"""Serve the codec demo in a browser.

    python scripts/build_demo.py     # once: codec bundles + example clips
    python scripts/run_demo.py       # then open http://127.0.0.1:8000

The page plays original / this codec / H.264-at-the-same-bitrate side by side,
shows each frame's bytes, and steps through one frame's motion, prediction,
residual and error. "Upload a clip" encodes a new video on the spot.

The server has no login and accepts uploads, so it binds to this machine only.
Use --allow-remote with a non-loopback --host only on a network you trust.

Needs the `demo` extra: pip install -e .[demo]
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Callable

from nvc.demo.clips import DEFAULT_MAX_FRAMES

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the codec demo page.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                        help="Frames kept from an uploaded video.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab.")
    parser.add_argument("--allow-remote", action="store_true",
                        help="Permit a --host other than this machine (no login is required "
                             "to upload, so only on a trusted network).")
    return parser


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None, *, serve: Callable[..., None] | None = None) -> int:
    """`serve(app, host=, port=)` defaults to `uvicorn.run`; tests pass their own."""
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        print(f"[ERROR] repository folder not found: {repo_root}", file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        print("[ERROR] --port must be between 1 and 65535.", file=sys.stderr)
        return 1
    if not is_loopback(args.host) and not args.allow_remote:
        print(f"[ERROR] --host {args.host} would expose the demo, which accepts uploads without "
              "a login, to other machines. Pass --allow-remote if that is intended.",
              file=sys.stderr)
        return 1
    try:
        from nvc.demo.server import create_app
    except ImportError as exc:
        print(f"[ERROR] the demo needs FastAPI ({exc}). Install it with: pip install -e .[demo]",
              file=sys.stderr)
        return 1
    if serve is None:
        import uvicorn
        serve = uvicorn.run

    app = create_app(repo_root, max_frames=args.max_frames,
                     device=None if args.device == "auto" else args.device)
    state = app.state.demo
    bundles = state.bundles()
    if not bundles:
        print("Note: no codec bundle found. Run python scripts/build_demo.py first; "
              "the page will say the same.")
    else:
        print(f"Codec bundles: {', '.join(b.label for b in bundles)}")
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"Demo at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    serve(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
