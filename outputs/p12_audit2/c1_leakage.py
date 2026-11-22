"""docs/PLAN_audit2.md, check C1: are any test-video frames also in the Vimeo training data?

Every frame of DAVIS TEST (original 480p JPEGs), UVG (first 96 frames) and MCL-JCV (first 96
frames) gets a 64-bit perceptual hash (pHash: grayscale, 32x32, 2-D DCT, the 8x8 low band
against its median). Then each Kaggle Vimeo-90k chunk (wangsally/vimeo-90k-1..10, the chunks
the models' training lineage used) is downloaded, ONE at a time. All 7 frames of every clip are
hashed straight from the zip (nothing is extracted), and the zip is deleted before the next
chunk is fetched.

Rule, fixed in the plan before running: a pair at Hamming distance <= 6 of 64 bits is a
candidate. Every candidate is saved as a side-by-side thumbnail and checked by eye. The
closest Vimeo frame to each test video is reported too, whatever its distance. Resumable:
finished chunks are kept in c1_state.json.

    .venv/Scripts/python -u outputs/p12_audit2/c1_leakage.py
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

O = Path("outputs/p12_audit2")
SCRATCH = Path("C:/audit2_scratch/vimeo")
STATE = O / "c1_state.json"
THUMBS = O / "c1_thumbs"
THRESHOLD = 6
CHUNKS = range(1, 11)
DAVIS_TEST = ["bmx-bumps", "car-turn", "cat-girl", "cows", "drift-chicane", "drone",
              "gold-fish", "schoolgirls", "surf"]
SETS = {"uvg": Path("C:/uvg_scratch/uvg"), "mcljcv": Path("C:/mcljcv_scratch/mcljcv")}

_N = 32
_D = np.cos(np.pi * (2 * np.arange(_N)[None, :] + 1) * np.arange(_N)[:, None] / (2 * _N))


def phash(image: Image.Image) -> int:
    g = np.asarray(image.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    low = (_D @ g @ _D.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view(">u8")[0])


def test_frames():
    for name in DAVIS_TEST:
        for path in sorted(Path(f"data/external/DAVIS/JPEGImages/480p/{name}").glob("*.jpg")):
            yield "davis", name, path
    for tag, root in SETS.items():
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["items"]:
            for i in range(1, item["frame_count"] + 1):
                yield tag, item["source_name"], root / item["frame_directory"] / \
                    item["frame_filename_pattern"].format(index=i)


def load_state():
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    print("hashing test frames ...", flush=True)
    rows = [(tag, name, str(path), phash(Image.open(path))) for tag, name, path in test_frames()]
    print(f"  {len(rows)} test frames", flush=True)
    state = {"test": rows, "chunks": {}, "candidates": [], "nearest": {}}
    save_state(state)
    return state


def save_state(state) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, STATE)


def thumb(test_path: str, zip_path: Path, member: str, distance: int, out: Path) -> None:
    left = Image.open(test_path).convert("RGB")
    with zipfile.ZipFile(zip_path) as z:
        right = Image.open(io.BytesIO(z.read(member))).convert("RGB")
    h = 256
    left = left.resize((round(left.width * h / left.height), h))
    right = right.resize((round(right.width * h / right.height), h))
    canvas = Image.new("RGB", (left.width + right.width + 8, h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 8, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> int:
    O.mkdir(parents=True, exist_ok=True)
    state = load_state()
    test_hash = np.array([r[3] for r in state["test"]], dtype=np.uint64)
    video_of = [f"{r[0]}/{r[1]}" for r in state["test"]]
    env = dict(os.environ, PATH=str(Path(".venv/Scripts").resolve()) + os.pathsep + os.environ["PATH"])
    for chunk in CHUNKS:
        if str(chunk) in state["chunks"]:
            print(f"[chunk {chunk}] done before, skipping", flush=True)
            continue
        dest = SCRATCH / f"chunk_{chunk}"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        started = time.time()
        print(f"[chunk {chunk}] downloading wangsally/vimeo-90k-{chunk} ...", flush=True)
        subprocess.run([str(Path(".venv/Scripts/kaggle.exe").resolve()), "datasets", "download", "-d", f"wangsally/vimeo-90k-{chunk}",
                        "-p", str(dest)], check=True, env=env)
        zip_path = next(dest.glob("*.zip"))
        downloaded = time.time() - started
        members, hashes = [], []
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.filename.endswith(".png") and "/im" in info.filename:
                    members.append(info.filename)
                    hashes.append(phash(Image.open(io.BytesIO(z.read(info)))))
        vh = np.array(hashes, dtype=np.uint64)
        np.save(O / f"c1_vimeo_hashes_chunk{chunk}.npy", vh)
        found = 0
        best = np.full(len(test_hash), 65, dtype=np.int64)
        arg = np.zeros(len(test_hash), dtype=np.int64)
        for start in range(0, len(vh), 4096):
            d = np.bitwise_count(vh[start:start + 4096, None] ^ test_hash[None, :]).astype(np.int64)
            k = d.argmin(0)
            m = d[k, np.arange(d.shape[1])]
            better = m < best
            best[better], arg[better] = m[better], start + k[better]
            for i, j in zip(*np.nonzero(d <= THRESHOLD)):
                cand = {"distance": int(d[i, j]), "test_frame": state["test"][j][2],
                        "video": video_of[j], "vimeo": members[start + i], "chunk": chunk}
                state["candidates"].append(cand)
                found += 1
                if found <= 200:
                    thumb(cand["test_frame"], zip_path, cand["vimeo"], cand["distance"],
                          THUMBS / "candidates" / f"c{chunk}_{found:03d}_d{cand['distance']}.png")
        for key in sorted(set(video_of)):
            idx = [j for j, v in enumerate(video_of) if v == key]
            j = min(idx, key=lambda q: best[q])
            old = state["nearest"].get(key)
            if old is None or best[j] < old["distance"]:
                state["nearest"][key] = {"distance": int(best[j]), "test_frame": state["test"][j][2],
                                         "vimeo": members[arg[j]], "chunk": chunk}
                thumb(state["test"][j][2], zip_path, members[arg[j]], int(best[j]),
                      THUMBS / "nearest" / f"{key.replace('/', '_')}.png")
        state["chunks"][str(chunk)] = {"frames": len(members), "candidates": found,
                                       "download_s": round(downloaded),
                                       "hash_s": round(time.time() - started - downloaded)}
        save_state(state)
        shutil.rmtree(dest, ignore_errors=True)
        print(f"[chunk {chunk}] {len(members)} frames, {found} candidates "
              f"(download {downloaded:.0f} s); chunk deleted", flush=True)
    by_video = {}
    for c in state["candidates"]:
        by_video[c["video"]] = by_video.get(c["video"], 0) + 1
    summary = {"threshold": THRESHOLD, "test_frames": len(state["test"]),
               "vimeo_frames": sum(c["frames"] for c in state["chunks"].values()),
               "candidates": len(state["candidates"]), "candidates_by_video": by_video,
               "nearest": state["nearest"], "chunks": state["chunks"]}
    (O / "c1_results.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "nearest"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
