"""docs/PLAN_audit2.md, check C3: split integrity.

1. DAVIS TRAIN, VAL-A, VAL-B and TEST video lists are disjoint (VAL-B = odd-indexed half of
   VAL, as m21_refinement.val_b_sequences picks it; VAL-A = the even half).
2. The frame folders hold only their own split's videos (by file-name prefix).
3. UVG and MCL-JCV video names never appear in any DAVIS or Vimeo manifest.
4. Near-duplicate frames between DAVIS TEST and DAVIS TRAIN/VAL (same pHash and threshold as
   C1, <= 6 of 64 bits). DAVIS has same-event families (drift-*, bmx-*, car-*), so this is
   reported per video pair.

    .venv/Scripts/python outputs/p12_audit2/c3_splits.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "outputs/p12_audit2")
sys.path.insert(0, "scripts")
from c1_leakage import THRESHOLD, phash  # noqa: E402
from nvc.evaluation.sequences import discover_sequences  # noqa: E402
from nvc.utils.config import load_default_config  # noqa: E402

O = Path("outputs/p12_audit2")


def main() -> int:
    manifest = load_default_config().processed_data_dir / "manifest.json"
    split = {s: [q.sequence_id for q in discover_sequences(manifest, split=s)]
             for s in ("train", "val", "test")}
    val_b = split["val"][1::2]
    val_a = split["val"][0::2]
    sets = {"train": set(split["train"]), "val_a": set(val_a), "val_b": set(val_b),
            "test": set(split["test"])}
    overlaps = {f"{a}&{b}": sorted(sets[a] & sets[b]) for a in sets for b in sets if a < b}
    out = {"counts": {k: len(v) for k, v in sets.items()}, "val_a": val_a, "val_b": val_b,
           "test": split["test"], "overlaps": overlaps,
           "disjoint": not any(overlaps.values())}
    # 2. folder contents by prefix
    folders = {}
    for folder, names in (("train", sets["train"]), ("val", sets["val_a"] | sets["val_b"]),
                          ("test", sets["test"])):
        prefixes = Counter(p.name.rsplit("_", 1)[0] for p in Path(f"data/frames/{folder}").glob("*.png"))
        folders[folder] = {"videos_in_folder": len(prefixes),
                           "foreign": sorted(set(prefixes) - names)}
    out["folders"] = folders
    # 3. external test names in training/validation manifests
    external = [i["source_name"] for root in ("C:/uvg_scratch/uvg", "C:/mcljcv_scratch/mcljcv")
                for i in json.loads(Path(root, "manifest.json").read_text(encoding="utf-8"))["items"]]
    texts = {p.name: p.read_text(encoding="utf-8") for p in Path("data/processed").glob("*.json")}
    out["external_names_in_manifests"] = {n: [f for f, t in texts.items() if n in t]
                                          for n in external if any(n in t for t in texts.values())}
    # 4. near-duplicates DAVIS TEST vs TRAIN+VAL
    def hashes(names):
        rows = []
        for n in sorted(names):
            for p in sorted(Path(f"data/external/DAVIS/JPEGImages/480p/{n}").glob("*.jpg")):
                rows.append((n, p.name, phash(Image.open(p))))
        return rows
    test_rows = hashes(sets["test"])
    other_rows = hashes(sets["train"] | sets["val_a"] | sets["val_b"])
    th = np.array([r[2] for r in test_rows], dtype=np.uint64)
    oh = np.array([r[2] for r in other_rows], dtype=np.uint64)
    d = np.bitwise_count(th[:, None] ^ oh[None, :])
    pairs = Counter()
    closest = {}
    for i, j in zip(*np.nonzero(d <= THRESHOLD)):
        pairs[f"{test_rows[i][0]} ~ {other_rows[j][0]}"] += 1
    for n in sets["test"]:
        idx = [i for i, r in enumerate(test_rows) if r[0] == n]
        k = int(d[idx].min())
        i, j = np.unravel_index(int(d[idx].argmin()), (len(idx), len(other_rows)))
        closest[n] = {"distance": k, "frame": test_rows[idx[i]][1], "other": other_rows[j][0],
                      "other_frame": other_rows[j][1]}
    out["davis_near_duplicates"] = {"threshold": THRESHOLD, "pairs": dict(pairs),
                                    "closest_per_test_video": closest}
    (O / "c3_splits.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
