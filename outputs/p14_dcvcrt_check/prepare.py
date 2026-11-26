"""docs/PLAN_dcvcrt_check.md: hard-link the exact PNG frames our codec codes into DCVC-RT's
layout (one folder per video, im00001.png ...) and write its test config (RGB path, I-frame
every 30 via --force_intra_period on the command line)."""
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path("C:/dcvc_scratch/data")
SETS = {"DAVIS": Path("data/processed/manifest.json"), "UVG": Path("C:/uvg_scratch/uvg/manifest.json")}
config = {"root_path": str(ROOT), "test_classes": {}}
total = {}
for name, manifest in SETS.items():
    m = json.loads(manifest.read_text(encoding="utf-8"))
    seqs = {}
    for item in (i for i in m["items"] if i.get("split") == "test"):
        src = manifest.parent / item["frame_directory"]
        dst = ROOT / name / item["source_name"]
        dst.mkdir(parents=True, exist_ok=True)
        n = item["frame_count"]
        for k in range(1, n + 1):
            target = dst / f"im{k:05d}.png"
            if not target.exists():
                os.link(src / item["frame_filename_pattern"].format(index=k), target)
        w, h = Image.open(dst / "im00001.png").size
        seqs[item["source_name"]] = {"width": w, "height": h, "frames": n, "intra_period": 30}
    config["test_classes"][name] = {"test": 1, "base_path": name, "src_type": "png", "sequences": seqs}
    total[name] = (len(seqs), sum(s["frames"] for s in seqs.values()), {(s["width"], s["height"]) for s in seqs.values()})
Path("outputs/p14_dcvcrt_check/dcvc_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
print(total)
