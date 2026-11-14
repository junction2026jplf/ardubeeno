"""Add EV2 bee crops (Zenodo 13771384, CC BY 4.0) to the per-bee classifier's training split.

EV2's PNGs are already bee crops, one per labelled video frame:
- dataset_infested/: varroa_visible = yes                                   -> mite
- dataset_free/ from varroa_free/ videos                                    -> clean
- dataset_free/ from varroa_infested/ videos (infested, mite not visible)   -> dropped (ambiguous)
Rows are mostly consecutive frames of 32 videos, so every k-th frame per video is kept, capped per video.
Crops are landscape and larger than ours: they are rotated to portrait and letterboxed to the same
square as convert_cls.py. Our val/test images are linked unchanged; only train gains EV2 images.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from convert_cls import letterbox_square


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ev2-dir", type=Path, default=Path.home() / "ev2")
    p.add_argument("--base", type=Path, default=Path("yolo_cls"), help="our classifier dataset (convert_cls.py)")
    p.add_argument("--out", type=Path, default=Path("yolo_cls_ev2"))
    p.add_argument("--every", type=int, default=5, help="keep every k-th labelled frame per video")
    p.add_argument("--cap", type=int, default=60, help="max frames per video after subsampling")
    p.add_argument("--size", type=int, default=280)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.ev2_dir / "labels.txt") if l.strip()]
    per_video = defaultdict(list)
    for r in rows:
        video = r["video"]
        visible = r["varroa_visible"] == "yes"
        if not visible and not video.startswith("varroa_free/"):
            continue  # infested bee without a visible mite: ambiguous for a single-image classifier
        name = f"{Path(video).name}_{r['id'].replace('_', '')}.png"
        folder = "dataset_infested" if visible else "dataset_free"
        frame = int(r["id"].split("_")[-1])
        per_video[video].append((frame, args.ev2_dir / folder / name, "mite" if visible else "clean"))

    rng = random.Random(args.seed)
    for split in ["train", "val", "test"]:
        (args.out / split).mkdir(parents=True, exist_ok=True)
    for cls in ["mite", "clean"]:
        d = args.out / "train" / cls
        d.mkdir(exist_ok=True)
        # Our own training crops, as symlinks.
        for src in sorted((args.base / "train" / cls).glob("*.png")):
            link = d / src.name
            if not link.exists():
                os.symlink(src.resolve(), link)
    # Ultralytics follows symlinked files but not symlinked folders, so link file by file.
    for split in ["val", "test"]:
        for cls in ["mite", "clean"]:
            d = args.out / split / cls
            d.mkdir(exist_ok=True)
            for src in sorted((args.base / split / cls).glob("*.png")):
                link = d / src.name
                if not link.exists():
                    os.symlink(src.resolve(), link)

    added, per_video_kept = Counter(), {}
    for video, items in sorted(per_video.items()):
        items.sort()
        kept = items[::args.every]
        if len(kept) > args.cap:
            step = len(kept) / args.cap
            kept = [kept[int(i * step)] for i in range(args.cap)]
        per_video_kept[video] = len(kept)
        for frame, src, cls in kept:
            img = cv2.imread(str(src))
            if img is None:
                raise SystemExit(f"missing {src}")
            if img.shape[1] > img.shape[0]:  # landscape -> portrait like our upright bees, either direction
                img = cv2.rotate(img, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
            out = args.out / "train" / cls / f"ev2_{Path(video).name}_{frame}.png"
            cv2.imwrite(str(out), letterbox_square(img, args.size))
            added[cls] += 1

    base = {c: len(list((args.base / "train" / c).glob("*.png"))) for c in ["mite", "clean"]}
    print(f"EV2 rows {len(rows)}, usable videos {len(per_video)}, every {args.every}, cap {args.cap}")
    print(f"added: {added['mite']} mite, {added['clean']} clean  |  "
          f"train now: {base['mite'] + added['mite']} mite, {base['clean'] + added['clean']} clean")
    print("kept per video:", ", ".join(f"{Path(v).name}={n}" for v, n in per_video_kept.items()))


if __name__ == "__main__":
    main()
