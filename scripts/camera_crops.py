"""Turn camera-on-screen recordings into mite training data, cut out exactly like the multi-bee pipeline does.

Each recorded frame has bee boxes (class 0) and mite boxes (class 1). Every bee is cut out with the app's
crop_bee, its mites are moved into the crop's coordinates, and the crops are written as:

  yolo_cam/images|labels/<split>/   mite detector data (YOLO, one class)
  yolo_cam_cls/<split>/mite|clean/  classifier data

plus two dataset definitions that mix the camera crops with the original crops, so fine-tuning doesn't forget them:

  varroa_cam.yaml   detector: original train+val + camera train, val/test on camera crops
  yolo_cam_mix/     classifier: original train + camera train, val/test on camera crops (symlinks)
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "applab" / "ardubeeno" / "python"))
from bees import crop_bee  # noqa: E402


def read_labels(path, W, H):
    """Return (bee boxes, mite boxes) in pixels as (x, y, w, h)."""
    bees, mites = [], []
    for line in Path(path).read_text().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        (bees if c == "0" else mites).append((cx - w / 2, cy - h / 2, w, h))
    return bees, mites


def mite_in_crop(shape, bee, mite):
    """The mite's box in crop pixels, found by cutting a mask of it out with the same crop_bee."""
    mask = np.zeros(shape[:2], np.uint8)
    x, y, w, h = mite
    cv2.rectangle(mask, (round(x), round(y)), (round(x + w) - 1, round(y + h) - 1), 255, -1)
    ys, xs = np.nonzero(crop_bee(mask, bee) >= 128)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def owner(bees, mite):
    """Index of the bee box that contains the mite's centre (the smallest one if several do), or None."""
    mx, my = mite[0] + mite[2] / 2, mite[1] + mite[3] / 2
    inside = [i for i, (x, y, w, h) in enumerate(bees) if x <= mx <= x + w and y <= my <= y + h]
    return min(inside, key=lambda i: bees[i][2] * bees[i][3]) if inside else None


def convert(rec, split, out_det, out_cls, stats):
    rec = Path(rec)
    tag = rec.name.replace("screen-", "")
    for d in (out_det / "images" / split, out_det / "labels" / split, out_cls / split / "mite", out_cls / split / "clean"):
        d.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(rec / "manifest.csv")))
    for row in rows:
        frame = cv2.imread(str(rec / row["image"]))
        H, W = frame.shape[:2]
        bees, mites = read_labels(rec / row["label"], W, H)
        flags = row["source_labels"].split(";") if row["source_labels"] else []  # gt_one.csv flag, "0" = no mite
        per_bee = [[] for _ in bees]
        for m in mites:
            i = owner(bees, m)
            if i is None:
                stats["orphan_mites"] += 1
            else:
                per_bee[i].append(m)
        if len(flags) == len(bees):
            stats["label_mismatch_bees"] += sum((f != "0") != bool(p) for f, p in zip(flags, per_bee))
        stem = Path(row["image"]).stem
        for i, bee in enumerate(bees):
            crop = crop_bee(frame, bee)
            ch, cw = crop.shape[:2]
            lines = []
            for m in per_bee[i]:
                b = mite_in_crop(frame.shape, bee, m)
                if b is None:
                    stats["lost_mites"] += 1
                    continue
                x0, y0, x1, y1 = b
                lines.append(f"0 {(x0 + x1) / 2 / cw:.6f} {(y0 + y1) / 2 / ch:.6f} {(x1 - x0) / cw:.6f} {(y1 - y0) / ch:.6f}")
            name = f"cam{tag}_{stem}_{i:02d}"
            cv2.imwrite(str(out_det / "images" / split / f"{name}.png"), crop)
            (out_det / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            cls = "mite" if lines else "clean"
            os.link(out_det / "images" / split / f"{name}.png", out_cls / split / cls / f"{name}.png")
            stats[f"{split}_{cls}"] += 1
    stats[f"{split}_frames"] += len(rows)


def link_dir(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if not target.exists():
            target.symlink_to(f.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", default=["captures/screen-20261114-124423"])
    p.add_argument("--val", nargs="+", default=["captures/screen-20261114-130127"])
    p.add_argument("--test", nargs="+", default=["captures/screen-20261114-122148"])
    p.add_argument("--out-det", default="yolo_cam")
    p.add_argument("--out-cls", default="yolo_cam_cls")
    p.add_argument("--mix-cls", default="yolo_cam_mix")
    p.add_argument("--yaml", default="varroa_cam.yaml")
    args = p.parse_args()

    root = Path.cwd()
    out_det, out_cls, mix = root / args.out_det, root / args.out_cls, root / args.mix_cls
    for d in (out_det, out_cls, mix):
        if d.exists():
            sys.exit(f"{d} already exists, delete it first")
    from collections import Counter
    stats = Counter()
    for split in ("train", "val", "test"):
        for rec in getattr(args, split):
            convert(rec, split, out_det, out_cls, stats)

    # Classifier mix: original train crops + camera train crops; val/test camera only.
    for cls in ("mite", "clean"):
        link_dir(root / "yolo_cls" / "train" / cls, mix / "train" / cls)
        link_dir(out_cls / "train" / cls, mix / "train" / cls)
        for split in ("val", "test"):
            link_dir(out_cls / split / cls, mix / split / cls)

    Path(args.yaml).write_text(
        f"path: {root}\n"
        f"train: [yolo/images/train, yolo/images/val, {args.out_det}/images/train]\n"
        f"val: {args.out_det}/images/val\n"
        f"test: {args.out_det}/images/test\n"
        "names:\n  0: varroa\n")

    for split in ("train", "val", "test"):
        print(f"{split:5s}  {stats[f'{split}_frames']:4d} frames  {stats[f'{split}_mite']:5d} mite  "
              f"{stats[f'{split}_clean']:5d} clean")
    print(f"mites outside every bee box: {stats['orphan_mites']}, lost in cropping: {stats['lost_mites']}, "
          f"bees whose mite yes/no differs from the source label: {stats['label_mismatch_bees']}")
    print(f"wrote {args.out_det}/, {args.out_cls}/, {args.mix_cls}/, {args.yaml}")


if __name__ == "__main__":
    main()
