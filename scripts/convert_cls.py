"""Build a per-bee classification dataset (mite / clean) from the YOLO detection dataset.

Crops are letterboxed to squares first: Ultralytics centre-crops classification images, which would
cut the top and bottom off the tall 160x280 bee crops (and with them some mites).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def letterbox_square(img, size: int):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--out", type=Path, default=Path("yolo_cls"))
    p.add_argument("--size", type=int, default=280, help="side of the square images written")
    p.add_argument("--trainval", action="store_true", help="also put the val bees into train (final model)")
    p.add_argument("--three-class", action="store_true",
                   help="separate class mite_underwing for gt_one.csv flag 3 (mite under a wing)")
    p.add_argument("--gt-dir", type=Path, default=Path("data"), help="original data/, for --three-class")
    args = p.parse_args()

    sources = {"train": ["train"] + (["val"] if args.trainval else []), "val": ["val"], "test": ["test"]}
    for split, src_splits in sources.items():
        counts = {"mite": 0, "clean": 0, **({"mite_underwing": 0} if args.three_class else {})}
        for cls in counts:
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)
        for src in src_splits:
            flags = {}
            if args.three_class:
                flags = {Path(r[0]).stem: r[1] for r in (l.split() for l in open(args.gt_dir / src / "gt_one.csv")) if r}
            for img_path in sorted((args.yolo_dir / "images" / src).glob("*.png")):
                label = args.yolo_dir / "labels" / src / f"{img_path.stem}.txt"
                cls = "mite" if label.exists() and label.read_text().strip() else "clean"
                if cls == "mite" and flags.get(img_path.stem) == "3":
                    cls = "mite_underwing"
                out = args.out / split / cls / img_path.name
                if not out.exists():
                    cv2.imwrite(str(out), letterbox_square(cv2.imread(str(img_path)), args.size))
                counts[cls] += 1
        print(f"{split:>5}: " + ", ".join(f"{v} {k}" for k, v in counts.items()) + f"  (from {', '.join(src_splits)})")


if __name__ == "__main__":
    main()
