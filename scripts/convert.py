"""Convert the Varroa bee-crop dataset (gt_one.csv) into YOLO format.

Images are symlinked, not copied. Writes yolo/{images,labels}/{split} and varroa.yaml.
"""
import argparse
import os
from pathlib import Path

from PIL import Image

SPLITS = ["train", "val", "test"]


def convert_split(src: Path, dst: Path, split: str, two_class: bool = False) -> tuple[int, int]:
    img_dir = dst / "images" / split
    lbl_dir = dst / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0
    for line in open(src / split / "gt_one.csv"):
        parts = line.split()
        if not parts:
            continue
        rel, coords = parts[0], list(map(float, parts[2:]))
        # gt_one.csv flag: 1 = visible mite, 3 = mite under or through a wing.
        cls = 1 if two_class and parts[1] == "3" else 0
        img_src = (src / split / rel).resolve()
        name = Path(rel).stem
        w, h = Image.open(img_src).size

        img_link = img_dir / f"{name}.png"
        if not img_link.exists():
            os.symlink(img_src, img_link)

        rows = []
        for i in range(0, len(coords) - 3, 4):
            x1, y1, x2, y2 = coords[i:i + 4]
            x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            rows.append(
                f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        (lbl_dir / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n_images += 1
        n_boxes += len(rows)
    return n_images, n_boxes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("data"))
    p.add_argument("--dst", type=Path, default=Path("yolo"))
    p.add_argument("--yaml", type=Path, default=Path("varroa.yaml"))
    p.add_argument("--two-class", action="store_true",
                   help="class 1 for mites under a wing (flag 3), e.g. --dst yolo2 --yaml varroa_2cls.yaml")
    args = p.parse_args()

    for split in SPLITS:
        n_images, n_boxes = convert_split(args.src, args.dst, split, args.two_class)
        print(f"{split:>5}: {n_images} images, {n_boxes} mite boxes")

    names = "names:\n  0: varroa\n" + ("  1: varroa_underwing\n" if args.two_class else "")
    args.yaml.write_text(
        f"path: {args.dst.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"{names}"
    )
    print(f"wrote {args.yaml}")

    # Final-model variant: train on train+val (val is then only a sanity check, not held out).
    trainval = args.yaml.with_name(args.yaml.stem + "_trainval.yaml")
    trainval.write_text(
        f"path: {args.dst.resolve()}\n"
        "train: [images/train, images/val]\n"
        "val: images/val\n"
        "test: images/test\n"
        f"{names}"
    )
    print(f"wrote {trainval}")


if __name__ == "__main__":
    main()
