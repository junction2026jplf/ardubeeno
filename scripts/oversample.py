"""Build a training split that repeats the hard "mite under a wing" bees (gt_one.csv flag 3).

On the test set the baseline finds 94% of visible mites but only 58% of mites under a wing, which
make up about a fifth of the mite bees. Repeating them makes the model see them more often.
Writes yolo/{images,labels}/<name>/ (symlinks + label copies) and varroa_<name>.yaml.
"""
import argparse
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--gt-dir", type=Path, default=Path("data"))
    p.add_argument("--splits", default="train", help="comma-separated source splits, e.g. train,val")
    p.add_argument("--factor", type=int, default=3, help="total copies of each flag-3 bee")
    p.add_argument("--name", default="train_os")
    args = p.parse_args()

    img_out = args.yolo_dir / "images" / args.name
    lbl_out = args.yolo_dir / "labels" / args.name
    for d in (img_out, lbl_out):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    n_images = n_extra = 0
    for split in args.splits.split(","):
        flags = {Path(r[0]).stem: r[1] for r in (l.split() for l in open(args.gt_dir / split / "gt_one.csv")) if r}
        for img in sorted((args.yolo_dir / "images" / split).glob("*.png")):
            label = args.yolo_dir / "labels" / split / f"{img.stem}.txt"
            copies = args.factor if flags.get(img.stem) == "3" else 1
            for k in range(copies):
                stem = img.stem if k == 0 else f"{img.stem}_dup{k}"
                os.symlink(img.resolve(), img_out / f"{stem}.png")
                shutil.copy(label, lbl_out / f"{stem}.txt")
            n_images += copies
            n_extra += copies - 1

    yaml = args.yolo_dir.parent / f"varroa_{args.name}.yaml"
    yaml.write_text(
        f"path: {args.yolo_dir.resolve()}\n"
        f"train: images/{args.name}\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: varroa\n"
    )
    print(f"{args.name}: {n_images} images from {args.splits} ({n_extra} extra copies of flag-3 bees), wrote {yaml}")


if __name__ == "__main__":
    main()
