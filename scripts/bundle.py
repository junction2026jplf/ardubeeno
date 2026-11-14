"""Bundle the ONNX model, device_check.py and a small labelled test sample for the UNO Q."""
import argparse
import random
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("export/varroa.onnx"))
    p.add_argument("--cls-model", type=Path, default=Path("export/varroa_cls.onnx"),
                   help="optional classifier, bundled as varroa_cls.onnx if it exists")
    p.add_argument("--out", type=Path, default=Path("export/device"))
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=100, help="number of bees with mites, and without")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not args.model.exists():
        raise SystemExit(f"{args.model} not found, run `make export` first")

    labels = sorted((args.yolo_dir / "labels" / args.split).glob("*.txt"))
    pos = [l for l in labels if l.read_text().strip()]
    neg = [l for l in labels if not l.read_text().strip()]
    rng = random.Random(args.seed)
    sample = rng.sample(pos, min(args.n, len(pos))) + rng.sample(neg, min(args.n, len(neg)))

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "sample" / "images").mkdir(parents=True)
    (args.out / "sample" / "labels").mkdir(parents=True)
    shutil.copy(args.model, args.out / "varroa.onnx")
    if args.cls_model.exists():
        shutil.copy(args.cls_model, args.out / "varroa_cls.onnx")
    shutil.copy(Path(__file__).with_name("device_check.py"), args.out)
    shutil.copy("requirements-device.txt", args.out)
    for label in sample:
        image = args.yolo_dir / "images" / args.split / f"{label.stem}.png"
        shutil.copy(image, args.out / "sample" / "images" / image.name)  # follows the symlink
        shutil.copy(label, args.out / "sample" / "labels" / label.name)

    n_pos = min(args.n, len(pos))
    cls = f" + {args.cls_model}" if args.cls_model.exists() else ""
    print(f"bundled {args.model}{cls} + {len(sample)} test bees ({n_pos} with mites) into {args.out}/")


if __name__ == "__main__":
    main()
