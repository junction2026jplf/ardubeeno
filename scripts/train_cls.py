"""Train a per-bee mite / clean classifier on the dataset from convert_cls.py."""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="yolo_cls")
    p.add_argument("--model", default="yolo11n-cls.pt")
    p.add_argument("--imgsz", type=int, default=288)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="mps")
    p.add_argument("--name", default="varroa-cls")
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience, 0 = never stop early")
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache", default="", help="'ram' or 'disk' to cache decoded images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto-augment", default="randaugment", help="randaugment, autoaugment, augmix or none")
    # Colour jitter only applies with --auto-augment none (Ultralytics disables it under RandAugment).
    # --auto-augment none --hsv-h 0 --hsv-s 0 --hsv-v 0 trains with geometric augmentation only.
    p.add_argument("--hsv-h", type=float, default=0.02)
    p.add_argument("--hsv-s", type=float, default=0.5)
    p.add_argument("--hsv-v", type=float, default=0.4)
    args = p.parse_args()

    YOLO(args.model).train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=True,
        workers=args.workers,
        cache=args.cache or False,
        seed=args.seed,
        auto_augment=None if args.auto_augment == "none" else args.auto_augment,
        deterministic=False,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=15,
        flipud=0.5,
        fliplr=0.5,
        # RandomResizedCrop keeps 80-100% of the area, so the mite stays in the crop.
        scale=0.2,
        # Random erasing could erase the only mite and leave a "mite" label on a clean bee.
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
