"""Train a YOLO mite detector on the converted dataset."""
import argparse

from ultralytics import YOLO


def screen_augmentations():
    """Albumentations that mimic a camera filming a screen: blur, sensor noise, JPEG, resampling, exposure."""
    import albumentations as A

    return [
        A.OneOf([A.MotionBlur(blur_limit=(3, 5)), A.GaussianBlur(blur_limit=(3, 5)), A.Defocus(radius=(1, 2))], p=0.3),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        A.ImageCompression(quality_range=(40, 90), p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.RandomGamma(gamma_limit=(70, 130), p=0.2),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="varroa.yaml")
    p.add_argument("--model", default="yolo11n.pt", help="base weights, or a best.pt to fine-tune")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="mps", help="mps, cpu, or a CUDA index")
    p.add_argument("--name", default="varroa")
    p.add_argument("--fraction", type=float, default=1.0, help="use a subset of train for quick tests")
    p.add_argument("--patience", type=int, default=15, help="early-stopping patience, 0 = never stop early")
    p.add_argument("--optimizer", default="auto", help="auto, AdamW, SGD, ...")
    p.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (ignored with --optimizer auto)")
    p.add_argument("--cos-lr", action="store_true", help="cosine instead of linear learning-rate decay")
    p.add_argument("--close-mosaic", type=int, default=10, help="final epochs trained without mosaic")
    p.add_argument("--workers", type=int, default=8, help="dataloader workers")
    p.add_argument("--cache", default="", help="'ram' or 'disk' to cache decoded images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--screen-aug", action="store_true", help="stronger blur/noise/JPEG augmentation (albumentations)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU for the per-epoch validation")
    p.add_argument("--save-period", type=int, default=-1, help="also save a checkpoint every N epochs")
    args = p.parse_args()

    extra = {"augmentations": screen_augmentations()} if args.screen_aug else {}
    model = YOLO(args.model)
    model.train(
        **extra,
        iou=args.iou,
        save_period=args.save_period,
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        exist_ok=True,
        fraction=args.fraction,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        workers=args.workers,
        cache=args.cache or False,
        seed=args.seed,
        # MPS has no deterministic kernels for some ops; avoids per-epoch warnings.
        deterministic=False,
        # Augmentations that help when the camera films a screen.
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=15,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )


if __name__ == "__main__":
    main()
