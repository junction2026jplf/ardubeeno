"""Ceiling test: fine-tune a large pretrained backbone (timm) as the per-bee classifier.

Same data as train_cls.py (yolo_cls from convert_cls.py) and the same protocol as evaluate_ens.py: the
threshold is chosen on val and applied to test, for the classifier alone and fused (mean) with the detector's
top box score. If a much stronger backbone doesn't beat YOLO11n-cls, the limit is the data, not the model;
if it does, it can teach the small model. Writes <out>/{best,last}.pt and <out>/probs.npz (val/test scores).

Needs `pip install timm`. Examples:
  python scripts/train_timm.py --model convnext_tiny.fb_in22k_ft_in1k --det-weights ... --det-weights-val ...
  python scripts/train_timm.py --model vit_small_patch14_dinov2.lvd142m --imgsz 280 --lr 5e-5 ...
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from evaluate_ens import THRESHOLDS, cls_probs, det_top, f1_at


class Crops(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform
        self.labels = [int(q.parent.name.startswith("mite")) for q in paths]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        for x, _ in loader:
            out.append(model(x.to(device, non_blocking=True)).float().softmax(1)[:, 1].cpu())
    return torch.cat(out).numpy()


def best_f1(actual, score):
    t = max(THRESHOLDS, key=lambda t: f1_at(actual, score, t)[0])
    return f1_at(actual, score, t)[0], t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="convnext_tiny.fb_in22k_ft_in1k")
    p.add_argument("--data", type=Path, default=Path("yolo_cls"))
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--imgsz", type=int, default=288, help="use a multiple of 14 (e.g. 280) for DINOv2 ViTs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.998, help="EMA decay of the weights that are evaluated")
    p.add_argument("--no-randaugment", dest="randaugment", action="store_false")
    p.add_argument("--det-weights", help="detector to fuse with on test, e.g. n320-tv-e40-screen last.pt")
    p.add_argument("--det-weights-val", help="train-only detector for the val threshold, e.g. n320-long best.pt")
    p.add_argument("--det-imgsz", default="320")
    p.add_argument("--cls-weights", help="YOLO11n-cls weights, to also report the timm + YOLO-cls ensemble")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="default runs/timm/<model>-s<seed>")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_device = "0" if device == "cuda" else "cpu"
    out = args.out or Path("runs/timm") / f"{args.model.split('.')[0]}-s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    try:  # ViTs need the input size to resize their position embeddings
        model = timm.create_model(args.model, pretrained=True, num_classes=2, drop_path_rate=args.drop_path,
                                  img_size=args.imgsz)
    except TypeError:
        model = timm.create_model(args.model, pretrained=True, num_classes=2, drop_path_rate=args.drop_path)
    model.to(device)
    cfg = resolve_data_config({}, model=model)
    norm = [T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize(cfg["mean"], cfg["std"])]
    # Same augmentation as train_cls.py: crops keep 80-100% of the area (the mite stays in), no erasing.
    train_tf = T.Compose([
        T.RandomResizedCrop(args.imgsz, scale=(0.8, 1.0), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(15, fill=114),
        T.ColorJitter(brightness=0.4, saturation=0.5, hue=0.02),
        *([T.RandAugment()] if args.randaugment else []), *norm])
    eval_tf = T.Compose([T.Resize(args.imgsz, antialias=True), *norm])

    paths = {s: sorted((args.data / s).glob("*/*.png")) for s in ["train", "val", "test"]}
    actual = {s: np.array([q.parent.name.startswith("mite") for q in paths[s]]) for s in ["val", "test"]}
    train = DataLoader(Crops(paths["train"], train_tf), args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    loaders = {s: DataLoader(Crops(paths[s], eval_tf), 128, num_workers=args.workers, pin_memory=True)
               for s in ["val", "test"]}

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps, warmup = args.epochs * len(train), len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))))
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = ModelEmaV2(model, decay=args.ema)
    best = -1.0
    print(f"{args.model}: {len(paths['train'])} train, {len(paths['val'])} val, {len(paths['test'])} test -> {out}")
    for epoch in range(args.epochs):
        model.train()
        start, total = time.time(), 0.0
        for x, y in train:
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = loss_fn(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.update(model)
            total += loss.item()
        vf, vt = best_f1(actual["val"], predict(ema.module, loaders["val"], device))
        if vf > best:
            best = vf
            torch.save(ema.module.state_dict(), out / "best.pt")
        print(f"epoch {epoch + 1:3d}  loss {total / len(train):.4f}  val F1 {vf:.4f} @ {vt:.2f}  "
              f"({time.time() - start:.0f}s)", flush=True)
    torch.save(ema.module.state_dict(), out / "last.pt")

    scores = {}
    for ckpt in ["best", "last"]:
        ema.module.load_state_dict(torch.load(out / f"{ckpt}.pt", map_location=device))
        scores[ckpt] = {s: predict(ema.module, loaders[s], device) for s in ["val", "test"]}
    if args.cls_weights:
        scores["yolo-cls"] = {s: cls_probs(args.cls_weights, paths[s], 288, yolo_device) for s in ["val", "test"]}
    top = None
    if args.det_weights:
        det_imgsz = [int(s) for s in args.det_imgsz.split(",")]
        det_imgsz = det_imgsz[0] if len(det_imgsz) == 1 else det_imgsz
        top = {s: det_top(args.det_weights_val if s == "val" and args.det_weights_val else args.det_weights,
                          [args.yolo_dir / "images" / s / q.name for q in paths[s]], det_imgsz, yolo_device)
               for s in ["val", "test"]}
    np.savez(out / "probs.npz", **{f"{k}_{s}": v[s] for k, v in scores.items() for s in v},
             **({f"det_{s}": top[s] for s in top} if top else {}),
             **{f"names_{s}": np.array([q.name for q in paths[s]]) for s in ["val", "test"]},
             **{f"actual_{s}": actual[s] for s in actual})

    combos = [[k] for k in scores] + ([["best", "yolo-cls"], ["last", "yolo-cls"]] if args.cls_weights else [])
    print(f"\n{'combination':40s} val F1  thr   test F1 (P / R)")
    for combo in combos:
        for fused in ([False, True] if top else [False]):
            s = {sp: np.mean([scores[k][sp] for k in combo], 0) for sp in ["val", "test"]}
            if fused:
                s = {sp: (s[sp] + top[sp]) / 2 for sp in s}
            vf, t = best_f1(actual["val"], s["val"])
            tf, tp, tr = f1_at(actual["test"], s["test"], t)
            label = " + ".join(combo) + (" + det" if fused else "")
            print(f"{label:40s} {vf:.3f}  {t:.2f}  {tf:.3f} ({tp:.3f} / {tr:.3f})")
            print("SUMMARY " + json.dumps({"model": args.model, "seed": args.seed, "combo": label, "fused": fused,
                                           "val_f1": round(vf, 4), "thr": float(t), "test_f1": round(tf, 4),
                                           "test_p": round(tp, 4), "test_r": round(tr, 4)}))


if __name__ == "__main__":
    main()
