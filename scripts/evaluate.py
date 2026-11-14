"""Evaluate a trained model: YOLO box metrics, per-bee "has mite?" accuracy and mite counts.

Ends with a `SUMMARY {...}` JSON line so results from many runs are easy to collect.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO


THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]


def load_flags(csv: Path) -> dict:
    """Image stem -> annotation flag from gt_one.csv (1 = visible mite, 3 = mite under a wing)."""
    if not csv.exists():
        return {}
    return {Path(p[0]).stem: p[1] for p in (l.split() for l in csv.open()) if p}


def per_bee_metrics(model: YOLO, images: Path, labels: Path, imgsz, conf: float, iou: float, device: str,
                    flags: dict) -> dict:
    """Per-bee "has a mite?" metrics over a range of thresholds, plus count accuracy at `conf`."""
    # A bee counts as positive at threshold t if its highest-scoring box is >= t,
    # so one low-threshold pass covers every threshold.
    gt_counts, scores = [], []
    paths = sorted(images.glob("*.png"))
    bee_flags = [flags.get(p.stem, "?") for p in paths]
    for r in model.predict(paths, imgsz=imgsz, conf=min(THRESHOLDS), iou=iou, device=device,
                           agnostic_nms=True, batch=32, stream=True, verbose=False):
        label = labels / (Path(r.path).stem + ".txt")
        lines = label.read_text().splitlines() if label.exists() else []
        gt_counts.append(sum(1 for l in lines if l.strip()))
        scores.append(r.boxes.conf.cpu().numpy())

    actual = [n > 0 for n in gt_counts]
    top = [float(s.max()) if len(s) else 0.0 for s in scores]
    print(f"\nPer-bee classification ({len(actual)} bees, {sum(actual)} with mites):")
    print("  conf  accuracy  precision  recall     F1     TP    FP    TN    FN")
    rows = {}
    for t in sorted(set(THRESHOLDS + [conf])):
        tp = sum(a and s >= t for a, s in zip(actual, top))
        fp = sum(not a and s >= t for a, s in zip(actual, top))
        fn = sum(a and s < t for a, s in zip(actual, top))
        tn = len(actual) - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[t] = dict(accuracy=(tp + tn) / len(actual), precision=precision, recall=recall, f1=f1)
        mark = "  <- CONF" if t == conf else ""
        print(f"  {t:.2f}  {(tp + tn) / len(actual):8.3f}  {precision:9.3f}  {recall:6.3f}  {f1:6.3f}"
              f"  {tp:5d} {fp:5d} {tn:5d} {fn:5d}{mark}")

    gt = np.array(gt_counts)
    pred = np.array([int((s >= conf).sum()) for s in scores])
    exact, mae = float(np.mean(gt == pred)), float(np.abs(gt - pred).mean())
    print(f"\nMite counts at conf={conf}, iou={iou}: exact {exact:.3f}, MAE {mae:.3f}, "
          f"total {pred.sum()} predicted vs {gt.sum()} labelled")

    by_flag = {}
    for flag, name in [("1", "visible"), ("3", "underwing")]:
        hits = [s >= conf for s, f, a in zip(top, bee_flags, actual) if a and f == flag]
        if hits:
            by_flag[f"recall_{name}"] = sum(hits) / len(hits)
    if by_flag:
        print("Recall by mite type at conf={}: ".format(conf) + ", ".join(f"{k[7:]} {v:.3f}" for k, v in by_flag.items()))

    best_t = max(rows, key=lambda t: rows[t]["f1"])
    return {**by_flag,
        "best_f1": rows[best_t]["f1"], "best_f1_conf": best_t,
        "f1": rows[conf]["f1"], "recall": rows[conf]["recall"], "precision": rows[conf]["precision"],
        "accuracy": rows[conf]["accuracy"], "count_exact": exact, "count_mae": mae,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="runs/detect/varroa/weights/best.pt")
    p.add_argument("--data", default="varroa.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--imgsz", default="320", help="square size, or height,width")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU; 0.7 leaves duplicate boxes on small mites")
    p.add_argument("--device", default="mps")
    p.add_argument("--skip-val", action="store_true", help="skip the (slow) YOLO box metrics")
    p.add_argument("--gt-dir", type=Path, default=Path("data"), help="original data/, for recall by mite type")
    args = p.parse_args()

    imgsz = [int(s) for s in args.imgsz.split(",")]
    imgsz = imgsz[0] if len(imgsz) == 1 else imgsz
    model = YOLO(args.weights)
    summary = {"weights": args.weights, "split": args.split, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou}

    if not args.skip_val:
        m = model.val(data=args.data, split=args.split, imgsz=imgsz, iou=args.iou, device=args.device, plots=True)
        print(f"\nBox metrics ({args.split}): P={m.box.mp:.3f} R={m.box.mr:.3f} "
              f"mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")
        summary.update(box_p=m.box.mp, box_r=m.box.mr, map50=m.box.map50, map=m.box.map)

    summary.update(per_bee_metrics(
        model,
        args.yolo_dir / "images" / args.split,
        args.yolo_dir / "labels" / args.split,
        imgsz,
        args.conf,
        args.iou,
        args.device,
        load_flags(args.gt_dir / args.split / "gt_one.csv"),
    ))
    print("SUMMARY " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}))


if __name__ == "__main__":
    main()
