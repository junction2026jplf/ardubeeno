"""Evaluate a per-bee classifier, optionally fused with a detector's top box score.

Ends with a `SUMMARY {...}` JSON line per scoring method.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def table(name: str, actual: np.ndarray, score: np.ndarray, conf: float) -> dict:
    rows = {}
    for t in sorted(set(THRESHOLDS.tolist() + [conf])):
        pred = score >= t
        tp, fp = int(np.sum(pred & actual)), int(np.sum(pred & ~actual))
        fn, tn = int(np.sum(~pred & actual)), int(np.sum(~pred & ~actual))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows[t] = dict(accuracy=(tp + tn) / len(actual), precision=precision, recall=recall, f1=f1)
    best_t = max(rows, key=lambda t: rows[t]["f1"])
    b, c = rows[best_t], rows[conf]
    print(f"{name:<10} best F1 {b['f1']:.3f} at {best_t:.2f} (P {b['precision']:.3f} R {b['recall']:.3f} "
          f"acc {b['accuracy']:.3f}) | at {conf:.2f}: F1 {c['f1']:.3f} P {c['precision']:.3f} R {c['recall']:.3f}")
    return {"method": name, "best_f1": b["f1"], "best_f1_conf": best_t, "best_precision": b["precision"],
            "best_recall": b["recall"], "f1": c["f1"], "precision": c["precision"], "recall": c["recall"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="runs/classify/varroa-cls/weights/best.pt")
    p.add_argument("--data", type=Path, default=Path("yolo_cls"))
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=288)
    p.add_argument("--det-weights", help="detector weights to fuse with, e.g. runs/detect/varroa/weights/best.pt")
    p.add_argument("--det-imgsz", default="320")
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"), help="detection dataset, for --det-weights")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    cls_model = YOLO(args.weights)
    mite = [i for i, n in cls_model.names.items() if n.startswith("mite")]  # mite, mite_underwing
    paths = sorted((args.data / args.split).glob("*/*.png"))
    actual = np.array([p.parent.name.startswith("mite") for p in paths])
    prob = np.array([float(r.probs.data[mite].sum()) for r in cls_model.predict(
        paths, imgsz=args.imgsz, device=args.device, batch=32, stream=True, verbose=False)])

    print(f"{len(paths)} bees ({actual.sum()} with mites), split {args.split}")
    results = [table("classifier", actual, prob, args.conf)]

    if args.det_weights:
        det_imgsz = [int(s) for s in args.det_imgsz.split(",")]
        det_imgsz = det_imgsz[0] if len(det_imgsz) == 1 else det_imgsz
        det_paths = [args.yolo_dir / "images" / args.split / p.name for p in paths]
        top = np.array([float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in YOLO(args.det_weights).predict(
            det_paths, imgsz=det_imgsz, conf=0.01, device=args.device, batch=32, stream=True, verbose=False)])
        results.append(table("detector", actual, top, args.conf))
        results.append(table("mean", actual, (prob + top) / 2, args.conf))
        results.append(table("max", actual, np.maximum(prob, top), args.conf))
        results.append(table("geomean", actual, np.sqrt(prob * top), args.conf))

    for r in results:
        r.update(weights=args.weights, det_weights=args.det_weights, split=args.split)
        print("SUMMARY " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))


if __name__ == "__main__":
    main()
