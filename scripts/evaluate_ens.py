"""Evaluate an ensemble of per-bee classifiers (mean probability), optionally fused with a detector.

Thresholds are chosen on val and then applied to test, so the test numbers are honest. Prints every
single classifier, every subset of 2+ classifiers, and each of those fused (mean) with the detector.
Ends with one `SUMMARY {...}` JSON line per combination.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def f1_at(actual, score, t):
    pred = score >= t
    tp, fp, fn = np.sum(pred & actual), np.sum(pred & ~actual), np.sum(~pred & actual)
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)


def cls_probs(weights, paths, imgsz, device):
    model = YOLO(weights)
    mite = [i for i, n in model.names.items() if n.startswith("mite")]
    return np.array([float(r.probs.data[mite].sum()) for r in model.predict(
        paths, imgsz=imgsz, device=device, batch=32, stream=True, verbose=False)])


def det_top(weights, paths, imgsz, device):
    return np.array([float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in YOLO(weights).predict(
        paths, imgsz=imgsz, conf=0.01, device=device, batch=32, stream=True, verbose=False)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cls-weights", nargs="+", required=True)
    p.add_argument("--det-weights", help="detector to fuse with (mean of its top box score and the ensemble)")
    p.add_argument("--det-weights-val", help="detector for choosing the val threshold (default --det-weights); use a "
                   "train-only detector when --det-weights was trained on train+val")
    p.add_argument("--data", type=Path, default=Path("yolo_cls"))
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--imgsz", type=int, default=288)
    p.add_argument("--det-imgsz", default="320")
    p.add_argument("--device", default="0")
    args = p.parse_args()

    det_imgsz = [int(s) for s in args.det_imgsz.split(",")]
    det_imgsz = det_imgsz[0] if len(det_imgsz) == 1 else det_imgsz
    data = {}
    for split in ["val", "test"]:
        paths = sorted((args.data / split).glob("*/*.png"))
        actual = np.array([q.parent.name.startswith("mite") for q in paths])
        probs = [cls_probs(w, paths, args.imgsz, args.device) for w in args.cls_weights]
        top = None
        if args.det_weights:
            det_paths = [args.yolo_dir / "images" / split / q.name for q in paths]
            weights = args.det_weights_val if split == "val" and args.det_weights_val else args.det_weights
            top = det_top(weights, det_paths, det_imgsz, args.device)
        data[split] = (actual, probs, top)

    names = [Path(w).parent.parent.name + "/" + Path(w).stem for w in args.cls_weights]
    combos = [c for k in range(1, len(names) + 1) for c in itertools.combinations(range(len(names)), k)]
    print(f"{'combination':60s} val F1  thr   test F1 (P / R)")
    for combo in combos:
        for fused in ([False, True] if args.det_weights else [False]):
            scores = {}
            for split, (actual, probs, top) in data.items():
                s = np.mean([probs[i] for i in combo], 0)
                scores[split] = (s + top) / 2 if fused else s
            va, ta = data["val"][0], data["test"][0]
            t = max(THRESHOLDS, key=lambda t: f1_at(va, scores["val"], t)[0])
            vf = f1_at(va, scores["val"], t)[0]
            tf, tp, tr = f1_at(ta, scores["test"], t)
            label = " + ".join(names[i] for i in combo) + (" + det" if fused else "")
            print(f"{label:60s} {vf:.3f}  {t:.2f}  {tf:.3f} ({tp:.3f} / {tr:.3f})")
            print("SUMMARY " + json.dumps({"combo": label, "n_cls": len(combo), "fused": fused, "val_f1": round(vf, 4),
                                           "thr": float(t), "test_f1": round(tf, 4), "test_p": round(tp, 4),
                                           "test_r": round(tr, 4)}))


if __name__ == "__main__":
    main()
