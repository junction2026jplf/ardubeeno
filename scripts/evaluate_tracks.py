"""Measure multi-frame aggregation per bee on EV2 (Zenodo 13771384), which our models never saw.

Every EV2 video follows one bee over consecutive frames: videos under varroa_infested/ show an infested bee
(with a per-frame "mite visible" flag), videos under varroa_free/ a clean one. Each frame crop is scored
like a deployed bee crop (rotated to portrait, classifier on the letterboxed square, optional detector
fusion). Then windows of N consecutive frames per video are aggregated (mean / max / fraction of frames
over the threshold) to simulate a bee walking past the hive camera, and we report how many infested bees
are caught and how many clean bees are falsely flagged, for each N.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from convert_cls import letterbox_square


def load_rows(ev2_dir: Path):
    rows = []
    for line in open(ev2_dir / "labels.txt"):
        if not line.strip():
            continue
        r = json.loads(line)
        visible = r["varroa_visible"] == "yes"
        folder = "dataset_infested" if visible else "dataset_free"
        rows.append({
            "video": r["video"],
            "frame": int(r["id"].split("_")[-1]),
            "visible": visible,
            "infested": r["video"].startswith("varroa_infested/"),
            "path": ev2_dir / folder / f"{Path(r['video']).name}_{r['id'].replace('_', '')}.png",
        })
    return rows


def portrait(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if img.shape[1] > img.shape[0] else img


def score_frames(rows, cls_weights, det_weights, det_imgsz, device, batch=64):
    cls = YOLO(cls_weights)
    mite = [i for i, n in cls.names.items() if n.startswith("mite")]
    det = YOLO(det_weights) if det_weights else None
    probs, tops = [], []
    for i in range(0, len(rows), batch):
        imgs = [portrait(cv2.imread(str(r["path"]))) for r in rows[i:i + batch]]
        squares = [letterbox_square(im, 280) for im in imgs]
        probs += [float(r.probs.data[mite].sum()) for r in cls.predict(squares, imgsz=288, device=device, verbose=False)]
        if det:
            tops += [float(r.boxes.conf.max()) if len(r.boxes) else 0.0
                     for r in det.predict(imgs, imgsz=det_imgsz, conf=0.01, iou=0.45, device=device, verbose=False)]
    return np.array(probs), (np.array(tops) if det else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ev2-dir", type=Path, default=Path.home() / "ev2")
    p.add_argument("--cls-weights", default="runs/classify/cls-n288/weights/best.pt")
    p.add_argument("--det-weights", help="optional detector for mean fusion, e.g. n320-tv-e40-screen last.pt")
    p.add_argument("--det-imgsz", default="320")
    p.add_argument("--thr", type=float, help="per-frame threshold (default 0.46 classifier alone, 0.38 fused, "
                                              "both chosen on our val split)")
    p.add_argument("--windows", default="1,3,5,10,20")
    p.add_argument("--samples", type=int, default=200, help="random windows per video and N")
    p.add_argument("--stride", type=int, default=1,
                   help="use every k-th labelled frame inside a window (EV2 frames are ~25-30 fps; a board running "
                        "at a few FPS sees more varied frames)")
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = load_rows(args.ev2_dir)
    det_imgsz = [int(s) for s in args.det_imgsz.split(",")]
    det_imgsz = det_imgsz[0] if len(det_imgsz) == 1 else det_imgsz
    prob, top = score_frames(rows, args.cls_weights, args.det_weights, det_imgsz, args.device)
    score = (prob + top) / 2 if top is not None else prob
    thr = args.thr if args.thr is not None else (0.38 if top is not None else 0.46)
    mode = "classifier + detector (mean)" if top is not None else "classifier alone"
    print(f"EV2: {len(rows)} frames, {len({r['video'] for r in rows})} videos; {mode}; per-frame threshold {thr}")

    flag = score >= thr
    inf_vis = np.array([r["infested"] and r["visible"] for r in rows])
    inf_hid = np.array([r["infested"] and not r["visible"] for r in rows])
    free = np.array([not r["infested"] for r in rows])
    print("\nSingle frames:")
    print(f"  infested, mite visible   {inf_vis.sum():5d} frames  flagged {flag[inf_vis].mean():.3f}")
    print(f"  infested, mite hidden    {inf_hid.sum():5d} frames  flagged {flag[inf_hid].mean():.3f}")
    print(f"  clean bee                {free.sum():5d} frames  flagged {flag[free].mean():.3f}  (false alarms)")

    # Separation independent of the threshold: is it a calibration shift on foreign footage, or no signal?
    # Video-level caveat: clean frames come from only 10 videos, so treat these as indicative.
    def auc(pos, neg):
        values = np.concatenate([pos, neg])
        ranks = np.empty(len(values))
        ranks[values.argsort(kind="stable")] = np.arange(1, len(values) + 1)
        _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
        ranks = (np.bincount(inv, weights=ranks) / counts)[inv]  # average rank for ties
        return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    print(f"\nSeparation (AUC): visible vs clean {auc(score[inf_vis], score[free]):.3f}, "
          f"visible+hidden vs clean {auc(score[inf_vis | inf_hid], score[free]):.3f}")
    for fpr in [0.01, 0.05]:
        t = float(np.quantile(score[free], 1 - fpr))
        print(f"  threshold for {fpr:.0%} false alarms on clean frames: {t:.3f} -> visible flagged "
              f"{(score[inf_vis] > t).mean():.3f}, hidden flagged {(score[inf_hid] > t).mean():.3f}")
    print(f"  score medians: visible {np.median(score[inf_vis]):.3f}, hidden {np.median(score[inf_hid]):.3f}, "
          f"clean {np.median(score[free]):.3f}")

    per_video = defaultdict(list)
    for r, s in zip(rows, score):
        per_video[r["video"]].append((r["frame"], s))
    rng = random.Random(args.seed)
    print(f"\nWindows of N frames per bee, every {args.stride}. labelled frame ({args.samples} random windows per video):")
    print("     N   rule                infested bees caught   clean bees flagged")
    for n in [int(x) for x in args.windows.split(",")]:
        for rule in ["mean >= thr", "max >= thr", ">= half of frames"]:
            caught, fa = [], []
            for video, items in per_video.items():
                s = np.array([v for _, v in sorted(items)])
                span = (n - 1) * args.stride + 1
                if len(s) < span:
                    continue
                hits = []
                for _ in range(args.samples):
                    start = rng.randrange(len(s) - span + 1)
                    w = s[start:start + span:args.stride]
                    hit = {"mean >= thr": w.mean() >= thr, "max >= thr": w.max() >= thr,
                           ">= half of frames": (w >= thr).mean() >= 0.5}[rule]
                    hits.append(hit)
                (caught if video.startswith("varroa_infested/") else fa).append(np.mean(hits))
            print(f"  {n:4d}   {rule:18s}  {np.mean(caught):.3f} ({len(caught)} videos)   "
                  f"{np.mean(fa):.3f} ({len(fa)} videos)")


if __name__ == "__main__":
    main()
