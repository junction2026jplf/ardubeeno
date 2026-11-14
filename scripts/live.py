"""Live camera detection. Runs on the laptop or the UNO Q (point the camera at the slideshow).

The model runs on a centre crop with the training crops' 4:7 shape (outlined in cyan), so a bee that
fills the crop reaches the model at the same scale as in training. Aim the camera so the bee fills it.

With a classifier (--cls-weights, default export/varroa_cls.onnx if present), the verdict is
mean(classifier mite probability, top box score) >= --fuse-conf, as in device_check.py.
--window N --min-hits K shows MITE only when at least K of the last N frames were positive: on EV2 videos,
aggregating over frames caught about 3x more infested bees than single frames, and requiring K >= 2 keeps
per-frame false alarms from adding up.

Press Esc or q to quit, s to save the current frame to captures/ (for screen fine-tuning).
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

from device_check import Classifier

# Training crops are 160x280.
ROI_ASPECT = 160 / 280


def center_roi(frame):
    h, w = frame.shape[:2]
    rw = min(w, round(h * ROI_ASPECT))
    x0 = (w - rw) // 2
    return frame[:, x0:x0 + rw], x0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="export/varroa.onnx")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--imgsz", default="320", help="square size, or height,width of the exported model")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU; 0.7 leaves duplicate boxes")
    p.add_argument("--device", default="cpu")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-roi", dest="roi", action="store_false", help="use the full frame, not the centre crop")
    p.add_argument("--cls-weights", nargs="+", default=["export/varroa_cls.onnx"],
                   help="per-bee classifier ONNX file(s), used if they exist; several are averaged")
    p.add_argument("--fuse-conf", type=float, default=0.38)
    p.add_argument("--window", type=int, default=1, help="aggregate the verdict over the last N frames")
    p.add_argument("--min-hits", type=int, default=1, help="positive frames needed within the window")
    p.add_argument("--captures", type=Path, default=Path("captures"))
    args = p.parse_args()

    imgsz = [int(s) for s in args.imgsz.split(",")]
    imgsz = imgsz[0] if len(imgsz) == 1 else imgsz
    model = YOLO(args.weights, task="detect")
    cls = [Classifier(w, threads=0) for w in args.cls_weights if Path(w).exists()] or None
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    recent = deque(maxlen=args.window)
    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi, x0 = center_roi(frame) if args.roi else (frame, 0)
        r = model.predict(roi, imgsz=imgsz, conf=min(args.conf, 0.01) if cls else args.conf, iou=args.iou,
                          agnostic_nms=True, device=args.device, verbose=False)[0]
        scores = r.boxes.conf.cpu().numpy()
        top = float(scores.max()) if len(scores) else 0.0
        keep = scores >= args.conf

        out = frame.copy()
        if args.roi:
            cv2.rectangle(out, (x0, 0), (x0 + roi.shape[1] - 1, out.shape[0] - 1), (255, 255, 0), 2)
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy()[keep], scores[keep]):
            cv2.rectangle(out, (int(x1) + x0, int(y1)), (int(x2) + x0, int(y2)), (0, 0, 255), 2)
            cv2.putText(out, f"{s:.2f}", (int(x1) + x0, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        now = time.time()
        fps, last = 1 / max(now - last, 1e-6), now
        n = int(keep.sum())
        if cls:
            fused = (top + sum(c(roi) for c in cls) / len(cls)) / 2
            recent.append(fused >= args.fuse_conf)
        else:
            recent.append(n > 0)
        hits = sum(recent)
        mite = hits >= args.min_hits if args.window > 1 else recent[-1]
        label = f"{'MITE' if mite else 'clean'} ({fused:.2f})  boxes: {n}" if cls else f"Varroa: {n}"
        if args.window > 1:
            label += f"  [{hits}/{len(recent)} frames]"
        color = (0, 0, 255) if mite else (0, 200, 0)
        cv2.putText(out, label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(out, f"{fps:.1f} FPS", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("ArduBeeno", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            args.captures.mkdir(exist_ok=True)
            path = args.captures / f"{int(now * 1000)}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
