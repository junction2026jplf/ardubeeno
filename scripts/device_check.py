"""Check the exported model on the Arduino UNO Q (or any machine) without PyTorch.

Needs only numpy, opencv-python-headless and onnxruntime (requirements-device.txt).

  python device_check.py accuracy --images sample/images --labels sample/labels --csv board.csv
  python device_check.py compare laptop.csv board.csv
  python device_check.py bench --seconds 60
  python device_check.py camera --frames 100 --save out
  python device_check.py --cls-model varroa_cls.onnx calibrate --clean new_cam/clean --mite new_cam/mite --fpr 0.01

With --cls-model, a per-bee mite/clean classifier runs next to the detector and the bee's verdict is
the mean of the classifier's mite probability and the detector's top box score (>= --fuse-conf).
Repeating --cls-model gives an ensemble: their probabilities are averaged first.
With --cascade the classifier runs first and the detector only for bees with --cls-lo <= probability < --cls-hi.
"""
import argparse
import ast
import os
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Training crops are 160x280; the camera ROI uses the same width:height ratio.
ROI_ASPECT = 160 / 280
# The classifier was trained on crops letterboxed to this square (convert_cls.py).
CLS_SQUARE = 280


def session(model: str, threads: int, spin: bool = True):
    opts = ort.SessionOptions()
    opts.add_free_dimension_override_by_name("batch", 1)  # export/fast models: fixed shapes run faster
    if threads:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    if not spin:  # idle pool threads don't busy-wait, leaving the cores to the other sessions
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])


class Detector:
    """YOLO ONNX model: letterbox -> inference -> confidence filter -> NMS."""

    def __init__(self, model: str, conf: float, iou: float, threads: int, spin: bool = True):
        self.session = session(model, threads, spin)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])
        self.conf, self.iou = conf, iou

    def preprocess(self, img):
        h, w = img.shape[:2]
        r = min(self.h / h, self.w / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.h - nh) // 2, (self.w - nw) // 2
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), r, left, top

    def __call__(self, img):
        """Return [(x, y, w, h, score), ...] in original image pixels."""
        x, r, left, top = self.preprocess(img)
        out = self.session.run(None, {self.input_name: x})[0][0]  # (4 + classes, candidates)
        scores = out[4:].max(0)
        keep = scores >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[:4, keep]
        scores = scores[keep]
        xywh = np.stack([(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r, bw / r, bh / r], 1)
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.iou)).flatten()
        return [(*xywh[i].tolist(), float(scores[i])) for i in idx]


class Classifier:
    """YOLO classification ONNX model: square letterbox -> resize -> mite probability."""

    def __init__(self, model: str, threads: int, spin: bool = True):
        self.session = session(model, threads, spin)
        inp = self.session.get_inputs()[0]
        self.input_name, self.size = inp.name, int(inp.shape[2])
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map.get("names", "{}"))
        # "mite", plus "mite_underwing" for a three-class model.
        self.mite = [i for i, n in names.items() if str(n).startswith("mite")] or [1]

    def __call__(self, img) -> float:
        h, w = img.shape[:2]
        r = CLS_SQUARE / max(h, w)
        nh, nw = round(h * r), round(w * r)
        square = np.full((CLS_SQUARE, CLS_SQUARE, 3), 114, np.uint8)
        top, left = (CLS_SQUARE - nh) // 2, (CLS_SQUARE - nw) // 2
        square[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x = cv2.resize(square, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return float(self.session.run(None, {self.input_name: x})[0][0][self.mite].sum())


class Pipeline:
    """Detector, optionally fused with a classifier, giving boxes and a per-bee verdict.

    With args.cascade (and a classifier), the classifier runs first: below cls_lo the bee is clean without the
    detector, at cls_hi or above it is a mite (the detector still runs, to draw the mite boxes), and in between
    the fused score decides as usual. On the test bees this keeps F1 and skips the detector for ~96% of bees.
    """

    def __init__(self, args):
        self.conf, self.fuse_conf = args.conf, args.fuse_conf
        spin = getattr(args, "spin", True)
        self.cls = [Classifier(m, args.threads, spin) for m in args.cls_model] if args.cls_model else None
        self.cascade = bool(getattr(args, "cascade", False) and self.cls)
        self.cls_lo, self.cls_hi = getattr(args, "cls_lo", 0.31), getattr(args, "cls_hi", 0.6)
        # Fusion needs the detector's top score even when it is below --conf.
        det_conf = min(args.conf, 0.01) if self.cls else args.conf
        self.det = Detector(args.model, det_conf, args.iou, args.threads, spin)
        self.last_ms, self.skipped = (0.0, 0.0), False

    def __call__(self, img):
        """Return (boxes >= conf, detector top score, classifier probability or None, fused score, verdict)."""
        t0 = time.perf_counter()
        prob = float(np.mean([c(img) for c in self.cls])) if self.cls else None
        t1 = time.perf_counter()
        self.skipped = self.cascade and prob < self.cls_lo
        all_boxes = [] if self.skipped else self.det(img)
        t2 = time.perf_counter()
        self.last_ms = ((t2 - t1) * 1000, (t1 - t0) * 1000)  # (detector, classifier); detector 0 when skipped
        boxes = [b for b in all_boxes if b[4] >= self.conf]
        top = max((b[4] for b in all_boxes), default=0.0)
        if self.cls is None:
            return boxes, top, None, top, bool(boxes)
        fused = (top + prob) / 2
        if self.cascade and prob >= self.cls_hi:
            return boxes, top, prob, fused, True
        return boxes, top, prob, fused, fused >= self.fuse_conf


def cpu_temp():
    temps = []
    for f in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(f.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    return f"{max(temps):.1f} °C" if temps else "n/a"


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024


def summarize(name, ms):
    ms = np.array(ms)
    print(f"{name:<14} mean {ms.mean():6.1f} ms   p50 {np.percentile(ms, 50):6.1f}   "
          f"p95 {np.percentile(ms, 95):6.1f}   -> {1000 / ms.mean():5.1f} FPS")


def print_system(pipe, args):
    print(f"host           {platform.node()} ({platform.machine()}, {os.cpu_count()} CPUs)")
    print(f"python         {platform.python_version()}, onnxruntime {ort.__version__}, opencv {cv2.__version__}")
    print(f"model          {args.model} (input {pipe.det.w}x{pipe.det.h}, threads {args.threads or 'auto'})")
    if pipe.cls:
        print(f"classifier     {', '.join(args.cls_model)} (input {pipe.cls[0].size}x{pipe.cls[0].size}), "
              f"verdict = mean(det, cls) >= {args.fuse_conf}")
        if pipe.cascade:
            print(f"cascade        classifier first: clean below {pipe.cls_lo}, mite from {pipe.cls_hi}")
    print(f"cpu temp       {cpu_temp()}\n")


def accuracy(pipe, args):
    images = sorted(Path(args.images).glob("*.png"))
    if not images:
        raise SystemExit(f"no .png images in {args.images}")
    tp = fp = tn = fn = n_boxes = 0
    times, det_ms, cls_ms, rows = [], [], [], []
    for p in images:
        img = cv2.imread(str(p))
        t = time.perf_counter()
        boxes, top, prob, fused, predicted = pipe(img)
        times.append((time.perf_counter() - t) * 1000)
        det_ms.append(pipe.last_ms[0])
        cls_ms.append(pipe.last_ms[1])

        label = Path(args.labels) / f"{p.stem}.txt"
        actual = label.exists() and label.read_text().strip() != ""
        tp += actual and predicted
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not actual and not predicted
        n_boxes += len(boxes)
        rows.append(f"{p.name},{int(actual)},{len(boxes)},{top:.4f},{-1 if prob is None else prob:.4f},"
                    f"{fused:.4f},{int(predicted)}")

    total = len(images)
    rule = f"mean(det, cls) >= {args.fuse_conf}" if pipe.cls else f"detector conf >= {args.conf}"
    print(f"Per-bee classification, {rule} ({total} bees, {n_boxes} boxes at conf {args.conf}):")
    print(f"  accuracy  {(tp + tn) / total:.3f}")
    print(f"  precision {tp / (tp + fp) if tp + fp else 0:.3f}")
    print(f"  recall    {tp / (tp + fn) if tp + fn else 0:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}\n")
    summarize("per image", times)
    if pipe.cls:
        summarize("  detector", det_ms)
        summarize("  classifier", cls_ms)
    print(f"peak memory    {peak_rss_mb():.0f} MB")
    if args.csv:
        Path(args.csv).write_text("image,actual,boxes,top_score,cls_prob,fused,verdict\n" + "\n".join(rows) + "\n")
        print(f"wrote {args.csv}")


def bench(pipe, args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"could not read {args.image}")
    elif sample := sorted(Path("sample/images").glob("*.png"))[:1]:
        img = cv2.imread(str(sample[0]))
    else:
        print("no --image given, using random noise\n")
        img = np.random.default_rng(0).integers(0, 255, (280, 160, 3), dtype=np.uint8)

    for _ in range(5):
        pipe(img)
    temp_start = cpu_temp()
    times, det_ms, cls_ms = [], [], []
    deadline = time.perf_counter() + args.seconds
    while time.perf_counter() < deadline:
        t = time.perf_counter()
        pipe(img)
        times.append((time.perf_counter() - t) * 1000)
        det_ms.append(pipe.last_ms[0])
        cls_ms.append(pipe.last_ms[1])

    q = max(len(times) // 4, 1)
    print(f"{len(times)} runs in {args.seconds:.0f}s")
    summarize("all", times)
    summarize("first 25%", times[:q])
    summarize("last 25%", times[-q:])
    if pipe.cls:
        summarize("  detector", det_ms)
        summarize("  classifier", cls_ms)
    print(f"cpu temp       {temp_start} -> {cpu_temp()}")
    print(f"peak memory    {peak_rss_mb():.0f} MB")


def center_roi(frame):
    h, w = frame.shape[:2]
    rw = min(w, round(h * ROI_ASPECT))
    x0 = (w - rw) // 2
    return frame[:, x0:x0 + rw], x0


def camera(pipe, args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")
    save = Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)

    grab_ms, infer_ms, verdicts = [], [], []
    start = time.perf_counter()
    for i in range(args.frames):
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            print("camera read failed")
            break
        t1 = time.perf_counter()
        roi, x0 = center_roi(frame) if args.roi else (frame, 0)
        boxes, top, prob, fused, verdict = pipe(roi)
        t2 = time.perf_counter()
        grab_ms.append((t1 - t0) * 1000)
        infer_ms.append((t2 - t1) * 1000)
        verdicts.append(verdict)

        if save and i % args.save_every == 0:
            out = frame.copy()
            if args.roi:
                cv2.rectangle(out, (x0, 0), (x0 + roi.shape[1] - 1, out.shape[0] - 1), (255, 255, 0), 2)
            for x, y, w, h, s in boxes:
                p1, p2 = (int(x + x0), int(y)), (int(x + x0 + w), int(y + h))
                cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
                cv2.putText(out, f"{s:.2f}", (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(out, f"{'MITE' if verdict else 'clean'} {fused:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255) if verdict else (0, 200, 0), 2)
            cv2.imwrite(str(save / f"frame_{i:04d}.jpg"), out)
    elapsed = time.perf_counter() - start
    cap.release()

    if not verdicts:
        raise SystemExit("no frames captured")
    print(f"{len(verdicts)} frames at {frame.shape[1]}x{frame.shape[0]} in {elapsed:.1f}s "
          f"-> {len(verdicts) / elapsed:.1f} FPS end to end")
    summarize("capture", grab_ms)
    summarize("inference", infer_ms)
    print(f"frames w/ mite {sum(verdicts)} / {len(verdicts)}")
    print(f"cpu temp       {cpu_temp()}")
    print(f"peak memory    {peak_rss_mb():.0f} MB")
    if save:
        print(f"annotated frames in {save}/")


def calibrate(pipe, args):
    """Pick the per-bee threshold for a new camera from labelled crops of clean (and optionally mite) bees.

    On foreign footage (EV2) the scores shift down: at the default threshold only 27% of visible-mite crops
    were flagged, but at the threshold giving 1% false alarms on clean bees, 63% were.
    """
    def scores(folder):
        paths = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in Path(folder).glob(ext))
        if not paths:
            raise SystemExit(f"no images in {folder}")
        return np.array([pipe(cv2.imread(str(p)))[3] for p in paths])

    clean = scores(args.clean)
    thr = float(np.quantile(clean, 1 - args.fpr))
    option = "--fuse-conf" if pipe.cls else "--conf"
    current = args.fuse_conf if pipe.cls else args.conf
    print(f"{len(clean)} clean bees: threshold for {args.fpr:.1%} false alarms = {thr:.3f} (current {current})")
    print(f"  clean flagged at current threshold: {(clean >= current).mean():.3f}")
    if args.mite:
        mite = scores(args.mite)
        print(f"{len(mite)} mite bees: flagged at {thr:.3f}: {(mite > thr).mean():.3f}, "
              f"at current {current}: {(mite >= current).mean():.3f}")
    print(f"\n-> use {option} {thr:.3f} for this camera" + (" (check it on more images before relying on it)"
                                                           if len(clean) < 100 else ""))


def compare(args):
    def read(path):
        lines = Path(path).read_text().splitlines()
        header = lines[0].split(",")
        out = {}
        for line in lines[1:]:
            row = dict(zip(header, line.split(",")))
            verdict = int(row["verdict"]) if "verdict" in row else int(row["boxes"]) > 0
            out[row["image"]] = (verdict, float(row.get("fused", row["top_score"])))
        return out

    a, b = read(args.a), read(args.b)
    common = sorted(a.keys() & b.keys())
    if not common:
        raise SystemExit("no images in common")
    flips = [k for k in common if a[k][0] != b[k][0]]
    diff = max(abs(a[k][1] - b[k][1]) for k in common)
    print(f"{len(common)} images: {len(flips)} different mite yes/no answers, "
          f"max score difference {diff:.4f}")
    for k in flips[:10]:
        print(f"  {k}: {args.a} mite={a[k][0]} score={a[k][1]:.3f} | {args.b} mite={b[k][0]} score={b[k][1]:.3f}")
    print("OK: board matches laptop" if not flips and diff < 0.01 else "CHECK: results differ")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="varroa.onnx")
    p.add_argument("--cls-model", action="append", help="per-bee classifier, e.g. varroa_cls.onnx; repeat for an ensemble")
    p.add_argument("--conf", type=float, default=0.25, help="detector box threshold")
    p.add_argument("--fuse-conf", type=float, default=0.38, help="per-bee threshold on mean(det, cls)")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU; 0.7 leaves duplicate boxes")
    p.add_argument("--threads", type=int, default=0, help="onnxruntime threads, 0 = auto")
    p.add_argument("--cascade", action="store_true", help="classifier first, detector only when it is unsure")
    p.add_argument("--cls-lo", type=float, default=0.31, help="cascade: clean below this classifier probability")
    p.add_argument("--cls-hi", type=float, default=0.6, help="cascade: mite at or above this probability")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("accuracy", help="per-bee accuracy on labelled images")
    a.add_argument("--images", default="sample/images")
    a.add_argument("--labels", default="sample/labels")
    a.add_argument("--csv", help="write per-image results, for `compare`")

    b = sub.add_parser("bench", help="sustained inference speed, CPU temperature, memory")
    b.add_argument("--image", help="defaults to the first image in sample/images")
    b.add_argument("--seconds", type=float, default=60)

    c = sub.add_parser("camera", help="capture + inference from the camera, saves annotated frames")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--frames", type=int, default=100)
    c.add_argument("--no-roi", dest="roi", action="store_false", help="use the full frame, not the centre crop")
    c.add_argument("--save", help="directory for annotated frames")
    c.add_argument("--save-every", type=int, default=10)

    k = sub.add_parser("calibrate", help="threshold for a new camera from labelled clean (and mite) bee crops")
    k.add_argument("--clean", required=True, help="folder of crops of clean bees from the new camera")
    k.add_argument("--mite", help="optional folder of crops of bees with a visible mite")
    k.add_argument("--fpr", type=float, default=0.01, help="target false-alarm rate on clean bees")

    d = sub.add_parser("compare", help="compare two --csv outputs, e.g. laptop vs board")
    d.add_argument("a")
    d.add_argument("b")

    args = p.parse_args()
    if args.cmd == "compare":
        return compare(args)

    for m in args.cls_model or []:
        if not Path(m).exists():
            raise SystemExit(f"{m} not found")
    pipe = Pipeline(args)
    print_system(pipe, args)
    {"accuracy": accuracy, "bench": bench, "camera": camera, "calibrate": calibrate}[args.cmd](pipe, args)


if __name__ == "__main__":
    main()
