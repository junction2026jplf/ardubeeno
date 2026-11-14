"""Speed (and accuracy) of all Varroa model variants in one table. Runs on the laptop and on the UNO Q.

Needs only numpy, opencv-python-headless and onnxruntime. Models are looked up in --dir (default: fast/).

  python fastbench.py speed                      # ms per bee, 1 and 4 threads, batch 1 and 4
  python fastbench.py speed --seconds 5 --threads 1 2 4 --batch 1 4 8
  python fastbench.py accuracy --images sample/images --labels sample/labels
  python fastbench.py accuracy --images ../yolo/images/test --labels ../yolo/labels/test \
      --val-images ../yolo/images/val --val-labels ../yolo/labels/val      # laptop, full protocol

Per-bee verdict as deployed: mean(classifier mite probability, detector top box score) >= threshold.
The top box score is the max over all candidates, so the verdict needs no NMS (NMS only for drawing boxes).
"""
import argparse
import os
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLS_SQUARE = 280  # classifier training crops were letterboxed to this square (convert_cls.py)

# name: (detector file, classifier file). None = model not used. "deployed" files sit next to fast/.
VARIANTS = {
    "deployed":          ("../varroa.onnx", "../varroa_cls.onnx"),
    "fp32-b":            ("det_320x192_fp32.onnx", "cls_288x288_fp32.onnx"),
    "det288x160":        ("det_288x160_fp32.onnx", "cls_288x288_fp32.onnx"),
    "det256x160":        ("det_256x160_fp32.onnx", "cls_288x288_fp32.onnx"),
    "det256x128":        ("det_256x128_fp32.onnx", "cls_288x288_fp32.onnx"),
    "cls288x160":        ("det_320x192_fp32.onnx", "cls_288x160_fp32.onnx"),
    "cls224x128":        ("det_320x192_fp32.onnx", "cls_224x128_fp32.onnx"),
    "cls224":            ("det_320x192_fp32.onnx", "cls_224x224_fp32.onnx"),
    "cls192":            ("det_320x192_fp32.onnx", "cls_192x192_fp32.onnx"),
    "cls160":            ("det_320x192_fp32.onnx", "cls_160x160_fp32.onnx"),
    "int8":              ("det_320x192_int8qdq.onnx", "cls_288x288_int8qdq.onnx"),
    "int8-qop":          ("det_320x192_int8qop.onnx", "cls_288x288_int8qop.onnx"),
    "int8-qdqall":       ("det_320x192_int8qdqall.onnx", "cls_288x288_int8qdqall.onnx"),
    "det-int8":          ("det_320x192_int8qdq.onnx", "cls_288x288_fp32.onnx"),
    "cls-int8":          ("det_320x192_fp32.onnx", "cls_288x288_int8qdq.onnx"),
    "int8-det288":       ("det_288x160_int8qdq.onnx", "cls_288x288_int8qdq.onnx"),
    "int8-det288-qop":   ("det_288x160_int8qop.onnx", "cls_288x288_int8qop.onnx"),
    "small-fp32":        ("det_288x160_fp32.onnx", "cls_224x224_fp32.onnx"),
    "small-int8":        ("det_288x160_int8qdq.onnx", "cls_224x224_int8qdq.onnx"),
    "small-int8-qop":    ("det_288x160_int8qop.onnx", "cls_224x224_int8qop.onnx"),
    # classifiers retrained at a smaller input (runs/verda/classify/cls-nNNN)
    "c224":              ("det_320x192_fp32.onnx", "cls224_224x224_fp32.onnx"),
    "c192":              ("det_320x192_fp32.onnx", "cls192_192x192_fp32.onnx"),
    "c160":              ("det_320x192_fp32.onnx", "cls160_160x160_fp32.onnx"),
    "c224-int8":         ("det_320x192_int8qdq.onnx", "cls224_224x224_int8qdq.onnx"),
    "c192-int8":         ("det_320x192_int8qdq.onnx", "cls192_192x192_int8qdq.onnx"),
    "c160-int8":         ("det_320x192_int8qdq.onnx", "cls160_160x160_int8qdq.onnx"),
    "c160-int8all":      ("det_320x192_int8qdqall.onnx", "cls160_160x160_int8qdqall.onnx"),
    "c160-det-int8":     ("det_320x192_int8qdq.onnx", "cls160_160x160_fp32.onnx"),
    # camera fine-tunes (runs/verda/detect/det-cam-ft + classify/cls-cam-ft160|288 last.pt, export/deploy/);
    # their thresholds: --fuse-conf 0.16 --cascade 0.02 0.30 (cam160), see export/deploy/deploy.json
    "cam160":            ("camdet_320x192_fp32.onnx", "cam160_160x160_fp32.onnx"),
    "cam160-int8all":    ("camdet_320x192_int8qdqall.onnx", "cam160_160x160_int8qdqall.onnx"),
    "cam288-int8all":    ("camdet_320x192_int8qdqall.onnx", "cam288_288x288_int8qdqall.onnx"),
    # classifiers trained on letterboxed crops (classify/cls-cam-sq160|288 last.pt); cam-sq288 = export/deploy/ set.
    # thresholds (sensitive point): cam-sq288 --fuse-conf 0.21 --cascade 0.20 0.73, cam-sq160 --fuse-conf 0.13
    # --cascade 0.12 0.73; max-F1: 0.34 / 0.33 0.73 and 0.38 / 0.37 0.73
    "cam-sq160":         ("camdet_320x192_int8qdqall.onnx", "camsq160_160x160_int8qdqall.onnx"),
    "cam-sq288":         ("camdet_320x192_int8qdqall.onnx", "camsq288_288x288_int8qdqall.onnx"),
    "det-only":          ("det_320x192_fp32.onnx", None),
    "cls-only":          (None, "cls_288x288_fp32.onnx"),
}


def letterbox_det(imgs, h, w):
    """device_check.Detector.preprocess for a list of crops -> (N,3,h,w) float32."""
    out = np.full((len(imgs), h, w, 3), 114, np.uint8)
    for i, img in enumerate(imgs):
        ih, iw = img.shape[:2]
        r = min(h / ih, w / iw)
        nh, nw = round(ih * r), round(iw * r)
        top, left = (h - nh) // 2, (w - nw) // 2
        out[i, top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out[..., ::-1].transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def letterbox_cls(imgs, h, w):
    """Square model: device_check.Classifier (280 square letterbox, then resize). Rectangular: direct letterbox."""
    out = np.empty((len(imgs), h, w, 3), np.uint8)
    for i, img in enumerate(imgs):
        ih, iw = img.shape[:2]
        S, Sw = (CLS_SQUARE, CLS_SQUARE) if h == w else (h, w)
        r = min(S / ih, Sw / iw)
        nh, nw = round(ih * r), round(iw * r)
        canvas = np.full((S, Sw, 3), 114, np.uint8)
        top, left = (S - nh) // 2, (Sw - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        out[i] = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR) if h == w else canvas
    return np.ascontiguousarray(out[..., ::-1].transpose(0, 3, 1, 2), dtype=np.float32) / 255.0


def make_session(path, threads, spin=True, xnnpack=False, batch=None):
    o = ort.SessionOptions()
    if batch:  # fixed shapes are much faster than a symbolic batch dimension
        o.add_free_dimension_override_by_name("batch", batch)
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if not spin:
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    providers = ["CPUExecutionProvider"]
    if xnnpack:
        o.intra_op_num_threads = 1  # XNNPACK uses its own pool
        providers = [("XnnpackExecutionProvider", {"intra_op_num_threads": threads}), "CPUExecutionProvider"]
    return ort.InferenceSession(str(path), o, providers=providers)


class Model:
    def __init__(self, path, kind, threads, **kw):
        self.path, self.kind = Path(path), kind
        self.s = make_session(path, threads, **kw)
        i = self.s.get_inputs()[0]
        self.name, self.h, self.w = i.name, int(i.shape[2]), int(i.shape[3])
        self.max_batch = i.shape[0] if isinstance(i.shape[0], int) else None
        self.prep = letterbox_det if kind == "det" else letterbox_cls
        self.int8 = any(n.op_type in ("QuantizeLinear", "QLinearConv", "ConvInteger") for n in self._nodes())

    def _nodes(self):
        try:
            import onnx
            return onnx.load(str(self.path), load_external_data=False).graph.node
        except ImportError:  # board: guess from the file name
            class N:
                op_type = "QLinearConv" if "int8" in self.path.name else "Conv"
            return [N()]

    def run(self, x):
        if self.max_batch == 1 and len(x) > 1:
            return np.concatenate([self.s.run(None, {self.name: x[i:i + 1]})[0] for i in range(len(x))])
        return self.s.run(None, {self.name: x})[0]

    def score(self, out):
        """det: top score per image (N,); cls: mite probability (N,)."""
        if self.kind == "cls":
            return out[:, 1]
        top = out[:, 4:].max((1, 2))
        return np.where(top >= 0.01, top, 0.0)  # as device_check: boxes below conf 0.01 are dropped


_CACHE = {}


def cached_model(path, kind, threads, **kw):
    key = (str(path), threads, tuple(sorted((k, str(v)) for k, v in kw.items())))
    if key not in _CACHE:
        _CACHE[key] = Model(path, kind, threads, **kw)
    return _CACHE[key]


class Variant:
    def __init__(self, name, d, threads, **kw):
        det, cls = VARIANTS[name]
        self.name = name
        self.det = cached_model(d / det, "det", threads, **kw) if det else None
        self.cls = cached_model(d / cls, "cls", threads, **kw) if cls else None

    def describe(self):
        size = lambda m: f"{m.h}x{m.w}" if m else "-"
        q = sorted({("int8" if m.int8 else "fp32") for m in (self.det, self.cls) if m})
        mb = sum(m.path.stat().st_size for m in (self.det, self.cls) if m) / 1e6
        return size(self.det), size(self.cls), "/".join(q), mb

    def scores(self, imgs):
        """(det top, cls prob, timings ms: pre, det, cls) for a batch of crops."""
        t0 = time.perf_counter()
        xd = self.det.prep(imgs, self.det.h, self.det.w) if self.det else None
        xc = self.cls.prep(imgs, self.cls.h, self.cls.w) if self.cls else None
        t1 = time.perf_counter()
        top = self.det.score(self.det.run(xd)) if self.det else None
        t2 = time.perf_counter()
        prob = self.cls.score(self.cls.run(xc)) if self.cls else None
        t3 = time.perf_counter()
        return top, prob, ((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)


def fused(top, prob):
    if top is None:
        return prob
    if prob is None:
        return top
    return (top + prob) / 2


def load_images(folder, n=None):
    paths = sorted(Path(folder).glob("*.png"))[:n]
    if not paths:
        raise SystemExit(f"no .png images in {folder}")
    return paths, [cv2.imread(str(p)) for p in paths]


def labels_for(paths, folder):
    return np.array([(Path(folder) / f"{p.stem}.txt").exists() and
                     (Path(folder) / f"{p.stem}.txt").read_text().strip() != "" for p in paths])


def f1_at(actual, score, t):
    pred = score >= t
    tp, fp, fn = np.sum(pred & actual), np.sum(pred & ~actual), np.sum(~pred & actual)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def best_threshold(actual, score):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    return max(grid, key=lambda t: f1_at(actual, score, t))


def all_scores(v, imgs, batch=16):
    tops, probs = [], []
    for i in range(0, len(imgs), batch):
        t, p, _ = v.scores(imgs[i:i + batch])
        tops.append(t)
        probs.append(p)
    cat = lambda a: None if a[0] is None else np.concatenate(a)
    return cat(tops), cat(probs)


def system_line():
    temp = []
    for f in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temp.append(int(f.read_text()) / 1000)
        except (OSError, ValueError):
            pass
    t = f", cpu {max(temp):.0f} C" if temp else ""
    return (f"{platform.node()} {platform.machine()} {os.cpu_count()} CPUs, python {platform.python_version()}, "
            f"onnxruntime {ort.__version__} {ort.get_available_providers()}{t}")


def variant_names(args):
    names = args.variants or list(VARIANTS)
    return [n for n in names if all((args.dir / f).exists() for f in VARIANTS[n] if f)]


def cascade_scores(top, prob, lo, hi):
    """Classifier first; the detector only decides bees with lo <= prob < hi (skipped bees get 0 or 1)."""
    return np.where(prob >= hi, 1.0, np.where(prob < lo, 0.0, (top + prob) / 2))


def pick_cascade(actual, top, prob, thr, tol=0.001):
    """Widest skip band on val whose F1 is within tol of the full pipeline."""
    base, best = f1_at(actual, (top + prob) / 2, thr), (0.0, 2 * thr, 0.0)
    for lo in np.arange(0.0, thr, 0.01):
        for hi in (2 * thr, 0.7, 0.6, 0.5):
            skip = np.mean((prob >= hi) | (prob < lo))
            if skip > best[2] and f1_at(actual, cascade_scores(top, prob, lo, hi), thr) >= base - tol:
                best = (round(lo, 2), hi, skip)
    return best[:2]


def accuracy(args):
    paths, imgs = load_images(args.images, args.n)
    actual = labels_for(paths, args.labels)
    val = None
    if args.val_images:
        vp, vi = load_images(args.val_images, args.n)
        val = (labels_for(vp, args.val_labels), vi)
    print(system_line())
    print(f"{len(imgs)} bees ({actual.sum()} with mites) from {args.images}"
          + (f"; thresholds also chosen on {len(val[1])} val bees" if val else "") + "\n")
    print(f"{'variant':<16} {'F1@' + str(args.fuse_conf):>8} {'acc':>6}  cascade lo,hi -> F1, detector skipped"
          + ("   | val-chosen: thr F1, cascade lo,hi F1 skip" if val else ""))
    ref = None
    for name in variant_names(args):
        v = Variant(name, args.dir, threads=args.threads[0])
        top, prob = all_scores(v, imgs)
        s = fused(top, prob)
        line = f"{name:<16} {f1_at(actual, s, args.fuse_conf):8.3f} {np.mean((s >= args.fuse_conf) == actual):6.3f}"
        if top is not None and prob is not None:
            lo, hi = args.cascade
            line += (f"  {lo:.2f},{hi:.2f} -> {f1_at(actual, cascade_scores(top, prob, lo, hi), args.fuse_conf):.3f}, "
                     f"{np.mean((prob >= hi) | (prob < lo)):4.0%}")
        if val:
            vt, vpr = all_scores(v, val[1])
            thr = best_threshold(val[0], fused(vt, vpr))
            line += f"   | {thr:.2f} {f1_at(actual, s, thr):.3f}"
            if top is not None and prob is not None:
                lo, hi = pick_cascade(val[0], vt, vpr, args.fuse_conf)
                line += (f", {lo:.2f},{hi:.2f} {f1_at(actual, cascade_scores(top, prob, lo, hi), args.fuse_conf):.3f}"
                         f" {np.mean((prob >= hi) | (prob < lo)):.0%}")
        if name == "deployed":
            ref = s
        elif ref is not None:
            line += f"  flips vs deployed {np.sum((ref >= args.fuse_conf) != (s >= args.fuse_conf))}"
        print(line, flush=True)
        if args.save_scores:
            np.savez(Path(args.save_scores) / f"{name}.npz", top=top if top is not None else [],
                     prob=prob if prob is not None else [], actual=actual)
        _CACHE.clear()


def batches(imgs, batch):
    xs = [imgs[i:i + batch] for i in range(0, len(imgs) - batch + 1, batch)]
    return xs or [imgs[:1] * batch]


def time_round(v, xs, seconds, concurrent=False):
    """One timing window: per-run ms lists for pre-processing, detector, classifier, wall."""
    pre, det, cls, wall = [], [], [], []
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end or n < 3:
        x = xs[n % len(xs)]
        t = time.perf_counter()
        if concurrent:
            xd = v.det.prep(x, v.det.h, v.det.w)
            xc = v.cls.prep(x, v.cls.h, v.cls.w)
            th = threading.Thread(target=v.cls.run, args=(xc,))
            th.start()
            v.det.run(xd)
            th.join()
        else:
            _, _, (a, b, c) = v.scores(x)
            pre.append(a), det.append(b), cls.append(c)
        wall.append((time.perf_counter() - t) * 1e3)
        n += 1
    return pre, det, cls, wall


def speed(args):
    _, imgs = load_images(args.images, 64)
    print(system_line())
    print(f"ms per bee: best of {args.rounds} rounds of {args.seconds:g} s per row (median per round). total = wall time incl. pre-processing. bat = bee crops per run.\n"
          f"cascade = classifier on every bee, detector only on the {1 - args.skip:.1%} it can't decide "
          f"(test set; --skip): pre + cls + {1 - args.skip:.3f} x det\n")
    kws = [("", {})]
    if args.xnnpack and "XnnpackExecutionProvider" in ort.get_available_providers():
        kws.append(("+xnn", {"xnnpack": True}))
    if args.nospin:
        kws.append(("+nospin", {"spin": False}))
    xs = {b: batches(imgs, b) for b in args.batch}
    print(f"{'variant':<18} {'det in':>7} {'cls in':>7} {'quant':>9} {'MB':>5} {'thr':>4} {'bat':>3} "
          f"{'pre':>6} {'det':>6} {'cls':>6} {'total':>7} {'bees/s':>7} {'cascade':>8}", flush=True)
    names = variant_names(args)
    if "deployed" in names and len(names) > 1:
        names.append("deployed")  # again at the end: shows thermal throttling
    for name in names:
        configs = []
        for th in args.threads:
            for tag, kw in kws:
                for b in args.batch:
                    if b > 1 and name == "deployed":  # fixed batch 1
                        continue
                    configs.append((name + tag, str(th), b, (name, th, b, kw), False))
                if args.concurrent and th >= 2 and all(VARIANTS[name]):
                    b = args.batch[-1]
                    if b == 1 or name != "deployed":
                        configs.append((name + tag + "+par", f"{th // 2}+{th // 2}", b, (name, th // 2, b, kw), True))
        for label, th, b, (vn, vth, vb, kw), c in configs:
            v = Variant(vn, args.dir, threads=vth, batch=vb, **kw)
            time_round(v, xs[b], 0, c)  # warm-up
            best = None
            for _ in range(args.rounds):
                res = [float(np.median(a)) / b if a else float("nan") for a in time_round(v, xs[b], args.seconds, c)]
                if best is None or res[3] < best[3]:
                    best = res
            ds, cs, q, mb = v.describe()
            p, d, cl, tot = best
            f = lambda x: f"{x:6.2f}" if x == x else f"{'':>6}"
            casc = f"{p + cl + (1 - args.skip) * d:8.2f}" if v.det and v.cls and not c else f"{'':>8}"
            print(f"{label:<18} {ds:>7} {cs:>7} {q:>9} {mb:5.1f} {th:>4} {b:3d} {f(p)} {f(d)} {f(cl)} {tot:7.2f} "
                  f"{1000 / tot:7.1f} {casc}", flush=True)
            _CACHE.clear()  # one config's sessions at a time: the board has 1.7 GB
        _CACHE.clear()
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"peak memory {rss / 1024 / (1024 if platform.system() == 'Darwin' else 1):.0f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", type=Path, default=Path("fast"), help="folder with the variant .onnx files")
    common.add_argument("--variants", nargs="*", help=f"subset of: {' '.join(VARIANTS)}")
    common.add_argument("--threads", type=int, nargs="*", default=[1, 4])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speed", parents=[common])
    s.add_argument("--images", default="sample/images")
    s.add_argument("--batch", type=int, nargs="*", default=[1, 4])
    s.add_argument("--seconds", type=float, default=1.0, help="per row and round")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--skip", type=float, default=0.957, help="share of bees where the cascade skips the detector")
    s.add_argument("--xnnpack", action="store_true", help="also try the XNNPACK EP if this onnxruntime has it")
    s.add_argument("--nospin", action="store_true", help="also try without intra-op thread spinning")
    s.add_argument("--concurrent", action="store_true", help="also run det and cls in parallel, threads/2 each")
    a = sub.add_parser("accuracy", parents=[common])
    a.add_argument("--images", default="sample/images")
    a.add_argument("--labels", default="sample/labels")
    a.add_argument("--val-images")
    a.add_argument("--val-labels")
    a.add_argument("--n", type=int)
    a.add_argument("--fuse-conf", type=float, default=0.38)
    a.add_argument("--save-scores")
    a.add_argument("--cascade", type=float, nargs=2, default=[0.31, 0.6], metavar=("LO", "HI"),
                   help="run the detector only when LO <= classifier prob < HI")
    args = p.parse_args()
    {"speed": speed, "accuracy": accuracy}[args.cmd](args)


if __name__ == "__main__":
    main()
