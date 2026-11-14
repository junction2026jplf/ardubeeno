"""Evaluate the camera fine-tuned Varroa pairs and build the deployment set in export/deploy/.

  python scripts/deploy_build.py export     # fp32 ONNX of every candidate (fast_build.py recipe) -> export/deploy/work
  python scripts/deploy_build.py score      # per-bee scores (det top score, cls mite prob) cached as .npy
  python scripts/deploy_build.py table      # thresholds on camera val -> camera test + original test
  python scripts/deploy_build.py points     # max-F1 and sensitive (high-recall) points + cascade bands, cam detector
  python scripts/deploy_build.py quantize --det cam --cls cam160_last   # int8 "qdqall", calibrated on camera crops

All scoring runs on onnxruntime CPU with the board's own pre-processing (fastbench.letterbox_*), so the numbers are
those of the deployed files. Detectors trained on train+val get their val scores from the train-only twin.
"""
import argparse
import itertools
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx

sys.path.insert(0, str(Path(__file__).parent))
from fastbench import Model, cascade_scores, f1_at, letterbox_cls, letterbox_det  # noqa: E402
from fast_build import fix_spatial  # noqa: E402

WORK = Path("export/deploy/work")
R = "runs/verda"
# name: (weights, imgsz (h, w), kind)
DETS = {
    "base":    (f"{R}/detect/n320-tv-e40-screen/weights/last.pt", (320, 192)),
    "base_tr": (f"{R}/detect/n320-screen/weights/last.pt", (320, 192)),
    "cam":     (f"{R}/detect/det-cam-ft/weights/last.pt", (320, 192)),
    "cam_tr":  (f"{R}/detect/det-cam-ft-tr/weights/last.pt", (320, 192)),
}
VAL_TWIN = {"base": "base_tr", "cam": "cam_tr"}  # train+val detector -> train-only detector for val scores
CLSS = {
    "base288":     (f"{R}/classify/cls-n288/weights/best.pt", (288, 288)),
    "base160":     (f"{R}/classify/cls-n160/weights/best.pt", (160, 160)),
    "cam288_best": (f"{R}/classify/cls-cam-ft288/weights/best.pt", (288, 288)),
    "cam288_last": (f"{R}/classify/cls-cam-ft288/weights/last.pt", (288, 288)),
    "cam160_best": (f"{R}/classify/cls-cam-ft160/weights/best.pt", (160, 160)),
    "cam160_last": (f"{R}/classify/cls-cam-ft160/weights/last.pt", (160, 160)),
    # trained on letterboxed camera crops (yolo_cam_mix_sq), i.e. the app's own classifier pre-processing
    "sq288_best":  (f"{R}/classify/cls-cam-sq288/weights/best.pt", (288, 288)),
    "sq288_last":  (f"{R}/classify/cls-cam-sq288/weights/last.pt", (288, 288)),
    "sq160_best":  (f"{R}/classify/cls-cam-sq160/weights/best.pt", (160, 160)),
    "sq160_last":  (f"{R}/classify/cls-cam-sq160/weights/last.pt", (160, 160)),
}
SETS = {
    "camval":   ("yolo_cam/images/val", "yolo_cam/labels/val"),
    "camtest":  ("yolo_cam/images/test", "yolo_cam/labels/test"),
    "origtest": ("yolo/images/test", "yolo/labels/test"),
}
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def onnx_path(name, kind, tag="fp32"):
    h, w = (DETS if kind == "det" else CLSS)[name][1]
    return WORK / f"{kind}_{name}_{h}x{w}_{tag}.onnx"


def export(args):
    from ultralytics import YOLO
    for kind, table in (("det", DETS), ("cls", CLSS)):
        for name, (pt, (h, w)) in table.items():
            out = onnx_path(name, kind)
            if out.exists():
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = WORK / "src" / f"{kind}_{name}.pt"
            tmp.parent.mkdir(exist_ok=True)
            shutil.copy(pt, tmp)
            p = YOLO(str(tmp)).export(format="onnx", imgsz=[h, w], dynamic=True, simplify=True, batch=8)
            shutil.move(p, out)
            fix_spatial(out, h, w)
            print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB", flush=True)


def load_set(name):
    img_dir, lab_dir = map(Path, SETS[name])
    paths = sorted(img_dir.glob("*.png"))
    actual = np.array([(lab_dir / f"{p.stem}.txt").exists() and (lab_dir / f"{p.stem}.txt").read_text().strip() != ""
                       for p in paths])
    return paths, actual


def scores_of(path, kind, imgs, threads, batch=32):
    m = Model(path, kind, threads)
    return np.concatenate([m.score(m.run(m.prep(imgs[i:i + batch], m.h, m.w))) for i in range(0, len(imgs), batch)])


def score(args):
    cache = WORK / "scores"
    cache.mkdir(parents=True, exist_ok=True)
    models = [(n, "det", onnx_path(n, "det")) for n in DETS] + [(n, "cls", onnx_path(n, "cls")) for n in CLSS]
    models += [(p.stem, "det" if p.name.startswith("det") else "cls", p) for p in map(Path, args.extra or [])]
    for s in args.sets or SETS:
        paths, actual = load_set(s)
        np.save(cache / f"{s}_actual.npy", actual)
        imgs = None
        for name, kind, path in models:
            out = cache / f"{s}_{kind}_{name}.npy"
            if out.exists() or not path.exists():
                continue
            imgs = imgs if imgs is not None else [cv2.imread(str(p)) for p in paths]
            np.save(out, scores_of(path, kind, imgs, args.threads))
            print(f"{s} {kind} {name}: {len(imgs)} bees", flush=True)


def prf(actual, score, t):
    pred = score >= t
    tp, fp, fn = np.sum(pred & actual), np.sum(pred & ~actual), np.sum(~pred & actual)
    return (2 * tp / (2 * tp + fp + fn) if tp else 0.0), (tp / (tp + fp) if tp + fp else 0.0), \
        (tp / (tp + fn) if tp + fn else 0.0)


def load(s, kind, name, cache=None):
    return np.load((cache or WORK / "scores") / f"{s}_{kind}_{name}.npy")


def fbeta(actual, score, t, beta=2.0):
    pred = score >= t
    tp, fp, fn = np.sum(pred & actual), np.sum(pred & ~actual), np.sum(~pred & actual)
    b2 = beta * beta
    return (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp) if tp else 0.0


def pick_cascade(actual, top, prob, thr, tol=0.01, metric=f1_at, hi_min=0.0):
    """Widest skip band (lo, hi on a 0.01 grid) whose camera-val metric stays within tol of the full pipeline.

    With metric=f1_at, hi_min=0 this is the rule the 2026-11-14 cam160 band (0.02 / 0.30) was chosen with;
    fastbench.pick_cascade is the coarse tol=0.001 version used by the board benchmark. hi_min keeps cls_hi a
    "confident mite" level (bees.py counts one check with prob >= cls_hi as a mite at once).
    """
    base, best = metric(actual, (top + prob) / 2, thr), (0.0, max(round(2 * thr, 2), hi_min), 0.0)
    for lo in np.round(np.arange(0.0, thr, 0.01), 2):
        for hi in np.round(np.arange(max(thr, hi_min), 1.0, 0.01), 2):
            skip = np.mean((prob >= hi) | (prob < lo))
            if skip > best[2] and metric(actual, cascade_scores(top, prob, lo, hi), thr) >= base - tol:
                best = (float(lo), float(hi), skip)
    return best[:2]


HI_PRECISION = 0.95   # cls_hi >= lowest prob at which the classifier alone has this precision on camera val
SENS_RECALL, SENS_PRECISION = 0.93, 0.80


def operating_points(det, cls, cache=None, val_det=None):
    """Max-F1 and sensitive operating points, chosen on camera val, applied to camera and original test.

    sensitive = the less aggressive (higher threshold) of: max F2, and the highest threshold with val recall >= 0.93
    at precision >= 0.80 (if any). Its cascade band keeps val F2 within 0.01; the max-F1 band keeps val F1 within 0.01.
    Both bands have cls_hi >= the HI_PRECISION level.
    """
    c = cache or WORK / "scores"
    act = {s: np.load(c / f"{s}_actual.npy") for s in SETS}
    a = act["camval"]
    vt, vp = load("camval", "det", val_det or VAL_TWIN.get(det, det), c), load("camval", "cls", cls, c)
    fv = (vt + vp) / 2
    hi_min = float(min((h for h in GRID if np.sum(vp >= h) and np.mean(a[vp >= h]) >= HI_PRECISION), default=0.5))
    t_f1 = float(max(GRID, key=lambda t: prf(a, fv, t)[0]))
    t_f2 = float(max(GRID, key=lambda t: fbeta(a, fv, t)))
    ok = [float(t) for t in GRID if prf(a, fv, t)[2] >= SENS_RECALL and prf(a, fv, t)[1] >= SENS_PRECISION]
    t_sens = max([t_f2] + ok[-1:])
    out = {"hi_min": hi_min, "sens_rule": "recall>=0.93" if ok and ok[-1] >= t_f2 else "max F2"}
    for name, thr, metric in (("maxf1", t_f1, f1_at), ("sensitive", t_sens, fbeta)):
        lo, hi = pick_cascade(a, vt, vp, thr, metric=metric, hi_min=hi_min)
        r = {"fuse_conf": thr, "cls_lo": lo, "cls_hi": hi, "camval": prf(a, fv, thr),
             "camval_skip": float(np.mean((vp >= hi) | (vp < lo)))}
        for s in ("camtest", "origtest"):
            t, p = load(s, "det", det, c), load(s, "cls", cls, c)
            r[s] = prf(act[s], (t + p) / 2, thr)
            r[s + "_casc"] = prf(act[s], cascade_scores(t, p, lo, hi), thr)
            r[s + "_skip"] = float(np.mean((p >= hi) | (p < lo)))
            r[s + "_hi_precision"] = float(np.mean(act[s][p >= hi])) if np.sum(p >= hi) else float("nan")
        out[name] = r
    return out


def points(args):
    print(f"{'cls':<34} {'point':<9} {'thr':>4} {'lo,hi':>9} | {'cam val F1/P/R':>17} | {'cam test F1 / P / R':>19} "
          f"{'casc F1/P/R':>17} {'skip':>4} {'P@hi':>5} | {'orig test F1 / P / R':>19} {'casc':>5} {'skip':>4}")
    for cls in args.cls:
        o = operating_points(args.det, cls)
        for name in ("maxf1", "sensitive"):
            r = o[name]
            f = lambda x: " / ".join(f"{v:.3f}" for v in x)  # noqa: E731
            print(f"{cls:<34} {name:<9} {r['fuse_conf']:.2f} {r['cls_lo']:.2f},{r['cls_hi']:.2f} | {f(r['camval'])} | "
                  f"{f(r['camtest'])} {f(r['camtest_casc'])} {r['camtest_skip']:4.0%} {r['camtest_hi_precision']:5.3f} | "
                  f"{f(r['origtest'])} {r['origtest_casc'][0]:5.3f} {r['origtest_skip']:4.0%}")
        print(f"{'':<34} sensitive rule: {o['sens_rule']}, cls_hi >= {o['hi_min']:.2f} (val precision {HI_PRECISION})")


def evaluate(det, cls, cache=None, val_det=None, deployed=(0.38, 0.31, 0.60)):
    """Threshold + cascade chosen on camera val (train-only twin detector), applied to camera and original test."""
    c = cache or WORK / "scores"
    act = {s: np.load(c / f"{s}_actual.npy") for s in SETS}
    vd = val_det or VAL_TWIN.get(det, det)
    vt, vp = load("camval", "det", vd, c), load("camval", "cls", cls, c)
    fv = (vt + vp) / 2
    thr = float(max(GRID, key=lambda t: prf(act["camval"], fv, t)[0]))
    lo, hi = pick_cascade(act["camval"], vt, vp, thr)
    row = {"det": det, "cls": cls, "thr": thr, "val_f1": prf(act["camval"], fv, thr)[0], "lo": float(lo),
           "hi": float(hi)}
    for s in ("camtest", "origtest"):
        t, p = load(s, "det", det, c), load(s, "cls", cls, c)
        f = (t + p) / 2
        row[s] = prf(act[s], f, thr)
        row[s + "_casc"] = prf(act[s], cascade_scores(t, p, lo, hi), thr)[0]
        row[s + "_skip"] = float(np.mean((p >= hi) | (p < lo)))
        d_thr, d_lo, d_hi = deployed
        row[s + "_dep"] = prf(act[s], f, d_thr)[0]
        row[s + "_depcasc"] = prf(act[s], cascade_scores(t, p, d_lo, d_hi), d_thr)[0]
        row[s + "_cls"] = prf(act[s], p, float(max(GRID, key=lambda x: prf(act["camval"], vp, x)[0])))[0]
    return row


def table(args):
    act = np.load(WORK / "scores" / "camval_actual.npy")
    print("classifier alone on camera val (best F1 over thresholds, AUC-free):")
    for c in CLSS:
        p = load("camval", "cls", c)
        print(f"  {c:<12} val F1 {max(prf(act, p, t)[0] for t in GRID):.3f}")
    print(f"\n{'det':<6} {'cls':<12} {'thr':>4} {'valF1':>6} | {'cam test F1 / P / R':>21} {'casc':>6} {'skip':>5} | "
          f"{'orig test F1 / P / R':>21} {'casc':>6} {'skip':>5} | lo,hi | @0.38: cam orig, casc cam orig | cls-only cam")
    for det, cls in itertools.product(["base", "cam"], CLSS):
        r = evaluate(det, cls)
        cf, cp, cr = r["camtest"]
        of, op, orr = r["origtest"]
        print(f"{det:<6} {cls:<12} {r['thr']:.2f} {r['val_f1']:6.3f} | {cf:.3f} / {cp:.3f} / {cr:.3f} "
              f"{r['camtest_casc']:6.3f} {r['camtest_skip']:5.0%} | {of:.3f} / {op:.3f} / {orr:.3f} "
              f"{r['origtest_casc']:6.3f} {r['origtest_skip']:5.0%} | {r['lo']:.2f},{r['hi']:.2f} | "
              f"{r['camtest_dep']:.3f} {r['origtest_dep']:.3f}, {r['camtest_depcasc']:.3f} {r['origtest_depcasc']:.3f}"
              f" | {r['camtest_cls']:.3f}", flush=True)


class Reader:
    """fast_build.Reader with a selectable crop folder: n crops, half with mites, same letterbox as the board."""

    def __init__(self, kind, h, w, n, root="yolo_cam", seed=0):
        imgs = sorted(Path(root, "images/train").glob("*.png"))
        labels = Path(root, "labels/train")
        pos = [p for p in imgs if (labels / f"{p.stem}.txt").exists() and (labels / f"{p.stem}.txt").read_text().strip()]
        neg = sorted(set(imgs) - set(pos))
        rng = random.Random(seed)
        pick = rng.sample(pos, n // 2) + rng.sample(neg, n // 2)
        prep = letterbox_det if kind == "det" else letterbox_cls
        self.data = iter([{"images": prep([cv2.imread(str(p))], h, w)} for p in pick])

    def get_next(self):
        return next(self.data, None)


def quantize_one(src: Path, out: Path, kind: str, n: int, root: str, minmax=False):
    """fast_build.py quantize, config ("qdq", "all", "int8qdqall"), with the calibration folder as a parameter."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quant_pre_process, quantize_static
    h, w = (int(x) for x in src.stem.split("_")[-2].split("x"))
    pre = WORK / "tmp" / f"{src.stem}_pre.onnx"
    pre.parent.mkdir(parents=True, exist_ok=True)
    try:
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
    except Exception:  # the detector's decode head defeats symbolic shape inference
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    m = onnx.load(str(pre))
    exclude = [x.name for x in m.graph.node if kind == "det" and "/model.23/" in x.name and x.op_type not in ("Conv",)]
    quantize_static(
        str(pre), str(out), Reader(kind, h, w, n, root),
        quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax if minmax else CalibrationMethod.Percentile,
        op_types_to_quantize=None, nodes_to_exclude=exclude,
        extra_options={"CalibPercentile": 99.999, "ActivationSymmetric": False, "WeightSymmetric": True},
    )
    print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB", flush=True)


def quantize(args):
    for kind, name in (("det", args.det), ("cls", args.cls)):
        src = onnx_path(name, kind)
        out = onnx_path(name, kind, f"int8qdqall-{args.calib.replace('/', '')}")
        if not out.exists():
            quantize_one(src, out, kind, args.n, args.calib, args.minmax)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    s = sub.add_parser("score")
    s.add_argument("--threads", type=int, default=4)
    s.add_argument("--sets", nargs="*")
    s.add_argument("--extra", nargs="*", help="more .onnx files to score (name = file stem)")
    sub.add_parser("table")
    pt = sub.add_parser("points", help="max-F1 and sensitive operating points with cascade bands")
    pt.add_argument("--det", default="cam")
    pt.add_argument("--cls", nargs="+", default=["cam160_last", "sq160_best", "sq160_last", "sq288_best", "sq288_last"])
    q = sub.add_parser("quantize")
    q.add_argument("--det", default="cam")
    q.add_argument("--cls", default="cam160_last")
    q.add_argument("--n", type=int, default=400, help="calibration crops (half with mites), as fast_build.py")
    q.add_argument("--calib", default="yolo_cam", help="crop folder with images/train and labels/train")
    q.add_argument("--minmax", action="store_true")
    args = p.parse_args()
    {"export": export, "score": score, "table": table, "points": points, "quantize": quantize}[args.cmd](args)


if __name__ == "__main__":
    main()
