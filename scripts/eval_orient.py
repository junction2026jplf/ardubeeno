"""Evaluate orient.crop_bee_oriented against bees.crop_bee on rotated bees.

Test crops (data/test, balanced mite/clean) are pasted with scripts/synth_bees.paste at ~250 px bee length onto a
grey-128 and a landing-board (bee-free region) 1280x720 canvas, rotated by each angle. The "bee detector box" is
the axis-aligned box of the paste ellipse (synth_bees convention). Both crops go through the app's mite Pipeline
with deploy.json's settings (as monitor.py builds it).

  .venv/bin/python scripts/eval_orient.py --n 300
"""
import argparse
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "applab" / "ardubeeno" / "python"
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "scripts"))
import synth_bees as sb  # noqa: E402
from bees import crop_bee  # noqa: E402
from device_check import Pipeline  # noqa: E402
import orient  # noqa: E402

ANGLES = [0, 15, 30, 45, 60, 75, 90, 120, 135, 150, 180]


def build_pipeline():
    sys.path.insert(0, str(APP))
    from monitor import DEFAULTS, deploy_settings
    o = {**DEFAULTS, **deploy_settings(APP / "deploy.json")}
    model, cls = APP / o["files"][0], APP / o["files"][1]
    return Pipeline(SimpleNamespace(model=str(model), cls_model=[str(cls)], conf=o["conf"], iou=0.45,
                                    fuse_conf=o["fuse_conf"], threads=2, cascade=o["cascade"], cls_lo=o["cls_lo"],
                                    cls_hi=o["cls_hi"], spin=False))


def pick(n, seed):
    rows = [l.split() for l in (ROOT / "data/test/gt_one.csv").read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(seed)
    mite = [r for r in rows if r[1] != "0"]
    clean = [r for r in rows if r[1] == "0"]
    sel = [mite[i] for i in rng.choice(len(mite), n // 2, replace=False)]
    sel += [clean[i] for i in rng.choice(len(clean), n - n // 2, replace=False)]
    return [(str(ROOT / "data/test" / r[0]), r[1] != "0") for r in sel]


def landing_bg(rng, src, size):
    """A landing-board canvas with no labelled bee near the centre."""
    for _ in range(50):
        img, polys = sb.bg_landing(rng, src)
        c = np.array([sb.W / 2, sb.H / 2])
        if all(np.abs(p.mean(0) - c).max() > size * 0.8 for p in polys):
            return np.ascontiguousarray(img)
    return np.ascontiguousarray(img)


def scene(crop, ang, bg, rng, src, size=250.0):
    canvas = np.full((sb.H, sb.W, 3), 128, np.uint8) if bg == "grey" else landing_bg(rng, src, size)
    cx, cy = sb.W / 2 + rng.uniform(-100, 100), sb.H / 2 + rng.uniform(-60, 60)
    s = float(size * rng.uniform(0.9, 1.1))
    sb.paste(canvas, crop, cx, cy, s, float(ang), False, False, rng)
    a, b = sb.ELL_FX * s / sb.ELL_FY * sb.CROP_W / sb.CROP_H / 2, s / 2
    x0, y0, x1, y1 = sb.poly_box(sb.ellipse_poly(cx, cy, a, b, math.radians(ang)))
    return canvas, (x0, y0, x1 - x0, y1 - y0)


def f1(pred, gt):
    pred, gt = np.array(pred), np.array(gt)
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    p, r = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r), r, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--angles", default=",".join(map(str, ANGLES)))
    ap.add_argument("--bgs", default="grey,landing")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--jitter", type=float, default=0.0, help="box edge noise, sigma as a fraction of the long side")
    ap.add_argument("--min-turn", type=float, default=orient.MIN_TURN)
    args = ap.parse_args()
    cv2.setNumThreads(1)
    pipe = build_pipeline()
    bees = pick(args.n, args.seed)
    crops = [cv2.imread(p) for p, _ in bees]
    gt = [m for _, m in bees]
    src = sb.image_list("mendeley_landing_board", "val")
    ref = [pipe(c)[4] for c in crops]
    F, R, P = f1(ref, gt)
    print(f"reference (original crops): F1 {F:.3f}  R {R:.3f}  P {P:.3f}  n={len(gt)}")
    print(f"{'bg':<8}{'angle':>6} | {'F1 old':>7}{'F1 new':>7} | {'R old':>6}{'R new':>6} | {'P old':>6}{'P new':>6}"
          f" | {'err med':>7}{'err p90':>8}{'turned':>7}{'fallbk':>7} | {'ms old':>7}{'ms new':>7}")
    sheet = []
    for bg in args.bgs.split(","):
        for ang in (int(a) for a in args.angles.split(",")):
            rng = np.random.default_rng([args.seed, ang, bg == "grey"])
            po, pn, errs, to, tn, turned, fb = [], [], [], [], [], 0, 0
            for i, crop in enumerate(crops):
                frame, box = scene(crop, ang, bg, rng, src)
                if args.jitter:
                    e = rng.normal(0, args.jitter * max(box[2:]), 4)
                    box = (box[0] + e[0], box[1] + e[1], box[2] - e[0] + e[2], box[3] - e[1] + e[3])
                t = time.perf_counter()
                co = crop_bee(frame, box)
                t1 = time.perf_counter()
                a = orient.bee_angle(frame, box, args.min_turn)
                cn = orient.crop_bee_oriented(frame, box, angle=a)
                t2 = time.perf_counter()
                to.append((t1 - t) * 1e3)
                tn.append((t2 - t1) * 1e3)
                est = orient.estimate_axis(frame, box)
                true_axis = (ang + 90) % 180  # crop +y rotated by ang (image coords)
                if est is None:
                    fb += 1
                else:
                    errs.append(abs((est[0] - true_axis + 90) % 180 - 90))
                turned += a is not None
                po.append(pipe(co)[4])
                pn.append(pipe(cn)[4])
                if args.sheet and i < 3 and bg == "landing":
                    sheet.append(np.hstack([co, cn]))
            Fo, Ro, Po = f1(po, gt)
            Fn, Rn, Pn = f1(pn, gt)
            e = np.array(errs) if errs else np.array([np.nan])
            print(f"{bg:<8}{ang:>6} | {Fo:7.3f}{Fn:7.3f} | {Ro:6.3f}{Rn:6.3f} | {Po:6.3f}{Pn:6.3f}"
                  f" | {np.median(e):7.1f}{np.percentile(e, 90):8.1f}{turned / len(gt):7.0%}{fb / len(gt):7.0%}"
                  f" | {np.median(to):7.2f}{np.median(tn):7.2f}", flush=True)
    if args.sheet:
        rows = [np.hstack(sheet[i:i + 6]) for i in range(0, len(sheet) - len(sheet) % 6, 6)]
        cv2.imwrite(args.sheet, np.vstack(rows))


if __name__ == "__main__":
    main()
