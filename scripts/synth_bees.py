"""Synthetic training data for the one-class BEE detector: cut-out bee crops pasted onto backgrounds.

The demo films a screen showing bees of ~80-500 px, a size the real datasets (landing board: tiny bees, beehive
close-ups: huge bees) barely cover. This pastes the 160x280 Varroa-dataset bee crops onto 1280x720 canvases:

  scatter (~80%)  1-20 bees at random angle, flip, size (long side 40-520 px, log-uniform, weighted to 80-350)
                  and position, pasted with a feathered elliptical mask (so the hive background of the crop
                  fades out and no rectangle is learned)
  grid    (~20%)  the demo slideshow look: upright full crop rectangles in a 2x1..4x3 grid on grey

Box convention (used everywhere, incl. datasets/cam_tiles_tight): the axis-aligned bounding box of the paste
ellipse, which is centred on the crop and covers ELL_FX x ELL_FY of it. For an upright crop that is just the crop
rectangle shrunk to 0.90 x 0.95 around its centre; for a rotated bee it is the box of the rotated ellipse.

Backgrounds: grey 128 +- jitter, flat colours / gradients, landing-board frames (their real bees keep their labels),
bee-free beehive crops, texture noise. Then screen-filming artefacts on the composite: optional perspective
(the screen inside a dark surround), colour/gamma jitter, blur, noise, moire, banding, JPEG 50-95.

Crop splits are strict: synthetic train/val/test use data/train/val/test crops. ~30% of the scatter bees in
TRAIN are camera-domain crops (yolo_cam/images/train, with crop_bee's 15% margin removed).

  .venv/bin/python scripts/synth_bees.py              # writes datasets/synth_bees/{images,labels}/{train,val,test}
  .venv/bin/python scripts/synth_bees.py --preview 12 --sheet out.jpg   # contact sheet only
"""
import argparse
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets"
W, H = 1280, 720
CROP_W, CROP_H = 160, 280
ELL_FX, ELL_FY = 0.90, 0.95      # paste ellipse = box convention, as fractions of the crop
CAM_FX, CAM_FY = 0.84, 0.91      # camera crops: smaller paste mask (screen edges in the corners), box still ELL_*
FEATHER = 0.15                   # alpha ramps 0 -> 1 between ellipse radius 1 and 1 - FEATHER
CAM_MARGIN = 0.15                # crop_bee's margin, removed from camera crops
SEED = 20261114
COUNTS = {"train": 6000, "val": 600, "test": 600}
SPLIT_ID = {"train": 0, "val": 1, "test": 2}
# Background sources per synthetic split (bee crops are what must not leak; landing board has no test split).
LANDING_SPLIT = {"train": "train", "val": "val", "test": "val"}
BEEHIVE_SPLIT = {"train": "train", "val": "val", "test": "test"}


# ----------------------------------------------------------------------------------------------- sources

def crop_paths(split):
    d = ROOT / "data" / split
    paths = [d / line.split()[0] for line in (d / "gt_one.csv").read_text().splitlines() if line.strip()]
    return sorted(str(p) for p in paths)


def read_yolo(path, w, h):
    """YOLO label file -> array of (x0, y0, x1, y1) pixels."""
    boxes = []
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return np.array(boxes, np.float64).reshape(-1, 4)


def image_list(name, split):
    d = DS / name / "images" / split
    return sorted(str(p) for p in d.glob("*.jpg")) if d.exists() else []


SOURCES = {}


def sources(split):
    if split not in SOURCES:
        SOURCES[split] = {
            "crops": crop_paths(split),
            "cam": sorted(str(p) for p in (ROOT / "yolo_cam" / "images" / "train").glob("*.png")) if split == "train" else [],
            "landing": image_list("mendeley_landing_board", LANDING_SPLIT[split]),
            "beehive": image_list("beehive", BEEHIVE_SPLIT[split]),
        }
    return SOURCES[split]


def load_crop(path, cam):
    img = cv2.imread(path)
    if img is None:
        return None
    if img.shape[:2] != (CROP_H, CROP_W):
        img = cv2.resize(img, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
    if cam:  # undo crop_bee's margin: the original tile is the central 1/1.15
        iw, ih = CROP_W / (1 + CAM_MARGIN), CROP_H / (1 + CAM_MARGIN)
        x0, y0 = round((CROP_W - iw) / 2), round((CROP_H - ih) / 2)
        img = cv2.resize(img[y0:y0 + round(ih), x0:x0 + round(iw)], (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
    return img


# ----------------------------------------------------------------------------------------------- backgrounds

def smooth_noise(rng, h, w, octaves=4, base=None):
    base = base or int(rng.integers(2, 8))
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = base * 2 ** o
        n = rng.random((s * h // w + 2, s + 2)).astype(np.float32)
        out += amp * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= float(rng.uniform(0.4, 0.7))
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def bg_grey(rng):
    g = 128 + rng.normal(0, 12) if rng.random() < 0.75 else rng.uniform(70, 215)
    img = np.full((H, W, 3), g, np.float32) + rng.normal(0, 4, 3)
    if rng.random() < 0.5:  # uneven screen brightness
        img += (smooth_noise(rng, H, W, 2, 1)[..., None] - 0.5) * rng.uniform(5, 30)
    return np.clip(img, 0, 255).astype(np.uint8)


def bg_flat(rng):
    c1, c2 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    if rng.random() < 0.4:
        return np.full((H, W, 3), c1, np.uint8)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if rng.random() < 0.7:  # linear gradient
        a = rng.uniform(0, 2 * math.pi)
        t = (xx * math.cos(a) + yy * math.sin(a))
    else:                   # radial
        t = np.hypot(xx - rng.uniform(0, W), yy - rng.uniform(0, H))
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return (c1 * (1 - t[..., None]) + c2 * t[..., None]).astype(np.uint8)


def bg_texture(rng):
    t = smooth_noise(rng, H, W, int(rng.integers(3, 7)), int(rng.integers(2, 40)))
    if rng.random() < 0.3:  # sharper, grainy
        t = np.clip(t + rng.normal(0, rng.uniform(0.02, 0.15), t.shape), 0, 1)
    c1, c2 = rng.uniform(0, 255, 3), rng.uniform(0, 255, 3)
    if rng.random() < 0.5:  # hive-like browns / greys
        c1, c2 = rng.uniform([20, 30, 40], [90, 110, 140]), rng.uniform([90, 110, 130], [200, 210, 230])
    return (c1 * (1 - t[..., None]) + c2 * t[..., None]).astype(np.float32).clip(0, 255).astype(np.uint8)


def bg_landing(rng, src):
    """Random 16:9 crop of a landing-board frame, resized to the canvas; its labelled bees come along."""
    path = src[int(rng.integers(len(src)))]
    img = cv2.imread(path)
    ih, iw = img.shape[:2]
    boxes = read_yolo(path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", iw, ih)
    cw = rng.uniform(0.33, 1.0) * iw
    ch = cw * H / W
    if ch > ih:
        ch, cw = ih, ih * W / H
    x0, y0 = rng.uniform(0, iw - cw), rng.uniform(0, ih - ch)
    s = W / cw
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    polys = [np.array([[a, b], [c, b], [c, d], [a, d]]) * s - [x0 * s, y0 * s] for a, b, c, d in boxes]
    if rng.random() < 0.5:
        out = out[:, ::-1].copy()
        polys = [np.column_stack([W - p[:, 0], p[:, 1]]) for p in polys]
    return out, polys


def bg_beehive(rng, src):
    """A bee-free 16:9 region of a beehive close-up (boxes there are huge, so tries several frames)."""
    for _ in range(25):
        path = src[int(rng.integers(len(src)))]
        img = cv2.imread(path)
        ih, iw = img.shape[:2]
        boxes = read_yolo(path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt", iw, ih)
        for _ in range(15):
            cw = rng.uniform(0.25, 1.0) * iw
            ch = cw * H / W
            x0, y0 = rng.uniform(0, iw - cw), rng.uniform(0, ih - ch)
            if all(b[2] <= x0 or b[0] >= x0 + cw or b[3] <= y0 or b[1] >= y0 + ch for b in boxes):
                out = cv2.resize(img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)], (W, H), interpolation=cv2.INTER_CUBIC)
                return out[:, ::-1].copy() if rng.random() < 0.5 else out
    return None


def background(rng, src, kind):
    if kind == "landing" and src["landing"]:
        return bg_landing(rng, src["landing"])
    if kind == "beehive" and src["beehive"]:
        img = bg_beehive(rng, src["beehive"])
        if img is not None:
            return img, []
    if kind == "grey":
        return bg_grey(rng), []
    if kind == "flat":
        return bg_flat(rng), []
    return bg_texture(rng), []


# ----------------------------------------------------------------------------------------------- pasting

ELL_T = np.linspace(0, 2 * math.pi, 72, endpoint=False)


def ellipse_poly(cx, cy, a, b, ang):
    """Points on an ellipse with semi-axes a (crop x) and b (crop y), rotated by ang (radians)."""
    x, y = a * np.cos(ELL_T), b * np.sin(ELL_T)
    c, s = math.cos(ang), math.sin(ang)
    return np.column_stack([cx + c * x - s * y, cy + s * x + c * y])


def alpha_mask(w, h, fx, fy):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx + 0.5 - w / 2) / (fx * w / 2)) ** 2 + ((yy + 0.5 - h / 2) / (fy * h / 2)) ** 2)
    t = np.clip((1 - r) / FEATHER, 0, 1)
    return t * t * (3 - 2 * t)


def bee_jitter(rng, crop):
    img = crop.astype(np.float32)
    img = img * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    img *= rng.uniform(0.93, 1.07, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def paste(canvas, crop, cx, cy, size, ang, flip, cam, rng):
    """Paste crop (bee long side = ellipse long axis = size px) centred at (cx, cy), rotated by ang degrees."""
    h = size / ELL_FY
    w = h * CROP_W / CROP_H
    iw, ih = max(4, round(w)), max(4, round(h))
    crop = bee_jitter(rng, crop)
    if flip:
        crop = crop[:, ::-1]
    img = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA if iw < CROP_W else cv2.INTER_CUBIC)
    alpha = alpha_mask(iw, ih, CAM_FX if cam else ELL_FX, CAM_FY if cam else ELL_FY)
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    # crop centre -> (cx, cy), rotated by a
    M = np.array([[c, -s, cx - (c * iw / 2 - s * ih / 2)], [s, c, cy - (s * iw / 2 + c * ih / 2)]])
    corners = np.array([[0, 0, 1], [iw, 0, 1], [iw, ih, 1], [0, ih, 1]], np.float64) @ M.T
    x0, y0 = max(0, int(corners[:, 0].min()) - 1), max(0, int(corners[:, 1].min()) - 1)
    x1, y1 = min(W, int(corners[:, 0].max()) + 2), min(H, int(corners[:, 1].max()) + 2)
    if x1 <= x0 or y1 <= y0:
        return
    Ml = M.copy()
    Ml[:, 2] -= [x0, y0]
    pi = cv2.warpAffine(img, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pa = cv2.warpAffine(alpha, Ml, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (roi * (1 - pa) + pi.astype(np.float32) * pa).astype(np.uint8)


def iou_and_cover(b, boxes):
    """IoU of box b with each box, and the overlap as a fraction of the smaller box."""
    if not len(boxes):
        return np.zeros(0), np.zeros(0)
    B = np.array(boxes)
    ix = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = ix * iy
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (area_b + area_B - inter + 1e-9), inter / (np.minimum(area_b, area_B) + 1e-9)


def poly_box(p):
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])


def sample_size(rng, lo=40, hi=520):
    if rng.random() < 0.65:
        lo, hi = 80, 350
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def distractors(rng, src, canvas):
    """0-3 bee-free elliptical patches (hive-surface-like noise, crop-coloured) pasted with the same feathered mask,
    so 'feathered grey ellipse' alone is not a bee."""
    for _ in range(int(rng.integers(1, 4))):
        size = sample_size(rng)
        crop = load_crop(src["crops"][int(rng.integers(len(src["crops"])))], False)
        ring = np.concatenate([crop[:12].reshape(-1, 3), crop[-12:].reshape(-1, 3)]).astype(np.float32)
        t = smooth_noise(rng, CROP_H, CROP_W, 4, int(rng.integers(3, 20)))[..., None]
        patch = ring.mean(0) + (t - 0.5) * ring.std(0) * rng.uniform(1, 3) + rng.normal(0, 4, (CROP_H, CROP_W, 1))
        patch = np.clip(patch, 0, 255).astype(np.uint8)
        paste(canvas, patch, rng.uniform(0, W), rng.uniform(0, H), size, float(rng.uniform(0, 360)), False, False, rng)


def scatter(rng, src, canvas, polys):
    """Random bees on the canvas; polys already holds background bees. Returns all bee polygons."""
    boxes = [poly_box(p) for p in polys]
    if rng.random() < 0.3:
        distractors(rng, src, canvas)
    n = int(rng.integers(1, 21))
    use_cam = bool(src["cam"])
    # largest first, so the big bees still find room on a filling canvas
    for size in sorted((sample_size(rng) for _ in range(n)), reverse=True):
        cam = use_cam and rng.random() < 0.30
        pool = src["cam"] if cam else src["crops"]
        crop = load_crop(pool[int(rng.integers(len(pool)))], cam)
        if crop is None:
            continue
        ang, flip = float(rng.uniform(0, 360)), bool(rng.random() < 0.5)
        a, b = ELL_FX * size / ELL_FY * CROP_W / CROP_H / 2, size / 2
        outside = rng.random() < 0.10
        for _ in range(30):
            if outside:
                cx, cy = rng.uniform(-0.3 * size, W + 0.3 * size), rng.uniform(-0.3 * size, H + 0.3 * size)
            else:
                cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            poly = ellipse_poly(cx, cy, a, b, math.radians(ang))
            box = poly_box(poly)
            if not outside and (box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H):
                continue
            iou, cover = iou_and_cover(box, boxes)
            if len(boxes) and (iou.max() > 0.25 or cover.max() > 0.40):
                continue
            paste(canvas, crop, cx, cy, size, ang, flip, cam, rng)
            polys.append(poly)
            boxes.append(box)
            break
    return polys


def grid(rng, src, canvas):
    """Slideshow look: upright full crops in a cols x rows grid (2x1 .. 4x3), bee long side 150-500 px."""
    cols, rows = int(rng.integers(2, 5)), int(rng.integers(1, 4))
    gap = rng.uniform(0.02, 0.25)
    # crop height that fits, then a random size below it within 150-500 px of bee (= ellipse long axis)
    fit_h = min(H / (rows + gap * (rows + 1)), W / (cols * CROP_W / CROP_H + gap * (cols + 1)))
    max_bee = min(500, fit_h * ELL_FY)
    if max_bee < 150:
        cols, rows = 2, 1
        fit_h = min(H / (1 + 2 * gap), W / (2 * CROP_W / CROP_H + 3 * gap))
        max_bee = min(500, fit_h * ELL_FY)
    bee = float(math.exp(rng.uniform(math.log(min(150, max_bee)), math.log(max_bee))))
    ch = bee / ELL_FY
    cw = ch * CROP_W / CROP_H
    sx, sy = (W - cols * cw) / (cols + 1), (H - rows * ch) / (rows + 1)
    ox, oy = rng.uniform(-0.5, 0.5) * sx * 0.8, rng.uniform(-0.5, 0.5) * sy * 0.8
    polys = []
    for r in range(rows):
        for c in range(cols):
            if rng.random() < 0.05:  # an empty slot now and then
                continue
            crop = load_crop(src["crops"][int(rng.integers(len(src["crops"])))], False)
            x0 = ox + sx + c * (cw + sx) + rng.normal(0, 2)
            y0 = oy + sy + r * (ch + sy) + rng.normal(0, 2)
            iw, ih = round(cw), round(ch)
            img = cv2.resize(bee_jitter(rng, crop), (iw, ih), interpolation=cv2.INTER_CUBIC)
            xi, yi = round(x0), round(y0)
            ax0, ay0, ax1, ay1 = max(0, xi), max(0, yi), min(W, xi + iw), min(H, yi + ih)
            if ax1 <= ax0 or ay1 <= ay0:
                continue
            canvas[ay0:ay1, ax0:ax1] = img[ay0 - yi:ay1 - yi, ax0 - xi:ax1 - xi]
            mx, my = xi + iw / 2, yi + ih / 2
            polys.append(ellipse_poly(mx, my, ELL_FX * iw / 2, ELL_FY * ih / 2, 0.0))
    return polys


# ----------------------------------------------------------------------------------------------- camera look

CANVAS = np.float32([[0, 0], [W, 0], [W, H], [0, H]])


def clip_polys(polys, real):
    """Clip each (convex) bee polygon to the canvas; drop bees with < 40% of their box area visible."""
    out, keep = [], []
    for p, r in zip(polys, real):
        full = poly_box(p)
        area, q = cv2.intersectConvexConvex(p.astype(np.float32), CANVAS)
        if q is None or area <= 0:
            continue
        q = q.reshape(-1, 2).astype(np.float64)
        b = poly_box(q)
        if (b[2] - b[0]) * (b[3] - b[1]) >= 0.4 * (full[2] - full[0]) * (full[3] - full[1]):
            out.append(q)
            keep.append(r)
    return out, keep


def perspective(rng, img, polys, surround):
    """The screen seen by a camera: canvas mapped to a jittered quad inside a surround image."""
    m = rng.uniform(0.0, 0.12)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * np.float32([W, H]) * m
    dst += rng.normal(0, 0.025, (4, 2)).astype(np.float32) * np.float32([W, H])
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, P, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((H, W), np.float32), P, (W, H))[..., None]
    out = (warped * mask + surround * (1 - mask)).astype(np.uint8)
    new = []
    for p in polys:
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float64), P).reshape(-1, 2)
        new.append(q)
    return out, new


def moire(rng, img):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = rng.uniform(0, math.pi)
    f = rng.uniform(0.15, 0.6)
    f2 = f * rng.uniform(0.95, 1.05)
    pat = np.sin((xx * math.cos(a) + yy * math.sin(a)) * f) * np.sin((xx * math.cos(a + 0.03) + yy * math.sin(a + 0.03)) * f2)
    return img + pat[..., None] * rng.uniform(2, 8)


def banding(rng, img):
    period = rng.uniform(20, 200)
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    return img + np.sin(yy / period * 2 * math.pi + rng.uniform(0, 6.3)) * rng.uniform(2, 8)


def photometric(rng, img):
    f = img.astype(np.float32)
    f = (f - 128) * rng.uniform(0.75, 1.25) + 128 + rng.uniform(-25, 25)
    f *= rng.uniform(0.88, 1.12, 3)
    if rng.random() < 0.15:
        f = moire(rng, f)
    if rng.random() < 0.10:
        f = banding(rng, f)
    f = np.clip(f, 0, 255)
    f = 255 * (f / 255) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.7:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.3, 1.5))
    if rng.random() < 0.7:
        f += rng.normal(0, rng.uniform(1, 8), f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------- one image

def make(split, idx):
    rng = np.random.default_rng([SEED, SPLIT_ID[split], idx])
    src = sources(split)
    if rng.random() < 0.20:
        layout = "grid"
        canvas = bg_grey(rng)
        polys = grid(rng, src, canvas)
        n_real = 0
        persp = rng.random() < 0.5
    else:
        layout = "scatter"
        kind = rng.choice(["grey", "flat", "landing", "beehive", "texture"], p=[0.25, 0.15, 0.25, 0.15, 0.20])
        canvas, polys = background(rng, src, kind)
        canvas = np.ascontiguousarray(canvas)
        n_real = len(polys)  # landing-board bees, labelled in the source
        polys = scatter(rng, src, canvas, list(polys))
        persp = rng.random() < 0.15
    polys, real = clip_polys(polys, [i < n_real for i in range(len(polys))])
    if persp:
        surround = bg_texture(rng).astype(np.float32) * rng.uniform(0.15, 0.6)
        canvas, polys = perspective(rng, canvas, polys, surround)
    img = photometric(rng, canvas)
    lines, sizes = [], []
    for p, r in zip(polys, real):
        x0, y0, x1, y1 = poly_box(p)
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            continue
        lines.append(f"0 {(cx0 + cx1) / 2 / W:.6f} {(cy0 + cy1) / 2 / H:.6f} {(cx1 - cx0) / W:.6f} {(cy1 - cy0) / H:.6f}")
        sizes.append((max(x1 - x0, y1 - y0), r))
    return img, lines, int(rng.integers(50, 96)), layout, sizes


def work(job):
    split, idx, out = job
    img, lines, q, layout, sizes = make(split, idx)
    name = f"synth_{split}_{idx:05d}"
    cv2.imwrite(str(Path(out) / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])
    (Path(out) / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return split, layout, sizes


def draw(img, lines):
    img = img.copy()
    for line in lines:
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        cv2.rectangle(img, (round((cx - bw / 2) * W), round((cy - bh / 2) * H)),
                      (round((cx + bw / 2) * W), round((cy + bh / 2) * H)), (0, 0, 255), 2)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(DS / "synth_bees"))
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--preview", type=int, default=0, help="only render N random train images to --sheet")
    p.add_argument("--sheet", default="synth_sheet.jpg")
    p.add_argument("--start", type=int, default=0, help="preview index offset")
    args = p.parse_args()
    cv2.setNumThreads(1)

    if args.preview:
        tiles = []
        for i in range(args.preview):
            img, lines, q, _, _ = make("train", args.start + i * 487)
            img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
            tiles.append(cv2.resize(draw(img, lines), (640, 360), interpolation=cv2.INTER_AREA))
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(args.sheet, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
        print("wrote", args.sheet)
        return

    out = Path(args.out)
    if out.exists() and any(out.rglob("*.jpg")):
        sys.exit(f"{out} already has images, delete it first")
    for s in COUNTS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
    jobs = [(s, i, str(out)) for s, n in COUNTS.items() for i in range(n)]
    from collections import Counter, defaultdict
    layouts, sizes = Counter(), defaultdict(list)
    with Pool(min(6, args.procs)) as pool:
        for k, (s, layout, sz) in enumerate(pool.imap_unordered(work, jobs, chunksize=16)):
            layouts[s, layout] += 1
            sizes[s] += sz
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    edges = [0, 40, 80, 120, 160, 200, 250, 300, 350, 400, 450, 520, 10000]
    for s in COUNTS:
        a = np.array([v for v, r in sizes[s] if not r])
        real = sum(r for _, r in sizes[s])
        h, _ = np.histogram(a, edges)
        print(f"{s}: {COUNTS[s]} images ({layouts[s, 'grid']} grid), {len(a)} pasted bees "
              f"(median box long side {np.median(a):.0f} px) + {real} real landing-board bees")
        print("   pasted, box long side px: " + "  ".join(
            f"{edges[i]}-{edges[i + 1] if edges[i + 1] < 10000 else ''}:{h[i]}" for i in range(len(h))))


if __name__ == "__main__":
    main()
