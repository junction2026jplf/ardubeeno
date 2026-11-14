"""Bee crops along the bee's own axis, for bees at any angle.

crop_bee (bees.py) cuts the axis-aligned box and only turns it by 90 degrees, so a bee at 45 degrees ends up small,
tilted and surrounded by background in the 160x280 crop, and the mite models (trained on near-upright crops with
+-15 degrees of rotation augmentation) miss mites. crop_bee_oriented has the same contract but:

1. Angle from pixels: a square patch around the detector box (1.25x its long side), downscaled to SEG_PX, is split
   into bee and background: saturation plus the colour distance to the patch border's median Lab colour, Otsu
   threshold, morphological open + close, limited to the detector box; the largest component near the centre is
   the bee. Its second moments give the major axis.
2. Rotation: one warpAffine of the full-resolution frame maps that axis to vertical (head up or down, whichever
   needs less turning; the training crops have both), into the 4:7 window with crop_bee's margin semantics, then
   INTER_AREA down to 160x280.
   Window size: the detector box is the axis-aligned box of an ellipse with the bee's aspect BEE_ASPECT (the box
   convention of the bee detector's training data), so the bee length L follows from the box and the angle alone:
   (w/2)^2 + (h/2)^2 = a^2 + b^2 for any ellipse angle, with a = L/2, b = BEE_ASPECT * a. An upright box gives
   exactly crop_bee's window.
3. Fallbacks to crop_bee: segmentation unreliable (component < 12% or > 90% of the box, elongation < 1.3, more than
   15% of it on the patch border, none near the centre, or an axis more than BOX_TOL degrees from both angles the
   box's own shape allows for a bee-shaped ellipse), or the axis within MIN_TURN degrees of crop_bee's own
   orientation (upright for h >= w, lying for w > h): the models handle that range and crop_bee is then the
   bit-identical old behaviour.

Training data must be cut the same way (scripts/camera_crops.py): bee_angle() once per bee on the frame, then
crop_bee_oriented(frame, box, angle=a) and crop_bee_oriented(mite_mask, box, angle=a) with the same angle.
"""
import math

import cv2
import numpy as np

CROP_W, CROP_H = 160, 280         # the training crops' size (= bees.CROP_W/CROP_H)
ROI_ASPECT = CROP_W / CROP_H      # = device_check.ROI_ASPECT
BEE_ASPECT = (0.90 * CROP_W) / (0.95 * CROP_H)  # bee box convention: ellipse 0.90 x 0.95 of an upright crop
SEG_PX = 64                       # side of the downscaled patch the angle is measured on
PATCH_PAD = 1.25                  # patch side = PATCH_PAD x the box's long side
MIN_TURN = 15.0                   # degrees: closer to crop_bee's orientation than this -> crop_bee
MIN_ELONG, MIN_FILL, MAX_FILL, MAX_BORDER = 1.3, 0.12, 0.90, 0.15
BOX_TOL = 20.0                    # degrees: the axis must fit the box's shape (box_angles) this well
BORDER_VALUE = (114, 114, 114)    # crop_bee's padding colour

_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _crop_bee(frame, box, margin):
    from bees import crop_bee  # lazy: bees.py may import this module
    return crop_bee(frame, box, margin)


def bee_map(patch):
    """uint8 'how bee-like' map of a small BGR patch: HSV saturation (bees are yellow-brown, hive floor and screens
    grey) plus the Lab distance to the patch border's median colour (lightness half, colour double weight)."""
    sat = cv2.normalize(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[..., 1], None, 0, 255, cv2.NORM_MINMAX,
                        cv2.CV_32F)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.concatenate([lab[:2].reshape(-1, 3), lab[-2:].reshape(-1, 3), lab[:, :2].reshape(-1, 3),
                           lab[:, -2:].reshape(-1, 3)])
    d = (lab - np.median(ring, 0)) * np.float32([0.5, 2, 2])
    dist = cv2.normalize(np.sqrt((d * d).sum(2)), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.normalize(sat + dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def box_angles(box):
    """The two axis angles (degrees, mirror images) for which an ellipse of aspect BEE_ASPECT has this box."""
    X2, Y2 = (box[2] / 2) ** 2, (box[3] / 2) ** 2
    k = (1 + BEE_ASPECT ** 2) / (1 - BEE_ASPECT ** 2)
    p = math.degrees(0.5 * math.acos(min(1.0, max(-1.0, (X2 - Y2) / (X2 + Y2) * k))))
    return p, (180.0 - p) % 180.0


def _adiff(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def estimate_axis(frame, box):
    """Major axis of the bee in the box: (angle in degrees in [0, 180), image coords, 0 = along +x, 90 = vertical;
    elongation) or None when the segmentation is unreliable."""
    x, y, w, h = (float(v) for v in box[:4])
    if w < 4 or h < 4 or frame.ndim != 3:
        return None
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * PATCH_PAD / 2
    H, W = frame.shape[:2]
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(W, int(math.ceil(cx + half))), min(H, int(math.ceil(cy + half)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    s = SEG_PX / (2 * half)
    pw, ph = max(8, round((x1 - x0) * s)), max(8, round((y1 - y0) * s))
    step = max(1, int(1 / (2 * s)))  # subsample to ~2x SEG_PX first: INTER_AREA on the full patch costs ~1 ms
    patch = cv2.resize(frame[y0:y1:step, x0:x1:step], (pw, ph), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(bee_map(patch), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # only inside the detector box (in patch pixels)
    bx0, by0 = round((x - x0) * s), round((y - y0) * s)
    bx1, by1 = round((x + w - x0) * s), round((y + h - y0) * s)
    inside = np.zeros_like(mask)
    inside[max(0, by0):max(0, by1), max(0, bx0):max(0, bx1)] = 255
    mask &= inside
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    pcx, pcy = (cx - x0) * s, (cy - y0) * s
    near = 0.25 * SEG_PX
    best = max((i for i in range(1, n) if abs(cent[i, 0] - pcx) < near and abs(cent[i, 1] - pcy) < near),
               key=lambda i: stats[i, cv2.CC_STAT_AREA], default=None)
    if best is None:
        return None
    area = stats[best, cv2.CC_STAT_AREA]
    box_area = max(1, (bx1 - bx0) * (by1 - by0))
    if not MIN_FILL <= area / box_area <= MAX_FILL:
        return None
    comp = (lbl == best).astype(np.uint8)
    on_border = comp[0].sum() + comp[-1].sum() + comp[1:-1, 0].sum() + comp[1:-1, -1].sum()
    if on_border > MAX_BORDER * area:
        return None
    m = cv2.moments(comp, binaryImage=True)
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    d = math.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    l1, l2 = (mu20 + mu02 + d) / 2, (mu20 + mu02 - d) / 2
    elong = math.sqrt(l1 / max(l2, 1e-9))
    if elong < MIN_ELONG:
        return None
    ang = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02)) % 180.0
    if min(_adiff(ang, b) for b in box_angles(box)) > BOX_TOL:  # the box's shape disagrees: segmentation failed
        return None
    return ang, elong


def bee_angle(frame, box, min_turn=MIN_TURN):
    """The bee's axis angle (degrees, see estimate_axis) if crop_bee_oriented should turn the bee, else None
    (crop_bee's crop is good enough or the angle is not reliable). Call once per bee and pass it on as `angle` to cut
    the frame and any masks (mite boxes) identically."""
    est = estimate_axis(frame, box)
    if est is None:
        return None
    ang = est[0]
    ref = 0.0 if box[2] > box[3] else 90.0  # crop_bee's long axis: lying when w > h, else upright
    return ang if _adiff(ang, ref) > min_turn else None


def oriented_matrix(box, angle, margin=0.15):
    """2x3 matrix from crop pixels (160x280) to frame pixels for a bee at `angle` degrees in `box`, plus the frame
    pixels per crop pixel."""
    x, y, w, h = (float(v) for v in box[:4])
    cx, cy = x + w / 2, y + h / 2
    a = math.sqrt(((w / 2) ** 2 + (h / 2) ** 2) / (1 + BEE_ASPECT ** 2))  # half bee length (box = rotated ellipse)
    long, short = 2 * a, 2 * a * BEE_ASPECT
    if short / long > ROI_ASPECT:
        long = short / ROI_ASPECT
    long = long * (1 + margin)
    k = long / CROP_H
    t = math.radians(angle % 180.0)  # in [0, pi): crop +y = (cos t, sin t) points down-ish, the smaller turn
    ex, ey = (math.sin(t), -math.cos(t)), (math.cos(t), math.sin(t))
    M = np.array([[k * ex[0], k * ey[0], 0.0], [k * ex[1], k * ey[1], 0.0]])
    M[:, 2] = (cx, cy) - M[:, :2] @ (CROP_W / 2, CROP_H / 2)
    return M, k


def crop_bee_oriented(frame, box, margin=0.15, angle="auto"):
    """Drop-in for bees.crop_bee: the bee cut out in the training crops' 4:7 portrait shape (CROP_W x CROP_H),
    turned so its long axis is vertical. angle="auto" estimates it (bee_angle); None means crop_bee; a number is
    used as is (for cutting masks with a frame's angle)."""
    if isinstance(angle, str):
        angle = bee_angle(frame, box)
    if angle is None:
        return _crop_bee(frame, box, margin)
    M, k = oriented_matrix(box, angle, margin)
    border = BORDER_VALUE if frame.ndim == 3 else BORDER_VALUE[0]
    if k <= 1.0:  # upsampling or ~1:1: straight into the crop
        return cv2.warpAffine(frame, M, (CROP_W, CROP_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    big = (round(CROP_W * k), round(CROP_H * k))  # full resolution window, then INTER_AREA like crop_bee
    Mb = M.copy()
    Mb[:, :2] = M[:, :2] / k * np.array([[CROP_W * k / big[0], CROP_H * k / big[1]]])
    win = cv2.warpAffine(frame, Mb, big, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    return cv2.resize(win, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
