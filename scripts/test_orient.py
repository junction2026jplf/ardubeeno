"""Standalone check of applab/ardubeeno/python/orient.py on drawn bees (no data or models needed).

  .venv/bin/python scripts/test_orient.py
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "applab" / "ardubeeno" / "python"))
from bees import crop_bee  # noqa: E402
from orient import BEE_ASPECT, bee_angle, crop_bee_oriented, estimate_axis  # noqa: E402


def drawn_bee(axis_deg, length=250, size=(720, 1280)):
    """Grey frame with a striped orange ellipse whose long axis is at axis_deg (0 = along +x, 90 = vertical)."""
    frame = np.full((*size, 3), 128, np.uint8)
    c = (size[1] // 2 + 37, size[0] // 2 - 21)
    a, b = length / 2, length / 2 * BEE_ASPECT
    cv2.ellipse(frame, (c, (2 * a, 2 * b), axis_deg), (40, 150, 220), -1)
    t = math.radians(axis_deg)
    for k in range(-3, 4):  # dark stripes across the body
        p = (c[0] + k * 0.12 * length * math.cos(t), c[1] + k * 0.12 * length * math.sin(t))
        cv2.ellipse(frame, (p, (0.04 * length, 1.6 * b), axis_deg), (20, 50, 80), -1)
    pts = cv2.ellipse2Poly(c, (round(a), round(b)), round(axis_deg), 0, 360, 2)
    x, y, w, h = cv2.boundingRect(pts)
    return frame, (x, y, w, h)


def main():
    fails = 0
    for axis in (90, 100, 120, 135, 150, 170, 0, 30, 45, 60, 80):  # 15 = MIN_TURN itself is a coin flip
        frame, box = drawn_bee(axis)
        est = estimate_axis(frame, box)
        err = None if est is None else abs((est[0] - axis + 90) % 180 - 90)
        a = bee_angle(frame, box)
        crop = crop_bee_oriented(frame, box)
        ok = crop.shape == (280, 160, 3) and err is not None and err < 5
        upright = min(abs((axis - 90 + 90) % 180 - 90), abs((axis + 90) % 180 - 90))
        if upright < 15:  # near crop_bee's own orientation: must be crop_bee exactly
            ok &= a is None and np.array_equal(crop, crop_bee(frame, box))
        else:  # turned: the drawn body must now be vertical and centred in the crop
            m = cv2.inRange(crop, (30, 130, 200), (50, 170, 240))
            mu = cv2.moments(m, binaryImage=True)
            ang = math.degrees(0.5 * math.atan2(2 * mu["mu11"], mu["mu20"] - mu["mu02"])) % 180
            cx, cy = mu["m10"] / mu["m00"], mu["m01"] / mu["m00"]
            ok &= a is not None and abs(ang - 90) < 6 and abs(cx - 80) < 8 and abs(cy - 140) < 12
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.circle(mask, (box[0] + box[2] // 2, box[1] + box[3] // 2), 10, 255, -1)
            mc = crop_bee_oriented(mask, box, angle=a)  # masks: same angle, single channel
            ys, xs = np.nonzero(mc >= 128)
            ok &= mc.shape == (280, 160) and len(xs) > 0 and abs(xs.mean() - 80) < 3 and abs(ys.mean() - 140) < 3
        print(f"axis {axis:3d}  est {'-' if est is None else f'{est[0]:6.1f}'}  turned {a is not None!s:5}  "
              f"{'ok' if ok else 'FAIL'}")
        fails += not ok
    print("all ok" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
