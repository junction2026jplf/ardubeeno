"""Full-screen slideshow of held-out test bees, for filming with the UNO Q camera.

Shows one bee at a time, scaled to fill the screen height. Space/right = next, left = previous,
Esc/q = quit. Mite-positive bees are interleaved so the demo shows both cases.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def load_paths(yolo_dir: Path, split: str, seed: int) -> list[Path]:
    images = sorted((yolo_dir / "images" / split).glob("*.png"))
    labels = yolo_dir / "labels" / split
    pos = [p for p in images if (labels / f"{p.stem}.txt").read_text().strip()]
    neg = [p for p in images if p not in set(pos)]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    # Alternate: two positives, one negative.
    out = []
    while pos or neg:
        out += [pos.pop() for _ in range(2) if pos]
        if neg:
            out.append(neg.pop())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"))
    p.add_argument("--split", default="test")
    p.add_argument("--interval", type=float, default=3.0, help="seconds per slide, 0 = manual only")
    p.add_argument("--screen-height", type=int, default=1080)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = load_paths(args.yolo_dir, args.split, args.seed)
    win = "slideshow"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    i = 0
    delay = int(args.interval * 1000) if args.interval > 0 else 0
    while True:
        img = cv2.imread(str(paths[i % len(paths)]))
        h = args.screen_height
        w = int(img.shape[1] * h / img.shape[0])
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # Mid-grey background, close to the dataset's backdrop.
        canvas = np.full((h, int(h * 16 / 9), 3), 90, np.uint8)
        x = (canvas.shape[1] - w) // 2
        canvas[:, x:x + w] = img
        cv2.imshow(win, canvas)

        key = cv2.waitKeyEx(delay)
        if key in (27, ord("q")):
            break
        if key in (2424832, 63234, 65361):  # left arrow (Windows / macOS / Linux)
            i -= 1
        else:
            i += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
