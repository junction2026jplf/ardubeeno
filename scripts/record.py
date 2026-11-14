"""Screen-to-camera recorder: show dataset images full-screen and film them, carrying the labels over.

Shows each image (or a tile of several bee crops) full-screen on a black background, films the screen with
a camera, and saves the video plus settled still frames with YOLO labels in camera coordinates:

  1. Calibration: a grid of ArUco markers is shown, detected in the camera image, and a homography
     screen -> camera is fitted. It is repeated every --recalib-every slides, when the camera seems to have
     moved (the rectified camera view stops matching the slide), on key c, and once at the end (drift check).
  2. Slides: after --settle seconds, a frame is kept only once the screen in the camera view matches the
     new slide better than the old one and is steady (no transition blends).

Sources: a data split with gt_one.csv (data/test, or the csv itself), a YOLO dataset (folder with images/
and labels/, or its yaml), or a folder of plain images (no labels). Output classes: 0 bee, 1 varroa, then
any other source class names. Output goes to captures/screen-<time>/:
video.mp4, images/*.jpg, labels/*.txt, manifest.csv, calibration.json + calib_*.jpg, data.yaml.

  python scripts/record.py --list-cameras          # find the USB camera's index
  python scripts/record.py data/test --camera 1 --duration 1.5 --limit 200 --shuffle
  python scripts/record.py datasets/foo --camera 1 --screen 1
  python scripts/record.py data/test --camera sim --headless --limit 10   # dry run, simulated camera

--camera takes a device index, a video file (replayed in real time), sim[:move_at=SECONDS] (renders the
slides through a known homography) or none (slideshow only). Keys: Esc/q quit, c recalibrate.
macOS: the terminal app running this needs Camera permission (System Settings > Privacy & Security > Camera).
"""
import argparse
import csv
import json
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
OUT_NAMES = ["bee", "varroa"]
ALIASES = {"bee": "bee", "bees": "bee", "honeybee": "bee", "honey bee": "bee", "apis mellifera": "bee",
           "varroa": "varroa", "mite": "varroa", "mites": "varroa", "varroa mite": "varroa",
           "varroa_mite": "varroa", "varroa destructor": "varroa"}
ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
WIN = "record"


# ---------------------------------------------------------------- sources

@dataclass
class Item:
    path: Path
    boxes: list = field(default_factory=list)  # (name, x1, y1, x2, y2), in pixels or 0..1 if norm
    norm: bool = False
    whole: str | None = None  # class of a box covering the whole image (single-bee crops: "bee")
    tag: str = ""  # source label for the manifest: gt_one.csv flag or label file

    @property
    def mite(self) -> bool:
        return any(b[0] == "varroa" for b in self.boxes) or self.tag in ("1", "3")


def load_crops(csv_path: Path) -> list[Item]:
    """gt_one.csv: <png relative to the csv> <flag 0|1|3> [x1 y1 x2 y2 ...] mite boxes in crop pixels."""
    items = []
    for line in open(csv_path):
        parts = line.split()
        if not parts:
            continue
        c = list(map(float, parts[2:]))
        boxes = [("varroa", min(c[i], c[i + 2]), min(c[i + 1], c[i + 3]), max(c[i], c[i + 2]), max(c[i + 1], c[i + 3]))
                 for i in range(0, len(c) - 3, 4)]
        items.append(Item(csv_path.parent / parts[0], boxes, whole="bee", tag=parts[1]))
    return items


def yolo_names(cfg_names) -> dict[int, str]:
    if isinstance(cfg_names, dict):
        return {int(k): str(v) for k, v in cfg_names.items()}
    return dict(enumerate(map(str, cfg_names or [])))


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_label(path: Path, names: dict[int, str], class_map: dict[str, str]) -> list:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        cls, c = int(float(v[0])), list(map(float, v[1:]))
        if len(c) == 4:  # x y w h
            x, y, w, h = c
            box = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        else:  # polygon
            xs, ys = c[0::2], c[1::2]
            box = (min(xs), min(ys), max(xs), max(ys))
        name = names.get(cls, f"class{cls}")
        name = class_map.get(name, ALIASES.get(name.lower().strip(), name))
        boxes.append((name, *box))
    return boxes


def image_files(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."))


def load_yolo(src: Path, split: str | None, class_map: dict[str, str]) -> list[Item]:
    """A YOLO dataset folder (images/ + labels/ anywhere below it) or its yaml."""
    import yaml

    if src.suffix in (".yaml", ".yml"):
        cfg = yaml.safe_load(src.read_text())
        base = Path(cfg.get("path") or ".")
        base = base if base.is_absolute() else src.parent / base
        dirs = []
        for s in [split] if split else ["train", "val", "test"]:
            entries = cfg.get(s) or []
            for e in entries if isinstance(entries, list) else [entries]:
                p = base / e
                dirs += [Path(l.strip()) if Path(l.strip()).is_absolute() else base / l.strip()
                         for l in p.read_text().splitlines() if l.strip()] if p.suffix == ".txt" else [p]
        images = [f for d in dirs for f in (image_files(d) if d.is_dir() else [d])]
        names = yolo_names(cfg.get("names"))
    else:
        images = [p for p in image_files(src) if "images" in p.relative_to(src).parts]
        if split:
            images = [p for p in images if split in p.relative_to(src).parts]
        names = {}
        yamls = sorted(src.glob("*.yaml")) + sorted(src.glob("*.yml"))
        if yamls:
            names = yolo_names(yaml.safe_load(yamls[0].read_text()).get("names"))
        elif (src / "classes.txt").exists():
            names = dict(enumerate((src / "classes.txt").read_text().split("\n")))
    items = []
    for img in images:
        lbl = label_path(img) if "images" in img.parts else img.with_suffix(".txt")
        items.append(Item(img, read_yolo_label(lbl, names, class_map), norm=True, tag=str(lbl)))
    return items


def load_source(src: Path, split: str, class_map: dict[str, str]) -> list[Item]:
    if src.suffix == ".csv":
        return load_crops(src)
    if src.is_dir() and (src / "gt_one.csv").exists():
        return load_crops(src / "gt_one.csv")
    if src.is_dir() and (src / split / "gt_one.csv").exists():
        return load_crops(src / split / "gt_one.csv")
    if src.suffix in (".yaml", ".yml"):
        return load_yolo(src, split, class_map)
    for y in sorted(src.glob("*.y*ml")) if src.is_dir() else []:  # a dataset folder with its yaml: use the yaml (ignores raw/ copies of the images)
        import yaml
        cfg = yaml.safe_load(y.read_text())
        if isinstance(cfg, dict) and any(k in cfg for k in ("train", "val", "test")):
            return load_yolo(y, split if split in cfg else None, class_map)
    if src.is_dir() and any(p.is_dir() and p.name == "images" for p in src.rglob("images")):
        has_split = any(split in p.relative_to(src).parts for p in src.rglob(split))
        return load_yolo(src, split if has_split else None, class_map)
    if src.is_dir():
        return [Item(p) for p in image_files(src)]
    raise SystemExit(f"unknown source {src}: expected gt_one.csv, a data split, a YOLO folder/yaml or images")


# ---------------------------------------------------------------- rendering

def parse_size(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


def render(items: list[Item], size: tuple[int, int], tile: tuple[int, int], fill: float, bg: int, gap: int = 0):
    """Lay out the images in a cols x rows grid. Returns the canvas and boxes (name, x1, y1, x2, y2) in it."""
    W, H = size
    cols, rows = tile
    canvas = np.full((H, W, 3), bg, np.uint8)
    boxes = []
    cw, ch = (W - gap * (cols + 1)) / cols, (H - gap * (rows + 1)) / rows
    for k, item in enumerate(items):
        img = cv2.imread(str(item.path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {item.path}")
        h, w = img.shape[:2]
        s = min(cw / w, ch / h) * fill
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        r, c = divmod(k, cols)
        x0 = round(gap + c * (cw + gap) + (cw - nw) / 2)
        y0 = round(gap + r * (ch + gap) + (ch - nh) / 2)
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if item.whole:
            boxes.append((item.whole, x0, y0, x0 + nw, y0 + nh))
        sx, sy = (nw, nh) if item.norm else (nw / w, nh / h)
        for name, x1, y1, x2, y2 in item.boxes:
            boxes.append((name, x0 + x1 * sx, y0 + y1 * sy, x0 + x2 * sx, y0 + y2 * sy))
    return canvas, boxes


def calib_canvas(size: tuple[int, int], grid: tuple[int, int], bg: int = 0):
    """ArUco markers (each on a white quiet zone) in a cols x rows grid. Returns canvas, {id: 4x2 corners}."""
    W, H = size
    cols, rows = grid
    cw, ch = W / cols, H / rows
    s = int(min(cw, ch) * 0.7)
    q = max(4, s // 7)
    canvas = np.full((H, W, 3), bg, np.uint8)
    ref = {}
    for i in range(cols * rows):
        r, c = divmod(i, cols)
        x, y = round(c * cw + (cw - s) / 2), round(r * ch + (ch - s) / 2)
        canvas[y - q:y + s + q, x - q:x + s + q] = 255
        canvas[y:y + s, x:x + s] = cv2.cvtColor(cv2.aruco.generateImageMarker(ARUCO, i, s), cv2.COLOR_GRAY2BGR)
        ref[i] = np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], np.float32)  # aruco corner order
    return canvas, ref


def fit_homography(ref: dict, corners, ids):
    """Homography screen -> camera from detected markers. Returns H, number of markers, RMS error (px)."""
    if ids is None:
        return None, 0, 0.0
    pairs = [(ref[int(i)], c.reshape(4, 2)) for c, i in zip(corners, ids.flatten()) if int(i) in ref]
    if len(pairs) < 2:
        return None, len(pairs), 0.0
    src = np.concatenate([p[0] for p in pairs])
    dst = np.concatenate([p[1] for p in pairs]).astype(np.float32)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None, len(pairs), 0.0
    err = np.linalg.norm(cv2.perspectiveTransform(src[None], H)[0] - dst, axis=1)
    return H, len(pairs), float(np.sqrt((err[inl.ravel() > 0] ** 2).mean()))


def map_boxes(boxes, H, frame_size, min_visible=0.5):
    """Map canvas boxes through H. Returns [(name, x1, y1, x2, y2)] in frame pixels, clipped to the frame."""
    W, Hf = frame_size
    out = []
    for name, x1, y1, x2, y2 in boxes:
        pts = cv2.perspectiveTransform(np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float64), H)[0]
        a1, b1 = pts.min(0)
        a2, b2 = pts.max(0)
        c1, d1, c2, d2 = max(a1, 0), max(b1, 0), min(a2, W), min(b2, Hf)
        if c2 <= c1 or d2 <= d1 or (c2 - c1) * (d2 - d1) < min_visible * (a2 - a1) * (b2 - b1):
            continue
        out.append((name, c1, d1, c2, d2))
    return out


def yolo_lines(boxes, names: list[str], frame_size) -> list[str]:
    W, H = frame_size
    return [f"{names.index(n)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
            for n, x1, y1, x2, y2 in boxes]


def long_sides(boxes, name="bee") -> list[float]:
    return [max(x2 - x1, y2 - y1) for n, x1, y1, x2, y2 in boxes if n == name]


def corner_drift(H1, H2, size) -> float:
    """Max distance (camera px) between where H1 and H2 put the canvas corners."""
    W, H = size
    c = np.array([[[0, 0], [W, 0], [W, H], [0, H]]], np.float64)
    return float(np.linalg.norm(cv2.perspectiveTransform(c, H1) - cv2.perspectiveTransform(c, H2), axis=2).max())


class ScreenView:
    """Small blurred grey views of canvas and camera frame (rectified via H), to compare them."""

    def __init__(self, H, canvas_size, frame_size, width=256):
        W, Hc = canvas_size
        self.size = (width, max(8, round(width * Hc / W)))
        self.f = min(1.0, 2.5 * width / frame_size[0])
        S = np.diag([width / W, self.size[1] / Hc, 1.0])
        self.M = S @ np.linalg.inv(np.diag([self.f, self.f, 1.0]) @ H)

    @staticmethod
    def _prep(gray):
        return cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.5)

    def canvas(self, canvas):
        return self._prep(cv2.resize(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), self.size, interpolation=cv2.INTER_AREA))

    def frame(self, frame):
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=self.f, fy=self.f, interpolation=cv2.INTER_AREA)
        return self._prep(cv2.warpPerspective(small, self.M, self.size, flags=cv2.INTER_LINEAR))


def edges(gray):
    """Gradient magnitude: compares slides by their structure, not just where the bright bee blob is."""
    return cv2.GaussianBlur(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)), (5, 5), 1)


def ncc(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


# ---------------------------------------------------------------- display and cameras

def list_screens() -> list[tuple[int, int, int, int, float]]:
    """macOS screens as (x, y, w, h, scale) in points, top-left origin."""
    js = ('ObjC.import("AppKit"); var s=$.NSScreen.screens, o=[]; for (var i=0;i<s.count;i++)'
          '{var f=s.objectAtIndex(i).frame; o.push([f.origin.x,f.origin.y,f.size.width,f.size.height,'
          's.objectAtIndex(i).backingScaleFactor].join(" "))}; o.join("\\n")')
    try:
        out = subprocess.run(["osascript", "-l", "JavaScript", "-e", js], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = [list(map(float, l.split())) for l in out.splitlines() if l.strip()]
    main_h = rows[0][3] if rows else 0
    return [(int(x), int(main_h - y - h), int(w), int(h), s) for x, y, w, h, s in rows]


class Display:
    def __init__(self, headless: bool, canvas: str | None, screen: int | None, bg: int):
        self.headless, self.history, self.quit, self.recal, self.go = headless, [], False, False, False
        if headless:
            self.size = parse_size(canvas or "1920x1080")
            return
        screens = list_screens()
        sx, sy, sw, sh, scale = screens[screen if screen is not None and screen < len(screens) else 0] \
            if screens else (0, 0, 1920, 1080, 1.0)
        # The window is sized to the whole screen before going full screen: macOS (Cocoa) only applies the
        # full-screen property to a window that already shows an image.
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # images are letterboxed by render(), never stretched
        cv2.imshow(WIN, np.full((sh, sw, 3), bg, np.uint8))
        cv2.waitKey(50)
        cv2.moveWindow(WIN, sx, sy)
        cv2.resizeWindow(WIN, sw, sh)
        cv2.waitKey(50)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.wait(1.5)  # macOS full-screen animation
        _, _, w, h = cv2.getWindowImageRect(WIN)
        if canvas:
            self.size = parse_size(canvas)
        elif w >= sw * 0.9 and h >= sh * 0.8:
            self.size = (round(w * scale), round(h * scale))
        else:
            # Full screen didn't take (window rect stays small); still draw slides at the screen's resolution.
            print(f"warning: the window is {w}x{h} points, not full screen ({sw}x{sh}). Press the green button in "
                  "its title bar or Ctrl+Cmd+F, then c to recalibrate")
            self.size = (round(sw * scale), round(sh * scale))

    def show(self, canvas) -> float:
        if not self.headless:
            cv2.imshow(WIN, canvas)
        self.poll(1)
        t = time.monotonic()
        self.history = self.history[-30:] + [(t, canvas)]
        return t

    def at(self, t):
        """The canvas that was on screen at time t (for the simulated camera)."""
        for ts, c in reversed(self.history):
            if ts <= t:
                return c
        return self.history[0][1] if self.history else None

    def poll(self, ms: int = 1):
        if self.headless:
            time.sleep(ms / 1000)
            return
        key = cv2.waitKey(ms) & 0xFF
        if key in (27, ord("q")):
            self.quit = True
        elif key == ord("c"):
            self.recal = True
        elif key in (32, 13):  # space / enter
            self.go = True

    def wait(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.quit:
            self.poll(max(1, min(10, int((end - time.monotonic()) * 1000))))

    def close(self):
        if not self.headless:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)


def sim_homography(canvas_size, frame_size, shift=(0.0, 0.0)):
    """A plausible view: the screen fills ~80% of the frame, slightly tilted, optionally shifted."""
    W, H = canvas_size
    fw, fh = frame_size
    dst = np.float32([[0.12, 0.10], [0.90, 0.06], [0.94, 0.92], [0.08, 0.86]]) * [fw, fh] + np.float32(shift) * [fw, fh]
    return cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), dst.astype(np.float32))


class SimCamera:
    """Renders what the display showed `latency` seconds ago through a known homography, with blur and noise."""

    def __init__(self, display: Display, size=(1920, 1080), fps=30.0, latency=0.12, move_at=None):
        self.display, self.size, self.dt, self.latency = display, size, 1 / fps, latency
        self.H0 = sim_homography(display.size, size)
        self.H1 = sim_homography(display.size, size, (0.03, 0.02))
        self.move_at, self.t0, self.next = move_at, time.monotonic(), time.monotonic()
        rng = np.random.default_rng(0)
        self.noise, self.n = [rng.normal(0, 4, (size[1], size[0], 3)).astype(np.int16) for _ in range(3)], 0

    def H_at(self, t):
        return self.H1 if self.move_at is not None and t - self.t0 >= self.move_at else self.H0

    def read(self):
        self.next += self.dt
        time.sleep(max(0.0, self.next - time.monotonic()))
        t = time.monotonic()
        canvas = self.display.at(t - self.latency)
        if canvas is None:
            canvas = np.zeros((self.display.size[1], self.display.size[0], 3), np.uint8)
        f = cv2.warpPerspective(canvas, self.H_at(t), self.size, borderValue=(35, 30, 30))
        f = cv2.convertScaleAbs(cv2.GaussianBlur(f, (0, 0), 1.2), alpha=0.8, beta=12)
        self.n += 1
        return True, cv2.add(f, self.noise[self.n % len(self.noise)], dtype=cv2.CV_8U)

    def release(self):
        pass


class FileCamera:
    """A video file replayed in real time (looped), standing in for a camera."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open video {path}")
        self.dt = 1 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.next = time.monotonic()

    def read(self):
        self.next += self.dt
        time.sleep(max(0.0, self.next - time.monotonic()))
        ok, f = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return ok, f

    def release(self):
        self.cap.release()


def open_device(index: int, size: tuple[int, int] | None, ask_max: bool = True):
    """Open a camera at the requested size, or the largest it offers (asks for 4K and reads back)."""
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    if size or ask_max:
        w, h = size or (3840, 2160)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(40):  # first frames can be empty while the camera starts
        ok, f = cap.read()
        if ok and f is not None:
            return cap
        time.sleep(0.05)
    cap.release()
    return open_device(index, None, False) if size is None and ask_max else None


def list_cameras(n: int, out: Path):
    names = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True).stdout \
        if sys.platform == "darwin" else ""
    names = [l.strip().rstrip(":") for l in names.splitlines() if l.startswith("    ") and not l.startswith("      ")
             and l.strip().endswith(":")]
    if names:
        print("cameras known to macOS (index order is usually, not always, the same):", ", ".join(names))
    out.mkdir(parents=True, exist_ok=True)
    found = False
    for i in range(n):
        cap = open_device(i, None)
        if cap is None:
            continue
        ok, f = cap.read()
        cap.release()
        if ok:
            found = True
            path = out / f"camera{i}.jpg"
            cv2.imwrite(str(path), f)
            print(f"  --camera {i}: {f.shape[1]}x{f.shape[0]}, snapshot {path}")
    if not found:
        print("no camera delivered frames; on macOS give your terminal Camera permission "
              "(System Settings > Privacy & Security > Camera) and run again")


def open_camera(spec: str, size, display: Display):
    if spec == "none":
        return None
    if spec.startswith("sim"):
        opts = dict(kv.split("=") for kv in spec[4:].split(",") if "=" in kv)
        return SimCamera(display, size or (1920, 1080), move_at=float(opts["move_at"]) if "move_at" in opts else None,
                         latency=float(opts.get("latency", 0.12)))
    if spec.isdigit():
        cap = open_device(int(spec), size)
        if cap is None:
            raise SystemExit(f"camera {spec} delivered no frames: check --list-cameras, and on macOS that your terminal "
                             "has Camera permission (System Settings > Privacy & Security > Camera)")
        return cap
    if Path(spec).exists():
        return FileCamera(spec)
    raise SystemExit(f"--camera {spec}: not an index, a video file, sim or none")


class Recorder(threading.Thread):
    """Reads the camera continuously, keeps the latest frame and writes all frames to a video (own thread)."""

    def __init__(self, cam, video: Path | None):
        super().__init__(daemon=True)
        self.cam, self.cond, self.running = cam, threading.Condition(), True
        self.frame, self.t, self.idx, self.dropped, self.written, self.queued = None, 0.0, -1, 0, 0, 0
        self.video_index = {}  # recent camera frame index -> frame index in video.mp4 (frames can be dropped)
        # Warm up (auto exposure) and measure the real frame rate.
        ts = []
        for _ in range(1 if isinstance(cam, FileCamera) else 60):  # keep a video file's first frames
            ok, f = cam.read()
            if ok:
                ts.append(time.monotonic())
                self.frame = f
            if len(ts) >= 2 and ts[-1] - ts[0] > 1.0:
                break
        if self.frame is None:
            raise SystemExit("the camera delivered no frames")
        self.fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 else 1 / getattr(cam, "dt", 1 / 30)
        self.size = (self.frame.shape[1], self.frame.shape[0])
        self.t0 = time.monotonic()
        self.queue, self.writer_thread = None, None
        if video:
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), round(self.fps, 2), self.size)
            if not writer.isOpened():
                raise SystemExit(f"could not open video writer for {video}")
            self.queue = queue.Queue(maxsize=max(4, int(self.fps * 2)))
            self.writer_thread = threading.Thread(target=self._write, args=(writer,), daemon=True)
            self.writer_thread.start()

    def _write(self, writer):
        while (f := self.queue.get()) is not None:
            writer.write(f)
            self.written += 1
        writer.release()

    def run(self):
        while self.running:
            ok, f = self.cam.read()
            if not ok or f is None:
                time.sleep(0.01)
                continue
            vidx = None
            if self.queue is not None:
                try:
                    self.queue.put_nowait(f)
                    vidx, self.queued = self.queued, self.queued + 1
                except queue.Full:
                    self.dropped += 1
            with self.cond:
                self.frame, self.t, self.idx = f, time.monotonic(), self.idx + 1
                self.video_index[self.idx] = vidx
                self.video_index.pop(self.idx - 600, None)
                self.cond.notify_all()

    def next_frame(self, after: int, timeout: float = 1.0):
        """The first frame with index > after: (idx, time, frame), or None on timeout."""
        with self.cond:
            if not self.cond.wait_for(lambda: self.idx > after, timeout):
                return None
            return self.idx, self.t, self.frame

    def stop(self):
        self.running = False
        self.join(timeout=2)
        if self.queue is not None:
            self.queue.put(None)
            self.writer_thread.join(timeout=30)
        self.cam.release()


# ---------------------------------------------------------------- recording

def calibrate(display: Display, rec: Recorder, grid, out: Path, tag: str, timeout: float = 8.0):
    """Show the marker grid, detect it, fit the homography. Returns dict or None (debug frame saved)."""
    canvas, ref = calib_canvas(display.size, grid)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(ARUCO, params)
    t_show = display.show(canvas)
    display.wait(0.3)
    best, idx, seen = None, rec.idx, 0
    while time.monotonic() - t_show < timeout and not display.quit:
        got = rec.next_frame(idx, 1.0)
        display.poll(1)
        if got is None:
            continue
        idx, t, frame = got
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        n = 0 if ids is None else len(ids)
        seen += 1
        if best is None or n >= best[0]:  # the latest of the best frames: most settled
            best = (n, corners, ids, frame, idx)
        if (n == len(ref) and seen >= 3) or (seen >= 15 and best[0] >= 4):
            break
    if best is None:
        return None
    n, corners, ids, frame, idx = best
    H, n_used, err = fit_homography(ref, corners, ids)
    dbg = frame.copy()
    cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
    if H is not None:
        W, Hc = display.size
        outline = cv2.perspectiveTransform(np.float64([[[0, 0], [W, 0], [W, Hc], [0, Hc]]]), H)
        cv2.polylines(dbg, [outline.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.imwrite(str(out / f"calib_{tag}.jpg"), dbg)
    if H is None:
        return None
    if n_used < 4:
        print(f"warning: calibration {tag} used only {n_used} markers, labels may be inaccurate")
    return {"tag": tag, "H": H, "markers": n_used, "rms_px": err, "camera_frame": idx,
            "time_s": round(rec.t - rec.t0, 3)}


def aim(display: Display, rec: Recorder, grid):
    """Show the marker grid with a live camera preview in the middle until the user presses space.

    The preview shows the detected markers, so the camera can be pointed and focused before recording starts.
    """
    canvas, ref = calib_canvas(display.size, grid)
    detector = cv2.aruco.ArucoDetector(ARUCO, cv2.aruco.DetectorParameters())
    W, H = display.size
    pw = W // 4
    shown = None
    idx, display.go = rec.idx, False
    print("aim the camera at the screen until (nearly) all markers are found, then press space (q quits)")
    while not display.go and not display.quit:
        got = rec.next_frame(idx, 1.0)
        if got is None:
            display.poll(10)
            continue
        idx, _, frame = got
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        ph = round(pw * frame.shape[0] / frame.shape[1])
        if shown is None:  # markers hidden behind the preview can't be seen while aiming; count only the others
            box = ((W - pw) // 2 - 10, (H - ph) // 2 - 70, (W + pw) // 2 + 10, (H + ph) // 2 + 10)
            shown = {i for i, c in ref.items() if c[:, 0].max() < box[0] or c[:, 0].min() > box[2]
                     or c[:, 1].max() < box[1] or c[:, 1].min() > box[3]}
        n = 0 if ids is None else len(set(ids.flatten().tolist()) & shown)
        view = frame.copy()
        cv2.aruco.drawDetectedMarkers(view, corners, ids)
        view = cv2.resize(view, (pw, ph), interpolation=cv2.INTER_AREA)
        out = canvas.copy()
        x0, y0 = (W - pw) // 2, (H - ph) // 2
        out[y0 - 70:y0 + ph + 10, x0 - 10:x0 + pw + 10] = 40
        out[y0:y0 + ph, x0:x0 + pw] = view
        ok = n >= len(shown) * 0.75
        color = (80, 220, 80) if ok else (60, 160, 255)
        text = f"markers {n}/{len(shown)}  " + ("press SPACE to start" if ok else "point the camera at the whole screen")
        cv2.putText(out, text, (x0, y0 - 25), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, pw / 900), color, 2)
        display.show(out)
    display.go = False
    return not display.quit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?", type=Path, default=Path("data/test"),
                   help="data split (gt_one.csv), YOLO folder or yaml, or an image folder")
    p.add_argument("--split", default="test", help="split for data/ or YOLO folders with split subfolders")
    p.add_argument("--camera", default="0", help="device index, video file, sim[:move_at=S] or none")
    p.add_argument("--cam-size", help="camera resolution WxH (default: the largest the camera offers)")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--list-screens", action="store_true")
    p.add_argument("--screen", type=int, help="show on this screen (see --list-screens)")
    p.add_argument("--canvas", help="slide size WxH (default: the full-screen window size in pixels)")
    p.add_argument("--headless", action="store_true", help="no window (with --camera sim, for testing)")
    p.add_argument("--duration", type=float, default=1.5, help="seconds per slide")
    p.add_argument("--settle", type=float, default=0.5, help="seconds after a slide change before a still is taken")
    p.add_argument("--stills", type=int, default=2, help="still frames saved per slide")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", choices=["all", "mite", "clean"], default="all")
    p.add_argument("--limit", type=int, default=0, help="number of slides, 0 = all")
    p.add_argument("--tile", default="1x1", help="COLSxROWS crops per slide, e.g. 4x2 for a multi-bee scene")
    p.add_argument("--fill", type=float, default=1.0, help="image size as a fraction of its cell")
    p.add_argument("--bg", type=int, default=0, help="background grey level")
    p.add_argument("--class-map", default="", help="source=output class names, e.g. Varroa=varroa,apis=bee")
    p.add_argument("--calib-grid", default="4x3", help="ArUco markers COLSxROWS on the calibration screen")
    p.add_argument("--recalib-every", type=int, default=100, help="slides between calibrations, 0 = start/end only")
    p.add_argument("--max-drift", type=float, default=4.0, help="warn when calibrations differ by more (camera px)")
    p.add_argument("--min-corr", type=float, default=0.6,
                   help="edge correlation of the slide and the rectified camera view needed to keep a frame")
    p.add_argument("--max-motion", type=float, default=0.02, help="1 - corr between consecutive frames to accept")
    p.add_argument("--min-bee-px", type=float, default=40, help="warn when the median bee is smaller in the camera image")
    p.add_argument("--no-check", action="store_true", help="accept frames after --settle without the image check")
    p.add_argument("--no-aim", action="store_true", help="start calibrating at once, without the live aiming preview")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--ext", default="jpg", help="still frame format: jpg or png")
    p.add_argument("--out", type=Path, help="default captures/screen-<time>")
    args = p.parse_args()

    if args.list_screens:
        for i, (x, y, w, h, s) in enumerate(list_screens()):
            print(f"  --screen {i}: {w}x{h} points at ({x}, {y}), scale {s:g}")
        return
    if args.list_cameras:
        return list_cameras(6, Path("captures/cameras"))
    if args.out and any(args.out.resolve().is_relative_to(Path(d).resolve()) for d in ("data", "datasets")):
        raise SystemExit("--out must not be inside data/ or datasets/")

    class_map = dict(kv.split("=") for kv in args.class_map.split(",") if "=" in kv)
    items = load_source(args.src, args.split, class_map)
    if args.only != "all":
        items = [it for it in items if it.mite == (args.only == "mite")]
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    cols, rows = parse_size(args.tile)
    slides = [items[i:i + cols * rows] for i in range(0, len(items), cols * rows)]
    slides = slides[:args.limit] if args.limit else slides
    if not slides:
        raise SystemExit(f"no images in {args.src}")
    names = OUT_NAMES + sorted({b[0] for it in items for b in it.boxes} - set(OUT_NAMES))
    print(f"{len(items)} images from {args.src} ({sum(it.mite for it in items)} with mites), {len(slides)} slides, "
          f"classes {names}")

    out = args.out or Path("captures") / time.strftime("screen-%Y%m%d-%H%M%S")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images\nval: images\nnames: {names}\n")
    (out / "run.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    display = Display(args.headless, args.canvas, args.screen, args.bg)
    cam_size = parse_size(args.cam_size) if args.cam_size else None
    cam = open_camera(args.camera, cam_size, display)
    rec = None
    calibs, rows_written, skipped = [], 0, 0
    corrs, bee_px = [], []
    try:
        if cam is not None:
            display.show(np.full((display.size[1], display.size[0], 3), args.bg, np.uint8))
            rec = Recorder(cam, None if args.no_video else out / "video.mp4")
            rec.start()
            print(f"camera {args.camera}: {rec.size[0]}x{rec.size[1]} at {rec.fps:.1f} fps, slides {display.size[0]}x"
                  f"{display.size[1]}, output {out}")

        def recalibrate(tag):
            c = calibrate(display, rec, parse_size(args.calib_grid), out, tag)
            if c is None:
                return None
            msg = f"calibration {tag}: {c['markers']} markers, {c['rms_px']:.2f} px RMS"
            if calibs:
                c["drift_px"] = corner_drift(calibs[-1]["H"], c["H"], display.size)
                msg += f", {c['drift_px']:.1f} px from the previous one"
                if c["drift_px"] > args.max_drift:
                    msg += f"  WARNING: the camera moved (> {args.max_drift} px); labels since " \
                           f"calibration {calibs[-1]['tag']} may be off"
            print(msg)
            calibs.append(c)
            return c

        calib = None
        if rec is not None and not display.headless and not args.no_aim and args.camera.isdigit():
            if not aim(display, rec, parse_size(args.calib_grid)):
                raise SystemExit("stopped while aiming")
        if rec is not None:
            calib = recalibrate("start")
            if calib is None:
                raise SystemExit(f"calibration failed: markers not found, see {out}/calib_start.jpg. Make sure the "
                                 "camera sees the whole screen (or most markers), is in focus, and not overexposed.")
            view = ScreenView(calib["H"], display.size, rec.size)
            # Expected bee size in the camera image, from the first slides' boxes (images are only read for their size).
            sizes = []
            for group in slides[:30]:
                _, boxes = render(group, display.size, (cols, rows), args.fill, args.bg)
                sizes += long_sides(map_boxes(boxes, calib["H"], rec.size))
            if sizes:
                med = float(np.median(sizes))
                print(f"expected bee size in the camera image: median {med:.0f} px (long side, first {min(30, len(slides))} "
                      f"slides)" + (f"  WARNING: below {args.min_bee_px:g} px, move the camera closer, zoom in, or use a "
                                    "larger --tile cell / fewer bees per screen" if med < args.min_bee_px else ""))

        mf = open(out / "manifest.csv", "w", newline="")
        manifest = csv.writer(mf)
        manifest.writerow(["image", "label", "slide", "shot", "camera_frame", "video_frame", "time_s", "sources", "source_labels",
                           "calibration", "corr", "motion", "bees", "mites", "other"])
        prev_small, misses = None, 0
        for si, group in enumerate(slides):
            if display.quit:
                break
            canvas, boxes = render(group, display.size, (cols, rows), args.fill, args.bg)
            if rec is not None and (display.recal or misses >= 2 or
                                    (args.recalib_every and si and si % args.recalib_every == 0)):
                reason = "key c" if display.recal else "camera moved?" if misses >= 2 else "periodic"
                display.recal, misses = False, 0
                print(f"recalibrating before slide {si} ({reason})")
                c = recalibrate(f"{si:05d}")
                if c is None:
                    print(f"recalibration before slide {si} ({reason}) failed, keeping the previous one")
                else:
                    calib, view = c, ScreenView(c["H"], display.size, rec.size)
                prev_small = None
            t_show = display.show(canvas)
            t_end = t_show + args.duration
            if rec is None:
                display.wait(args.duration)
                continue

            exp = edges(view.canvas(canvas))
            check_old = prev_small is not None and ncc(exp, prev_small) < 0.98
            mapped = map_boxes(boxes, calib["H"], rec.size)
            labels = yolo_lines(mapped, names, rec.size)
            counts = [sum(b[0] == "bee" for b in mapped), sum(b[0] == "varroa" for b in mapped)]
            counts.append(len(mapped) - sum(counts))
            span = max(0.0, args.duration - args.settle - 0.1)
            targets = [t_show + args.settle + span * k / max(1, args.stills) for k in range(args.stills)]
            idx, last_small, best_corr, saved = rec.idx, None, -1.0, 0
            for k, target in enumerate(targets):
                display.wait(target - 0.1 - time.monotonic())  # from just before, for the motion check
                deadline = targets[k + 1] if k + 1 < len(targets) else t_end - 0.05
                while time.monotonic() < deadline and not display.quit:
                    got = rec.next_frame(idx, max(0.01, deadline - time.monotonic()))
                    display.poll(1)
                    if got is None:
                        continue
                    idx, t, frame = got
                    small = view.frame(frame)
                    motion = 1 - ncc(small, last_small) if last_small is not None else 1.0
                    last_small = small
                    if t < target:
                        continue
                    edge = edges(small)
                    corr = ncc(edge, exp)
                    best_corr = max(best_corr, corr)
                    if not args.no_check and (corr < args.min_corr or motion > args.max_motion or
                                              (check_old and ncc(edge, prev_small) >= corr)):
                        continue
                    name = f"{si:05d}_{k}"
                    cv2.imwrite(str(out / "images" / f"{name}.{args.ext}"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95] if args.ext == "jpg" else [])
                    (out / "labels" / f"{name}.txt").write_text("".join(l + "\n" for l in labels))
                    vidx = rec.video_index.get(idx)
                    manifest.writerow([f"images/{name}.{args.ext}", f"labels/{name}.txt", si, k, idx,
                                       "" if vidx is None else vidx,
                                       round(t - rec.t0, 3), ";".join(str(it.path) for it in group),
                                       ";".join(it.tag for it in group), calib["tag"], round(corr, 3),
                                       round(motion, 4), *counts])
                    rows_written += 1
                    saved += 1
                    corrs.append(corr)
                    bee_px += long_sides(mapped)
                    break
            display.wait(t_end - time.monotonic())
            if saved < len(targets):
                skipped += len(targets) - saved
            # A slide with no usable frame although the image matched badly suggests the camera moved.
            misses = misses + 1 if saved == 0 and best_corr < args.min_corr else 0
            prev_small = exp
            if si % 20 == 0:
                mf.flush()
                print(f"slide {si + 1}/{len(slides)}: {rows_written} frames saved, {skipped} skipped", flush=True)
        mf.close()
        if rec is not None and not display.quit:
            recalibrate("end")
    finally:
        if rec is not None:
            rec.stop()
        display.close()

    if rec is not None:
        (out / "calibration.json").write_text(json.dumps(
            [{**c, "H": np.asarray(c["H"]).round(8).tolist()} for c in calibs], indent=1))
        print(f"{rows_written} frames with labels in {out}, {skipped} shots skipped (transition/mismatch); "
              f"median slide match {np.median(corrs) if corrs else 0:.2f}")
        if bee_px:
            med = float(np.median(bee_px))
            print(f"bee size in saved frames: median {med:.0f} px, 10th percentile {np.percentile(bee_px, 10):.0f} px "
                  f"(long side, {len(bee_px)} boxes)" + (f"  WARNING: below {args.min_bee_px:g} px" if med < args.min_bee_px else ""))
        if rec.dropped:
            print(f"warning: the video writer dropped {rec.dropped} frames (encoder too slow, try --cam-size 1920x1080 "
                  "or --no-video); stills are unaffected, manifest video_frame is their frame in video.mp4")


if __name__ == "__main__":
    main()
