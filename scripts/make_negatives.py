"""Hard negatives for the bee detector: people, faces, rooms, screens/phones without bees, purple stage light.

The board's camera at the demo also sees the presenter's face and the venue (purple/blue LED light), and the bee
detector fine-tuned on datasets/bees_synth.yaml fired on a face. This builds datasets/negatives/ (YOLO, 1280x720):

  coco      COCO val2017 images with a person (largest person box >= 6% of the image, or an indoor/screen object
            next to the person), no "bird"; 16:9 crop, resized. Split by image id.
  phone     a dark rounded "phone"/"monitor" showing a COCO image or code-like text, on a dark room background
  capture   full frames of the captures/screen-* recordings with every labelled box (bee, varroa) inpainted away;
            runs split as in bees_synth.yaml (124423/135503/121719 -> train, 130127/123026 -> val, 122148 = test
            run, not used)
  calib     the calib_*.jpg frames of the captures/screen-* runs (marker grid on the screen, no bees), full + crops
  camera    captures/cameras/*.jpg (board camera test shots), random crops (train only, there are just 4)

Augmentation: ~35% get a purple/magenta cast with bright LED streaks (the venue light). ~30% of all images also get
1-3 bees pasted with scripts/synth_bees.py's paste() and box convention (train crops -> train, val crops -> val);
those are named mix_* and have labels, all others have EMPTY label files.

  .venv/bin/python scripts/make_negatives.py                 # writes datasets/negatives (+ sources.csv)
  .venv/bin/python scripts/make_negatives.py --sheet x.jpg   # also a contact sheet
"""
import argparse
import csv
import json
import math
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_bees as sb  # noqa: E402

ROOT = sb.ROOT
OUT = sb.DS / "negatives"
COCO = ROOT / "data" / "coco_dl"
W, H = sb.W, sb.H
SEED = 20261115
CAP_RUNS = {"train": ["screen-20261114-124423", "screen-20261114-135503", "screen-20261114-121719"],
            "val": ["screen-20261114-130127", "screen-20261114-123026"]}
CALIB_RUNS = {"train": CAP_RUNS["train"] + ["screen-20261114-122148"], "val": CAP_RUNS["val"]}
COUNTS = {"train": {"coco": 1250, "phone": 220, "capture": 360, "calib_variants": 4, "camera_variants": 10},
          "val": {"coco": 125, "phone": 25, "capture": 50, "calib_variants": 3, "camera_variants": 0}}
P_MIX, P_PURPLE = 0.30, 0.35
BIRD, PERSON = 16, 1
INDOOR = {62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86}  # furniture, tv, laptop...
SCREEN_CATS = {72, 73, 77}  # tv, laptop, cell phone


# ----------------------------------------------------------------------------------------------- source lists

def coco_pool():
    """[(file, image_id, license_url, reason)] of val2017 images with a person and no bird."""
    with zipfile.ZipFile(COCO / "ann.zip") as z:
        ann = json.loads(z.read("annotations/instances_val2017.json"))
    lic = {l["id"]: l["url"] for l in ann["licenses"]}
    imgs = {i["id"]: i for i in ann["images"]}
    cats, big = {}, {}
    for a in ann["annotations"]:
        i = a["image_id"]
        cats.setdefault(i, set()).add(a["category_id"])
        if a["category_id"] == PERSON:
            im = imgs[i]
            big[i] = max(big.get(i, 0), a["bbox"][2] * a["bbox"][3] / (im["width"] * im["height"]))
    pool = []
    for i, c in sorted(cats.items()):
        if PERSON not in c or BIRD in c:
            continue
        reason = "person_large" if big[i] >= 0.06 else ("person_indoor" if c & INDOOR else None)
        if reason:
            pool.append((imgs[i]["file_name"], i, lic[imgs[i]["license"]], reason, bool(c & SCREEN_CATS)))
    return pool


def capture_frames(split):
    out = []
    for run in CAP_RUNS[split]:
        out += sorted(str(p) for p in (ROOT / "captures" / run / "images").glob("*.jpg"))
    return out


# ----------------------------------------------------------------------------------------------- image helpers

def crop16x9(rng, img, min_frac=0.6, full=False):
    ih, iw = img.shape[:2]
    cw = iw if full else rng.uniform(min_frac, 1.0) * iw
    ch = cw * H / W
    if ch > ih:
        ch, cw = ih, ih * W / H
    x0, y0 = rng.uniform(0, iw - cw), rng.uniform(0, ih - ch)
    c = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
    out = cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA if c.shape[1] > W else cv2.INTER_CUBIC)
    return out[:, ::-1].copy() if rng.random() < 0.5 else out


def coco_img(name):
    with zipfile.ZipFile(COCO / "val2017.zip") as z:
        buf = np.frombuffer(z.read(f"val2017/{name}"), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def masked_capture(path):
    """Capture frame with every labelled box (expanded) inpainted from the surroundings."""
    img = cv2.imread(path)
    ih, iw = img.shape[:2]
    lab = path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    boxes = sb.read_yolo(lab, iw, ih)
    s = 0.5
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = np.zeros(small.shape[:2], np.uint8)
    for x0, y0, x1, y1 in boxes:
        mx, my = (x1 - x0) * 0.08 + 6, (y1 - y0) * 0.08 + 6
        cv2.rectangle(mask, (int((x0 - mx) * s), int((y0 - my) * s)), (int((x1 + mx) * s), int((y1 + my) * s)), 255, -1)
    if mask.any():
        # normalized convolution: a smooth local average of the unmasked surroundings (TELEA dragged the dark bezel
        # into boxes at the screen edge as sawtooth notches), coarse scales only where fine ones see nothing
        valid = (mask == 0).astype(np.float32)
        f = small.astype(np.float32)
        fill, have = np.zeros_like(f), np.zeros(valid.shape, np.float32)
        for sig in (15, 40, 120):
            num, den = cv2.GaussianBlur(f * valid[..., None], (0, 0), sig), cv2.GaussianBlur(valid, (0, 0), sig)
            ok = (den > 0.05) & (have == 0)
            fill[ok] = num[ok] / den[ok, None]
            have[ok] = 1
        fill[have == 0] = f[valid > 0].mean(0)
        fill += np.random.default_rng(len(boxes)).normal(0, 2.5, fill.shape).astype(np.float32)
        fill = cv2.resize(np.clip(fill, 0, 255).astype(np.uint8), (iw, ih), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)[..., None] > 0
        img = np.where(m, fill, img)
    return img, len(boxes)


def purple(rng, img):
    """Venue look: magenta/purple cast, darker, glow, bright LED tube streaks."""
    f = img.astype(np.float32)
    f *= rng.uniform(0.35, 0.9)                                         # darker room
    f *= np.array([rng.uniform(1.05, 1.5), rng.uniform(0.45, 0.8), rng.uniform(0.9, 1.35)], np.float32)  # BGR
    glow = sb.smooth_noise(rng, H, W, 2, int(rng.integers(1, 4)))[..., None]
    tint = np.array([rng.uniform(150, 255), rng.uniform(0, 60), rng.uniform(90, 230)], np.float32)
    f += glow * tint * rng.uniform(0.1, 0.45)
    streak = np.zeros((H, W, 3), np.float32)
    for _ in range(int(rng.integers(1, 7))):
        col = [(255, 40, 40), (255, 60, 220), (220, 80, 255), (255, 255, 255), (60, 160, 255)][int(rng.integers(5))]
        x0, y0 = rng.uniform(-100, W + 100), rng.uniform(-100, H * 0.7)
        ang, ln = rng.uniform(0, math.pi), rng.uniform(80, 700)
        p1 = (int(x0), int(y0))
        p2 = (int(x0 + ln * math.cos(ang)), int(y0 + ln * math.sin(ang)))
        cv2.line(streak, p1, p2, col, int(rng.integers(2, 10)), cv2.LINE_AA)
    f += cv2.GaussianBlur(streak, (0, 0), rng.uniform(8, 25)) * rng.uniform(0.6, 1.5) + streak * rng.uniform(0.5, 1.0)
    return np.clip(f, 0, 255).astype(np.uint8)


def text_screen(rng, w, h):
    img = np.full((h, w, 3), rng.uniform(10, 50, 3), np.uint8)
    y, lh = int(rng.integers(8, 20)), int(rng.integers(12, 26))
    sc = lh / 30
    chars = "abcdefghijklmnopqrstuvwxyz_(){}[]=+-.:;0123456789     "
    while y < h:
        x = int(rng.integers(4, 60))
        s = "".join(chars[int(rng.integers(len(chars)))] for _ in range(int(rng.integers(5, 70))))
        col = tuple(int(v) for v in rng.uniform(120, 255, 3))
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
        y += lh
    return img


def device(rng, content_fn, bg):
    """A phone (portrait/landscape) or monitor with rounded dark bezel showing content_fn(w, h), on bg."""
    kind = rng.choice(["phone", "phone_land", "monitor"], p=[0.45, 0.25, 0.30])
    if kind == "phone":
        sh = rng.uniform(300, 700)
        sw = sh * rng.uniform(0.46, 0.56)
    elif kind == "phone_land":
        sw = rng.uniform(350, 900)
        sh = sw * rng.uniform(0.46, 0.56)
    else:
        sw = rng.uniform(500, 1150)
        sh = sw * rng.uniform(0.56, 0.66)
    bez = max(6, int(min(sw, sh) * rng.uniform(0.03, 0.08)))
    sw, sh = int(sw), int(sh)
    bw, bh = sw + 2 * bez, sh + 2 * bez
    body = np.full((bh, bw, 3), rng.uniform(5, 45), np.uint8)
    body[bez:bez + sh, bez:bez + sw] = content_fn(sw, sh)
    r = int(min(bw, bh) * (rng.uniform(0.06, 0.14) if kind != "monitor" else rng.uniform(0.0, 0.03)))
    mask = np.zeros((bh, bw), np.uint8)
    cv2.rectangle(mask, (r, 0), (bw - r, bh), 255, -1)
    cv2.rectangle(mask, (0, r), (bw, bh - r), 255, -1)
    for cx, cy in [(r, r), (bw - r - 1, r), (r, bh - r - 1), (bw - r - 1, bh - r - 1)]:
        cv2.circle(mask, (cx, cy), r, 255, -1)
    # rotate + perspective into the canvas
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    scale = min(1.0, 0.95 * W / bw, 0.95 * H / bh)
    ang = math.radians(rng.normal(0, 12))
    cx, cy = rng.uniform(bw * scale / 2, W - bw * scale / 2), rng.uniform(bh * scale / 2, H - bh * scale / 2)
    c, s = math.cos(ang), math.sin(ang)
    pts = (src - [bw / 2, bh / 2]) * scale
    dst = np.float32([[cx + c * x - s * y, cy + s * x + c * y] for x, y in pts])
    dst += rng.normal(0, 0.03, (4, 2)).astype(np.float32) * [bw * scale, bh * scale]
    P = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(body, P, (W, H), flags=cv2.INTER_AREA)
    m = cv2.warpPerspective(mask, P, (W, H)).astype(np.float32)[..., None] / 255
    return (bg * (1 - m) + warped * m).astype(np.uint8)


def fit(img, w, h):
    ih, iw = img.shape[:2]
    r = max(w / iw, h / ih)
    t = cv2.resize(img, (max(w, round(iw * r)), max(h, round(ih * r))), interpolation=cv2.INTER_AREA)
    y0, x0 = (t.shape[0] - h) // 2, (t.shape[1] - w) // 2
    return t[y0:y0 + h, x0:x0 + w]


# ----------------------------------------------------------------------------------------------- bees

def add_bees(rng, src, canvas):
    """1-3 bees, same pasting and ellipse-bbox convention as synth_bees.scatter(). Returns YOLO lines."""
    polys, boxes = [], []
    use_cam = bool(src["cam"])
    for size in sorted((sb.sample_size(rng) for _ in range(int(rng.integers(1, 4)))), reverse=True):
        cam = use_cam and rng.random() < 0.30
        pool = src["cam"] if cam else src["crops"]
        crop = sb.load_crop(pool[int(rng.integers(len(pool)))], cam)
        if crop is None:
            continue
        ang, flip = float(rng.uniform(0, 360)), bool(rng.random() < 0.5)
        a, b = sb.ELL_FX * size / sb.ELL_FY * sb.CROP_W / sb.CROP_H / 2, size / 2
        for _ in range(30):
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            poly = sb.ellipse_poly(cx, cy, a, b, math.radians(ang))
            box = sb.poly_box(poly)
            if box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H:
                continue
            iou, cover = sb.iou_and_cover(box, boxes)
            if len(boxes) and (iou.max() > 0.25 or cover.max() > 0.40):
                continue
            sb.paste(canvas, crop, cx, cy, size, ang, flip, cam, rng)
            polys.append(poly)
            boxes.append(box)
            break
    lines = []
    for x0, y0, x1, y1 in boxes:
        lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return lines


# ----------------------------------------------------------------------------------------------- jobs

def build(job):
    """job = (split, idx, kind, payload) -> (name, row) after writing image + label."""
    split, idx, kind, payload = job
    rng = np.random.default_rng([SEED, sb.SPLIT_ID[split], idx])
    origin, note, photo_p, purple_p = "", "", 0.5, P_PURPLE
    if kind == "coco":
        name_, iid, lic, reason, _ = payload
        img = crop16x9(rng, coco_img(name_), 0.7)
        origin, note = f"coco:val2017/{name_} id={iid}", f"{reason} license={lic}"
        photo_p, purple_p = 0.5, 0.40
    elif kind == "phone":
        content, bgsrc, cap = payload
        if content[0] == "text" :
            fn = lambda w, h: text_screen(rng, w, h)  # noqa: E731
            origin = "text"
        else:
            ci = coco_img(content[1][0])
            fn = lambda w, h: fit(ci, w, h)  # noqa: E731
            origin = f"coco:val2017/{content[1][0]} id={content[1][1]}"
            note = f"license={content[1][2]}"
        if bgsrc[0] == "coco":
            bg = crop16x9(rng, coco_img(bgsrc[1][0]), 0.6).astype(np.float32) * rng.uniform(0.15, 0.5)
            origin += f" on coco:{bgsrc[1][0]}"
        elif bgsrc[0] == "capture":
            bg = crop16x9(rng, masked_capture(bgsrc[1])[0], 0.5).astype(np.float32) * rng.uniform(0.3, 0.8)
            origin += f" on {Path(bgsrc[1]).relative_to(ROOT)}"
        else:
            bg = sb.bg_texture(rng).astype(np.float32) * rng.uniform(0.1, 0.4)
            origin += " on texture"
        img = device(rng, fn, bg)
        photo_p, purple_p = 0.8, 0.35
    elif kind == "capture":
        full, nbox = masked_capture(payload)
        img = crop16x9(rng, full, 0.5, full=rng.random() < 0.6)
        origin, note = str(Path(payload).relative_to(ROOT)), f"{nbox} boxes inpainted"
        photo_p, purple_p = 0.3, 0.20
    elif kind in ("calib", "camera"):
        path, v = payload
        img = crop16x9(rng, cv2.imread(path), 0.4 if kind == "camera" else 0.5, full=(v == 0))
        origin = str(Path(path).relative_to(ROOT)) + f" variant {v}"
        photo_p, purple_p = 0.4, 0.25
    img = np.ascontiguousarray(img)
    lines = []
    if rng.random() < P_MIX:
        lines = add_bees(rng, sb.sources(split), img)
    aug = []
    if rng.random() < purple_p:
        img = purple(rng, img)
        aug.append("purple")
    if rng.random() < photo_p:
        img = sb.photometric(rng, img)
        aug.append("photometric")
    name = f"{'mix' if lines else 'neg'}_{kind}_{split}_{idx:05d}"
    cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"file": f"images/{split}/{name}.jpg", "split": split, "source": kind, "origin": origin, "note": note,
            "augment": "+".join(aug), "bees": len(lines)}


def jobs():
    pool = coco_pool()
    is_val = lambda iid: (iid * 2654435761) % 1000 < 100  # noqa: E731  ~10% of image ids
    coco = {"train": [p for p in pool if not is_val(p[1])], "val": [p for p in pool if is_val(p[1])]}
    print("coco pool:", {s: len(v) for s, v in coco.items()}, flush=True)
    out = []
    for split, cnt in COUNTS.items():
        rng = np.random.default_rng([SEED, 99, sb.SPLIT_ID[split]])
        c = coco[split]
        order = rng.permutation(len(c))
        n_coco = min(cnt["coco"], len(c))
        for k in order[:n_coco]:
            out.append((split, "coco", c[k]))
        caps = capture_frames(split)
        for k in rng.choice(len(caps), min(cnt["capture"], len(caps)), replace=False):
            out.append((split, "capture", caps[k]))
        rest = [c[k] for k in order[n_coco:]] or c  # phone content: unused COCO images if there are any
        screens = [p for p in c if p[4]]
        for _ in range(cnt["phone"]):
            r = rng.random()
            if r < 0.3:
                content = ("text", None)
            else:
                src = screens if r < 0.45 and screens else rest  # sometimes a screen showing a screen
                content = ("coco", src[int(rng.integers(len(src)))])
            b = rng.random()
            bgsrc = ("coco", c[int(rng.integers(len(c)))]) if b < 0.45 else \
                ("capture", caps[int(rng.integers(len(caps)))]) if b < 0.8 else ("texture", None)
            out.append((split, "phone", (content, bgsrc, None)))
        for run in CALIB_RUNS[split]:
            for p in sorted((ROOT / "captures" / run).glob("calib_*.jpg")):
                for v in range(cnt["calib_variants"]):
                    out.append((split, "calib", (str(p), v)))
        if cnt["camera_variants"]:
            for p in sorted((ROOT / "captures" / "cameras").glob("*.jpg")):
                for v in range(cnt["camera_variants"]):
                    out.append((split, "camera", (str(p), v)))
    return [(s, i, k, p) for i, (s, k, p) in enumerate(out)]


def sheet(rows, path, n=24):
    rng = np.random.default_rng(1)
    pick = []
    for src in ["coco", "phone", "capture", "calib", "camera"]:
        r = [x for x in rows if x["source"] == src]
        pick += [r[i] for i in rng.choice(len(r), min(len(r), 5 if src in ("coco", "phone", "capture") else 3), replace=False)]
    mix = [x for x in rows if x["bees"]]
    pick += [mix[i] for i in rng.choice(len(mix), min(len(mix), n - len(pick)), replace=False)]
    tiles = []
    for x in pick[:n]:
        img = cv2.imread(str(OUT / x["file"]))
        lab = OUT / x["file"].replace("images/", "labels/").rsplit(".", 1)[0]
        for line in Path(str(lab) + ".txt").read_text().splitlines():
            _, cx, cy, bw, bh = (float(v) for v in line.split())
            cv2.rectangle(img, (round((cx - bw / 2) * W), round((cy - bh / 2) * H)),
                          (round((cx + bw / 2) * W), round((cy + bh / 2) * H)), (0, 255, 0), 3)
        t = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{x['source']} {x['augment']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(t, f"{x['source']} {x['augment']}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--procs", type=int, default=6)
    p.add_argument("--sheet", default="")
    args = p.parse_args()
    cv2.setNumThreads(1)
    if OUT.exists() and any(OUT.rglob("*.jpg")):
        sys.exit(f"{OUT} already has images, delete it first")
    for s in COUNTS:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)
    js = jobs()
    rows = []
    with Pool(args.procs) as pool:
        for k, row in enumerate(pool.imap_unordered(build, js, chunksize=8)):
            rows.append(row)
            if (k + 1) % 250 == 0:
                print(f"{k + 1}/{len(js)}", flush=True)
    rows.sort(key=lambda r: r["file"])
    with open(OUT / "sources.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    c = Counter((r["split"], r["source"], "mix" if r["bees"] else "neg") for r in rows)
    for k in sorted(c):
        print(k, c[k])
    if args.sheet:
        sheet(rows, args.sheet)


if __name__ == "__main__":
    main()
