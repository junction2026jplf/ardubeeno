"""Multi-bee mode: find every bee in the frame, follow it across frames and check each bee for mites.

Per frame:
1. The bee detector (bees.onnx, one class) runs on the whole frame every `det_every` processed frames. On the UNO Q
   it takes ~0.5 s, so ~13 camera frames pass between two runs; in between, each box moves on by its velocity in
   pixels per camera frame (frame sequence numbers, not processed frames).
2. A tracker gives each bee an ID (Tracker). Before matching, every track is moved to the detection's frame; a
   detection continues the track whose predicted box it overlaps (IoU) or whose predicted centre is close (distance
   in box sizes), the gate growing with every detector run the track was missed. A bee counts as observed after
   `confirm` frames' worth of detections. A track that is not matched is kept for `lost_s` seconds and `lost_runs`
   detector runs (whichever is longer), so a bee the detector missed once keeps its ID; a track predicted to have
   left the frame ends at once.
3. Each detected bee that is large enough and not cut off at the frame edge is cut out of the full-resolution
   frame in the training crops' 4:7 portrait shape (turned upright if it lies sideways) and checked with the
   existing mite pipeline (classifier-first cascade by default), every `check_every` frames up to `max_checks`
   times. At most `max_checks_frame` crops wait for or are in the check; bees with fewer than two checks go first
   and take the place of a waiting crop of a bee checked more often, so a fast bee gets its first checks in time.
   The checks run in a worker thread next to the bee detector (or inline with background=False).
4. A bee is MITE once `min_hits` of its checks were positive, or after a single strong positive check (fused score
   at or above `strong`; default the cascade's cls_hi, else fuse_conf + 0.25).

Counts: bees in view now, bees observed since the last reset, bees checked, bees with mites. A bee that leaves
the picture and comes back is counted again.
"""
import json
import math
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace

import cv2

from device_check import ROI_ASPECT, Detector, Pipeline
from orient import crop_bee_oriented

RED, GREEN, GREY = (40, 40, 230), (60, 180, 60), (160, 160, 160)
CROP_W, CROP_H = 160, 280  # the training crops' size
CROP_MARGIN = 0.15  # context around the bee box in the crop, for the current bee detector's box tightness


def bee_settings(bee_model):
    """Settings that belong to the bee detector, from a JSON file next to it (bees.onnx -> bees.json), e.g.
    {"crop_margin": 0.21} for a detector with tighter boxes. Unknown keys are ignored; {} without the file."""
    path = Path(bee_model).with_suffix(".json")
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {k: float(d[k]) for k in ("crop_margin",) if d.get(k) is not None}
    print(f"{path.name}: {out}", flush=True)
    return out


def iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def clip(box, size):
    """The part of an (x, y, w, h, ...) box inside a (w, h) frame; zero width or height when it is outside."""
    if size is None:
        return box
    W, H = size
    x0, y0 = min(max(box[0], 0), W), min(max(box[1], 0), H)
    x1, y1 = max(min(box[0] + box[2], W), 0), max(min(box[1] + box[3], H), 0)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0), *box[4:])


def ref_point(a, b, size, edge=2):
    """A point to follow from box a to box b: the centre, but on an axis where either box is cut off by the frame
    edge, the box side away from that edge (a bee coming in or going out grows or shrinks from the edge)."""
    out = []
    for i, n in ((0, 0), (1, 1)):
        lim = size[n] if size else None
        lo = any(box[i] <= edge for box in (a, b))
        hi = lim is not None and any(box[i] + box[i + 2] >= lim - edge for box in (a, b))
        if lo and not hi:
            out.append((a[i] + a[i + 2], b[i] + b[i + 2]))
        elif hi and not lo:
            out.append((a[i], b[i]))
        else:
            out.append((a[i] + a[i + 2] / 2, b[i] + b[i + 2] / 2))
    return out


class Track:
    def __init__(self, tid, box, t, now):
        self.id, self.box = tid, box
        self.hits, self.missed, self.confirmed = 1, 0, False  # missed: detector runs since the last match
        self.checks, self.positives, self.strong, self.since_check = 0, 0, False, 10 ** 6
        self.best_score, self.best_jpeg, self.listed = -1.0, None, False  # most mite-like check; shown on the page
        self.anchor, self.t0, self.seen = box, t, now  # last detected box, its frame number and time (s)
        self.v = (0.0, 0.0)  # pixels per camera frame (exponential moving average over the detections)
        self.pending = False  # a crop of this bee waits for or is in the mite check

    @property
    def size(self):
        return max(math.sqrt(self.anchor[2] * self.anchor[3]), 1.0)


class Tracker:
    """Greedy matching of detections to tracks predicted to the detection's frame.

    Times are camera frame numbers (velocity in pixels per frame) and seconds (how long a lost track is kept), so
    an irregular detector cadence does not matter. Each (track, detection) pair costs the centre distance between
    the predicted box and the detection in track box sizes (sqrt(w*h)) plus 1 - IoU; it is a candidate if the
    sizes differ by at most `size_ratio` and IoU >= min_iou or the distance is at most max_dist, times
    (1 + detector runs missed). A track without a velocity yet (one detection) may also be as far as a bee moving
    max_speed box sizes per frame gets in the frames since.
    Lowest cost first; a confirmed, currently seen track wins over a new or lost one at equal distance.
    A track is confirmed after `confirm` detections. An unmatched track is dropped once it has missed lost_runs
    detector runs and was last seen lost_s seconds ago (an unconfirmed one after a single run), or as soon as its
    predicted box has left the frame.
    """

    def __init__(self, confirm=2, min_iou=0.1, max_dist=1.5, max_speed=0.3, size_ratio=2.0, lost_s=1.5, lost_runs=3,
                 motion=True, smooth=0.5):
        self.confirm, self.min_iou, self.max_dist, self.max_speed = confirm, min_iou, max_dist, max_speed
        self.size_ratio, self.lost_s, self.lost_runs = size_ratio, lost_s, lost_runs
        self.motion, self.smooth = motion, smooth
        self.tracks, self.next_id, self.observed, self.t = [], 1, 0, 0

    def predict(self, t):
        """Move every track to camera frame t."""
        self.t = t
        for tr in self.tracks:
            if self.motion:
                dt = t - tr.t0
                x, y, w, h = tr.anchor[:4]
                tr.box = (x + tr.v[0] * dt, y + tr.v[1] * dt, w, h, *tr.anchor[4:])

    def cost(self, tr, b, size):
        """The cost of continuing track tr with detection b, or None if b is outside its gate."""
        p = clip(tr.box, size)
        if p[2] <= 0 or p[3] <= 0:
            return None
        ratio = math.sqrt(max(b[2] * b[3], 1.0) / max(p[2] * p[3], 1.0))
        if not 1 / self.size_ratio <= ratio <= self.size_ratio:
            return None
        o = iou(p, b)
        d = math.hypot(p[0] + p[2] / 2 - b[0] - b[2] / 2, p[1] + p[3] / 2 - b[1] - b[3] / 2) / tr.size
        gate = self.max_dist * (1 + tr.missed)
        if tr.hits == 1 and self.motion:  # no velocity yet: the bee may have moved anywhere a bee gets in the time
            gate = max(gate, self.max_speed * (self.t - tr.t0))
        if o < self.min_iou and d > gate:
            return None
        # prefer the established track: new tracks and lost ones pay a little
        return d + (1 - o) + 0.25 * (not tr.confirmed) + 0.25 * tr.missed

    def gone(self, tr, size):
        """Whether the track's predicted box has (mostly) left the frame."""
        if size is None or not self.motion:
            return False
        x, y, w, h = tr.box[:4]
        cx, cy = x + w / 2, y + h / 2
        return not (0 <= cx < size[0] and 0 <= cy < size[1])

    def update(self, boxes, t=None, now=None, size=None):
        """Match the detections of camera frame t (default: the next frame); size = (w, h) of the frame.
        Return (confirmed tracks seen in this frame, tracks dropped in this frame)."""
        t = self.t + 1 if t is None else t
        now = time.monotonic() if now is None else now
        self.predict(t)
        pairs = sorted((c, ti, bi) for ti, tr in enumerate(self.tracks) for bi, b in enumerate(boxes)
                       if (c := self.cost(tr, b, size)) is not None)
        used_t, used_b = set(), set()
        for _, ti, bi in pairs:
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            tr, b = self.tracks[ti], boxes[bi]
            dt = max(t - tr.t0, 1)
            (a0, b0), (a1, b1) = ref_point(tr.anchor, b, size)
            v = ((b0 - a0) / dt, (b1 - a1) / dt)
            k = 1.0 if tr.hits == 1 else self.smooth
            tr.v = (k * v[0] + (1 - k) * tr.v[0], k * v[1] + (1 - k) * tr.v[1])
            tr.box, tr.anchor, tr.t0, tr.seen, tr.hits, tr.missed = b, b, t, now, tr.hits + 1, 0
        for ti, tr in enumerate(self.tracks):
            if ti not in used_t:
                tr.missed += 1
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                self.tracks.append(Track(self.next_id, b, t, now))
                self.next_id += 1
        for tr in self.tracks:
            if not tr.confirmed and tr.hits >= self.confirm:
                tr.confirmed = True
                self.observed += 1

        def drop(tr):
            if tr.missed == 0:
                return False
            if self.gone(tr, size):
                return True
            if not tr.confirmed:  # a ghost without a velocity would pick up the next bee passing
                return True
            return tr.missed >= self.lost_runs and now - tr.seen >= self.lost_s

        dropped = [tr for tr in self.tracks if drop(tr)]
        self.tracks = [tr for tr in self.tracks if tr not in dropped]
        return [tr for tr in self.tracks if tr.confirmed and tr.missed == 0], dropped


def crop_bee(frame, box, margin=CROP_MARGIN):
    """Cut the bee out in the training crops' 4:7 portrait shape, turned upright if it lies sideways."""
    x, y, w, h = box[:4]
    rotate = w > h
    long, short = max(w, h), min(w, h)
    if short / long > ROI_ASPECT:
        long = short / ROI_ASPECT
    else:
        short = long * ROI_ASPECT
    long, short = long * (1 + margin), short * (1 + margin)
    cw, ch = (long, short) if rotate else (short, long)
    cx, cy = x + w / 2, y + h / 2
    x0, y0, x1, y1 = round(cx - cw / 2), round(cy - ch / 2), round(cx + cw / 2), round(cy + ch / 2)
    H, W = frame.shape[:2]
    pad = max(0, -x0, -y0, x1 - W, y1 - H)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    crop = frame[y0:y1, x0:x1]
    if rotate:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)


class BeeCounter:
    """Bee detection, tracking, counting and the per-bee mite check on full frames."""

    def __init__(self, bee_model, mite_model, cls_model=None, bee_conf=0.35, conf=0.25, iou=0.45, fuse_conf=0.38,
                 threads=2, min_bee_px=100, check_every=3, max_checks=8, min_hits=2, confirm=3, gallery=24,
                 bee_threads=2, det_every=3, max_checks_frame=2, background=True, motion=True, cascade=True,
                 cls_lo=0.31, cls_hi=0.6, strong=None, crop_margin=None, match_iou=0.1, match_dist=1.5,
                 lost_s=1.5, lost_runs=3, det_min_ms=0, oriented=False, cut=0.0):
        """mite_model: a device_check.Pipeline to share, or the mite detector file (then cls_model, conf, ...).
        The bee detector runs every `det_every` frames; at most `max_checks_frame` bee crops wait for or are in the
        mite check at a time (the bees with the fewest checks first). background=True runs the mite checks in a
        worker thread next to the bee detector, otherwise they run on the frames between detections.
        strong: a single check with a fused score at or above it makes the bee MITE (None: the cascade's cls_hi,
        else fuse_conf + 0.25). crop_margin: None = bees.json next to the bee model, else CROP_MARGIN.
        match_iou, match_dist, lost_s, lost_runs: see Tracker. det_min_ms (testing): make every bee detector run
        take at least this long, e.g. 550 to behave like the UNO Q on a laptop. oriented: turn diagonal bees upright
        for the mite check (orient.crop_bee_oriented), else crop_bee's 90-degree steps only. cut: a picture change
        between two bee detector runs above this (mean grey level difference of a 64x36 thumbnail) is a scene cut,
        e.g. a new slide with other bees at the same spots: all tracks end there (0 = off). Same scene ~0.4-5,
        new slide of the tile slideshow >= 9."""
        self.bees = Detector(str(bee_model), bee_conf, 0.5, bee_threads, spin=False)
        self.mites = mite_model if isinstance(mite_model, Pipeline) else Pipeline(SimpleNamespace(
            model=str(mite_model), cls_model=[str(cls_model)] if cls_model else None, conf=conf, iou=iou,
            fuse_conf=fuse_conf, threads=threads, cascade=cascade, cls_lo=cls_lo, cls_hi=cls_hi, spin=False))
        self.min_bee_px, self.check_every, self.max_checks, self.min_hits = min_bee_px, check_every, max_checks, min_hits
        self.confirm, self.gallery_size, self.motion, self.strong = confirm, gallery, motion, strong
        self.crop_margin = crop_margin if crop_margin is not None else \
            bee_settings(bee_model).get("crop_margin", CROP_MARGIN)
        self.match_iou, self.match_dist, self.lost_s, self.lost_runs = match_iou, match_dist, lost_s, lost_runs
        self.crop = crop_bee_oriented if oriented else crop_bee
        self.cut, self.thumb, self.cuts = cut, None, 0
        self.det_every, self.max_checks_frame, self.det_min_ms = max(1, det_every), max_checks_frame, det_min_ms
        self.last_ms = (0.0, 0.0)  # (bee detector, last mite check)
        self.ms = {k: deque(maxlen=200) for k in ("bees", "check", "cls", "det")}  # recent timings per stage
        self.n_checks, self.n_skipped = 0, 0  # mite checks, and those where the cascade skipped the detector
        self.lock, self.done, self.frame_no, self.seq = threading.Lock(), [], 0, 0
        self.jobs_cv, self.waiting, self.running, self.stopping = threading.Condition(), [], 0, False
        self.worker = threading.Thread(target=self.work, daemon=True) if background else None
        if self.worker:
            self.worker.start()
        self.reset()

    def reset(self):
        old = getattr(self, "tracker", None)
        # confirm is in frames; the tracker counts detections
        self.tracker = Tracker(confirm=max(2, round(self.confirm / self.det_every)), min_iou=self.match_iou,
                               max_dist=self.match_dist, lost_s=self.lost_s, lost_runs=self.lost_runs,
                               motion=self.motion)
        if old:  # track IDs stay unique for the whole run: each one is at most one bee passage
            self.tracker.next_id, self.tracker.t = old.next_id, old.t
        self.finished = []  # confirmed tracks that ended, until taken by end_tracks()
        self.leaving = []  # confirmed tracks that ended while a crop of them was in the mite check
        self.gone_checked, self.gone_mites = 0, 0
        self.gallery = OrderedDict()  # track id -> track, most recently checked last

    @property
    def strong_score(self):
        """A single check with a fused score at or above this makes the bee MITE."""
        if self.strong is not None:
            return self.strong
        return self.mites.cls_hi if self.mites.cascade else self.mites.fuse_conf + 0.25

    def is_mite(self, t):
        return t.strong or t.positives >= self.min_hits

    def status(self, t):
        return "mite" if self.is_mite(t) else "clean" if t.checks else "unchecked"

    def assessment(self, t, min_clean_checks=2):
        """The bee passage's verdict: mites_detected at min_hits positive checks or one strong positive check,
        not_detected only with at least min_clean_checks checks and no positive one, otherwise (too few checks, or
        some positives) uncertain."""
        if self.is_mite(t):
            return "mites_detected"
        return "not_detected" if t.checks >= min_clean_checks and t.positives == 0 else "uncertain"

    def end_tracks(self, all=False):
        """Take the confirmed tracks that ended since the last call, each once; all=True also ends those still
        tracked or waiting for a mite check (before a reset or at shutdown)."""
        ended, self.finished = self.finished, []
        if all:
            ended += self.leaving + [t for t in self.tracker.tracks if t.confirmed]
            self.leaving = []
        return [(t.id, self.assessment(t), {"checks": t.checks, "positives": t.positives, "strong": t.strong,
                                            "hits": t.hits, "score": round(t.best_score, 3),
                                            "min_hits": self.min_hits}) for t in ended]

    def scene_cut(self, frame):
        """End every track when the whole picture changed since the last detection (a new slide): without this, the
        bees of the next slide at the same spots would continue the old tracks and inherit their checks."""
        if not self.cut:
            return
        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
        prev, self.thumb = self.thumb, thumb
        if prev is None or cv2.absdiff(prev, thumb).mean() <= self.cut:
            return
        self.cuts += 1
        for t in self.tracker.tracks:
            if t.confirmed and t.pending:  # wait for its check before it becomes a passage
                self.leaving.append(t)
            elif t.confirmed:
                self.finish(t)
        self.tracker.tracks = []

    def finish(self, t):
        """A confirmed track ended: its verdict stays in the totals and it becomes a passage."""
        self.gone_checked += t.checks > 0
        self.gone_mites += self.is_mite(t)
        self.finished.append(t)

    def checkable(self, box, w, h, edge=2):
        x, y, bw, bh = box[:4]
        return max(bw, bh) >= self.min_bee_px and x > edge and y > edge and x + bw < w - edge and y + bh < h - edge

    def run_check(self, crop):
        """The mite pipeline on one bee crop (worker thread or inline); returns its result."""
        t0 = time.perf_counter()
        result = self.mites(crop)
        ms = (time.perf_counter() - t0) * 1000
        det_ms, cls_ms = self.mites.last_ms
        self.ms["check"].append(ms)
        self.ms["cls"].append(cls_ms)
        if not self.mites.skipped:
            self.ms["det"].append(det_ms)
        self.n_checks += 1
        self.n_skipped += self.mites.skipped
        self.last_ms = (self.last_ms[0], ms)
        return result

    def next_job(self):
        """The waiting crop of the bee with the fewest checks (oldest first among equals), or None."""
        with self.jobs_cv:
            if not self.waiting:
                return None
            job = min(self.waiting, key=lambda j: j[0].checks)
            self.waiting.remove(job)
            self.running += 1
            return job

    def work(self):
        while True:
            with self.jobs_cv:
                self.jobs_cv.wait_for(lambda: self.waiting or self.stopping)
                if self.stopping:
                    return
            job = self.next_job()
            if job is None:
                continue
            t, crop = job
            result = self.run_check(crop)
            with self.lock:
                self.done.append((t, crop, result))
            with self.jobs_cv:
                self.running -= 1

    def stop(self):
        """End the worker thread after the current check (benchmarks build many counters)."""
        if self.worker:
            with self.jobs_cv:
                self.stopping = True
                self.jobs_cv.notify_all()
            self.worker.join()

    def queue_checks(self, frame, seen, w, h):
        """Cut out the detected bees due for a mite check, fewest checks first, while fewer than max_checks_frame
        wait or are checked. A bee with fewer than two checks takes the place of a waiting crop of a bee checked
        more often, so new (fast) bees are not starved by the ones already in view."""
        due = sorted((t for t in seen if not t.pending and not self.is_mite(t) and t.checks < self.max_checks
                      and t.since_check >= self.check_every and self.checkable(t.box, w, h)),
                     key=lambda t: (t.checks, not t.confirmed, -t.since_check))
        with self.jobs_cv:
            for t in due:
                if len(self.waiting) + self.running >= self.max_checks_frame:
                    if t.checks >= 2 or not self.waiting:
                        break
                    worst = max(self.waiting, key=lambda j: j[0].checks)
                    if worst[0].checks <= t.checks:
                        break
                    self.waiting.remove(worst)
                    worst[0].pending, worst[0].since_check = False, self.check_every
                    if worst[0] in self.leaving:  # it left the frame: a passage without this check
                        self.leaving.remove(worst[0])
                        self.finish(worst[0])
                t.pending, t.since_check = True, 0
                self.waiting.append((t, self.crop(frame, t.box, self.crop_margin)))
            self.jobs_cv.notify_all()

    def check(self, t, crop, result):
        """Count a finished mite check of the bee crop; keep the most mite-like crop for the page."""
        boxes, top, prob, fused, verdict = result
        t.pending = False
        leaving = t in self.leaving
        if not leaving and t not in self.tracker.tracks:  # the counts were reset (or a new track was dropped)
            return
        t.checks += 1
        t.positives += bool(verdict)
        t.strong = t.strong or bool(verdict and fused >= self.strong_score)
        if fused > t.best_score:
            out = crop.copy()
            for x, y, bw, bh, s in boxes:
                cv2.rectangle(out, (int(x), int(y)), (int(x + bw), int(y + bh)), RED, 2)
                cv2.putText(out, f"{s:.2f}", (int(x), max(int(y) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1)
            t.best_score, t.best_jpeg = fused, cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        if t.confirmed:
            self.list(t)
        if leaving:  # the bee left during its check: now it is a passage
            self.leaving.remove(t)
            self.finish(t)

    def list(self, t):
        """Show the bee's crop on the page (newest last)."""
        t.listed = True
        self.gallery[t.id] = t
        self.gallery.move_to_end(t.id)
        while len(self.gallery) > self.gallery_size:
            self.gallery.popitem(last=False)

    def process(self, frame, seq=None, now=None):
        """Detect, track and check bees; return the confirmed tracks visible in this frame.
        seq: the frame's camera sequence number (velocities are per camera frame; default: one per call)."""
        h, w = frame.shape[:2]
        self.seq = self.seq + 1 if seq is None else seq
        with self.lock:
            done, self.done = self.done, []
        for t, crop, result in done:
            self.check(t, crop, result)
        detect = self.frame_no % self.det_every == 0
        self.frame_no += 1
        if detect:
            t0 = time.perf_counter()
            boxes = self.bees(frame)
            ms = (time.perf_counter() - t0) * 1000
            if ms < self.det_min_ms:  # testing: as slow as on the board
                time.sleep((self.det_min_ms - ms) / 1000)
            self.ms["bees"].append(ms)
            self.last_ms = (ms, self.last_ms[1])
            self.scene_cut(frame)
            visible, dropped = self.tracker.update(boxes, self.seq, now, (w, h))
            for t in dropped:
                if t.confirmed and t.pending:  # wait for its check before it becomes a passage
                    self.leaving.append(t)
                elif t.confirmed:
                    self.finish(t)
            for t in visible:  # checked before it was confirmed
                if t.checks and not t.listed:
                    self.list(t)
        else:  # the boxes move on by their velocity
            self.tracker.predict(self.seq)
            visible = [t for t in self.tracker.tracks if t.confirmed and t.missed == 0]
        for t in self.tracker.tracks:
            t.since_check += 1
        if detect:  # crops come from detected boxes only; new bees too, so that fast ones get checked in time
            self.queue_checks(frame, [t for t in self.tracker.tracks if t.missed == 0], w, h)
        if self.worker is None and (not detect or self.det_every == 1):
            for _ in range(self.max_checks_frame):  # one heavy step per frame: detection or mite checks
                job = self.next_job()
                if job is None:
                    break
                self.check(*job, self.run_check(job[1]))
                with self.jobs_cv:
                    self.running -= 1
        return visible

    def stats(self):
        """Median ms per stage over the recent runs, and the share of mite checks without the mite detector."""
        med = lambda d: round(sorted(d)[len(d) // 2], 1) if d else None
        return {**{f"{k}_ms": med(list(v)) for k, v in self.ms.items()}, "checks": self.n_checks,
                "det_skipped": round(self.n_skipped / self.n_checks, 3) if self.n_checks else None}

    def overlay(self, visible):
        """What the stream draws for each visible bee: (box, velocity in px per camera frame, ID, color)."""
        colors = {"mite": RED, "clean": GREEN, "unchecked": GREY}
        return [(tuple(t.box[:4]), t.v if self.motion else (0.0, 0.0), t.id, colors[self.status(t)]) for t in visible]

    def counts(self, visible):
        tracks = [t for t in self.tracker.tracks if t.confirmed] + self.leaving
        return {"bees_now": len(visible), "bees_seen": self.tracker.observed,
                "bees_checked": self.gone_checked + sum(t.checks > 0 for t in tracks),
                "count": self.gone_mites + sum(self.is_mite(t) for t in tracks)}

    def settings(self):
        """The tracking and verdict settings, for the page and the passage revision."""
        return {"strong": round(self.strong_score, 3), "crop_margin": self.crop_margin, "match_iou": self.match_iou,
                "match_dist": self.match_dist, "lost_s": self.lost_s, "lost_runs": self.lost_runs,
                "oriented": self.crop is crop_bee_oriented, "cut": self.cut, "cuts": self.cuts}

    def gallery_state(self):
        """Newest first: what the page needs to list the checked bees and fetch their crops."""
        return [{"id": t.id, "status": self.status(t), "checks": t.checks, "positives": t.positives,
                 "score": round(t.best_score, 3), "v": t.checks} for t in reversed(self.gallery.values())]

    def crop_jpeg(self, tid):
        t = self.gallery.get(int(tid))
        return t.best_jpeg if t else None
