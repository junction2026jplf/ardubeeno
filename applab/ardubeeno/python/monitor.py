"""Shared by main.py (App Lab) and serve.py (plain Python): run the Varroa pipeline on camera frames,
aggregate the verdict over frames and serve the camera as MJPEG with the latest results drawn on top.

The stream and the model run apart: show() takes every camera frame, a render thread draws the latest overlay
(boxes, labels; made by process()) on each one and JPEG-encodes it while someone watches, and the processing loop
takes only the newest frame (next_frame()). So a slow model lowers the result rate, not the stream's frame rate;
between results the bee boxes move on by their tracked velocity.

The mite count goes up once each time the aggregated verdict changes from clean to MITE, i.e. once per
infested bee passing through the region, not once per frame.

Two modes:
- multi (when bees.onnx is present): every bee in the frame is found, tracked, counted and checked for mites
  on its own (bees.py). The page lists the checked bees' crops with the mite boxes.
- single: only the centre region with the training crops' 4:7 shape goes to the model (cyan in the stream),
  so a bee that fills it reaches the model at the same scale as in training.

When the receiver is configured (outbox.py), every bee track that ends in multi mode is recorded locally as a bee
passage and uploaded in the background, together with the recorded time (coverage) and heartbeats.
"""
import atexit
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2

from bees import BeeCounter
from device_check import ROI_ASPECT, Pipeline
from outbox import Recorder

HERE = Path(__file__).parent
RED, GREEN, CYAN, WHITE = (40, 40, 230), (60, 180, 60), (255, 255, 0), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Mite detector + classifier files in this folder (make applab copies them), picked with models= / --models.
# Default: deploy.json's files (camera-fine-tuned, from export/deploy/) when it exists, else c160-int8all (board F1
# 0.926 on the 200 sample bees, cascade 0.931; deployed 0.903). "deployed" = varroa.onnx + varroa_cls.onnx.
MODEL_SETS = {
    "deployed": ("varroa.onnx", "varroa_cls.onnx"),
    "int8-qdqall": ("det_320x192_int8qdqall.onnx", "cls_288x288_int8qdqall.onnx"),
    "c160-int8all": ("det_320x192_int8qdqall.onnx", "cls160_160x160_int8qdqall.onnx"),
}
DEFAULTS = {"conf": 0.25, "fuse_conf": 0.38, "cascade": True, "cls_lo": 0.31, "cls_hi": 0.6, "bee_conf": 0.35}


def deploy_settings(path=HERE / "deploy.json"):
    """Settings from deploy.json (export/deploy/, copied by make applab): conf, fuse_conf, bee_conf, cls_lo, cls_hi,
    cascade as true/false, [lo, hi] or {"lo": .., "hi": ..}, and the mite model files as model / cls_model (default
    varroa.onnx / varroa_cls.onnx). Unknown keys are ignored."""
    if not path.exists():
        return {}
    import json
    d = json.loads(path.read_text())
    out = {k: float(d[k]) for k in ("conf", "fuse_conf", "bee_conf", "cls_lo", "cls_hi", "strong") if d.get(k) is not None}
    c = d.get("cascade")
    if isinstance(c, (list, tuple)) and len(c) == 2:
        out.update(cascade=True, cls_lo=float(c[0]), cls_hi=float(c[1]))
    elif isinstance(c, dict):
        out.update(cascade=c.get("enabled", True), cls_lo=float(c.get("lo", c.get("cls_lo", DEFAULTS["cls_lo"]))),
                   cls_hi=float(c.get("hi", c.get("cls_hi", DEFAULTS["cls_hi"]))))
    elif isinstance(c, bool):
        out["cascade"] = c
    out.update(files=(d.get("model") or "varroa.onnx", d.get("cls_model") or "varroa_cls.onnx"))
    print(f"{path.name}: {out}", flush=True)
    return out


class Monitor:
    def __init__(self, model=None, cls_model=None, conf=None, iou=0.45, fuse_conf=None, window=10, min_hits=2,
                 threads=2, bee_model=HERE / "bees.onnx", mode="auto", min_bee_px=100, source="field",
                 models=None, cascade=None, cls_lo=None, cls_hi=None, bee_conf=None, bee_threads=2,
                 det_every=3, max_checks_frame=2, background=True, stream_fps=30, stream_width=None,
                 jpeg_quality=75, strong=None, crop_margin=None, match_iou=0.1, match_dist=1.5, lost_s=1.5,
                 lost_runs=3, det_min_ms=0, oriented=False, cut=0.0, source_override=False):
        """None = from deploy.json, else DEFAULTS. models: a MODEL_SETS name (None: deploy.json's files, else
        c160-int8all); model/cls_model override single files.
        threads/bee_threads: onnxruntime threads per session (1-2 on the UNO Q; 4 is 2-3x slower there).
        stream_fps caps the stream below the camera's rate (0 = no cap); stream_width scales the stream down.
        strong, crop_margin, match_iou, match_dist, lost_s, lost_runs, det_min_ms, oriented: see bees.BeeCounter (multi
        mode)."""
        # deploy.json's thresholds belong to its own model pair; a MODELS set with other files keeps the DEFAULTS.
        ds = deploy_settings()
        if models and tuple(MODEL_SETS[models]) != tuple(ds.get("files", ())):
            ds = {}
        o = {**DEFAULTS, **ds, **{k: v for k, v in dict(
            conf=conf, fuse_conf=fuse_conf, cascade=cascade, cls_lo=cls_lo, cls_hi=cls_hi, bee_conf=bee_conf).items()
            if v is not None}}
        conf, fuse_conf = o["conf"], o["fuse_conf"]
        files = MODEL_SETS[models] if models else o.get("files", MODEL_SETS["c160-int8all"])
        model, cls_model = Path(model or HERE / files[0]), Path(cls_model or HERE / files[1])
        for m in (model, cls_model):
            if not m.exists() and m.name != "varroa_cls.onnx":
                raise SystemExit(f"{m} not found (make applab copies the model files)")
        cls = [str(cls_model)] if Path(cls_model).exists() else None
        self.pipe = Pipeline(SimpleNamespace(model=str(model), cls_model=cls, conf=conf, iou=iou, fuse_conf=fuse_conf,
                                             threads=threads, cascade=o["cascade"], cls_lo=o["cls_lo"],
                                             cls_hi=o["cls_hi"], spin=False))
        self.counter = None
        if mode != "single" and Path(bee_model).exists():  # shares self.pipe: one copy of the mite models
            self.counter = BeeCounter(bee_model, self.pipe, bee_conf=o["bee_conf"], min_bee_px=min_bee_px,
                                      min_hits=min_hits, bee_threads=bee_threads, det_every=det_every,
                                      max_checks_frame=max_checks_frame, background=background,
                                      strong=strong if strong is not None else o.get("strong"),
                                      crop_margin=crop_margin, match_iou=match_iou, match_dist=match_dist,
                                      lost_s=lost_s, lost_runs=lost_runs, det_min_ms=det_min_ms, oriented=oriented, cut=cut)
        elif mode == "multi":
            raise SystemExit(f"multi-bee mode needs the bee detector {bee_model}")
        p, b = self.pipe, self.counter
        print(f"mite check: {model.name} {p.det.w}x{p.det.h}" + (f" + {cls_model.name} {p.cls[0].size}" if cls else "")
              + f", {threads} threads, " + (f"cascade {p.cls_lo}/{p.cls_hi}" if p.cascade else "always both")
              + f", fuse_conf {fuse_conf}", flush=True)
        print(f"bee detector: {Path(bee_model).name} {b.bees.w}x{b.bees.h} conf {b.bees.conf}, {bee_threads} threads, "
              f"every {b.det_every} frames, max {b.max_checks_frame} checks waiting, "
              f"{'background' if b.worker else 'inline'} checks, MITE at {b.min_hits} hits or one >= "
              f"{b.strong_score:.2f}, crop margin {b.crop_margin}, match IoU {b.match_iou} / {b.match_dist} sizes, "
              f"lost after {b.lost_s} s and {b.lost_runs} runs, "
              f"{'oriented' if b.settings()['oriented'] else '90-degree'} crops, scene cut at {b.cut or 'off'}" if b else
              f"bee detector: none ({bee_model} missing or mode single)", flush=True)
        self.multi = self.counter is not None
        self.min_hits = min_hits
        self.recent = deque(maxlen=window)
        self.cond = threading.Condition()
        self.frame_id, self.roi = 0, None
        self.fps, self.frame_dt, self.last_t, self.positive_frames = 0.0, 0.0, None, 0
        self.last_seq, self.gap = None, 1.0  # camera frames per processed frame (smoothed)
        # The stream's own lock: camera frames, overlay and JPEG never wait for the model (which holds self.cond)
        self.view = threading.Condition()
        self.cam, self.overlay = (0, None), None  # (sequence number, newest camera frame); latest results to draw
        self.viewers = 0  # open /stream connections; without any, frames are not drawn or JPEG-encoded
        self.jpeg, self.jpeg_id, self.encoded, self.encode_ms, self.view_fps = None, 0, 0, deque(maxlen=100), 0.0
        self.stream_fps, self.stream_width, self.quality = stream_fps, stream_width, jpeg_quality
        threading.Thread(target=self.render_loop, daemon=True).start()
        self.base_thresholds = (fuse_conf, o["cls_lo"], o["cls_hi"])
        self.model_meta = {Path(m).name: Path(m).stat().st_size for m in (model, cls_model, bee_model)
                           if Path(m).exists()}
        self.source = source
        revision = {"models": self.model_meta, **self.settings()}
        self.recorder = Recorder.from_env(source, revision, source_override=source_override)
        if self.recorder:
            self.source = self.recorder.source
            atexit.register(self.close)
        self.closed = False
        self.base_counts = dict(self.recorder.initial_counts) if self.recorder else {}
        self.count, self.was_mite = self.base_counts.get("count", 0), False
        self.state = {"frames": 0, "mite": False, **self.base_counts, "count": self.count,
                      **self.settings(), **self.upload_state()}

    def configure(self, fuse_conf=None, window=None, min_hits=None, mode=None, min_bee_px=None):
        changed_rev = any(x is not None for x in (fuse_conf, min_hits, min_bee_px))
        with self.cond:
            if mode in ("single", "multi"):
                self.multi = mode == "multi" and self.counter is not None
            if fuse_conf is not None:
                v = float(fuse_conf)
                base_f, base_lo, base_hi = self.base_thresholds
                self.pipe.fuse_conf = v
                if self.pipe.cascade:
                    if abs(v - base_f) < 1e-6:
                        self.pipe.cls_lo, self.pipe.cls_hi = base_lo, base_hi
                    else:
                        self.pipe.cls_lo = round(max(0.01, min(0.95, v - max(0.0, base_f - base_lo))), 3)
                        self.pipe.cls_hi = round(min(0.99, max(base_hi, v + 0.15)), 3)
            if self.counter is not None:
                if fuse_conf is not None:
                    self.counter.mites.fuse_conf = self.pipe.fuse_conf
                if min_hits is not None:
                    self.counter.min_hits = max(1, int(min_hits))
                if min_bee_px is not None:
                    self.counter.min_bee_px = max(0, int(min_bee_px))
            if window is not None:
                self.recent = deque(self.recent, maxlen=max(1, int(window)))
            if min_hits is not None:
                self.min_hits = max(1, int(min_hits))
            settings = self.settings()
        if changed_rev and self.recorder:
            self.recorder.update_revision({"models": self.model_meta, **settings})
        return settings

    def settings(self):
        return {"fuse_conf": self.pipe.fuse_conf, "window": self.recent.maxlen, "min_hits": self.min_hits,
                "classifier": self.pipe.cls is not None, "mode": "multi" if self.multi else "single",
                "multi_available": self.counter is not None, "source": getattr(self, "source", "field"),
                "min_bee_px": self.counter.min_bee_px if self.counter else None,
                "cascade": [self.pipe.cls_lo, self.pipe.cls_hi] if self.pipe.cascade else None,
                "det_every": self.counter.det_every if self.counter else None,
                **(self.counter.settings() if self.counter else {})}

    def tick(self, seq):
        now = time.perf_counter()
        if self.last_t is not None:
            # Average the frame time, not its inverse: a fast tracker-only frame after a slow detector frame would
            # otherwise dominate the mean.
            dt = max(now - self.last_t, 1e-6)
            self.frame_dt = 0.95 * self.frame_dt + 0.05 * dt if self.frame_dt else dt
            self.fps = 1 / self.frame_dt
        if seq is not None and self.last_seq is not None and seq > self.last_seq:
            self.gap = 0.8 * self.gap + 0.2 * (seq - self.last_seq)
        self.last_t, self.last_seq = now, seq

    def show(self, frame):
        """Hand over a camera frame (from the camera thread; never blocks on the model): the stream draws every
        one, processing takes the newest with next_frame()."""
        with self.view:
            self.cam = (self.cam[0] + 1, frame)
            self.view.notify_all()

    def next_frame(self, last, timeout=None):
        """Wait for a camera frame newer than sequence number `last`; return (seq, frame), frame None on timeout."""
        with self.view:
            if not self.view.wait_for(lambda: self.cam[0] != last, timeout):
                return last, None
            return self.cam

    def process(self, frame, seq=None):
        """Run the current mode on a frame (seq: its number from next_frame(), so the stream can move the boxes on);
        return whether a mite is detected right now (for the LED matrix)."""
        return self.process_multi(frame, seq) if self.multi else self.process_single(frame, seq)

    def process_multi(self, frame, seq=None):
        """Find, track and check every bee in the frame."""
        h, w = frame.shape[:2]
        with self.cond:  # configure(), reset_count() and close() synchronize on self.cond
            if self.closed:
                return False
            visible = self.counter.process(frame, seq)
            ended = self.counter.end_tracks()
            counts = self.counter.counts(visible, self.base_counts)
            gallery = self.counter.gallery_state()
            mite = any(self.counter.is_mite(t) for t in visible)
            bees = self.counter.overlay(visible)
            timing = self.counter.stats()
            self.record(ended, True)
        self.tick(seq)
        self.publish(seq, bees=bees, label=f"bees {counts['bees_now']}  seen {counts['bees_seen']}  "
                                           f"mites {counts['count']}", color=RED if mite else GREEN)
        with self.cond:
            if self.closed:
                return mite
            self.frame_id += 1
            self.roi = frame
            self.state = {
                "frames": self.frame_id, "mite": mite, **counts, "gallery": gallery,
                "boxes": [[round(v) for v in tr.box[:4]] + [tr.id] for tr in visible],
                "size": [w, h], "fps": round(self.fps, 1), "ms": [round(m, 1) for m in self.counter.last_ms],
                "timing": timing, **self.settings(), **self.upload_state(),
            }
            self.cond.notify_all()
        return mite

    def process_single(self, frame, seq=None):
        """Run the model on the frame's centre region; return the verdict aggregated over recent frames."""
        with self.cond:
            if self.closed:
                return False
        h, w = frame.shape[:2]
        rw = min(w, round(h * ROI_ASPECT))
        x0 = (w - rw) // 2
        roi = frame[:, x0:x0 + rw]
        boxes, top, prob, fused, verdict = self.pipe(roi)
        self.tick(seq)

        with self.cond:
            if self.closed:
                return False
            self.record([], False)  # no bee tracks: not recorded time
            self.recent.append(bool(verdict))
            hits = sum(self.recent)
            mite = hits >= min(self.min_hits, self.recent.maxlen)
            self.positive_frames += bool(verdict)
            if mite and not self.was_mite:
                self.count += 1
            self.was_mite = mite
            count = self.count
            fuse_conf = self.pipe.fuse_conf

        self.publish(seq, roi=(x0, rw), mites=[(x + x0, y, bw, bh, s) for x, y, bw, bh, s in boxes],
                     label=f"{'MITE' if mite else 'clean'}  {fused:.2f}  [{hits}/{len(self.recent)}]",
                     color=RED if mite else GREEN, count=count)
        with self.cond:
            self.frame_id += 1
            self.roi = roi.copy()
            self.state = {
                "frames": self.frame_id, "mite": mite, "frame_mite": bool(verdict), "fused": fused, "prob": prob,
                "top": top, "hits": hits, "recent": list(self.recent), "positive_frames": self.positive_frames,
                "count": count,
                "boxes": [[round(x + x0), round(y), round(bw), round(bh), round(s, 3)] for x, y, bw, bh, s in boxes],
                "size": [w, h], "roi": [x0, 0, rw, h], "fps": round(self.fps, 1),
                "ms": [round(m, 1) for m in self.pipe.last_ms], "fuse_conf": fuse_conf, **self.settings(),
                **self.upload_state(),
            }
            self.cond.notify_all()
        return mite

    def publish(self, seq, **overlay):
        """Replace the overlay the stream draws: plain data, made once per processed frame."""
        with self.view:
            self.overlay = SimpleNamespace(**{"bees": [], "roi": None, "mites": [], "count": None, **overlay},
                                           seq=seq, gap=self.gap, fps=self.fps)

    def render_loop(self):
        """Draw the latest overlay on every new camera frame and JPEG-encode it, only while someone watches."""
        last, due, prev = 0, 0.0, None
        while True:
            with self.view:
                self.view.wait_for(lambda: self.viewers and self.cam[0] != last)
                (last, frame), ov = self.cam, self.overlay
            now = time.perf_counter()
            if self.stream_fps:  # at most stream_fps frames a second; a little early is fine (camera jitter)
                period = 1 / self.stream_fps
                if now < due - 0.25 * period:
                    continue
                due = max(due, now - period) + period
            jpeg = self.render(frame, ov, last)
            ms = (time.perf_counter() - now) * 1000
            with self.view:
                if prev is not None:
                    dt = max(now - prev, 1e-6)
                    self.view_fps = 0.9 * self.view_fps + 0.1 / dt if self.view_fps else 1 / dt
                self.jpeg, self.jpeg_id, prev = jpeg, self.jpeg_id + 1, now
                self.encoded += 1
                self.encode_ms.append(ms)
                self.view.notify_all()

    def render(self, frame, ov, seq):
        """One stream frame: the camera frame (scaled to stream_width) with the overlay, as JPEG."""
        h, w = frame.shape[:2]
        s = self.stream_width / w if self.stream_width and self.stream_width < w else 1
        out = cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR) if s < 1 else frame.copy()
        H = out.shape[0]
        scale, t, tb = H / 720, max(2, H // 240), max(2, H // 360)
        if ov is not None:
            # camera frames since the overlay's frame (velocities are per camera frame), at most two processed frames'
            # worth (and one box size) so that boxes don't run away while the model stalls
            k = min(max(seq - ov.seq, -2 * ov.gap), 2 * ov.gap) if ov.seq is not None else 0
            for (x, y, bw, bh), (vx, vy), tid, color in ov.bees:
                x, y = (x + min(max(vx * k, -bw), bw)) * s, (y + min(max(vy * k, -bh), bh)) * s
                p1, p2 = (int(x), int(y)), (int(x + bw * s), int(y + bh * s))
                cv2.rectangle(out, p1, p2, color, tb)
                cv2.putText(out, f"#{tid}", (p1[0], max(p1[1] - 6, 14)), FONT, 0.5 * scale + 0.2, color, tb)
            if ov.roi:
                x0, rw = ov.roi
                cv2.rectangle(out, (int(x0 * s), 0), (int((x0 + rw) * s) - 1, H - 1), CYAN, t)
            for x, y, bw, bh, sc in ov.mites:
                p1, p2 = (int(x * s), int(y * s)), (int((x + bw) * s), int((y + bh) * s))
                cv2.rectangle(out, p1, p2, RED, t)
                cv2.putText(out, f"mite {sc:.2f}", (p1[0], max(p1[1] - 6, 14)), FONT, 0.7 * scale + 0.2, RED, t)
            if ov.roi:  # single mode: the verdict large, the count below
                cv2.putText(out, ov.label, (12, int(48 * scale) + 12), FONT, 1.4 * scale + 0.3, ov.color, t + 1)
                cv2.putText(out, f"mites counted: {ov.count}", (12, int(96 * scale) + 12), FONT, 0.9 * scale + 0.2,
                            WHITE, t)
            else:
                cv2.putText(out, ov.label, (12, int(40 * scale) + 12), FONT, 1.1 * scale + 0.2, ov.color, t)
            cv2.putText(out, f"model {ov.fps:.1f} FPS  view {self.view_fps:.0f} FPS", (12, H - 14), FONT,
                        0.8 * scale + 0.2, WHITE, t)
        return cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()

    def stream_stats(self):
        with self.view:
            ms = sorted(self.encode_ms)
            return {"fps": round(self.view_fps, 1) if self.viewers else 0, "viewers": self.viewers,
                    "encoded": self.encoded, "render_ms": round(ms[len(ms) // 2], 1) if ms else None}

    def record(self, ended, recording=None):
        """Hand ended bee tracks and frame health to the outbox; enqueues in memory, never touches the database."""
        recorder = self.recorder  # close() may set it to None
        if recorder:
            if recording:
                recorder.frame(True)
            for track_id, assessment, info in ended:
                recorder.passage(track_id, assessment, info)
            if recording is False:
                recorder.frame(False)

    def upload_state(self):
        recorder = self.recorder
        return {"upload": dict(recorder.status)} if recorder else {}

    def reset_count(self):
        """Resets the displayed counts on the page and LED matrix without dropping in-flight tracks or manufacturing
        unfinished passages; recorded history, sequences and unsent events remain intact."""
        with self.cond:
            if self.closed:
                return {"count": 0}
            ended = []
            if self.counter is not None:
                ended = self.counter.end_tracks(all=False)
                self.counter.mark_reset()
            self.base_counts = {}
            self.count = 0
            self.state = {**self.state, "bees_seen": 0, "bees_checked": 0, "bees_clean": 0,
                          "bees_uncertain": 0, "count": 0, "gallery": []}
            self.record(ended)
            if self.recorder:
                self.recorder.reset_baseline()
        return {"count": 0}

    def close(self):
        """On shutdown: finish pending checks, record confirmed bees (including those still in view), and drain."""
        with self.cond:
            if self.closed:
                return
            self.closed = True
            recorder = self.recorder
        if recorder is None:
            return
        if self.counter is not None:
            deadline = time.monotonic() + 0.3
            while True:
                with self.cond:
                    with self.counter.lock:
                        done, self.counter.done = self.counter.done, []
                    for t, crop, result in done:
                        self.counter.check(t, crop, result)
                    pending = any(t.pending for t in self.counter.leaving + self.counter.tracker.tracks if t.confirmed)
                if not pending or time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
        with self.cond:
            ended = self.counter.end_tracks(all=True) if self.counter is not None else []
            self.recorder = None
        for track_id, assessment, info in ended:
            recorder.passage(track_id, assessment, info)
        recorder.close()

    def crop(self, id):
        """JPEG of a checked bee's most mite-like crop, with the mite boxes drawn."""
        with self.cond:
            return self.counter.crop_jpeg(id) if self.counter else None

    def current(self):
        stream = self.stream_stats()
        with self.cond:
            return {**self.state, "stream": stream}

    def mjpeg(self):
        """Yield multipart MJPEG parts, one per rendered camera frame, starting with the next one."""
        with self.view:
            self.viewers += 1
            last = self.jpeg_id
            self.view.notify_all()
        try:
            while True:
                with self.view:
                    if not self.view.wait_for(lambda: self.jpeg_id != last, timeout=5):
                        continue
                    last, jpeg = self.jpeg_id, self.jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpeg), jpeg)
        finally:  # the client went away
            with self.view:
                self.viewers -= 1

    def snapshot(self, folder=HERE.parent / "captures"):
        """Save the current model region unannotated, e.g. to label your own footage for fine-tuning."""
        with self.cond:
            roi = self.roi
        if roi is None:
            return {"error": "no frame yet"}
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{self.frame_id}.png"
        cv2.imwrite(str(path), roi)
        return {"saved": str(path)}
