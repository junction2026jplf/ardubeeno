"""The same live preview without App Lab: on the UNO Q over SSH, or on a laptop with a webcam.

  python3 serve.py                      # first camera that delivers frames, then open http://<board-or-laptop-ip>:7000
  python3 serve.py --camera /dev/video2 # a specific camera (index or /dev/videoN)
  python3 serve.py --camera bees.mp4    # a video file instead of a camera, looped
  python3 serve.py --models deployed --det-every 4   # fp32 mite models, bee detector every 4th frame
  python3 serve.py --stream-width 960  # a lighter stream; it runs at the camera's rate (up to --stream-fps)
  python3 serve.py --focus 120          # manual focus from the start (autofocus off); the page's settings change it

Needs numpy, opencv-python(-headless) and onnxruntime (requirements-device.txt). Serves the same page
as the App Lab app (../assets/index.html), without the LED matrix.
"""
import argparse
import glob
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from camera_ctl import CameraControls, FakeControls, video_device
from monitor import MODEL_SETS, Monitor

PAGE = Path(__file__).parent.parent / "assets" / "index.html"


def open_camera(source, width, height):
    """Open a camera index, /dev/videoN or video file; return the capture only if it delivers a frame."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif source.startswith("/dev/video"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if cap.isOpened() and cap.read()[0]:
        return cap
    cap.release()
    return None


def find_camera(source, width, height):
    """`auto` tries every /dev/video* (Linux; the UNO Q's own video devices come first) or indices 0-4."""
    if source != "auto":
        cap = open_camera(source, width, height)
        if cap is None:
            raise SystemExit(f"could not read frames from {source}")
        return source, cap
    candidates = sorted(glob.glob("/dev/video*"), key=lambda d: int(d[10:]) if d[10:].isdigit() else 99) \
        or [str(i) for i in range(5)]
    for c in candidates:
        cap = open_camera(c, width, height)
        if cap is not None:
            return c, cap
    raise SystemExit(f"no camera delivered frames (tried {', '.join(candidates)}). Is the USB camera plugged in? "
                     "Check `ls /dev/video*` and that the user is in the `video` group.")


def grab(monitor, source, cap, lock):
    """Read the camera (or file, at its own speed) and hand every frame to the monitor: the stream shows each one,
    processing takes only the newest, so it never works through stale buffered frames."""
    is_file = not (source.isdigit() or source.startswith("/dev/video"))
    period = 1 / (cap.get(cv2.CAP_PROP_FPS) or 15) if is_file else 0
    while True:
        t = time.perf_counter()
        with lock:  # camera controls (OpenCV fallback) use the capture between reads
            ok, frame = cap.read()
        if not ok:
            if is_file:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                time.sleep(0.5)
            continue
        monitor.show(frame)
        time.sleep(max(0.0, period - (time.perf_counter() - t)))


def capture_loop(monitor, source, cap, lock, log_every=30, slow_ms=0):
    threading.Thread(target=grab, args=(monitor, source, cap, lock), daemon=True).start()
    seq, logged = 0, time.time()
    while True:
        seq, frame = monitor.next_frame(seq)
        monitor.process(frame, seq)
        if slow_ms:  # testing: a slower model
            time.sleep(slow_ms / 1000)
        if log_every and time.time() - logged > log_every:  # a status line for the terminal
            logged, s = time.time(), monitor.current()
            v = s["stream"]
            print(f"{s.get('mode')}: {s.get('fps')} FPS, frame {seq}, bees now {s.get('bees_now', '-')}, "
                  f"seen {s.get('bees_seen', '-')}, checked {s.get('bees_checked', '-')}, mites {s.get('count')}, "
                  f"ms {s.get('timing', s.get('ms'))}; stream {v['fps']} FPS to {v['viewers']} viewers, "
                  f"{v['render_ms']} ms per frame", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", default="auto", help="camera index, /dev/videoN or video file; auto = first that works")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mode", choices=["auto", "multi", "single"], default="auto",
                   help="multi: every bee in the frame (needs bees.onnx); single: one bee in the centre region")
    p.add_argument("--min-bee-px", type=int, default=100, help="smaller bees are counted but not checked for mites")
    p.add_argument("--models", choices=list(MODEL_SETS),
                   help="mite detector + classifier set (default: deploy.json's files, else c160-int8all)")
    p.add_argument("--model", help="mite detector file, overrides --models")
    p.add_argument("--cls-model", help="classifier file, overrides --models")
    p.add_argument("--fuse-conf", type=float,
                   help="per-bee threshold on mean(det, cls); default from deploy.json if present, else 0.38")
    p.add_argument("--no-cascade", dest="cascade", action="store_false", default=None,
                   help="always run detector and classifier (default: classifier first)")
    p.add_argument("--cls-lo", type=float,
                   help="cascade: clean below this classifier probability (default from deploy.json if present, else 0.31)")
    p.add_argument("--cls-hi", type=float,
                   help="cascade: mite at or above this classifier probability (default from deploy.json if present, else 0.60)")
    p.add_argument("--bee-model", default=str(Path(__file__).parent / "bees.onnx"))
    p.add_argument("--bee-conf", type=float, help="bee detector confidence (0.35)")
    p.add_argument("--det-every", type=int, default=3, help="run the bee detector every N frames")
    p.add_argument("--max-checks", type=int, default=2, help="bee crops waiting for or in the mite check at most")
    p.add_argument("--inline-checks", action="store_true", help="mite checks between detections, not in a thread")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--min-hits", type=int, default=2, help="positive checks that make a bee MITE")
    p.add_argument("--strong", type=float,
                   help="one check with a fused score at or above this makes a bee MITE (default: cls_hi with the "
                        "cascade, else fuse_conf + 0.25)")
    p.add_argument("--crop-margin", type=float,
                   help="context around the bee box in the mite crop (default: bees.json next to the bee model, "
                        "else 0.15)")
    p.add_argument("--orient", action="store_true",
                   help="turn diagonal bees upright for the mite check (default: 90-degree steps only, as tested on "
                        "the board)")
    p.add_argument("--cut", type=float, default=0.0,
                   help="end all tracks when the picture changes this much between detections (a new slide of the "
                        "tile slideshow: try 15); 0 = off (default)")
    p.add_argument("--match-iou", type=float, default=0.1,
                   help="tracking: a detection continues a track it overlaps by this IoU ...")
    p.add_argument("--match-dist", type=float, default=1.5,
                   help="... or whose predicted centre is this many box sizes away (x (1 + detector runs missed))")
    p.add_argument("--lost-s", type=float, default=1.5,
                   help="keep a bee the detector lost for this many seconds ...")
    p.add_argument("--lost-runs", type=int, default=3, help="... and this many bee detector runs")
    p.add_argument("--threads", type=int, default=2, help="onnxruntime threads per mite model (1-2 on the UNO Q)")
    p.add_argument("--bee-threads", type=int, default=2, help="onnxruntime threads for the bee detector")
    p.add_argument("--stream-fps", type=float, default=30, help="stream frame rate cap, 0 = the camera's rate")
    p.add_argument("--stream-width", type=int, help="scale the stream down to this width (default: camera width)")
    p.add_argument("--jpeg-quality", type=int, default=75, help="stream JPEG quality")
    p.add_argument("--slow-ms", type=float, default=0, help="testing: sleep this long after each processed frame")
    p.add_argument("--det-min-ms", type=float, default=0,
                   help="testing: every bee detector run takes at least this long (550 = the UNO Q)")
    p.add_argument("--log-every", type=float, default=30, help="seconds between status lines, 0 = off")
    p.add_argument("--focus", type=int, help="manual focus at start (turns autofocus off); range: GET /camera")
    p.add_argument("--autofocus", action="store_true", help="turn autofocus on at start")
    p.add_argument("--mains-hz", type=int, choices=[50, 60, 0], default=50,
                   help="camera anti-flicker for the local mains frequency (default 50 Hz, Europe; 0 = off)")
    p.add_argument("--exposure", type=int, help="manual exposure at start (turns auto exposure off; v4l2-ctl only)")
    p.add_argument("--auto-exposure", action="store_true", help="turn auto exposure on at start")
    p.add_argument("--fake-focus", action="store_true", help="testing: pretend the camera has focus controls")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7000)
    args = p.parse_args()

    is_file = not (args.camera == "auto" or args.camera.isdigit() or args.camera.startswith("/dev/video"))
    monitor = Monitor(fuse_conf=args.fuse_conf, window=args.window, min_hits=args.min_hits, threads=args.threads,
                      mode=args.mode, min_bee_px=args.min_bee_px, source="replay" if is_file else "field",
                      models=args.models, model=args.model, cls_model=args.cls_model, cascade=args.cascade,
                      cls_lo=args.cls_lo, cls_hi=args.cls_hi, bee_model=args.bee_model, bee_conf=args.bee_conf,
                      bee_threads=args.bee_threads, det_every=args.det_every, max_checks_frame=args.max_checks,
                      background=not args.inline_checks, stream_fps=args.stream_fps, stream_width=args.stream_width,
                      jpeg_quality=args.jpeg_quality, strong=args.strong, crop_margin=args.crop_margin,
                      match_iou=args.match_iou, match_dist=args.match_dist, lost_s=args.lost_s,
                      lost_runs=args.lost_runs, det_min_ms=args.det_min_ms, oriented=args.orient, cut=args.cut)
    if monitor.recorder:  # exit cleanly on stop so the recorded history is closed
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    source, cap = find_camera(args.camera, args.width, args.height)
    lock = threading.Lock()
    if args.fake_focus:
        controls = FakeControls()
    else:
        cam_file = not (source.isdigit() or source.startswith("/dev/"))
        controls = CameraControls(video_device(source), cap, lock, why_not="a video file, not a camera" if cam_file
                                  else None)
    if args.focus is not None or args.autofocus or args.exposure is not None or args.auto_exposure:
        r = controls.set(autofocus=True if args.autofocus else None, focus=args.focus,
                         auto_exposure=True if args.auto_exposure else None, exposure=args.exposure)
        if r.get("error"):
            print(f"camera controls at start: {r['error']}", flush=True)
    msg = controls.set_mains(args.mains_hz)
    if msg:
        print(msg, flush=True)
    print(controls.describe(), flush=True)
    threading.Thread(target=capture_loop, args=(monitor, source, cap, lock, args.log_every, args.slow_ms),
                     daemon=True).start()

    def camera(query):
        keys = ("autofocus", "focus", "auto_exposure", "exposure")
        if not any(query.get(k) not in (None, "") for k in keys):
            return controls.state()
        r = controls.set(**{k: query.get(k) for k in keys})
        print(f"camera: {', '.join(f'{k}={query[k]}' for k in keys if k in query)} -> autofocus "
              f"{r.get('autofocus')}, focus {r.get('focus')}" + (f", error: {r['error']}" if r.get("error") else ""),
              flush=True)
        return r

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, ctype="application/json"):
            body = json.dumps(body).encode() if ctype == "application/json" else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/":
                self.reply(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/state":
                self.reply(monitor.current())
            elif url.path == "/config":
                self.reply(monitor.configure(query.get("fuse_conf"), query.get("window"), query.get("min_hits"),
                                             query.get("mode"), query.get("min_bee_px")))
            elif url.path == "/crop":
                jpeg = monitor.crop(query.get("id", 0))
                self.reply(jpeg, "image/jpeg") if jpeg else self.send_error(404)
            elif url.path == "/snapshot":
                self.reply(monitor.snapshot())
            elif url.path == "/camera":  # GET: the focus state; with autofocus=0|1 and/or focus=N: set it
                self.reply(camera(query))
            elif url.path == "/reset":
                self.reply(monitor.reset_count())
            elif url.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for part in monitor.mjpeg():
                        self.wfile.write(part)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/camera":
                return self.send_error(404)
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode(errors="replace")
            try:  # a JSON object or a form/query string
                query = json.loads(body) if body.lstrip().startswith("{") else {}
            except ValueError:
                query = {}
            query = {**{k: v[0] for k, v in parse_qs(url.query).items()},
                     **{k: v[0] for k, v in parse_qs(body).items()}, **query}
            self.reply(camera(query))

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"{'multi-bee' if monitor.multi else 'single-bee'} mode, camera {source}; open http://localhost:{args.port} (or this machine's IP)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
