"""ArduBeeno for Arduino App Lab: live Varroa mite detection on the UNO Q's USB camera.

Open the app's page (App Lab shows the link, or http://<board-ip>:7000 from your laptop) to watch the camera.
With bees.onnx in this folder every bee is found, tracked, counted and checked for mites (green clean, red mite,
grey not checked), and the page shows each checked bee's crop with the mite boxes. Without it, only the cyan
centre region is checked.
The LED matrix shows how many mites were counted (a tick while none), and blinks a mite while one is detected.
Set CAMERA in the app's environment for another camera (default usb:0, e.g. csi:0 or /dev/video2).
MODELS picks the mite models (c160-int8all | int8-qdqall | deployed; default: deploy.json's files, else
c160-int8all); THREADS, BEE_THREADS, DET_EVERY and MAX_CHECKS tune the speed. The stream shows every camera
frame with the latest results, however slow the model; STREAM_FPS caps it (default 30), STREAM_WIDTH scales it down.
A bee is MITE after MIN_HITS positive checks (default 2) or one check scoring at least STRONG (default: cls_hi with
the cascade, else fuse_conf + 0.25). CROP_MARGIN is the context around each bee crop (default: bees.json next to
bees.onnx, else 0.15). Tracking: MATCH_IOU (0.1), MATCH_DIST (1.5 box sizes), and a lost bee is kept LOST_S
seconds (1.5) and LOST_RUNS bee detector runs (3), whichever is longer.
Camera focus: the page's settings (GET /camera) through v4l2-ctl on the camera's /dev/videoN, if installed; FOCUS=N
sets manual focus at start, AUTOFOCUS=1 turns autofocus on (EXPOSURE=N / AUTO_EXPOSURE=1 likewise). MAINS_HZ (50,
default; 60 or 0 = off) sets the camera's anti-flicker for the local mains frequency.
"""
import os
import signal
import sys
import threading
import time

from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App, Bridge
from fastapi.responses import Response, StreamingResponse

from camera_ctl import CameraControls, video_device
from monitor import Monitor

env = os.environ.get  # optional overrides in the app's environment; thresholds come from deploy.json if present
opt = lambda k: float(env(k)) if env(k) else None
monitor = Monitor(models=env("MODELS"), threads=int(env("THREADS", 2)),
                  bee_threads=int(env("BEE_THREADS", 2)), det_every=int(env("DET_EVERY", 3)),
                  max_checks_frame=int(env("MAX_CHECKS", 2)), stream_fps=float(env("STREAM_FPS", 30)),
                  stream_width=int(env("STREAM_WIDTH", 0)) or None, jpeg_quality=int(env("JPEG_QUALITY", 75)),
                  min_hits=int(env("MIN_HITS", 2)), strong=opt("STRONG"), crop_margin=opt("CROP_MARGIN"),
                  match_iou=float(env("MATCH_IOU", 0.1)), match_dist=float(env("MATCH_DIST", 1.5)),
                  lost_s=float(env("LOST_S", 1.5)), lost_runs=int(env("LOST_RUNS", 3)),
                  oriented=env("ORIENT", "0") == "1", cut=float(env("CUT", 0)),
                  source=env("ARDUBEENO_SOURCE") or env("SOURCE") or "field",
                  source_override=bool(env("SOURCE")))
camera = Camera(os.environ.get("CAMERA", "usb:0"), resolution=(1280, 720), fps=15)
ui = WebUI()
camera_lock = threading.Lock()


def find_capture(obj, depth=2):
    """App Lab's Camera has no focus API: look for the OpenCV capture inside it (fallback when v4l2-ctl is missing)."""
    try:
        import cv2
        for v in vars(obj).values():
            if isinstance(v, cv2.VideoCapture):
                return v
            if depth and hasattr(v, "__dict__") and not isinstance(v, type):
                found = find_capture(v, depth - 1)
                if found is not None:
                    return found
    except Exception:
        pass
    return None


def make_controls():
    """After camera.start(): v4l2-ctl on the camera's /dev/videoN, else the OpenCV capture inside Camera, else none."""
    try:
        spec = os.environ.get("CAMERA", "usb:0")
        dev, cap = video_device(spec), find_capture(camera)
        return CameraControls(dev, cap, camera_lock,
                              why_not=None if dev or cap is not None else f"no /dev/videoN found for CAMERA={spec}")
    except Exception as e:  # focus is optional: the app runs without it
        return CameraControls(why_not=f"camera controls failed: {e}")


controls = CameraControls(why_not="camera not started yet")


def configure(fuse_conf: float | None = None, window: int | None = None, min_hits: int | None = None,
              mode: str | None = None, min_bee_px: int | None = None):
    return monitor.configure(fuse_conf, window, min_hits, mode, min_bee_px)


def camera_ctl(autofocus: str | None = None, focus: str | None = None, auto_exposure: str | None = None,
               exposure: str | None = None):
    try:
        if autofocus is None and focus is None and auto_exposure is None and exposure is None:
            return controls.state()
        return controls.set(autofocus, focus, auto_exposure, exposure)
    except Exception as e:
        return {"available": False, "method": None, "reason": str(e)}


def crop(id: int = 0):
    jpeg = monitor.crop(id)
    return Response(jpeg, media_type="image/jpeg") if jpeg else Response(status_code=404)


ui.expose_api("GET", "/stream", lambda: StreamingResponse(
    monitor.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"))
ui.expose_api("GET", "/state", monitor.current)
ui.expose_api("GET", "/config", configure)
ui.expose_api("GET", "/snapshot", lambda: monitor.snapshot())
ui.expose_api("GET", "/reset", monitor.reset_count)
ui.expose_api("GET", "/crop", crop)
ui.expose_api("GET", "/camera", camera_ctl)
ui.expose_api("POST", "/camera", camera_ctl)

seq = 0
LED_REFRESH_S = 2  # resend the LED state this often even unchanged: a (re)started sketch shows the tick until told


def grab():
    """Every camera frame goes to the stream; loop() processes only the newest."""
    while True:
        try:
            with camera_lock:
                frame = camera.capture()
        except Exception as e:  # keep the thread alive; the camera may come back
            print(f"camera: {e}")
            frame = None
        if frame is None:
            time.sleep(0.05)
        else:
            monitor.show(frame)


def show_on_led():
    """The LED matrix follows the page's count and mite state: sent on every change (a /reset shows the tick, a
    restored count shows at startup, before the first frame) and again every LED_REFRESH_S seconds. Runs in its own
    thread with a short timeout, so a sketch that is not running (yet) never stalls the detection."""
    shown, sent_at, error = None, 0.0, None
    while True:
        state = monitor.current()
        led = (bool(state.get("mite")), int(state.get("count") or 0))
        if led != shown or time.monotonic() - sent_at >= LED_REFRESH_S:
            try:
                Bridge.call("set_count", led[1], timeout=1)
                Bridge.call("set_mite", led[0], timeout=1)
                shown, error = led, None
            except Exception as e:  # the app still works without the sketch; retried until it answers
                if str(e) != error:
                    print(f"LED matrix not updated: {e}")
                shown, error = None, str(e)
                time.sleep(1)
            sent_at = time.monotonic()
        time.sleep(0.1)


def loop():
    global seq
    seq, frame = monitor.next_frame(seq, timeout=1)
    if frame is not None:
        monitor.process(frame, seq)


if monitor.recorder:  # exit cleanly on stop so the recorded history is closed
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
camera.start()
controls = make_controls()
try:
    if env("FOCUS") or env("AUTOFOCUS") == "1" or env("EXPOSURE") or env("AUTO_EXPOSURE") == "1":
        r = controls.set(autofocus="1" if env("AUTOFOCUS") == "1" else None, focus=env("FOCUS"),
                         auto_exposure="1" if env("AUTO_EXPOSURE") == "1" else None, exposure=env("EXPOSURE"))
        if r.get("error"):
            print(f"camera controls at start: {r['error']}")
    msg = controls.set_mains(int(env("MAINS_HZ", 50)))
    if msg:
        print(msg)
    print(controls.describe())
except Exception as e:
    print(f"camera focus: {e}")
threading.Thread(target=grab, daemon=True).start()
threading.Thread(target=show_on_led, daemon=True).start()
App.run(user_loop=loop)
