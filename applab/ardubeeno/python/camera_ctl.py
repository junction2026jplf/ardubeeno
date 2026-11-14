"""Manual focus (and exposure) for the USB camera: v4l2-ctl on the device if installed, else OpenCV on the open capture.

  ssh arduino@<board> 'v4l2-ctl -d /dev/video2 --list-ctrls'   # what the camera offers
  sudo apt install v4l-utils                                   # if v4l2-ctl is missing (needs internet)

CameraControls(device, cap).state() and .set(autofocus=, focus=, auto_exposure=, exposure=) return the same dict:
  {"available", "method": "v4l2-ctl" | "opencv" | None, "autofocus", "focus", "min", "max", "step", "default",
   "range_known", "device", "reason", "error", "exposure": {...same keys for exposure, or available False}}
Nothing here raises: a failing control comes back as "error" (or "reason" when the camera has no such control).
Setting a focus value turns autofocus off first (most cameras ignore focus_absolute while it is on).
"""
import glob
import os
import re
import shutil
import subprocess
import threading

# Control names differ between kernels: newer first.
AF_NAMES = ("focus_automatic_continuous", "focus_auto")
FOCUS_NAMES = ("focus_absolute",)
AE_NAMES = ("auto_exposure", "exposure_auto")  # menu: 1 = manual, 3 = aperture priority (auto)
EXPOSURE_NAMES = ("exposure_time_absolute", "exposure_absolute")
AE_MANUAL = 1
MAINS = {0: 0, 50: 1, 60: 2}  # power_line_frequency menu: 0 = off, 1 = 50 Hz, 2 = 60 Hz

# One control of `v4l2-ctl --list-ctrls`, e.g.
#   focus_absolute 0x009a090a (int)    : min=0 max=250 step=5 default=0 value=0 flags=inactive
_CTRL = re.compile(r"^\s*([a-z0-9_]+)\s+(?:0x[0-9a-f]+\s+)?\((\w+)\)\s*:\s*(.*)$", re.I)
_KV = re.compile(r"(\w+)=(-?\w+)")


def parse_ctrls(text):
    """`v4l2-ctl --list-ctrls` output -> {name: {"type", "min", "max", "step", "default", "value", "flags"}}."""
    ctrls = {}
    for line in text.splitlines():
        m = _CTRL.match(line)
        if not m:
            continue
        name, ctype, rest = m.groups()
        c = {"type": ctype, "flags": []}
        for k, v in _KV.findall(rest):
            if k == "flags":
                c["flags"] = v.split(",")
            else:
                try:
                    c[k] = int(v, 0)
                except ValueError:
                    c[k] = v
        ctrls[name] = c
    return ctrls


def pick(ctrls, names):
    return next((n for n in names if n in ctrls), None)


def video_device(source):
    """/dev/videoN for a camera spec, or None: "/dev/video2", "2" (Linux index), "usb:0" (the first USB camera's
    capture node, as App Lab counts them). Anything else (csi:0, a video file, macOS) -> None."""
    source = str(source or "")
    if source.startswith("/dev/"):
        return source
    if source.isdigit():
        return f"/dev/video{source}" if os.path.exists(f"/dev/video{source}") else None
    m = re.fullmatch(r"usb:(\d+)", source)
    if m:
        usb = usb_capture_devices()
        k = int(m.group(1))
        return usb[k] if k < len(usb) else None
    return None


def usb_capture_devices():
    """The first video node of every USB camera (UVC adds a metadata node next to it), in /dev/videoN order."""
    found = []
    for path in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(re.sub(r"\D", "", p) or 0)):
        try:
            if "/usb" not in os.path.realpath(os.path.join(path, "device")):
                continue
            with open(os.path.join(path, "index")) as f:
                if f.read().strip() != "0":
                    continue
        except OSError:
            continue
        found.append("/dev/" + os.path.basename(path))
    return found


def _unavailable(reason, device=None, method=None):
    return {"available": False, "method": method, "autofocus": None, "focus": None, "min": None, "max": None,
            "step": None, "default": None, "range_known": False, "device": device, "reason": reason}


class V4l2Ctl:
    """Controls through the v4l2-ctl tool (v4l-utils); works while another process streams from the device."""
    method = "v4l2-ctl"

    def __init__(self, device, exe):
        self.device, self.exe = device, exe

    def run(self, *args):
        r = subprocess.run([self.exe, "-d", self.device, *args], capture_output=True, text=True, timeout=5)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip()
                               else f"v4l2-ctl exited with {r.returncode}")
        return r.stdout

    def ctrls(self):
        return parse_ctrls(self.run("--list-ctrls"))

    def set_ctrls(self, pairs):
        for name, value in pairs:  # one at a time: the order matters (autofocus off before the focus value)
            self.run(f"--set-ctrl={name}={int(value)}")


def _group(ctrls, auto_names, value_names, auto_is_on, device):
    """One {auto, value, range} block from parsed controls, or unavailable with the reason."""
    auto, val = pick(ctrls, auto_names), pick(ctrls, value_names)
    if not val:
        have = ", ".join(n for n in ctrls if any(w in n for w in ("focus", "exposure"))) or "none"
        return _unavailable(f"no {' / '.join(value_names)} control (camera's focus/exposure controls: {have})",
                            device, "v4l2-ctl")
    c = ctrls[val]
    return {"available": True, "method": "v4l2-ctl", "auto": auto_is_on(ctrls[auto]) if auto else None,
            "value": c.get("value"), "min": c.get("min"), "max": c.get("max"), "step": c.get("step") or 1,
            "default": c.get("default"), "range_known": True, "device": device, "controls": [auto, val]}


def describe_v4l2(ctrls, device):
    """state() for parsed v4l2 controls: the focus keys at the top level, exposure nested."""
    f = _group(ctrls, AF_NAMES, FOCUS_NAMES, lambda c: bool(c.get("value")), device)
    e = _group(ctrls, AE_NAMES, EXPOSURE_NAMES, lambda c: c.get("value") != AE_MANUAL, device)
    out = _focus_keys(f)
    out["exposure"] = e
    return out


def _focus_keys(g):
    """Rename a group's auto/value to the focus API's autofocus/focus."""
    out = dict(g)
    if "auto" in out:
        out["autofocus"] = out.pop("auto")
    if "value" in out:
        out["focus"] = out.pop("value")
    return out


def snap(value, lo, hi, step):
    """Clamp to [lo, hi] and round to the control's step."""
    value = int(round(float(value)))
    if lo is None or hi is None:
        return value
    step = step or 1
    value = max(lo, min(hi, value))
    return min(hi, lo + round((value - lo) / step) * step)


class CameraControls:
    """Focus/exposure for one camera. `device` is /dev/videoN (or None), `cap` an open cv2.VideoCapture (or None),
    `lock` the lock the frame grabber holds around cap.read() (OpenCV captures aren't thread-safe)."""

    def __init__(self, device=None, cap=None, lock=None, why_not=None):
        self.device, self.cap, self.lock = device, cap, lock or threading.Lock()
        self.why_not = why_not  # e.g. "video file": nothing to control
        self.v4l2 = None
        exe = shutil.which("v4l2-ctl")
        if device and exe and not why_not:
            self.v4l2 = V4l2Ctl(device, exe)
        self.missing_tool = bool(device and not exe)
        self.af_requested = None  # OpenCV can't always read autofocus back: remember what we set
        self.ae_requested = None

    def set_mains(self, hz):
        """Anti-flicker for the local mains frequency (50 Hz in Europe; a wrong value makes LED/fluorescent light
        flicker and band). v4l2-ctl only; returns a one-line message for the log."""
        if hz is None or not self.v4l2:
            return None
        try:
            ctrls = self.v4l2.ctrls()
            if "power_line_frequency" not in ctrls:
                return "mains frequency: the camera has no power_line_frequency control"
            self.v4l2.set_ctrls([("power_line_frequency", MAINS[hz])])
            return f"mains frequency: {hz or 'off'}" + (" Hz" if hz else "")
        except Exception as e:  # optional: never stop the app
            return f"mains frequency: setting {hz} Hz failed: {e}"

    # ---- read ----
    def state(self, error=None):
        try:
            s = self._state()
        except Exception as e:  # never break the caller
            s = _unavailable(f"reading the camera controls failed: {e}", self.device, self.method)
        if error:
            s["error"] = error
        return s

    @property
    def method(self):
        if self.why_not:
            return None
        if self.v4l2:
            return "v4l2-ctl"
        return "opencv" if self.cap is not None else None

    def _state(self):
        if self.why_not:
            return {**_unavailable(self.why_not, self.device), "exposure": _unavailable(self.why_not, self.device)}
        if self.v4l2:
            try:
                return describe_v4l2(self.v4l2.ctrls(), self.device)
            except Exception as e:
                if self.cap is None:
                    raise
                return self._opencv_state(note=f"v4l2-ctl failed ({e}), using OpenCV")
        if self.cap is not None:
            return self._opencv_state()
        reason = "no camera device to control"
        if self.missing_tool:
            reason = f"v4l2-ctl not installed (sudo apt install v4l-utils) and no OpenCV capture for {self.device}"
        return {**_unavailable(reason, self.device), "exposure": _unavailable(reason, self.device)}

    def _opencv_state(self, note=None):
        import cv2
        with self.lock:
            focus = self.cap.get(cv2.CAP_PROP_FOCUS)
            af = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
            backend = self._backend()
        hint = " (install v4l-utils for the exact range: sudo apt install v4l-utils)" if self.missing_tool else ""
        if focus is None or focus < 0:  # the backend has no focus property
            s = _unavailable(f"OpenCV ({backend}) reports no focus control{hint}", self.device, "opencv")
        else:
            hi = 1023 if focus > 255 else 255  # a guess: OpenCV doesn't expose the range
            s = {"available": True, "method": "opencv", "focus": int(focus),
                 "autofocus": bool(af) if af is not None and af >= 0 else self.af_requested,
                 "min": 0, "max": hi, "step": 1, "default": None, "range_known": False, "device": self.device,
                 "reason": f"range unknown: OpenCV ({backend}) doesn't report it, 0-{hi} is a guess{hint}"}
        if note:
            s["note"] = note
        s["exposure"] = _unavailable("exposure needs v4l2-ctl (sudo apt install v4l-utils)", self.device, "opencv")
        return s

    def _backend(self):
        try:
            return self.cap.getBackendName()
        except Exception:
            return "unknown backend"

    def describe(self):
        """One line for the startup log."""
        s = self.state()
        if not s["available"]:
            line = f"camera focus: not available ({s.get('reason')})"
        else:
            rng = f"{s['min']}-{s['max']} step {s['step']}" + ("" if s["range_known"] else " (guessed)")
            line = (f"camera focus via {s['method']} on {s['device'] or 'the capture'}: autofocus "
                    f"{_onoff(s['autofocus'])}, focus {s['focus']} (range {rng}, default {s['default']})")
        e = s.get("exposure") or {}
        if e.get("available"):
            line += (f"; exposure {'auto' if e.get('auto') else 'manual'} {e.get('value')} "
                     f"({e.get('min')}-{e.get('max')})")
        return line

    # ---- write ----
    def set(self, autofocus=None, focus=None, auto_exposure=None, exposure=None):
        """Apply what was given (strings from a query are fine) and return state() with the result.
        A focus value turns autofocus off unless autofocus=1 came with it (then the value is ignored)."""
        try:
            autofocus, auto_exposure = _bool(autofocus), _bool(auto_exposure)
            focus = None if focus in (None, "") else float(focus)
            exposure = None if exposure in (None, "") else float(exposure)
        except ValueError as e:
            return self.state(error=f"bad value: {e}")
        if focus is not None and autofocus is None:
            autofocus = False
        if focus is not None and autofocus:
            focus = None
        if exposure is not None and auto_exposure is None:
            auto_exposure = False
        if exposure is not None and auto_exposure:
            exposure = None
        if autofocus is None and focus is None and auto_exposure is None and exposure is None:
            return self.state()
        s = self.state()
        if not s["available"] and (autofocus is not None or focus is not None):
            return self.state(error=f"focus control not available: {s.get('reason')}")
        try:
            if s["method"] == "v4l2-ctl":
                self._set_v4l2(s, autofocus, focus, auto_exposure, exposure)
            else:
                err = self._set_opencv(s, autofocus, focus, auto_exposure, exposure)
                if err:
                    return self.state(error=err)
        except Exception as e:
            return self.state(error=str(e))
        return self.state()

    def _set_v4l2(self, s, autofocus, focus, auto_exposure, exposure):
        pairs = []
        af_name, focus_name = s.get("controls") or (None, None)
        if autofocus is not None and af_name:
            pairs.append((af_name, int(autofocus)))
        elif autofocus:
            raise RuntimeError("this camera has no autofocus control")
        if focus is not None:
            pairs.append((focus_name, snap(focus, s["min"], s["max"], s["step"])))
        e = s.get("exposure") or {}
        if auto_exposure is not None or exposure is not None:
            if not e.get("available"):
                raise RuntimeError(f"exposure control not available: {e.get('reason')}")
            ae_name, exp_name = e["controls"]
            if auto_exposure is not None and ae_name:
                ctrls = self.v4l2.ctrls()
                auto_value = ctrls[ae_name].get("default", 3)
                pairs.append((ae_name, AE_MANUAL if not auto_exposure else
                              (auto_value if auto_value != AE_MANUAL else 3)))
            if exposure is not None:
                pairs.append((exp_name, snap(exposure, e["min"], e["max"], e["step"])))
        self.v4l2.set_ctrls(pairs)

    def _set_opencv(self, s, autofocus, focus, auto_exposure, exposure):
        import cv2
        if auto_exposure is not None or exposure is not None:
            return "exposure needs v4l2-ctl (sudo apt install v4l-utils)"
        with self.lock:
            if autofocus is not None:
                ok = self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
                if not ok:
                    return f"the camera refused autofocus {'on' if autofocus else 'off'}"
                self.af_requested = autofocus
            if focus is not None:
                want = snap(focus, s["min"], s["max"], 1)
                ok = self.cap.set(cv2.CAP_PROP_FOCUS, want)
                got = self.cap.get(cv2.CAP_PROP_FOCUS)
                if not ok:
                    return f"the camera refused focus {want}"
                if got is not None and got >= 0 and abs(got - want) > 1:
                    return f"asked for focus {want}, the camera reports {int(got)}"
        return None


def _bool(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "on", "yes", "auto"):
        return True
    if s in ("0", "false", "off", "no", "manual"):
        return False
    raise ValueError(f"not a boolean: {v!r}")


def _onoff(v):
    return "unknown" if v is None else ("on" if v else "off")


class FakeControls:
    """Testing the page without a focusable camera (serve.py --fake-focus): a v4l2-like camera in memory."""

    def __init__(self, lo=0, hi=250, step=5, default=0):
        self.c = {"focus_automatic_continuous": {"type": "bool", "default": 1, "value": 1, "flags": []},
                  "focus_absolute": {"type": "int", "min": lo, "max": hi, "step": step, "default": default,
                                     "value": default, "flags": ["inactive"]},
                  "auto_exposure": {"type": "menu", "min": 0, "max": 3, "default": 3, "value": 3, "flags": []},
                  "exposure_time_absolute": {"type": "int", "min": 1, "max": 5000, "step": 1, "default": 157,
                                             "value": 157, "flags": ["inactive"]}}

    def state(self, error=None):
        s = describe_v4l2(self.c, "fake")
        s["method"] = s["exposure"]["method"] = "fake"
        if error:
            s["error"] = error
        return s

    def set(self, autofocus=None, focus=None, auto_exposure=None, exposure=None):
        try:
            af, ae = _bool(autofocus), _bool(auto_exposure)
        except ValueError as e:
            return self.state(error=str(e))
        if focus not in (None, "") and af is None:
            af = False
        if af is not None:
            self.c["focus_automatic_continuous"]["value"] = int(af)
        if focus not in (None, "") and not af:
            f = self.c["focus_absolute"]
            f["value"] = snap(focus, f["min"], f["max"], f["step"])
        if exposure not in (None, "") and ae is None:
            ae = False
        if ae is not None:
            self.c["auto_exposure"]["value"] = 3 if ae else AE_MANUAL
        if exposure not in (None, "") and not ae:
            e = self.c["exposure_time_absolute"]
            e["value"] = snap(exposure, e["min"], e["max"], e["step"])
        return self.state()

    def describe(self):
        return "camera focus: fake controls for testing (--fake-focus)"
