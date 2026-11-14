"""Standalone check of applab/ardubeeno/python/camera_ctl.py: the v4l2-ctl parser on real-looking outputs, and the
focus logic end to end against a fake v4l2-ctl (no camera or Linux needed).

  .venv/bin/python scripts/test_camera_ctl.py
"""
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "applab" / "ardubeeno" / "python"))
import camera_ctl  # noqa: E402
from camera_ctl import CameraControls, describe_v4l2, parse_ctrls, snap  # noqa: E402

# A typical UVC webcam (Logitech C920 style) on a recent kernel / v4l-utils 1.22+.
UVC_NEW = """
User Controls

                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
                       contrast 0x00980901 (int)    : min=0 max=255 step=1 default=128 value=128
                     saturation 0x00980902 (int)    : min=0 max=255 step=1 default=128 value=128
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
                           gain 0x00980913 (int)    : min=0 max=255 step=1 default=0 value=0
           power_line_frequency 0x00980918 (menu)   : min=0 max=2 default=2 value=2 (60 Hz)
      white_balance_temperature 0x0098091a (int)    : min=2000 max=6500 step=1 default=4000 value=4000 flags=inactive
                      sharpness 0x0098091b (int)    : min=0 max=255 step=1 default=128 value=128
         backlight_compensation 0x0098091c (int)    : min=0 max=1 step=1 default=0 value=0

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250 flags=inactive
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
                   pan_absolute 0x009a0908 (int)    : min=-36000 max=36000 step=3600 default=0 value=0
                  tilt_absolute 0x009a0909 (int)    : min=-36000 max=36000 step=3600 default=0 value=0
                 focus_absolute 0x009a090a (int)    : min=0 max=250 step=5 default=0 value=0 flags=inactive
     focus_automatic_continuous 0x009a090c (bool)   : default=1 value=1
                  zoom_absolute 0x009a090d (int)    : min=100 max=500 step=1 default=100 value=100
"""

# The same camera on an older kernel / v4l-utils (focus_auto, exposure_auto, exposure_absolute).
UVC_OLD = """
                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=1
              exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=166
                 focus_absolute 0x009a090a (int)    : min=0 max=1023 step=1 default=68 value=300
                     focus_auto 0x009a090c (bool)   : default=1 value=0
"""

# A fixed-focus camera: no focus controls at all.
NO_FOCUS = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                       contrast 0x00980901 (int)    : min=0 max=64 step=1 default=32 value=32
                          gamma 0x00980910 (int)    : min=72 max=500 step=1 default=100 value=100

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
"""


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"ok  {msg}")


def test_parser():
    c = parse_ctrls(UVC_NEW)
    f = c["focus_absolute"]
    check((f["min"], f["max"], f["step"], f["default"], f["value"]) == (0, 250, 5, 0, 0), "focus_absolute range")
    check(f["flags"] == ["inactive"] and f["type"] == "int", "flags and type")
    check(c["focus_automatic_continuous"] == {"type": "bool", "flags": [], "default": 1, "value": 1}, "bool control")
    check(c["auto_exposure"]["value"] == 3 and c["brightness"]["min"] == 0, "menu control (with label) and others")
    check(c["pan_absolute"]["min"] == -36000, "negative numbers")

    s = describe_v4l2(c, "/dev/video2")
    check(s["available"] and s["autofocus"] is True and s["focus"] == 0, "new names: autofocus on, focus 0")
    check((s["min"], s["max"], s["step"], s["default"]) == (0, 250, 5, 0), "new names: range")
    check(s["controls"] == ["focus_automatic_continuous", "focus_absolute"], "new names: control names")
    check(s["exposure"]["available"] and s["exposure"]["auto"] is True and s["exposure"]["max"] == 2047,
          "new names: exposure auto, 3-2047")

    s = describe_v4l2(parse_ctrls(UVC_OLD), "/dev/video2")
    check(s["controls"] == ["focus_auto", "focus_absolute"] and s["autofocus"] is False and s["focus"] == 300,
          "old names: focus_auto off, focus 300")
    check(s["max"] == 1023 and s["default"] == 68, "old names: range 0-1023")
    check(s["exposure"]["controls"] == ["exposure_auto", "exposure_absolute"] and s["exposure"]["auto"] is False,
          "old names: exposure_auto manual")

    s = describe_v4l2(parse_ctrls(NO_FOCUS), "/dev/video2")
    check(not s["available"] and "no focus_absolute" in s["reason"] and s["focus"] is None, "no focus: unavailable")
    check(s["exposure"]["available"], "no focus: exposure still available")
    check(parse_ctrls("") == {} and parse_ctrls("Failed to open /dev/video9: No such file") == {}, "empty / error")


def test_snap():
    check(snap(123, 0, 250, 5) == 125 and snap(-4, 0, 250, 5) == 0 and snap(999, 0, 250, 5) == 250, "clamp + step")
    check(snap("37.4", 3, 2047, 1) == 37 and snap(7, None, None, None) == 7, "strings, unknown range")


FAKE = r'''#!{python}
"""A fake v4l2-ctl: keeps the controls in a JSON file next to it."""
import json, sys
from pathlib import Path
db = Path(__file__).with_suffix(".json")
c = json.loads(db.read_text())
args = sys.argv[1:]
if "--list-ctrls" in args:
    for n, v in c.items():
        rng = f"min={{v['min']}} max={{v['max']}} " if "min" in v else ""
        rng += f"step={{v['step']}} " if "step" in v else ""
        print(f"{{n:>30}} 0x009a0900 ({{v['type']}}) : {{rng}}default={{v['default']}} value={{v['value']}}")
for a in args:
    if a.startswith("--set-ctrl="):
        name, val = a[len("--set-ctrl="):].split("=")
        if name not in c:
            sys.exit(f"unknown control '{{name}}'")
        if name == "focus_absolute" and c.get("focus_automatic_continuous", {{}}).get("value"):
            sys.exit("VIDIOC_S_EXT_CTRLS: failed: Permission denied")
        c[name]["value"] = int(val)
        with open(db.with_suffix(".log"), "a") as f:
            f.write(f"{{name}}={{val}}\n")
db.write_text(json.dumps(c))
'''


def test_with_fake_v4l2ctl():
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "v4l2-ctl"
        exe.write_text(FAKE.format(python=sys.executable))
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
        exe.with_suffix(".json").write_text(json.dumps(parse_ctrls(UVC_NEW)))
        log = exe.with_suffix(".log")
        os.environ["PATH"] = f"{d}{os.pathsep}{os.environ['PATH']}"

        cc = CameraControls("/dev/video2")
        s = cc.state()
        check(s["method"] == "v4l2-ctl" and s["available"] and s["autofocus"] is True, "fake v4l2-ctl: state")
        s = cc.set(focus="123")
        check(s["autofocus"] is False and s["focus"] == 125 and not s.get("error"),
              "focus=123 -> autofocus off first, then 125 (step 5)")
        check(log.read_text().split() == ["focus_automatic_continuous=0", "focus_absolute=125"], "order of set-ctrls")
        s = cc.set(focus=9999)
        check(s["focus"] == 250, "focus clamped to max")
        s = cc.set(autofocus="1")
        check(s["autofocus"] is True and s["focus"] == 250, "autofocus=1")
        s = cc.set(autofocus="1", focus="40")
        check(s["autofocus"] is True and s["focus"] == 250, "autofocus=1 with focus: focus ignored")
        s = cc.set(exposure=600)
        check(s["exposure"]["auto"] is False and s["exposure"]["value"] == 600, "exposure=600 -> manual 600")
        s = cc.set(auto_exposure=1)
        check(s["exposure"]["auto"] is True and s["exposure"]["value"] == 600, "auto_exposure=1 -> back to 3")
        s = cc.set(autofocus="maybe")
        check("error" in s and s["available"], "bad value -> error in the JSON, state still there")
        line = cc.describe()
        check(line.startswith("camera focus via v4l2-ctl on /dev/video2") and "0-250 step 5" in line,
              f"startup line: {line}")

        exe.with_suffix(".json").write_text(json.dumps(parse_ctrls(NO_FOCUS)))
        s = cc.set(focus=10)
        check(not s["available"] and "not available" in s["error"], "no focus control -> error, no crash")

        exe.write_text("#!/bin/sh\necho 'Cannot open device /dev/video2' >&2; exit 1\n")
        s = cc.state()
        check(not s["available"] and "Cannot open device" in s["reason"], f"v4l2-ctl failing: {s['reason']}")


def test_unavailable():
    os.environ["PATH"] = "/nonexistent"
    s = CameraControls("/dev/video2").state()
    check(not s["available"] and "v4l2-ctl not installed" in s["reason"], "no v4l2-ctl, no capture")
    s = CameraControls(None, None, why_not="a video file, not a camera").set(focus=3)
    check(not s["available"] and s["error"] and s["method"] is None, "video file: unavailable, error on set")
    check(camera_ctl.video_device("bees.mp4") is None and camera_ctl.video_device("/dev/video2") == "/dev/video2",
          "video_device")


if __name__ == "__main__":
    test_parser()
    test_snap()
    path = os.environ["PATH"]
    test_with_fake_v4l2ctl()
    os.environ["PATH"] = path
    test_unavailable()
    print("all camera_ctl checks passed")
