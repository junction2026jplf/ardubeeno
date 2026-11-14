# ArduBeeno Varroa: live preview on the UNO Q

Live Varroa mite detection on a USB camera plugged into the Arduino UNO Q. A web page shows the camera with:
- the centre region that goes to the model, in cyan (the 4:7 shape of the training crops, so fill it with one bee)
- every detected mite as a red box with its score, also listed with its position next to the video
- the verdict: MITE when `mean(classifier mite probability, top box score) >= threshold` in at least 2 of the last
  10 frames (you can change both in the page)

The LED matrix on the board shows how many mites were counted (a tick while none). The count goes up once per bee
that turns the verdict to MITE, not once per frame; reset it with the button in the page. While a mite is detected,
a mite icon alternates with the number.

By default it uses the files and thresholds in `python/deploy.json` (copied by `make applab` from `export/deploy/`):
the camera-fine-tuned detector and the letterbox-trained 288 px classifier, both int8 (`varroa.onnx`,
`varroa_cls.onnx`; at the deployed sensitive point camera test F1 0.859, recall 0.926, original test F1 0.873). `MODELS` overrides the model files (see Speed below).

## All bees or one bee

- **All bees** (when `python/bees.onnx` is present, copied by `make applab` from `export/bees/`): every bee in the
  1280x720 frame is found and followed from frame to frame. The page counts bees in view, bees observed, bees
  checked and bees with mites, and shows each checked bee's crop with the mite boxes. Bees shorter than the
  minimum size (default 100 px) or cut off at the edge are counted but not checked, because the mite models
  need bees about as large as their 160x280 training crops. A bee that leaves and comes back counts again.
- **Centre bee**: only the cyan centre region is checked, for a camera close enough that one bee fills it.

Switch between them in the page, or start `serve.py` with `--mode multi` / `--mode single`.

## Speed on the UNO Q

The board's 4 Cortex-A53 cores are the limit, so the multi-bee mode is tuned for them (all settable in `serve.py`,
or as environment variables in App Lab):

- **Bee detector** `bees.onnx`: YOLO11n trained at 416, exported at 416x256 for 16:9 frames (`BEE_MODEL` in the
  Makefile picks another file; the input size comes from the file). It runs every 3rd frame (`--det-every`,
  `DET_EVERY`); in between, the boxes move on by their velocity (pixels per camera frame, so the board's ~0.55 s
  per detector run doesn't matter).
- **Tracking**: each detection continues the track whose predicted box it overlaps (`--match-iou` / `MATCH_IOU`,
  0.1) or whose predicted centre is within `--match-dist` / `MATCH_DIST` box sizes (1.5, times 1 + detector runs
  missed); a track with a single detection may be as far as a bee moving 0.3 box sizes per frame gets. A lost bee
  keeps its ID for `--lost-s` / `LOST_S` seconds (1.5) and `--lost-runs` / `LOST_RUNS` detector runs (3), whichever
  is longer; one predicted to have left the frame ends at once. Each ended bee is one passage.
- **Mite verdict**: MITE after `--min-hits` / `MIN_HITS` positive checks (2), or after one check whose fused score
  is at least `--strong` / `STRONG` (default `cls_hi` with the cascade, else fuse_conf + 0.25). Bees get their first
  two checks first, from their first detection on. The crop's context around the bee box is `--crop-margin` /
  `CROP_MARGIN` (0.15), or `crop_margin` in a `bees.json` next to `bees.onnx` (the Makefile copies `foo.json`
  next to `BEE_MODEL=foo.onnx`), so the margin travels with a bee detector whose boxes are tighter.
- **Mite check**: classifier first (cascade): below `cls_lo` the bee is clean, from `cls_hi` it is a mite, and only
  in between does the mite detector decide too (deploy.json: 0.20 / 0.73, the detector is skipped for 89% of camera-test bees;
  without deploy.json: 0.31 / 0.60; `--cls-lo`, `--cls-hi`; `--no-cascade` runs both always). At most 2
  bee crops wait for the check at a time, fewest checks first (`--max-checks`, `MAX_CHECKS`); the checks run in a
  thread next to the bee detector.
- **Models**: by default the files and thresholds in `deploy.json` (fuse 0.21, cascade 0.20 / 0.73). `--models` /
  `MODELS` picks an older set from `export/fast/` instead: `c160-int8all` (int8, 160 px classifier, board F1 0.926
  on the 200 sample bees; the fallback without deploy.json), `int8-qdqall` (288 px classifier), `deployed`
  (`varroa.onnx` + `varroa_cls.onnx`, i.e. whichever pair `make applab` copied). deploy.json's thresholds only go
  with its own files (so `deployed` keeps them); the other sets use 0.38 / 0.31 / 0.60. `--fuse-conf`, `--cls-lo`,
  `--cls-hi` (or the page) override either.
- **Optional, off by default** (the defaults are the settings confirmed on the board): `--orient` / `ORIENT=1` turns
  diagonal bees upright for the mite check (orient.py); `--cut 12` / `CUT=12` ends all tracks when a new slide of the
  tile slideshow appears, so bees at the same spots aren't merged across slides. `bees_n416synthft` (export/bees/, with
  its bees.json margin 0.21) finds rotated bees, but tracked less steadily on the board than `bees_n416cam`.
- **Threads**: 2 per model (`--threads`, `--bee-threads`, `THREADS`, `BEE_THREADS`); 4 is 2-3x slower on the board.
- The stream runs apart from the model: every camera frame is shown with the latest results drawn on top (bee boxes
  move on by their velocity between results), so a slow model lowers the result rate, not the stream's. It is
  only drawn and JPEG-encoded while a page shows it (about 1.5-2 ms per 1280x720 frame on a Mac). `--stream-fps`
  / `STREAM_FPS` caps it (default 30), `--stream-width` / `STREAM_WIDTH` scales it down, `--jpeg-quality` /
  `JPEG_QUALITY` (75). `serve.py` prints a status line with the per-stage times and the stream's rate every 30 s;
  the page's `/state` has the same under `timing` and `stream`.

## Run it in Arduino App Lab

1. From the repo root, copy the models into the app: `make applab`.
2. Copy the app to the board. App Lab keeps apps in `/home/arduino/ArduinoApps/` on the board:
   `make applab-push BOARD=arduino@<board-ip>` (or `scp -rp applab/ardubeeno arduino@<board-ip>:ArduinoApps/`).
3. Plug the USB camera into the board (through a powered USB-C hub if the board also needs power or other
   devices), open App Lab, pick **ArduBeeno Varroa** under My Apps and press Run.
   On the first start, `onnxruntime` is installed from `python/requirements.txt`, which takes a minute.
4. Open the app's page: App Lab shows the link when the app runs, or open `http://<board-ip>:7000` on your laptop.
   Over a direct cable the board's name (e.g. `ardu-bee-no.local`) is more reliable than its 169.254.x.x address,
   which a Mac may route over Wi-Fi instead.

Another camera: set the `CAMERA` environment variable in the app, e.g. `csi:0` or `/dev/video2` (default `usb:0`).

## Run it without App Lab

The same page and model, without the LED matrix. It needs `numpy`, `opencv-python-headless` and `onnxruntime`:

- On the UNO Q over SSH: copy the folder (step 1 and 2), then
  `cd ~/ArduinoApps/ardubeeno/python && python3 serve.py`, and open `http://<board-ip>:7000`.
- On your laptop with its webcam: `make applab-local` from the repo root, then open http://localhost:7000.
  `python serve.py --camera some_video.mp4` plays a video file instead.

## Camera focus

**Settings & details → Camera focus** switches the camera between autofocus and manual focus, with a slider over
the camera's own range (and exposure, when the board can set it). With the panel closed, `f` toggles auto/manual
and `[` / `]` move the focus by one step. The server uses `v4l2-ctl` on the camera's `/dev/videoN` when it is
installed (`sudo apt install v4l-utils`), else OpenCV on the open capture, where the range is a guess. To see what
the camera offers: `ssh arduino@<board-ip> 'v4l2-ctl -d /dev/video2 --list-ctrls'`.
`serve.py --focus 120` starts with manual focus, and `--autofocus` starts with autofocus on. In App Lab, set `FOCUS` or `AUTOFOCUS=1`.
`GET /camera` returns the state, and `/camera?autofocus=0|1&focus=N` sets it (`auto_exposure`, `exposure` work the same way).

## Saving frames for fine-tuning

**Save this frame** in the page writes the cyan region, without drawings, to `captures/` in the app folder.
Label a few hundred of those from your own hive camera (mite / clean) to recalibrate the threshold
(`device_check.py calibrate`) or to fine-tune the model. See [docs/RESULTS.md](../../docs/RESULTS.md), "On a new camera".

## Files

| File | What it does |
|---|---|
| `app.yaml` | App Lab app definition (uses the `arduino:web_ui` brick) |
| `python/main.py` | App Lab entry point: camera, web routes, LED matrix updates |
| `python/monitor.py` | Shared: runs the model on the newest frame, aggregates frames, draws the results on every camera frame for the MJPEG stream |
| `python/serve.py` | The same preview as a plain Python server (no App Lab) |
| `python/camera_ctl.py` | Camera focus/exposure via v4l2-ctl or OpenCV (`/camera`) |
| `python/device_check.py`, `*.onnx`, `deploy.json`, `bees.json` | Build outputs: copied by `make applab` from `scripts/` and `export/`, .gitignored (edit `scripts/device_check.py`, not the copy) |
| `assets/index.html` | The page: live stream, verdict, mite list, settings, frame saving |
| `sketch/sketch.ino` | LED matrix: blinking mite or tick, set from Python over the Bridge |
