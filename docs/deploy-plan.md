# Deploy plan: ArduBeeno Varroa on the Arduino UNO Q

Tick each box when done. "command follows" means the exact command depends on work that is still in progress; fill it in when it lands.

Addresses in our setup (USB link; replace them with your board's and your laptop's IPs):

- Board: `arduino@169.254.100.2`, app page `http://169.254.100.2:7000`
- Mac (receiver/dashboard): `http://169.254.125.210:8000`

App settings that exist today (set in the App Lab app's environment, read by `applab/ardubeeno/python/main.py`): `CAMERA` (default `usb:0`), `MODELS` (default: the files in `deploy.json` when present, i.e. the camera fine-tune `cam160-int8all`; set it to `c160-int8all`, `int8-qdqall` or `deployed` to use an older set instead), `THREADS` (default 2), `BEE_THREADS` (default 2), `DET_EVERY` (default 3).

## Phase 1: integration on the Mac

- [ ] Merge-check the two-step pipeline change (bee detector, then per-bee mite check).
- [ ] Merge-check the uploader (`applab/ardubeeno/python/outbox.py`).
- [ ] Merge-check the new models in `export/deploy/` (`varroa.onnx`, `varroa_cls.onnx`, `deploy.json`).
- [ ] The three changes work together: no conflicts, the app starts.
- [ ] Local test run: `make applab-local` (webcam preview at `http://localhost:7000`). Full test command: command follows.
- [ ] Build the app bundle: `make applab`. Check that it prints "mite models from export/deploy/".
- [ ] Review the bundle in `applab/ardubeeno/python/`: `varroa.onnx`, `varroa_cls.onnx`, `deploy.json`, `bees.onnx`, int8 files if wanted; no stray captures or data.
- [ ] Board config file `demo/ardubeeno.env` exists (already created) with `ARDUBEENO_SERVER_URL`, `ARDUBEENO_DEVICE_ID`, `ARDUBEENO_DEVICE_TOKEN`.
- [ ] `demo/ardubeeno.env` is not in git (token inside).

## Phase 2: on the board

### Benchmarks

- [ ] Board benchmark of the new models (`cam160-int8all`): commands in [export/deploy/README.md](../export/deploy/README.md#board-benchmark). The old sets are already measured: `export/fast/board/board_speed.txt`, `board_accuracy.txt`.
- [ ] Record int8 vs fp32 speed and accuracy, and 1 vs 2 threads.
- [ ] **Decision: int8 or fp32** (sets `MODELS`) and **1 or 2 threads** (sets `THREADS`). Never 4.
- [ ] Bee detector speed per frame on the board: command follows.
- [ ] **Decision: how often the bee detector runs** (sets `DET_EVERY`, default every 3rd frame).

### Deploy

- [ ] Deploy: `make applab-push BOARD=arduino@169.254.100.2`
- [ ] Copy the config to the board (owner-only permissions) and check the receiver from there: `make applab-config BOARD=arduino@169.254.100.2`
- [ ] Set the decided `MODELS`, `THREADS`, `DET_EVERY` in the app.
- [ ] Start the app in App Lab (My Apps > ArduBeeno Varroa > Run). The first start installs `onnxruntime` and takes about a minute.
- [ ] Page loads at `http://169.254.100.2:7000`, camera image visible, LED matrix shows the tick.

### Dashboard link

- [ ] Start the receiver on the Mac: `demo/start-receiver.sh`
- [ ] Dashboard opens on the Mac at `http://127.0.0.1:8000`.
- [ ] From the board: `curl http://169.254.125.210:8000/health` returns `{"status":"ok","schema_version":1}`.
- [ ] App log shows no "upload off" or "HTTP 401/403" lines.
- [ ] A bee shown to the camera appears on the dashboard.

### Persistence

- [ ] Restart the app: config still read, outbox kept, pending events sent afterwards.
- [ ] Reboot the board: same checks after it comes back.
- [ ] Stop the receiver, show bees, start it again: the queued events arrive.

## Phase 3: demo readiness

- [ ] Live test: `demo/bee_no_mite_large.png` shown full-screen, board says clean.
- [ ] Live test: `demo/bee_with_mite_large.png` shown full-screen, board flags the mite, mite box shows, LED count goes up.
- [ ] Live test: multi-bee slideshow, bee count plausible. Command for the demo slideshow: command follows.
- [ ] Live test: landing-board images.
- [ ] Threshold tuning: fuse threshold (`deploy.json` or the page), cascade limits, min hits; write down the final values.
- [ ] Venue setup done: see [demo-venue-checklist.md](demo-venue-checklist.md).
- [ ] Backup video of the board working recorded and on the laptop.
- [ ] Rehearsal with the full script ([demo-script.md](demo-script.md)), timed, at least twice; once with the fallback.

## Phase 4: optional

- [ ] Smaller bee detectors with vertical flips (training on Verda now: `bees-n416-flip`, `bees-n320-flip`). Export and compare: command follows.
- [ ] Bee detector fine-tune on landing-board camera footage. Needs a new recording of the mendeley **train** split; the existing recording `captures/screen-20261114-123026` shows its val split. `make record-mendeley` records the val split; train split command follows.
- [ ] Uploader: fix the double count at reset.
- [ ] Uploader: page totals from committed events.
- [ ] RESULTS.md and pitch numbers (fill the TBDs in the demo script).
- [ ] Repo cleanup and commit.

## Decision points

| Decision | Based on | Choice |
|---|---|---|
| int8 vs fp32 | Phase 2 board benchmark (speed vs F1) | |
| 1 vs 2 threads | Phase 2 board benchmark | |
| Bee detector run interval | Phase 2 detector speed per frame | |
| Bee detector retrain or not | Phase 3 live tests on multi-bee and landing-board images | |
| Which optional items | Time left after Phase 3 | |
