# ArduBeeno

> Off-grid Edge AI on Arduino Uno Q that detects deadly Varroa mites in real time—protecting remote beehives without cloud connectivity.

**Junction 2026: European Tech Renaissance** · Track: *The Green European Renaissance* · Tech challenge: *Best use of Arduino Uno Q*

## The Problem: A Silent Threat to Global Food Security

Roughly **75% of global food crops** depend on animal pollination, sustaining over **8 billion people** and **$577 billion** in annual agricultural output. Yet honeybee populations face an existential biological threat: the parasitic mite ***Varroa destructor***.

*Varroa* mites attach to honeybees, feed on their vital fat bodies, weaken their immune systems, and transmit lethal pathogens such as Deformed Wing Virus (DWV). Without timely detection and intervention, infested colonies suffer up to **95% collapse rates within 3 to 4 years**.

## The Infrastructure Gap: Why Cloud AI Fails in the Field

Modern computer vision can spot *Varroa* mites, but existing solutions rely on cloud processing. In reality, **most beehives are located in remote meadows, forests, and rural fields with zero cellular or Wi-Fi connectivity**. Streaming continuous high-resolution video to the cloud is impossible off-grid, prohibitively expensive, and drains battery reserves far too quickly for remote solar-powered setups.

## What We Built: A Guard That Detects the Mites

ArduBeeno is an autonomous, real-time **Edge AI intruder detection system** running entirely on an **Arduino Uno Q** paired with a compact camera mounted at the beehive entrance.

By leveraging the Arduino Uno Q's hybrid architecture (a high-performance AI processor combined with a real-time STM32 microcontroller), our computer vision model inspects bees entering and exiting the hive locally in milliseconds—identifying *Varroa* mites directly on the edge with **zero internet connection required**.

### How It Works

1. **Capture:** A macro camera positioned at the hive landing board captures frames of returning forager bees.
2. **On-Device Inference:** Our lightweight vision model runs locally on the Arduino Uno Q to detect individual bees and spot parasitic *Varroa* mites on their bodies.
3. **Infestation Tracking:** The microcontroller tracks detection counts over time to calculate colony infestation rates and filter out false positives.
4. **Actionable Alerts:** When mite density crosses a critical threshold, the system logs the event and triggers a local alert so beekeepers can intervene immediately.

## Key Benefits

- **100% Off-Grid & Solar-Ready:** Operates reliably in remote forests and fields without cellular networks, cloud subscriptions, or high power draw.
- **Early Intervention:** Detects outbreaks weeks before visual symptoms appear, preventing winter colony collapse.
- **Precision Beekeeping:** Replaces blanket chemical miticide treatments with targeted, data-driven intervention—keeping hives and honey healthier.

## Real-World Pilot at 42 Heilbronn

We are a team of alumni and staff from **42 Heilbronn**, a peer-to-peer coding school in Germany. Our campus is home to **active beehives right outside our building**, giving us a living, real-world testbed to deploy, validate, and protect local bee colonies with self-built Edge AI.

## Screen-to-camera training data (`make record`)

`scripts/record.py` shows dataset images full-screen (black background), films the screen with a USB camera,
and saves the video plus settled still frames with YOLO labels (0 bee, 1 varroa) mapped into camera
coordinates, so a model can be fine-tuned on what a real camera sees of the screen.

```sh
make record-cameras                       # list screens and cameras (+ snapshots in captures/cameras/)
make record SRC=data/test CAMERA=1 REC_ARGS="--limit 300 --shuffle --duration 1.5"
make record SRC=data/test CAMERA=1 REC_ARGS="--tile 3x2 --only mite"     # 6 crops per screen = multi-bee scene
make record SRC=datasets/<name> CAMERA=1 REC_ARGS="--split val --screen 1"  # YOLO folder or its yaml
make record-mendeley CAMERA=1 REC_ARGS="--duration 1.5"   # Mendeley landing-board val split, 1,410 frames, shuffled
make record CAMERA=sim REC_ARGS="--headless --limit 10"                  # dry run with a simulated camera
```

- **macOS camera permission:** run it from your own Terminal/iTerm; the first run asks for Camera access
  (System Settings > Privacy & Security > Camera, enable the terminal app, then restart it). Without it
  the camera delivers no frames.
- **Setup:** point the camera at the screen so it sees all of it, focus, and lock exposure if the camera allows.
  The run starts with an ArUco marker screen; it fits the screen -> camera homography from it
  (`calib_start.jpg` shows what was detected). It recalibrates every `--recalib-every` slides, when the
  camera view stops matching the slides, on key `c`, and at the end, and warns if the camera moved.
  Keys: `Esc`/`q` quit.
- **Small bees:** images keep their aspect ratio (letterboxed on black). The run prints the median bee size in
  camera pixels (and warns below `--min-bee-px`, default 40). Mendeley bees are ~50–100 px at 1080p, so the screen
  should fill most of a 1080p+ camera frame. Missing YOLO label files count as frames without bees.
- **Frames:** per slide, `--stills` frames after `--settle` seconds, only once the rectified camera view matches
  the new slide and is steady (no transition blends).
- **Output** in `captures/screen-<time>/`: `video.mp4`, `images/`, `labels/` (YOLO), `data.yaml`, `manifest.csv`
  (source image(s), their gt_one.csv flag or label file, camera frame, time, calibration, match score, box counts),
  `calibration.json`. Single-bee crops get a bee box covering the whole crop.
- Sources: a data split with `gt_one.csv`, a YOLO dataset (class names bee/varroa/mite are recognised, others kept,
  `--class-map` to rename), or a folder of plain images. `--camera` also takes a video file or `none` (slideshow only).
