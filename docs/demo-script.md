# Demo script (2–3 minutes)

Two people: **Speaker** talks, **Operator** drives the laptop. Numbers in [brackets] are placeholders; fill them before the demo or drop the line.

| Time | On screen | Say |
|---|---|---|
| 0:00–0:20 | Title slide, board and camera visible on the table | "Varroa destructor is the main threat to honeybee colonies. Today, to check a colony, a beekeeper takes a sample of bees and does an alcohol wash or a sugar roll. It is slow, and the alcohol wash kills the bees." |
| 0:20–0:35 | Photo or sketch: camera at the hive entrance | "Our idea: a small camera at the hive entrance that checks every bee walking in. It runs on the device. No cloud needed." |
| 0:35–0:55 | Laptop: `bee_no_mite_large.png` full-screen. Projector: app page on the board | "This camera is filming the laptop screen. The Arduino UNO Q finds the bee and checks it. This one is clean." (Point at the verdict and the tick on the LED matrix.) |
| 0:55–1:20 | Laptop: `bee_with_mite_large.png`. Projector: app page, red mite box | "Now a bee with a mite. The board flags it, draws a box around the mite, and the LED matrix counts it." (Point at the box, then at the matrix.) |
| 1:20–1:45 | Laptop: multi-bee slideshow. Projector: app page with bee counts | "At a real entrance there are many bees at once. The board finds each bee, follows it, and counts bees and mites." |
| 1:45–2:00 | Dashboard on the laptop (`http://169.254.125.210:8000/?source=screen_demo`) | "Results go to a dashboard: bees seen, bees with mites, over time. The board keeps its own record if the connection drops, and sends it later." |
| 2:00–2:20 | Slide: pipeline diagram | "The tech, in one breath: a YOLO11-nano bee detector, then for each bee a classifier-first cascade with a mite detector. Int8 on the board's four Cortex-A53 cores, fine-tuned on camera-through-screen footage to close the gap between photos and a real camera." |
| 2:20–2:40 | Slide: numbers and limits | "Filmed through a camera, it finds 93% of the bees with mites (F1 0.86; 0.88 on the original photos). On the board: a live view at 25 frames per second, about 0.15 seconds to check a bee, and in most checks the classifier alone decides. Limits: it is trained on a public dataset of entrance videos from 2017. The next step is real hive footage." |
| 2:40–2:50 | Back to the live page | "A mite check for every bee at the hive door, without opening the hive and without killing bees. Thank you." |

## Notes for the speaker

- Point at the board and the LED matrix, not only the projector.
- If a verdict is wrong live, say so: "That one is wrong; this is why we need real hive footage." Then move on.
- Do not read the numbers slide; say two numbers at most.

## 30-second fallback (if the board fails)

Operator plays the backup video of the board working, then switches to the dashboard with recorded data.

> "Varroa mites are the main threat to honeybees, and checking for them today means killing a sample of bees. We put a camera at the hive entrance and an Arduino UNO Q that checks every bee on the device, no cloud. Here it is working: clean bee, bee with a mite, the box, the LED count. Results go to this dashboard. Through a camera it finds 93% of the bees with mites, and it checks a bee in about 0.15 seconds on the board. Next step: real hive footage."

## Numbers behind the script (14 Nov 2026)

| What | Value | Source |
|---|---|---|
| Mite recall / precision / F1, camera test (3,600 bees, int8) | 0.926 / 0.801 / 0.859 | export/deploy/deploy.json |
| F1, original test images (3,408 bees) | 0.873 (int8) | export/deploy/deploy.json |
| Live board test, phone pictures | 5 of 5 mite bees flagged, 0 false alarms (gallery checked by eye) | board run, 14 Nov |
| Live view / processing on the board | ~25 FPS / ~5 FPS | serve.py status lines |
| Mite check per bee / bee detector per frame | ~140–165 ms / ~0.5 s | serve.py status lines |
| Mite detector skipped (classifier sure) | 45–90%, depending on what is shown | serve.py status lines |
| Speed-optimized model set (older, not deployed) | ~30 ms per bee | export/fast/board/board_speed.txt |

Dashboard: choose Source "Screen demonstration" (the link above does it); "Field observations" stays empty for the demo.
