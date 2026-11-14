# Training the ArduBeeno Varroa models

This page explains how the models on the Arduino UNO Q were trained, evaluated and exported, and which command
runs each step. All paths are relative to the repo root. Numbers come from `RESULTS.md`,
`export/deploy/deploy.json` and `export/fast/board/board_*.txt`.

## The pipeline on the board

The UNO Q (4x Cortex-A53, onnxruntime on the CPU) films bees through a USB camera. In the demo it films a laptop screen.

```
camera frame (1280x720)
   |
   v
[1] bee detector  bees.onnx (YOLO11n, class "bee")
   |
   v
[2] tracker       follows each bee from frame to frame
   |
   v
[3] crop_bee      cuts each bee out as a 4:7 portrait crop (160x280)
   |
   v
[4] mite check (cascade)
      classifier varroa_cls.onnx (YOLO11n-cls, 160 px) -> mite prob p
        p <  cls_lo                -> clean (mite detector skipped)
        p >= cls_hi                -> mite
        otherwise                  -> mite detector varroa.onnx (YOLO11n, 320x192)
                                      mite if mean(top box score, p) >= fuse_conf
   |
   v
[5] a bee becomes MITE after min_hits positive checks (default 2)
   |
   v
LED matrix, web page (port 7000), dashboard API (outbox)
```

Deployed thresholds (`export/deploy/deploy.json`): `fuse_conf` 0.16, `cls_lo` 0.02, `cls_hi` 0.30.

## 1. Data

| Data | Where | What it is |
|---|---|---|
| Varroa bee crops | `data/{train,val,test}/` | 13,509 bee crops (160x280) cut from 2017 hive-entrance videos. Mite boxes in each split's `gt_one.csv` (flag 1 = visible mite, 3 = mite under or through a wing). Split by video: train 8,225 (6 videos), val 1,876 (5), test 3,408 (2). |
| YOLO / classifier versions | `yolo/`, `yolo_cls/` | Built by `scripts/convert.py` (symlinks, one class `varroa`) and `scripts/convert_cls.py` (mite/clean folders, crops letterboxed to 280x280 squares). |
| Bee-detector data | `datasets/beehive/`, `datasets/mendeley_landing_board/` | BEEHIVE close-ups (640x640, bees about 420 px long) and Mendeley landing-board frames (1920x1080, bees about 77 px). Combined in `datasets/bees_combined.yaml`; the landing-board val split is two held-out hives. |
| Camera-on-screen recordings | `captures/screen-*/` | Dataset images shown full-screen and filmed by the camera, with the labels mapped into camera coordinates. |
| Synthetic bees | `datasets/synth_bees/` | In progress, see below. |

**Caveat on the splits.** Train and val share recording sessions, so val scores are optimistic. Test (two
videos not used anywhere else) is the honest split. All thresholds are chosen on val and then applied once to test.

### Camera-on-screen recordings

`scripts/record.py` (via `make record`) shows images full-screen, films them, and saves settled still frames plus
YOLO labels (0 bee, 1 varroa). It starts with an ArUco marker screen to fit the screen-to-camera homography and
recalibrates during the run. See the "Screen-to-camera training data" section of `README.md` for all options.

```sh
make record-cameras                                     # list screens and cameras
make record SRC=data/test CAMERA=1 REC_ARGS="--tile 3x2 --only mite"
make record-mendeley CAMERA=1                           # Mendeley landing-board val split
```

`scripts/camera_crops.py` turns the recordings into mite training data. It cuts every bee out with the app's own
`crop_bee` (imported from `applab/ardubeeno/python/bees.py`), so training sees what the board sees:

```sh
.venv/bin/python scripts/camera_crops.py   # defaults: train/val/test = captures/screen-20261114-124423 / -130127 / -122148
```

| Output | Contents |
|---|---|
| `yolo_cam/` | Mite-detector crops (train 8,213, val 1,876, test 3,600) |
| `yolo_cam_cls/` | Classifier crops, mite/clean, raw 160x280 |
| `yolo_cam_mix/` | Classifier set: original train + camera train; val/test camera only |
| `varroa_cam.yaml` | Detector set: original train+val + camera train; val/test camera only |
| `yolo_cam_cls_sq/`, `yolo_cam_mix_sq/` | Letterboxed 280x280 variants of the two classifier sets (not used for the deployed models) |

For the bee detector, camera frames were also turned into `datasets/cam_tiles/` (tiled crop slides, bees about
320-480 px) and `datasets/cam_landing/` (landing-board recordings, bees about 55 px), combined in
`datasets/bees_combined_cam.yaml`. No script in `scripts/` builds these two folders.

### Synthetic bees (in progress)

`scripts/synth_bees.py` is writing `datasets/synth_bees/` right now (planned 6,000 / 600 / 600 images; at the
time of writing the train images are still being written and val/test are empty). It pastes cut-out bee crops onto 1280x720
backgrounds at any angle, flip and size, with a feathered elliptical mask, and adds screen-filming artefacts.
Boxes are the tight axis-aligned box of the paste ellipse. Bee crops keep their split (synthetic test uses only
`data/test` crops). A dataset yaml for it does not exist yet.

## 2. Models and why

| Choice | Why |
|---|---|
| Mite detector + classifier, fused | The mean of the detector's top box score and the classifier's mite probability beats either alone: test F1 0.901 vs 0.880 for the first detector-only model. Bigger classifiers (ConvNeXt, DINOv2, YOLO11s) did not do better; the limit is the data (mites under wings). |
| Classifier-first cascade | The classifier is cheap. It decides confident bees alone, and the detector runs only on the uncertain band. On camera test the detector is skipped for 86% of bees at a cost of about 0.01 F1. |
| 160 px classifier | On camera test the 160 px pair ties the 288 px pair (0.879 vs 0.874 fused) at about 5x less classifier time on the board. |
| int8 QDQ-all | Static onnxruntime quantization of all op types (detector head kept in float), calibrated on 400 camera crops. Costs at most 0.004 F1 and is the fastest set on the board. |
| Bee detector YOLO11n at 416, exported 256x416 | Finds bees on a 16:9 frame without letterbox bars; runs every 3rd frame, the tracker fills the gaps. |

## 3. Training steps

### Local (Mac, `DEVICE=mps` by default)

| Command | What it does |
|---|---|
| `make setup` | Creates `.venv` (Python 3.12, uv) and installs `requirements.txt` |
| `make convert` | `scripts/convert.py`: `data/` to `yolo/` + `varroa.yaml` |
| `make smoke` | 1 epoch on 5% of the data, to check the setup |
| `make train` | `scripts/train.py` mite detector (`EPOCHS=60 IMGSZ=320 BATCH=64 NAME=varroa`) |
| `make finetune` | Fine-tunes `WEIGHTS` on `FT_DATA` (default `varroa.yaml`) for `FT_EPOCHS` (default 15), AdamW, lr0 3e-4, cosine. Camera fine-tune: `make finetune FT_DATA=varroa_cam.yaml FT_EPOCHS=25 BATCH=128 WEIGHTS=...` (the run in section 4 also used `--close-mosaic 5 --patience 0`) |
| `make eval` | `scripts/evaluate.py`: box metrics and per-bee accuracy on test |
| `make export` / `make export-cls` | `scripts/export.py` to `export/varroa.onnx` / `export/varroa_cls.onnx` (`EXPORT_IMGSZ=320,192` for the rectangular detector input, `CLS_IMGSZ` default 288). The deployed 160 px set is built by `scripts/deploy_build.py` (section 6) |

The classifier has no Makefile target:

```sh
.venv/bin/python scripts/convert_cls.py                       # yolo/ -> yolo_cls/ (280x280 letterboxed)
.venv/bin/python scripts/train_cls.py --imgsz 160 --name cls-n160
```

### Remote GPU (Verda)

The GPU runs were done through a separate assistant thread. It rents Verda GPU instances, uploads the data
(symlinks dereferenced, since `yolo/` links into `data/`), runs the same `scripts/train.py` / `scripts/train_cls.py`
command lines with a CUDA `--device`, copies each run back to `runs/verda/<task>/<name>/` and deletes the instances.
There is no script for this in the repo.

### Hyperparameters of the deployed runs (from each run's `args.yaml`)

| Run | Start from | Data | Epochs | imgsz | Batch | Optimizer, lr0 |
|---|---|---|---|---|---|---|
| `detect/n320-tv-e40-screen` (base detector) | `yolo11n.pt` | `varroa_trainval.yaml` | 40 | 320 | 128 | AdamW, 1e-3, cosine |
| `classify/cls-n160` (base classifier) | `yolo11n-cls.pt` | `yolo_cls` | 60 | 160 | 256 | AdamW, 1e-3, cosine |
| `detect/det-cam-ft` (deployed) | n320-tv-e40-screen `last.pt` | `varroa_cam.yaml` | 25 | 320 | 128 | AdamW, 3e-4, cosine |
| `detect/det-cam-ft-tr` (threshold twin) | n320-screen `last.pt` | `varroa_cam_tr.yaml` | 25 | 320 | 128 | AdamW, 3e-4, cosine |
| `classify/cls-cam-ft160` (deployed) | cls-n160 `best.pt` | `yolo_cam_mix` | 25 | 160 | 256 | AdamW, 3e-4, cosine |
| `bees/bees-n416` (bee detector) | `yolo11n.pt` | `datasets/bees_combined.yaml` | 100 (patience 20) | 416 | 128 | auto, Ultralytics defaults |

Augmentation for the mite models (set in `train.py` / `train_cls.py`): HSV 0.02 / 0.5 / 0.4, rotation 15 degrees,
vertical and horizontal flips 0.5, RandAugment. Detector: scale 0.3, mosaic 1.0, erasing 0.4, and `--screen-aug`
(blur, noise, JPEG, downscale, exposure) for the "screen" runs. Classifier: scale 0.2 and no random erasing, so the
mite stays in the crop. The bee detector used Ultralytics defaults (no rotation, no vertical flip), so it was not
trained through `train.py`, whose augmentation is fixed.

## 4. Camera domain fine-tune

**Why.** A camera filming a screen adds blur, moire, colour shifts and perspective. The old deployed pair scored
F1 0.901 on the original test crops but only 0.725 on camera test crops. After the fine-tune: 0.877 (int8, fused).

**How.**

1. Record train, val and test splits from `data/train`, `data/val` and `data/test` with `make record`.
2. Cut out the crops with `scripts/camera_crops.py` (same `crop_bee` as the app).
3. Fine-tune from the deployed weights on original + camera crops, AdamW lr0 3e-4, 25 epochs:

```sh
.venv/bin/python scripts/train.py --model runs/verda/detect/n320-tv-e40-screen/weights/last.pt \
  --data varroa_cam.yaml --epochs 25 --batch 128 --imgsz 320 --optimizer AdamW --lr0 0.0003 --cos-lr \
  --close-mosaic 5 --patience 0 --device 0 --name det-cam-ft
.venv/bin/python scripts/train_cls.py --model runs/verda/classify/cls-n160/weights/best.pt \
  --data yolo_cam_mix --imgsz 160 --epochs 25 --batch 256 --lr0 0.0003 --device 0 --name cls-cam-ft160
```

4. **Thresholds without leakage.** `det-cam-ft` was trained on original train+val, so its val scores are useless
   for picking thresholds. A twin, `det-cam-ft-tr`, is trained the same way on train only (`varroa_cam_tr.yaml`),
   and its camera-val scores choose `fuse_conf` and the cascade band. Those thresholds are then applied to the real
   detector on test.

The price: on the original crops F1 drops from 0.901 to 0.849 at the camera threshold.

## 5. Evaluation

| Script | Use |
|---|---|
| `scripts/evaluate.py` | One detector: box metrics, per-bee F1 over thresholds, mite counts (`make eval`) |
| `scripts/evaluate_cls.py` | One classifier, optionally fused with a detector (`--det-weights`) |
| `scripts/evaluate_ens.py` | Classifier sets fused with a detector; thresholds on val, applied to test (`--det-weights-val` for a train-only twin) |
| `scripts/evaluate_tracks.py` | Multi-frame aggregation per bee on the external EV2 videos |
| `scripts/deploy_build.py` | The deployed set: `export`, `score`, `table` (thresholds on camera val, then camera test + original test), `quantize`. onnxruntime CPU with the board's pre-processing |
| `export/fast/board/fastbench.py` | On the board: `speed` (ms per bee) and `accuracy`. Commands in `export/deploy/README.md` |

Key numbers:

| Measure | Value | Source |
|---|---|---|
| Original test F1 (3,408 bees), old pair at 0.38 | 0.901 | RESULTS.md |
| Original test F1, deployed set (int8, fused / cascade) | 0.850 / 0.849 | deploy.json |
| Camera test F1 (3,600 bees), old pair at 0.38 | 0.725 | deploy.json |
| Camera test F1, deployed set (int8, fused / cascade) | 0.877 / 0.865 | deploy.json |
| Detector skipped by the cascade, camera test | 86% | deploy.json |
| Board, `c160-int8all`, 2 threads, batch 1: ms per bee, both models | 115.5 ms | board_speed.txt |
| Board, same, cascade estimate (detector on 4.3% of bees) | 28.9 ms | board_speed.txt |
| Board accuracy, `c160-int8all`, 200 sample bees, F1 at 0.38 | 0.926 (detector skipped 95%) | board_accuracy.txt |

The board files measure `c160-int8all`: same sizes and quantization as the deployed set, but the weights from
before the camera fine-tune. The camera-tuned set has not been benchmarked on the board yet; the commands are in
`export/deploy/README.md`.

## 6. Export and deploy

```sh
.venv/bin/python scripts/deploy_build.py export
.venv/bin/python scripts/deploy_build.py score
.venv/bin/python scripts/deploy_build.py table
.venv/bin/python scripts/deploy_build.py quantize --det cam --cls cam160_last
```

This writes `export/deploy/varroa.onnx`, `export/deploy/varroa_cls.onnx`, fp32 and 288 px alternatives in
`export/deploy/alternatives/`, and `export/deploy/deploy.json` with the thresholds. The quantization recipe is the
same as `scripts/fast_build.py` (which builds the speed candidates in `export/fast/`).

```sh
make applab                                   # copies export/deploy/ models, deploy.json and bees.onnx into applab/ardubeeno/python/
make applab-push BOARD=arduino@<board-ip>     # copies the app to ~/ArduinoApps/ardubeeno on the board
make applab-config BOARD=arduino@<board-ip>   # uploads demo/ardubeeno.env (receiver URL, device ID, token)
make applab-local                             # same app on the laptop webcam, http://localhost:7000
```

Then start the app in App Lab, or over SSH without App Lab:

```sh
cd ~/ArduinoApps/ardubeeno/python && python3 serve.py      # page on http://<board-ip>:7000
```

Details: [applab/ardubeeno/README.md](../applab/ardubeeno/README.md) and the checklist in
[deploy-plan.md](deploy-plan.md).

## 7. Known limits and next steps

- **Public 2017 data.** All mite labels come from one public dataset of 2017 entrance videos. Mites under a wing
  are the main miss (recall about 0.55-0.65 vs about 0.95 for visible mites).
- **Bee detector at demo scale.** The bee datasets have very small (landing board) or very large (close-up)
  bees. The demo shows bees in between, so the bee detector is being retrained with the synthetic data.
- **Rotation.** `crop_bee` only turns bees in 90 degree steps. A bee lying diagonally gives a crop unlike the
  upright training crops.
- **Real hive footage is needed.** The camera fine-tune covers a camera filming a screen, not a real hive
  entrance. On foreign footage (EV2) scores shift down; `RESULTS.md` ("On a new camera") describes how to
  recalibrate with `device_check.py calibrate` and fine-tune on labelled crops from the target camera.
