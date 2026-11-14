# Varroa model: results of the overnight tuning (2026-11-13/14)

**Deployed models:** the camera fine-tune (`det-cam-ft` + letterbox-trained 288 px `cls-cam-sq288`, both int8,
sensitive point `fuse_conf` 0.21, cascade 0.20 / 0.73): see [Camera fine-tune](#camera-fine-tune-camera-on-screen-crops-2026-11-14) and
`export/deploy/deploy.json`. The sections before it measure the original pair: detector
`runs/verda/detect/n320-tv-e40-screen/weights/last.pt` (YOLO11n, 320 px, train+val, 40 epochs, screen augmentation)
+ classifier `runs/verda/classify/cls-n288/weights/best.pt` (YOLO11n-cls, 288 px), infested when
`mean(classifier mite probability, detector top box score) >= 0.38`. Its ONNX exports: `export/varroa.onnx`
(320×192) and `export/varroa_cls.onnx` (288); board bundle in `export/device/`.
Try it in the browser: `.venv/bin/python scripts/app.py`, then open http://localhost:8000. It has random test bees with
their labels, your own bee crops, and the webcam with frame aggregation, on the same ONNX pipeline as the board.
On the UNO Q: the App Lab app in `applab/ardubeeno/` streams the board's camera with the mite boxes to
http://<board-ip>:7000 and blinks the LED matrix on a mite (`make applab-push BOARD=arduino@<board-ip>`).

## Test set (3,408 bees from 2 unseen videos), thresholds chosen on val

| | Bees correct | Precision | Recall | F1 | Missed / false alarms |
|---|---|---|---|---|---|
| First model (YOLO11n, Mac, conf 0.25) | 93.5% | 0.899 | 0.861 | 0.880 | 131 / 91 |
| **Deployed pair (board-side ONNX pipeline)** | **94.6%** | **0.918** | **0.884** | **0.901** | **109 / 74** |

- Realistic expectation is **F1 ≈ 0.89–0.90**: four classifier seeds give 0.885–0.901 fused (mean ≈ 0.894).
- Mite counts: duplicate boxes were inflating counts by 38%. NMS IoU 0.45 (instead of 0.7) gives the exact count
  for 93% of bees.
- Speed on an M4 CPU: about 4.4 ms detector + 2.8 ms classifier per bee. On the UNO Q (old sets, not yet the camera
  fine-tune): `c160-int8all` cascade ~29–42 ms per bee at 1–2 threads, F1 0.926 on 200 sample bees
  (`export/fast/board/board_speed.txt`, `board_accuracy.txt`).

## What limits accuracy
- `gt_one.csv` flag **3 = mite under or through a wing**. Recall: visible mites ~0.95, under-wing mites ~0.55–0.65.
  Most remaining misses are under-wing mites that are barely visible in a single crop.
- Test video `2017-10-17_1-39-36` has 31% under-wing mites (train ~20%) and gets F1 ~0.81 vs ~0.92 on the other.
- Seed noise: ±0.006 test F1 (detector last.pt), ±0.01 (classifier). Differences smaller than that are not real.
- It's the data, not the model: classifiers up to ~30× larger, pretrained on ImageNet-22k or with DINOv2, land on
  the same fused F1 (see below).

## Tried and rejected (all within noise or worse on test)
Longer training (100 epochs), 416 px, YOLO11s, YOLO11s at 416, YOLO26n/s, oversampling under-wing bees ×3,
a separate under-wing class (detector and classifier), screen augmentation (neutral, kept for the demo),
train+val classifier, bigger (s) or larger-input (384) classifier, no RandAugment, weighted/stacked fusion,
classifier seed ensembles, flip TTA (+0.005, 2× cost), 6-detector ensemble (0.894), a weight-averaged
"soup" of 4 classifier seeds (collapses; after BatchNorm recalibration 0.895, like an average seed), and
**external data**: adding the EV2 dataset ([Zenodo 13771384](https://zenodo.org/records/13771384), CC BY 4.0,
641 subsampled bee crops) to the classifier lowered the 3-seed mean (0.885 vs 0.893 fused) and under-wing
recall. EV2's infested bees show clearly visible mites, the case we already get right.
**Much larger pretrained classifiers** (`scripts/train_timm.py`, same crops and protocol, fused with the detector;
baseline 0.901 in the same run): ConvNeXt-tiny IN-22k 0.901, DINOv2 ViT-S 0.900, ConvNeXt-base IN-22k 0.905
(+ YOLO-cls). **Geometry-only augmentation** (no colour changes, to keep the faint red-brown of under-wing mites):
3-seed mean 0.885 vs 0.893.
YOLO26n train+val + classifier ties the deployed pair (0.9005) and is ~10% faster on CPU; its weights are in
`runs/verda/detect/y26n-tv-e40-screen/`. All evaluation logs: `runs/verda/evals/`.

## On a new camera (e.g. the hive): measured on EV2 footage the model never saw
- **Scores shift down on foreign footage.** At our threshold only 27% of EV2 crops with a visible mite are
  flagged (0 false alarms), but separation is still good (AUC 0.89 fused, 0.84 classifier alone).
  Recalibrating the threshold to 1% false alarms on clean bees lifts that to 63% (5% false alarms: 71%).
  Mites hidden under a wing stay at about 0% in single frames, whatever the threshold.
- **Combining frames per bee helps:** flagging a bee when any of N frames is positive caught ~3× more infested
  EV2 bees at N=20 than single frames, more with spread-out frames (every 5th: 38–42%) than consecutive ones.
  Averaging scores over frames did not help. On our camera single frames do give some false alarms, so use
  "at least 2 hits in N" or a higher per-frame threshold.
- Steps: (1) label 50–100 crops of clean bees (and some with mites) from the new camera, then
  `python device_check.py --cls-model varroa_cls.onnx calibrate --clean <dir> --mite <dir> --fpr 0.01`;
  (2) run with the suggested `--fuse-conf` and frame aggregation, e.g. `live.py --window 10 --min-hits 2`;
  (3) if recall is still too low, fine-tune on your own labelled crops.
- Caveat: EV2's clean bees come from only 10 videos, so these numbers are indicative.
- More video data from the same project: the Vit4V paper (CVPR 2025 workshop) releases code, models and the
  "EV2 Video Dataset" via https://github.com/kernel-machine/vit4v (Kaggle, labels per video only,
  CC BY-NC-ND 4.0, Kaggle login needed). It's the same EV2 camera domain tested above, so it's mainly useful
  for multi-frame experiments, and its license may not suit a product.

## Camera fine-tune (camera-on-screen crops, 2026-11-14)
Detector and classifiers fine-tuned 25 epochs on original + camera train crops (`det-cam-ft`, `cls-cam-ft160|288`).
Thresholds and cascade chosen on camera val (train-only detector twins for the train+val detectors), then applied
to camera test (3,600 bees) and the original test (3,408). onnxruntime CPU, board pre-processing
(`scripts/deploy_build.py`, score cache in `export/deploy/work/`).

| Detector + classifier | thr | Camera test F1 (P / R) | Original test F1 | Cascade lo,hi → camera F1, det skipped |
|---|---|---|---|---|
| Deployed pair (base + cls288), at 0.38 | 0.38 | 0.725 | 0.901 | 0.31,0.60 → 0.724 |
| base + cls288, val-tuned | 0.23 | 0.810 (0.930 / 0.718) | 0.884 | 0.01,0.46 → 0.810, 66% |
| base + cls160, val-tuned | 0.35 | 0.774 (0.922 / 0.667) | 0.893 | 0.14,0.70 → 0.774, 82% |
| cam + cam288 (last) | 0.29 | 0.874 (0.926 / 0.827) | 0.882 | 0.23,0.57 → 0.867, 94% |
| cam + cam160 (last), deployed until 2026-11-14 | 0.16 | 0.879 (0.861 / 0.897) | 0.849 | 0.02,0.30 → 0.869, 86% |
| base + cam160 (last) | 0.23 | 0.847 (0.946 / 0.767) | 0.867 | 0.02,0.46 → 0.846, 84% |
| cam + base288 | 0.16 | 0.872 (0.869 / 0.875) | 0.866 | 0.00,0.32 → 0.872, 20% |
| cam + sq160 (last), max-F1 | 0.38 | 0.866 (0.882 / 0.850) | 0.887 | 0.37,0.73 → 0.865, 91% |
| cam + sq288 (last), max-F1 | 0.34 | 0.881 (0.872 / 0.889) | 0.883 | 0.33,0.73 → 0.880, 93% |
| cam + sq160 (last), sensitive (max F2) | 0.13 | 0.817 (0.706 / 0.969) | 0.842 | 0.12,0.73 → 0.813, 80% |
| **cam + sq288 (last), sensitive (max F2), deployed set** | **0.21** | **0.857 (0.798 / 0.925)** | **0.881** | **0.20,0.73 → 0.857, 89%** |
| cam + cam160 (last), sensitive (max F2) | 0.06 | 0.834 (0.739 / 0.956) | 0.805 | 0.00,0.49 → 0.834, 18% |

- **Replaced set (until 2026-11-14):** det-cam-ft + cls-cam-ft160 last.pt, both int8 QDQ-all, `fuse_conf` 0.16,
  cascade 0.02 / 0.30 (int8: camera test 0.877 fused, 0.865 cascade). On the board (multi-bee run, tile bees) its
  cascade skipped the detector for only 30–47% of bees (laptop 86%), ~220–236 ms per check. Files kept in
  `export/deploy/alternatives/`.
- **Camera domain +0.15 F1** over the deployed pair at 0.38 (+0.07 over the same pair with val-tuned thresholds).
  The price is the original domain: 0.849 at the camera threshold vs 0.901 (recall 0.95, precision 0.77).
- best.pt vs last.pt (val only): cam160 fused val F1 0.861 vs 0.865 → last; cam288 0.873 vs 0.865, but the
  difference is within noise and last skips the detector more often.
- **Pre-processing mismatch:** the camera classifier crops (`yolo_cam_cls`) are raw 280×160, while the original ones
  are letterboxed 280×280 squares and the app letterboxes. Ultralytics trained and validated on centre-cropped views,
  so the fine-tuned classifiers score low on letterboxed camera bees (hence `fuse_conf` 0.16). Classifier alone on
  camera test: cam160 0.791 letterboxed vs 0.852 centre-cropped (cam288 0.837 vs 0.880). Fused with the detector the
  gap mostly closes (0.879 vs 0.869). Letterbox the camera crops like `convert_cls.py` before the next fine-tune.
- **Letterbox-trained classifiers (`cls-cam-sq160|288`, deployed 2026-11-14):** trained on `yolo_cam_mix_sq`
  (original train + letterboxed camera train crops), so training matches the app's pre-processing. Classifier alone
  on camera test (last.pt): 0.849 / 0.869 (raw-crop fine-tunes 0.791 / 0.811). Fused at max F1, sq288 ties the replaced cam160
  set on camera test (0.881 vs 0.879), is +0.03 on the original test (0.883 vs 0.849) and its probabilities are well
  separated: cascade band 0.33 / 0.73 skips the detector for 93% (cam160: 86% with a 0.02 / 0.30 band).
  sq160 is 0.015 below sq288 on camera test (int8: 0.025), outside the ~0.01 margin for preferring the faster
  160 px model, and below the replaced set. best vs last chosen on val: last for both (0.881 vs 0.879, 0.893 vs 0.891).
  **Deployed at a sensitive point** (board tests missed mites): max F2 on camera val (recall ≥ 0.93 at precision
  ≥ 0.80 is infeasible), `fuse_conf` 0.21, cascade 0.20 / 0.73 (F2 within 0.01 on val; `cls_hi` = 0.95 val precision,
  0.92 on test, a confident-mite level). Camera test recall 0.889 → 0.925 at precision 0.872 → 0.798, F1 0.857, with
  89% detector skips; the old pair at its own F2 point would skip only 18%. int8 vs fp32 within ±0.008 F1 (camera
  test sensitive 0.859 int8 vs 0.857 fp32, original 0.873 vs 0.881). Classifier calibrated on 400 letterboxed camera
  crops. The 288 px classifier costs ~100 ms on the board (160 px: ~20 ms); with ≥ 89% of detector runs skipped a
  check should still be well below the 220–236 ms measured with the replaced set, to be confirmed on the board.

## Next levers (need new data or hardware, not more tuning)
- Benchmark the deployed set (`cam-sq288`, vs `cam-sq160`) on the UNO Q, speed and detector skip rate on camera
  bees: commands in `export/deploy/README.md`.
- At the hive, aggregate verdicts over several frames of the same bee: under-wing mites show in some frames only.
- Record and label footage from your own hives (and the camera-on-screen demo setup) and fine-tune on it.
