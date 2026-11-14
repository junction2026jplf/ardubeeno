# ArduBeeno: Varroa mite detection on the Arduino UNO Q

ArduBeeno checks every bee in a USB camera's view for Varroa mites, on the Arduino UNO Q itself, with no
cloud connection. The board finds and tracks bees, runs a small mite classifier and detector on each bee crop,
shows the result on its LED matrix and a web page, and queues the events for an optional local dashboard.

Built at Junction 2026 (track "The Green European Renaissance", challenge "Best use of Arduino Uno Q") by a team
from 42 Heilbronn. So far it has been tested on bee images filmed off a laptop screen, not at a real hive.

```
USB camera (1280x720)
   |
   v
bee detector  bees.onnx (YOLO11n, every 3rd frame) --> tracker (one ID per bee)
   |
   v
per-bee crop (4:7, 160x280)
   |
   v
mite check, classifier first:  varroa_cls.onnx (YOLO11n-cls, 288 px, int8) -> p
   p < cls_lo -> clean   |   p >= cls_hi -> mite   |   else + varroa.onnx (YOLO11n, 320x192, int8)
   |
   v
bee = MITE after 2 positive checks --> LED matrix, web page :7000, outbox --> receiver + dashboard :8000
```

## Results

Numbers from [docs/RESULTS.md](docs/RESULTS.md) and `export/deploy/deploy.json`. Thresholds are chosen on val
and applied once to test; the test videos are not used anywhere else.

| Measure | Value |
|---|---|
| Original test set (3,408 bee crops from 2 held-out videos), first deployed pair | F1 0.901 (precision 0.918, recall 0.884) |
| Camera test set (3,600 bees filmed off a screen), that pair before the camera fine-tune | F1 0.725 |
| Camera test set, deployed int8 set (`camsq288-int8all-sensitive`) | F1 0.859 (precision 0.801, recall 0.926) |
| Original test set, deployed int8 set | F1 0.873 |
| Mite detector skipped by the cascade (camera test) | 89% of bees |
| On the board: live view / processing rate | ~25 FPS / ~5 FPS |
| On the board: mite check per bee / bee detector per frame | ~140-165 ms / ~0.5 s |

Main limit: mites under a wing (recall about 0.55-0.65, against about 0.95 for visible mites). Bigger classifiers
(ConvNeXt, DINOv2) did not do better; more and better-matched data is the next step. Details in [docs/RESULTS.md](docs/RESULTS.md).

## Repo layout

| Path | Contents |
|---|---|
| `applab/ardubeeno/` | The Arduino App Lab app: Python (`main.py`, `serve.py`, `monitor.py`, `bees.py`, `outbox.py`), web page, LED-matrix sketch |
| `dashboard/` | Receiver API (FastAPI + SQLite) and dashboard for the board's events, with tests |
| `scripts/` | Data conversion, training, evaluation, ONNX export and quantization, screen recording, laptop tools |
| `demo/` | `start-receiver.sh`; the local `*.env` config files are not in git |
| `docs/` | Results, training, Verda GPUs, uploads (see Documentation) |
| `Makefile` | Entry points (`make help`) |

## Not in this repo

- `data/`: the [VarroaDataset](https://zenodo.org/records/4085044) bee crops (see Data and models below).
- `datasets/`: bee-detector datasets (BEEHIVE, Mendeley landing boards, camera recordings, synthetic bees, COCO
  negatives) and their download/convert helpers.
- `runs/` (training runs), `export/` (ONNX models, `deploy.json`), `captures/` (recordings), `yolo*/` and
  `varroa*.yaml` (generated).

The models can be rebuilt with the scripts and the datasets below; the trained ONNX files can be shared on request.

## Quick start

Needs [uv](https://docs.astral.sh/uv/), Python 3.12, and for training a Mac (`DEVICE=mps`) or a CUDA GPU.

```sh
make setup                     # .venv with requirements.txt (ultralytics, onnxruntime, opencv, ...)
```

**Data.** Download the [VarroaDataset](https://zenodo.org/records/4085044) and put it in `data/{train,val,test}/`, each with `gt_one.csv` (one line per crop:
image path, flag 0/1/3, mite boxes) and the crops under `videos/<video>/`.

```sh
make convert                   # data/ -> yolo/ + varroa.yaml
make smoke                     # 1 epoch on 5% of the data
make train                     # mite detector (EPOCHS=60 IMGSZ=320)
make eval                      # box metrics and per-bee F1 on test
make export                    # export/varroa.onnx
```

The classifier, camera fine-tune, quantization and the deployed set (`scripts/deploy_build.py`) are described in
[docs/TRAINING.md](docs/TRAINING.md).

**Board.** Hardware: an Arduino UNO Q and a USB camera (a powered USB-C hub if needed), reachable over SSH.

```sh
make applab                                  # copy models + deploy.json into applab/ardubeeno/python/
make applab-push BOARD=arduino@<board-ip>    # copy the app to ~/ArduinoApps/ardubeeno on the board
make applab-config BOARD=arduino@<board-ip>  # upload demo/ardubeeno.env, check the receiver from the board
make applab-local                            # or: the same app on this laptop's webcam, http://localhost:7000
```

Then start **ArduBeeno Varroa** in App Lab and open `http://<board-ip>:7000`. Our board is reachable by name as
`ardu-bee-no.local` (`BOARD=arduino@ardu-bee-no.local`).

**Dashboard (optional).** Create the dashboard venv (see [dashboard/README.md](dashboard/README.md)) and two local
config files, which are .gitignored:

- `demo/receiver.env`: `ARDUBEENO_DEVICES_JSON='{"uno-q-01":{"name":"Hive 01","timezone":"Europe/Helsinki","token":"<long random token>"}}'`
  and `ARDUBEENO_DB=dashboard/var/receiver.sqlite3`
- `demo/ardubeeno.env`: `ARDUBEENO_SERVER_URL=http://<laptop-ip>:8000`, `ARDUBEENO_DEVICE_ID=uno-q-01`,
  `ARDUBEENO_DEVICE_TOKEN=<the same token>`, optionally `ARDUBEENO_SOURCE=screen_demo` (a camera filming a screen;
  other values: `field`, `replay`)

```sh
demo/start-receiver.sh         # receiver + dashboard on port 8000, all interfaces
```

## Documentation

| Topic | Document |
|---|---|
| The board app: modes, settings, speed, App Lab | [applab/ardubeeno/README.md](applab/ardubeeno/README.md) |
| Models: data, training, evaluation, export | [docs/TRAINING.md](docs/TRAINING.md) |
| Every experiment and measurement | [docs/RESULTS.md](docs/RESULTS.md) |
| The Verda GPU servers used for training | [docs/VERDA.md](docs/VERDA.md) |
| Receiver and dashboard | [dashboard/README.md](dashboard/README.md) |
| Uploads from the board: events, local history, retries | [docs/UPLOADS.md](docs/UPLOADS.md) |

## Data and models

| Source | Used for | License |
|---|---|---|
| [VarroaDataset](https://zenodo.org/records/4085044), Schurischuster and Kampel (TU Wien), Zenodo v1.2.0, doi:10.5281/zenodo.4085044: 13,509 bee crops (160x280), mite boxes in `gt_one.csv` | Mite detector and classifier, all test numbers | CC BY 4.0 |
| [BEEHIVE: a public dataset of Apis mellifera images](https://doi.org/10.17632/5yz78xxpmy.1), Micheli, Nuzzi, Papa, Mendeley Data v1 | Bee detector (close-ups) | CC BY 4.0 |
| [Labeled dataset for bee detection and direction estimation on beehive landing boards](https://doi.org/10.17632/8gb9r2yhfc.6), Sledevic, Mendeley Data v6 | Bee detector (landing boards), screen recordings | CC BY 4.0 |
| [EV2 dataset](https://zenodo.org/records/13771384), Zenodo 13771384 | Evaluation on foreign footage; one rejected training experiment | CC BY 4.0 |
| [COCO](https://cocodataset.org/) val2017 | Hard negatives for the bee detector (`scripts/make_negatives.py`) | Annotations CC BY 4.0; images under the Flickr terms of use of each image |
| [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) (`yolo11n.pt`, `yolo11n-cls.pt`) | Training framework and pretrained weights for all models | AGPL-3.0 (or an Ultralytics Enterprise license) |

Models trained with Ultralytics from its pretrained weights, and code that uses `ultralytics`, fall under
AGPL-3.0 obligations: if you distribute them or offer them over a network, the corresponding source has to be
available under AGPL-3.0, unless you hold an Ultralytics Enterprise license. The board app itself only needs
`numpy`, `opencv-python-headless` and `onnxruntime`, but its ONNX models are derived from the YOLO11 weights.

## License

No license has been chosen for this repository yet, so all rights are reserved by the authors for now. See the
Ultralytics note above before picking one.
