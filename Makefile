PYTHON  := .venv/bin/python
DEVICE  ?= mps
EPOCHS  ?= 60
IMGSZ   ?= 320
BATCH   ?= 64
CONF    ?= 0.25
IOU     ?= 0.45
NAME    ?= varroa
WEIGHTS ?= runs/detect/$(NAME)/weights/best.pt
CLS_WEIGHTS ?= runs/classify/varroa-cls/weights/best.pt
FUSE_CONF ?= 0.38
# make export-cls input size; the deployed classifier is 160 px and built by scripts/deploy_build.py
CLS_IMGSZ ?= 288
# make finetune: dataset and epochs (camera fine-tune: FT_DATA=varroa_cam.yaml FT_EPOCHS=25)
FT_DATA ?= varroa.yaml
FT_EPOCHS ?= 15
EXPORT_IMGSZ ?= $(IMGSZ)
BUNDLE  := export/device
APPLAB  := applab/ardubeeno
BOARD   ?=
# YOLO11n trained at 416, exported at 256x416 for 16:9 frames (no letterbox bars): see applab/ardubeeno/README.md
BEE_MODEL ?= export/bees/bees_n416cam_256x416.onnx
# Mite models + deploy.json (thresholds) from export/deploy/ when it has them, else export/
MODELS_DIR ?= $(if $(wildcard export/deploy/varroa.onnx),export/deploy,export)
SRC     ?= data/test
CAMERA  ?= 0
REC_ARGS ?=

.PHONY: all setup convert train smoke eval export export-cls live slideshow record record-mendeley record-cameras finetune device-check device-push applab applab-push applab-config applab-local clean help

help:
	@echo "make setup      create .venv (Python 3.12) and install dependencies"
	@echo "make convert    convert data/ to YOLO format in yolo/ + varroa.yaml"
	@echo "make smoke      1-epoch training on 5% of the data, to check everything works"
	@echo "make train      train the detector (EPOCHS=$(EPOCHS) DEVICE=$(DEVICE))"
	@echo "make finetune WEIGHTS=...  fine-tune a detector (FT_DATA=$(FT_DATA) FT_EPOCHS=$(FT_EPOCHS); camera: FT_DATA=varroa_cam.yaml FT_EPOCHS=25)"
	@echo "make eval       box metrics + per-bee accuracy on the test split"
	@echo "make export-cls export the classifier to export/varroa_cls.onnx (CLS_WEIGHTS=... CLS_IMGSZ=$(CLS_IMGSZ); the deployed 160 px one comes from scripts/deploy_build.py)"
	@echo "make export     export best.pt to export/varroa.onnx (EXPORT_IMGSZ=320,192 for the faster rectangular input)"
	@echo "make slideshow  full-screen test bees for filming"
	@echo "make record SRC=data/test CAMERA=1  show SRC full-screen, film it, save frames + YOLO labels to captures/"
	@echo "make record-mendeley CAMERA=1  the same for the Mendeley landing-board val split (datasets/mendeley_landing_board)"
	@echo "make record-cameras  list cameras (index, resolution, snapshot) and screens for make record"
	@echo "make live       live camera detection with the ONNX model"
	@echo "make device-check  bundle model + 200 test bees into $(BUNDLE)/, record laptop reference results"
	@echo "make device-push BOARD=user@host  copy the bundle to the UNO Q (~/ardubeeno)"
	@echo "make applab     copy the models into the App Lab app in $(APPLAB)/"
	@echo "make applab-push BOARD=arduino@host  copy the App Lab app to the UNO Q (~/ArduinoApps/ardubeeno)"
	@echo "make applab-config BOARD=arduino@host  copy demo/ardubeeno.env to the board and check the receiver from there"
	@echo "make applab-local  run the app's live preview on this laptop's webcam (http://localhost:7000)"
	@echo "make all        setup, convert, train, eval, export"

all: setup convert train eval export

setup: .venv/.installed

.venv/.installed: requirements.txt
	uv venv --allow-existing --python 3.12 .venv
	uv pip install --python $(PYTHON) -r requirements.txt
	touch $@

convert: varroa.yaml

varroa.yaml: scripts/convert.py | setup
	$(PYTHON) scripts/convert.py

smoke: convert
	$(PYTHON) scripts/train.py --epochs 1 --fraction 0.05 --device $(DEVICE) --name smoke --imgsz $(IMGSZ)

train: convert
	$(PYTHON) scripts/train.py --epochs $(EPOCHS) --batch $(BATCH) --device $(DEVICE) --name $(NAME) --imgsz $(IMGSZ)

eval: convert
	$(PYTHON) scripts/evaluate.py --weights $(WEIGHTS) --device $(DEVICE) --imgsz $(IMGSZ) --conf $(CONF) --iou $(IOU)

export: convert
	$(PYTHON) scripts/export.py --weights $(WEIGHTS) --imgsz $(EXPORT_IMGSZ)

export-cls: convert
	$(PYTHON) scripts/export.py --weights $(CLS_WEIGHTS) --imgsz $(CLS_IMGSZ) --out export/varroa_cls.onnx

slideshow: convert
	$(PYTHON) scripts/slideshow.py

# Screen -> camera training data: SRC = data split, YOLO folder/yaml or image folder; REC_ARGS e.g.
# "--duration 1.5 --limit 300 --shuffle --tile 3x2 --screen 1". CAMERA=sim REC_ARGS=--headless is a dry run.
record:
	$(PYTHON) scripts/record.py $(SRC) --camera $(CAMERA) $(REC_ARGS)

# Mendeley landing-board val split (2 unseen hives, 1,410 frames), e.g. make record-mendeley CAMERA=1
record-mendeley:
	$(PYTHON) scripts/record.py datasets/mendeley_landing_board/bees.yaml --split val --shuffle --camera $(CAMERA) $(REC_ARGS)

record-cameras:
	$(PYTHON) scripts/record.py --list-screens
	$(PYTHON) scripts/record.py --list-cameras

live:
	$(PYTHON) scripts/live.py --imgsz $(IMGSZ) --conf $(CONF) --iou $(IOU)

# Fine-tune an existing model, e.g. after adding labelled screen captures to the train split.
# Camera fine-tune: make finetune FT_DATA=varroa_cam.yaml FT_EPOCHS=25 WEIGHTS=runs/verda/detect/n320-tv-e40-screen/weights/last.pt
finetune: convert
	$(PYTHON) scripts/train.py --model $(WEIGHTS) --data $(FT_DATA) --epochs $(FT_EPOCHS) --batch $(BATCH) --device $(DEVICE) --name $(NAME)-ft --imgsz $(IMGSZ) \
		--optimizer AdamW --lr0 0.0003 --cos-lr

# Bundle for the UNO Q, and record the laptop's results on the same bees for comparison.
device-check: convert
	$(PYTHON) scripts/bundle.py --out $(BUNDLE)
	cd $(BUNDLE) && $(CURDIR)/$(PYTHON) device_check.py --conf $(CONF) --iou $(IOU) --fuse-conf $(FUSE_CONF) \
		$$(test -f varroa_cls.onnx && echo --cls-model varroa_cls.onnx) accuracy --csv laptop.csv

device-push:
	@test -n "$(BOARD)" || { echo "usage: make device-push BOARD=user@host"; exit 1; }
	@test -f $(BUNDLE)/laptop.csv || { echo "run make device-check first"; exit 1; }
	rsync -av $(BUNDLE)/ $(BOARD):ardubeeno/

# Arduino App Lab app: live camera preview with mite boxes in the browser, LED matrix as a warning light.
applab:
	@test -f $(MODELS_DIR)/varroa.onnx || { echo "$(MODELS_DIR)/varroa.onnx not found, run make export first"; exit 1; }
	@echo "mite models from $(MODELS_DIR)/, bee detector $(BEE_MODEL)"
	cp $(MODELS_DIR)/varroa.onnx scripts/device_check.py $(APPLAB)/python/
	test -f $(MODELS_DIR)/varroa_cls.onnx && cp $(MODELS_DIR)/varroa_cls.onnx $(APPLAB)/python/ || true
	@# export/deploy: its other .onnx files (for serve.py --model/--cls-model) and deploy.json (thresholds)
	if [ "$(MODELS_DIR)" != export ]; then cp $(MODELS_DIR)/*.onnx $(APPLAB)/python/; fi
	if [ -f $(MODELS_DIR)/deploy.json ]; then cp $(MODELS_DIR)/deploy.json $(APPLAB)/python/; \
		else rm -f $(APPLAB)/python/deploy.json; fi
	@# Optional int8 drop-ins for --models int8-qdqall / c160-int8all (MODELS= in App Lab)
	for f in det_320x192_int8qdqall.onnx cls_288x288_int8qdqall.onnx cls160_160x160_int8qdqall.onnx; do \
		test -f export/fast/$$f && cp export/fast/$$f $(APPLAB)/python/ || true; done
	@# The bee detector turns on the multi-bee mode; BEE_MODEL picks one of the trained sizes.
	test -n "$(BEE_MODEL)" && cp $(BEE_MODEL) $(APPLAB)/python/bees.onnx || true
	@# Its settings travel with it (crop_margin for its box tightness): foo.onnx -> foo.json, as bees.json. Always written
	@# ({} without foo.json): the push never deletes files on the board, so a stale bees.json would stay there.
	if [ -n "$(BEE_MODEL)" ]; then if [ -f "$(basename $(BEE_MODEL)).json" ]; then \
		cp "$(basename $(BEE_MODEL)).json" $(APPLAB)/python/bees.json; else echo "{}" > $(APPLAB)/python/bees.json; fi; fi

applab-push: applab
	@test -n "$(BOARD)" || { echo "usage: make applab-push BOARD=arduino@<board-ip>"; exit 1; }
	@# The board has no rsync: stream a tar over ssh. data/ (outbox, config) and captures/ on the board are left alone.
	COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -C $(APPLAB) --exclude __pycache__ --exclude .DS_Store --exclude .ruff_cache --exclude captures --exclude data -cf - . \
		| ssh $(BOARD) 'mkdir -p ArduinoApps/ardubeeno && tar -xf - -C ArduinoApps/ardubeeno'

# Upload config for the board (demo/ardubeeno.env: receiver URL, device ID and token), kept out of the app bundle.
applab-config:
	@test -n "$(BOARD)" || { echo "usage: make applab-config BOARD=arduino@<board-ip>"; exit 1; }
	@test -f demo/ardubeeno.env || { echo "demo/ardubeeno.env not found"; exit 1; }
	ssh $(BOARD) 'mkdir -p ArduinoApps/ardubeeno/data && umask 077 && cat > ArduinoApps/ardubeeno/data/ardubeeno.env' < demo/ardubeeno.env
	@url=$$(sed -n 's/^ARDUBEENO_SERVER_URL=//p' demo/ardubeeno.env); \
		ssh $(BOARD) "curl -s -m 5 $$url/health || wget -qO- -T 5 $$url/health || echo 'receiver not reachable from the board'"; echo

applab-local: applab
	cd $(APPLAB)/python && $(CURDIR)/$(PYTHON) serve.py --camera 0

clean:
	rm -rf yolo varroa.yaml runs export
