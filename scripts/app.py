"""Local test app: try the deployed model in the browser on test bees, your own bee crops or the webcam.

Runs the same ONNX pipeline as the board (device_check.py): detector + classifier, and a bee counts as
infested when mean(classifier mite probability, top box score) >= --fuse-conf. Only needs the device
requirements (numpy, opencv, onnxruntime).

  python scripts/app.py        # then open http://localhost:8000
"""
import argparse
import base64
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from device_check import Pipeline

PAGE = Path(__file__).with_name("app.html")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="export/varroa.onnx")
    p.add_argument("--cls-model", action="append", help="default export/varroa_cls.onnx; repeat for an ensemble")
    p.add_argument("--conf", type=float, default=0.25, help="boxes drawn at this detector score")
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--fuse-conf", type=float, default=0.38)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--yolo-dir", type=Path, default=Path("yolo"), help="source of the random test bees")
    p.add_argument("--gt", type=Path, default=Path("data/test/gt_one.csv"), help="flag 3 = mite under a wing")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    args.cls_model = args.cls_model or ["export/varroa_cls.onnx"]

    pipe, lock = Pipeline(args), threading.Lock()
    labels = args.yolo_dir / "labels" / "test"
    bees = {"mite": [], "clean": []}
    for img in sorted((args.yolo_dir / "images" / "test").glob("*.png")):
        label = labels / f"{img.stem}.txt"
        bees["mite" if label.exists() and label.read_text().strip() else "clean"].append(img)
    flags = {Path(r[0]).stem: r[1] for r in (l.split() for l in open(args.gt)) if r} if args.gt.exists() else {}

    def predict(img):
        with lock:
            boxes, top, prob, fused, verdict = pipe(img)
            ms = pipe.last_ms
        return {"verdict": bool(verdict), "fused": fused, "prob": prob, "top": top,
                "boxes": [[round(v, 1) for v in b[:4]] + [round(b[4], 3)] for b in boxes],
                "ms": [round(m, 1) for m in ms], "width": img.shape[1], "height": img.shape[0]}

    info = {"model": args.model, "cls_model": args.cls_model, "conf": args.conf, "fuse_conf": args.fuse_conf,
            "test_bees": {k: len(v) for k, v in bees.items()}}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, ctype="application/json"):
            body = json.dumps(body).encode() if ctype == "application/json" else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/info":
                self.reply(200, info)
            elif url.path == "/random":
                kind = parse_qs(url.query).get("kind", ["any"])[0]
                pool = bees.get(kind) or bees["mite"] + bees["clean"]
                path = random.choice(pool)
                img = cv2.imread(str(path))
                png = base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()
                self.reply(200, {**predict(img), "name": path.name, "image": f"data:image/png;base64,{png}",
                                 "actual": path in bees["mite"], "underwing": flags.get(path.stem) == "3"})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self.reply(404, {"error": "not found"})
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return self.reply(400, {"error": "could not decode the image (use PNG or JPEG)"})
            self.reply(200, predict(img))

        def log_message(self, *a):
            pass

    print(f"model {args.model} + {', '.join(args.cls_model)}, verdict mean >= {args.fuse_conf}; "
          f"{info['test_bees']['mite']} mite / {info['test_bees']['clean']} clean test bees")
    print(f"open http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
