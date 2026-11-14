"""Export trained detector or classifier weights to ONNX for the Arduino UNO Q, and check it loads."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="runs/detect/varroa/weights/best.pt")
    p.add_argument("--imgsz", default="320",
                   help="square size, or height,width (e.g. 320,192 to skip the letterbox padding)")
    p.add_argument("--out", type=Path, default=Path("export/varroa.onnx"))
    args = p.parse_args()

    imgsz = [int(s) for s in args.imgsz.split(",")]
    imgsz = imgsz[0] if len(imgsz) == 1 else imgsz
    model = YOLO(args.weights)
    task = model.task
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, args.out)

    # Smoke test: the exported model loads and runs on CPU.
    sample = str(next(Path("yolo/images/test").glob("*.png")))
    YOLO(str(args.out), task=task).predict(sample, imgsz=imgsz, device="cpu", verbose=False)
    print(f"exported {task} model {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
