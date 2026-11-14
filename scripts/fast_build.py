"""Build faster candidates of the deployed Varroa pair into export/fast/ (deployed files stay untouched).

  python scripts/fast_build.py export      # FP32 ONNX at several input sizes, dynamic batch
  python scripts/fast_build.py quantize    # static INT8 variants, calibrated on train crops

Every model keeps a dynamic batch dimension (several bee crops per run) with fixed H x W.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import onnx

sys.path.insert(0, str(Path(__file__).parent))
from fastbench import letterbox_cls, letterbox_det  # noqa: E402

OUT = Path("export/fast")
DET_PT = Path("runs/verda/detect/n320-tv-e40-screen/weights/last.pt")
CLS_PT = Path("runs/verda/classify/cls-n288/weights/best.pt")
DET_SIZES = [(320, 192), (288, 160), (256, 160), (256, 128)]
CLS_SIZES = [(288, 288), (224, 224), (192, 192), (160, 160), (288, 160), (224, 128)]
# Classifiers retrained at a smaller input (Verda), exported at their training size as clsNNN_<size>_fp32.onnx.
RETRAINED = {f"cls{n}": Path(f"runs/verda/classify/cls-n{n}/weights/best.pt") for n in (224, 192, 160)}


def fix_spatial(path: Path, h: int, w: int):
    """Dynamic export -> keep only the batch dimension dynamic."""
    m = onnx.load(str(path))
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_param = "batch"
    dims[1].dim_value = 3
    dims[2].dim_value, dims[3].dim_value = h, w
    for d in m.graph.output[0].type.tensor_type.shape.dim:
        if d.dim_param:
            d.Clear()
            d.dim_param = "?"
    m.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    m = SymbolicShapeInference.infer_shapes(m, auto_merge=True)
    import onnxslim
    m = onnxslim.slim(m)
    onnx.save(m, str(path))


def export(args):
    from ultralytics import YOLO
    src = OUT / "src"
    src.mkdir(parents=True, exist_ok=True)
    for pt, name in ((DET_PT, "det.pt"), (CLS_PT, "cls.pt")):
        if not (src / name).exists():
            shutil.copy(pt, src / name)
    jobs = [("det", s) for s in DET_SIZES] + [("cls", s) for s in CLS_SIZES]
    for kind, pt in RETRAINED.items():
        if pt.exists():
            n = int(kind[3:])
            if not (src / f"{kind}.pt").exists():
                shutil.copy(pt, src / f"{kind}.pt")
            jobs.append((kind, (n, n)))
    for kind, (h, w) in jobs:
        out = OUT / f"{kind}_{h}x{w}_fp32.onnx"
        if out.exists():
            continue
        p = YOLO(str(src / f"{kind}.pt")).export(format="onnx", imgsz=[h, w], dynamic=True, simplify=True,
                                                   batch=8)
        shutil.move(p, out)
        fix_spatial(out, h, w)
        print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


class Reader:
    def __init__(self, kind, h, w, n, seed=0):
        from onnxruntime.quantization import CalibrationDataReader  # noqa: F401
        imgs = sorted(Path("yolo/images/train").glob("*.png"))
        labels = Path("yolo/labels/train")
        pos = [p for p in imgs if (labels / f"{p.stem}.txt").exists() and (labels / f"{p.stem}.txt").read_text().strip()]
        neg = [p for p in imgs if p not in set(pos)]
        rng = random.Random(seed)
        pick = rng.sample(pos, n // 2) + rng.sample(neg, n // 2)
        prep = letterbox_det if kind == "det" else letterbox_cls
        self.data = iter([{"images": prep([cv2.imread(str(p))], h, w)} for p in pick])

    def get_next(self):
        return next(self.data, None)


def quantize(args):
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quant_pre_process,
                                          quantize_static)
    for src in sorted(OUT.glob("*_fp32.onnx")):
        kind, size = src.stem.split("_")[:2]
        calib_kind = "det" if kind == "det" else "cls"
        if args.only and not any(o in src.stem for o in args.only):
            continue
        h, w = map(int, size.split("x"))
        pre = OUT / "tmp" / f"{src.stem}_pre.onnx"
        pre.parent.mkdir(exist_ok=True)
        try:
            quant_pre_process(str(src), str(pre), skip_symbolic_shape=False)
        except Exception:  # the detector's decode head defeats symbolic shape inference
            quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        for fmt, ops, tag in args.configs:
            out = OUT / f"{kind}_{size}_{tag}.onnx"
            if out.exists():
                continue
            # Excluding the detector head's box/score decoding keeps scores in float.
            m = onnx.load(str(pre))
            exclude = [n.name for n in m.graph.node if kind == "det" and "/model.23/" in n.name
                       and n.op_type not in ("Conv",)] if ops == "all" else []
            quantize_static(
                str(pre), str(out), Reader(calib_kind, h, w, args.n),
                quant_format=QuantFormat.QDQ if fmt == "qdq" else QuantFormat.QOperator,
                per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax if args.minmax else CalibrationMethod.Percentile,
                op_types_to_quantize=["Conv", "MatMul"] if ops == "conv" else None,
                nodes_to_exclude=exclude,
                extra_options={"CalibPercentile": 99.999, "ActivationSymmetric": False,
                               "WeightSymmetric": True},
            )
            print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export")
    q = sub.add_parser("quantize")
    q.add_argument("--n", type=int, default=400, help="calibration crops (half with mites)")
    q.add_argument("--only", nargs="*", help="substrings of fp32 model names to quantize")
    q.add_argument("--minmax", action="store_true")
    args = p.parse_args()
    args.configs = [("qdq", "conv", "int8qdq"), ("qop", "conv", "int8qop"), ("qdq", "all", "int8qdqall")]
    {"export": export, "quantize": quantize}[args.cmd](args)


if __name__ == "__main__":
    main()
