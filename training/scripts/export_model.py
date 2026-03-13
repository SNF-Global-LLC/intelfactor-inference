#!/usr/bin/env python3
"""Export a trained YOLO .pt model to ONNX for TensorRT conversion on Jetson.

The exported .onnx file is placed in training/exports/ by default.
After export, SCP the file to the Jetson and run trtexec there.

Usage:
    python scripts/export_model.py --model runs/v1-medium-main/weights/best.pt
    python scripts/export_model.py --model runs/v1-medium-main/weights/best.pt \
        --output exports/yolov8m-metal-v1.onnx --imgsz 640
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import TRAINING_ROOT, EXPORTS_DIR, ensure_dirs

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export YOLO .pt → ONNX for TensorRT conversion on Jetson"
    )
    p.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    p.add_argument(
        "--output",
        help="Output .onnx path (default: exports/<model_stem>.onnx)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square, default: 640)",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17 — compatible with TRT 8.6+)",
    )
    p.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Run onnx-simplifier on the exported model (default: True)",
    )
    p.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Export with dynamic batch axis (not recommended for TRT; use static batch=1)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)

    ensure_dirs(EXPORTS_DIR)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = TRAINING_ROOT / output_path
    else:
        output_path = EXPORTS_DIR / (model_path.parent.parent.name + ".onnx")

    log.info("Exporting: %s", model_path)
    log.info("Output:    %s", output_path)
    log.info("imgsz=%d  opset=%d  simplify=%s  dynamic=%s",
             args.imgsz, args.opset, args.simplify, args.dynamic)

    model = YOLO(str(model_path))

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )

    # Ultralytics writes the ONNX next to the .pt file by default; move it.
    default_onnx = model_path.with_suffix(".onnx")
    if default_onnx.exists() and default_onnx != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        default_onnx.rename(output_path)
        log.info("Moved ONNX to: %s", output_path)
    elif not output_path.exists():
        log.warning("Expected ONNX not found at %s — check ultralytics export output", output_path)

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1024 / 1024
        log.info("Export complete: %.1f MB", size_mb)

    # Print Jetson conversion commands
    onnx_remote = f"/opt/intelfactor/models/{output_path.name}"
    engine_remote = onnx_remote.replace(".onnx", "_fp16.engine")

    print("\n" + "=" * 60)
    print("  NEXT STEPS: TensorRT conversion on Jetson")
    print("=" * 60)
    print(f"\n  1. Copy ONNX to Jetson:")
    print(f"     scp {output_path} tony@<JETSON_IP>:{onnx_remote}")
    print(f"\n  2. SSH to Jetson and convert:")
    print(f"     /usr/src/tensorrt/bin/trtexec \\")
    print(f"       --onnx={onnx_remote} \\")
    print(f"       --saveEngine={engine_remote} \\")
    print(f"       --fp16 --workspace=4096")
    print(f"\n  3. Benchmark (target < 25ms):")
    print(f"     /usr/src/tensorrt/bin/trtexec \\")
    print(f"       --loadEngine={engine_remote} --batch=1")
    print(f"\n  4. Run make doctor to validate the new engine.")
    print()


if __name__ == "__main__":
    main()
