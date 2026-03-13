#!/usr/bin/env python3
"""Evaluate a trained YOLO model on the test split.

Runs model.val() on the test dataset, writes metrics and confusion matrix
to runs/<name>/evaluation/.

Usage:
    python scripts/evaluate.py --model runs/v1-medium-main/weights/best.pt
    python scripts/evaluate.py --model runs/v1-medium-main/weights/best.pt --name v1-medium-eval
    python scripts/evaluate.py --model runs/v1-nano-sanity/weights/best.pt --split val
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import TRAINING_ROOT, CONFIG_DIR, ensure_dirs
from utils.yaml_io import load_yaml

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained YOLO model")
    p.add_argument("--model", required=True, help="Path to best.pt checkpoint")
    p.add_argument(
        "--data",
        default="datasets/combined/data.yaml",
        help="Path to data.yaml (relative to training/)",
    )
    p.add_argument(
        "--name",
        help="Evaluation output name (default: derived from model path)",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for evaluation",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS",
    )
    p.add_argument("--device", default=None, help="GPU index or 'cpu' (auto-detects if omitted)")
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

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = (TRAINING_ROOT / data_path).resolve()
    if not data_path.exists():
        log.error("data.yaml not found: %s", data_path)
        sys.exit(1)

    run_name = args.name or (model_path.parent.parent.name + "-eval")
    eval_dir = model_path.parent.parent / "evaluation"
    ensure_dirs(eval_dir)

    import torch
    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    log.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))

    log.info("Evaluating on '%s' split (conf=%.2f, iou=%.2f, device=%s)", args.split, args.conf, args.iou, device)
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=device,
        project=str(eval_dir),
        name=run_name,
        plots=True,
        save_json=True,
    )

    # Extract scalar metrics for summary
    summary = {
        "model": str(model_path),
        "split": args.split,
        "conf": args.conf,
        "iou": args.iou,
        "mAP50": float(metrics.box.map50) if hasattr(metrics, "box") else None,
        "mAP50_95": float(metrics.box.map) if hasattr(metrics, "box") else None,
        "precision": float(metrics.box.mp) if hasattr(metrics, "box") else None,
        "recall": float(metrics.box.mr) if hasattr(metrics, "box") else None,
    }

    summary_path = eval_dir / "metrics_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    log.info("Evaluation complete.")
    log.info("  mAP50:     %.4f", summary["mAP50"] or 0)
    log.info("  mAP50-95:  %.4f", summary["mAP50_95"] or 0)
    log.info("  Precision: %.4f", summary["precision"] or 0)
    log.info("  Recall:    %.4f", summary["recall"] or 0)
    log.info("  Output:    %s", eval_dir)
    log.info("Next step: python scripts/tune_threshold.py --model %s", model_path)

    # Sanity check
    map50 = summary["mAP50"] or 0.0
    if map50 < 0.30:
        log.warning("mAP50 < 0.30 — review dataset quality before proceeding to larger models.")
    elif map50 < 0.60:
        log.warning("mAP50 < 0.60 — consider more data, longer training, or model upgrade.")
    else:
        log.info("mAP50 looks healthy. Proceed to tune_threshold.py.")


if __name__ == "__main__":
    main()
