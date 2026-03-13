#!/usr/bin/env python3
"""Sweep confidence thresholds and output per-class optimal values.

For each class, finds the lowest confidence threshold that still meets
the --min-recall target.  Saves results to evaluation/confidence_thresholds.yaml.

This YAML is read by the production VisionProvider at runtime to apply
per-class thresholds instead of a single global cutoff.

Usage:
    python scripts/tune_threshold.py --model runs/v1-medium-main/weights/best.pt
    python scripts/tune_threshold.py --model runs/v1-medium-main/weights/best.pt --min-recall 0.90
    python scripts/tune_threshold.py --model runs/v1-medium-main/weights/best.pt --step 0.05
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import TRAINING_ROOT, ensure_dirs
from utils.taxonomy import CLASSES
from utils.yaml_io import save_yaml

log = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {cls: 0.25 for cls in CLASSES}


def sweep_thresholds(
    model,
    data_path: str,
    thresholds: list[float],
    split: str,
    device: str,
    min_recall: float,
) -> dict[str, float]:
    """Sweep confidence thresholds and return per-class optimal values.

    Strategy: for each threshold, run val() and collect per-class recall.
    Choose the lowest threshold where recall >= min_recall for each class.
    Falls back to 0.25 if no threshold achieves the target.
    """
    # {class_name: best_threshold}
    best: dict[str, float | None] = {cls: None for cls in CLASSES}

    for thresh in sorted(thresholds, reverse=True):  # high → low
        log.info("  Evaluating conf=%.2f ...", thresh)
        try:
            metrics = model.val(
                data=data_path,
                split=split,
                conf=thresh,
                iou=0.6,
                device=device,
                verbose=False,
                plots=False,
                save_json=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("  val() failed at conf=%.2f: %s", thresh, exc)
            continue

        # per-class recall is in metrics.box.r (array aligned to class order)
        if not hasattr(metrics, "box") or metrics.box.r is None:
            log.warning("  No per-class recall data at conf=%.2f", thresh)
            continue

        per_class_recall = metrics.box.r  # shape: (num_classes,)
        for i, cls in enumerate(CLASSES):
            if i < len(per_class_recall):
                recall = float(per_class_recall[i])
                if recall >= min_recall:
                    best[cls] = thresh  # lower threshold still meets recall target

    # Fill fallback for classes that never met recall target
    result = {}
    for cls in CLASSES:
        if best[cls] is not None:
            result[cls] = best[cls]
        else:
            result[cls] = 0.25  # default — operator must review
            log.warning(
                "Class '%s' never reached %.0f%% recall across all thresholds. "
                "Using fallback 0.25 — consider more training data for this class.",
                cls,
                min_recall * 100,
            )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep confidence thresholds per class")
    p.add_argument("--model", required=True, help="Path to best.pt")
    p.add_argument(
        "--data",
        default="datasets/combined/data.yaml",
        help="Path to data.yaml (relative to training/)",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Split to use for threshold sweep (use val; save test for final eval)",
    )
    p.add_argument(
        "--min-recall",
        type=float,
        default=0.85,
        help="Minimum per-class recall required (default: 0.85)",
    )
    p.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Threshold sweep step size (default: 0.05)",
    )
    p.add_argument(
        "--min-conf",
        type=float,
        default=0.10,
        help="Minimum threshold to sweep from (default: 0.10)",
    )
    p.add_argument(
        "--max-conf",
        type=float,
        default=0.90,
        help="Maximum threshold to sweep to (default: 0.90)",
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

    eval_dir = model_path.parent.parent / "evaluation"
    ensure_dirs(eval_dir)

    import torch
    device = args.device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(str(model_path))

    # Build threshold list
    thresholds = []
    t = args.min_conf
    while t <= args.max_conf + 1e-6:
        thresholds.append(round(t, 4))
        t += args.step

    log.info(
        "Sweeping %d thresholds (%.2f–%.2f step %.2f) on '%s' split, min_recall=%.0f%%",
        len(thresholds),
        args.min_conf,
        args.max_conf,
        args.step,
        args.split,
        args.min_recall * 100,
    )

    optimal = sweep_thresholds(
        model=model,
        data_path=str(data_path),
        thresholds=thresholds,
        split=args.split,
        device=device,
        min_recall=args.min_recall,
    )

    output = {
        "min_recall_target": args.min_recall,
        "sweep_split": args.split,
        "model": str(model_path),
        "thresholds": optimal,
        # Global fallback used by VisionProvider when class-level key absent
        "default_threshold": 0.25,
    }

    out_path = eval_dir / "confidence_thresholds.yaml"
    save_yaml(
        output,
        out_path,
        comment=(
            "IntelFactor per-class confidence thresholds\n"
            f"Generated by tune_threshold.py — min_recall={args.min_recall:.0%}\n"
            "Load this file into VisionProvider at runtime."
        ),
    )
    log.info("Thresholds written to %s", out_path)

    log.info("\nPer-class thresholds:")
    for cls, thresh in sorted(optimal.items()):
        log.info("  %-20s %.2f", cls, thresh)

    log.info("\nNext step: python scripts/export_model.py --model %s", model_path)


if __name__ == "__main__":
    main()
