#!/usr/bin/env python3
"""Run YOLO training using Ultralytics.

Loads config from train_config.yaml and supports per-run overrides at CLI.
All output goes to training/runs/<name>/.

Usage:
    # Sanity check (nano, 30 epochs)
    python scripts/train.py --name v1-nano-sanity --override model=yolov8n.pt epochs=30 batch=32

    # Main training run
    python scripts/train.py --name v1-medium-main --override model=yolov8m.pt

    # CPU smoke test (no GPU required)
    python scripts/train.py --name smoke --override model=yolov8n.pt epochs=1 batch=2 device=cpu imgsz=320
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import CONFIG_DIR, TRAINING_ROOT, RUNS_DIR, ensure_dirs
from utils.yaml_io import load_yaml, merge_yaml

log = logging.getLogger(__name__)


def parse_override(override_str: str) -> tuple[str, object]:
    """Parse 'key=value' into (key, typed_value)."""
    if "=" not in override_str:
        raise ValueError(f"Override must be key=value, got: {override_str!r}")
    key, raw = override_str.split("=", 1)
    # Type coercion: try int, then float, then bool, then str
    for coerce in (int, float):
        try:
            return key, coerce(raw)
        except ValueError:
            pass
    if raw.lower() in {"true", "false"}:
        return key, raw.lower() == "true"
    return key, raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLO model on IntelFactor defect data")
    p.add_argument(
        "--config",
        default=str(CONFIG_DIR / "train_config.yaml"),
        help="Training config YAML",
    )
    p.add_argument(
        "--name",
        required=True,
        help="Run name (output goes to runs/<name>/)",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="key=value",
        help="Override config values, e.g. --override epochs=50 batch=16",
    )
    p.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
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

    cfg = load_yaml(args.config)

    # Apply CLI overrides
    overrides: dict = {}
    for ov in (args.override or []):
        k, v = parse_override(ov)
        overrides[k] = v
    if args.name:
        overrides["name"] = args.name
    cfg = merge_yaml(cfg, overrides)

    # Resolve data.yaml path relative to TRAINING_ROOT
    data_path = cfg.get("data", "datasets/combined/data.yaml")
    if not Path(data_path).is_absolute():
        data_path = str(TRAINING_ROOT / data_path)
    cfg["data"] = data_path

    # Resolve project path
    project_path = cfg.get("project", "runs")
    if not Path(project_path).is_absolute():
        project_path = str(TRAINING_ROOT / project_path)
    cfg["project"] = project_path

    ensure_dirs(RUNS_DIR)

    model_name = cfg.pop("model", "yolov8n.pt")
    task = cfg.pop("task", "detect")

    log.info("Loading model: %s", model_name)
    model = YOLO(model_name)

    log.info("Starting training run: %s", args.name)
    log.info("Config: epochs=%s batch=%s imgsz=%s device=%s",
             cfg.get("epochs"), cfg.get("batch"), cfg.get("imgsz"), cfg.get("device"))

    # Pass all remaining cfg keys to model.train()
    results = model.train(**cfg)

    best_pt = Path(cfg["project"]) / args.name / "weights" / "best.pt"
    log.info("Training complete.")
    if best_pt.exists():
        log.info("Best model: %s", best_pt)
        log.info("Next step: python scripts/evaluate.py --model %s", best_pt)
    else:
        log.warning("best.pt not found at expected path: %s", best_pt)


if __name__ == "__main__":
    main()
