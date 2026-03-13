#!/usr/bin/env python3
"""Analyze a YOLO dataset and print health stats.

Checks:
  - Image and label counts per split
  - Missing labels
  - Empty label files (background images)
  - Malformed label lines
  - Class distribution and imbalance warnings
  - Split ratio summary

Usage:
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --data datasets/combined/data.yaml
    python scripts/analyze_dataset.py --data datasets/raw/roboflow/metal-defect-detection --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import COMBINED_DIR, CONFIG_DIR
from utils.taxonomy import CLASSES, NUM_CLASSES
from utils.yaml_io import load_yaml
from utils.dataset_checks import check_dataset, class_imbalance_warnings

log = logging.getLogger(__name__)


def print_split_report(result: dict) -> None:
    split = result["split"]
    print(f"\n{'=' * 60}")
    print(f"  Split: {split.upper()}")
    print(f"{'=' * 60}")
    print(f"  Images:  {result.get('image_count', 0)}")
    print(f"  Labels:  {result.get('label_count', 0)}")

    missing = result.get("missing_labels", [])
    if missing:
        print(f"  Missing labels: {len(missing)} (showing first 5)")
        for m in missing[:5]:
            print(f"    - {m}")

    empty = result.get("empty_labels", [])
    if empty:
        print(f"  Empty labels (background): {len(empty)}")

    invalid = result.get("invalid_lines", [])
    if invalid:
        print(f"  Malformed lines: {len(invalid)} (showing first 5)")
        for inv in invalid[:5]:
            print(f"    - {inv}")

    class_counts = result.get("class_counts", {})
    total_instances = sum(class_counts.values())
    if total_instances > 0:
        print(f"\n  Class distribution ({total_instances} instances):")
        for cls in CLASSES:
            count = class_counts.get(cls, 0)
            pct = count / total_instances * 100 if total_instances else 0
            bar = "#" * min(40, int(pct / 2))
            print(f"    {cls:<20} {count:>6}  {pct:5.1f}%  {bar}")

    for w in result.get("warnings", []):
        print(f"  WARNING: {w}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a YOLO dataset for health and balance")
    p.add_argument(
        "--data",
        default=str(COMBINED_DIR),
        help="Path to dataset root dir OR data.yaml file (default: datasets/combined/)",
    )
    p.add_argument(
        "--output-json",
        metavar="FILE",
        help="Save full analysis as JSON to FILE",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    data_path = Path(args.data)
    if data_path.is_file() and data_path.suffix == ".yaml":
        data_yaml = load_yaml(data_path)
        dataset_dir = Path(data_yaml.get("path", data_path.parent)).resolve()
    else:
        dataset_dir = data_path.resolve()

    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nIntelFactor Dataset Analysis")
    print(f"Dataset: {dataset_dir}")
    print(f"Taxonomy: {NUM_CLASSES} classes")

    results = check_dataset(dataset_dir)

    all_class_counts: dict[str, int] = {cls: 0 for cls in CLASSES}
    total_images = 0

    for result in results:
        print_split_report(result)
        for cls in CLASSES:
            all_class_counts[cls] += result.get("class_counts", {}).get(cls, 0)
        total_images += result.get("image_count", 0)

    # Global summary
    print(f"\n{'=' * 60}")
    print("  GLOBAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total images: {total_images}")
    total_instances = sum(all_class_counts.values())
    print(f"  Total instances: {total_instances}")

    imbalance_warnings = class_imbalance_warnings(all_class_counts)
    if imbalance_warnings:
        print("\n  CLASS IMBALANCE WARNINGS:")
        for w in imbalance_warnings:
            print(f"    ! {w}")
    else:
        print("  No severe class imbalance detected.")

    # Split ratio
    split_counts = {r["split"]: r.get("image_count", 0) for r in results}
    if total_images > 0:
        print("\n  Split ratios:")
        for split, count in split_counts.items():
            pct = count / total_images * 100
            print(f"    {split:<6} {count:>6} images  ({pct:.1f}%)")

    print()

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(
                {
                    "dataset": str(dataset_dir),
                    "splits": results,
                    "global_class_counts": all_class_counts,
                    "total_images": total_images,
                    "total_instances": total_instances,
                },
                f,
                indent=2,
            )
        print(f"Analysis saved to {out}")


if __name__ == "__main__":
    main()
