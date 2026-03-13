"""Dataset integrity checks used by analyze_dataset.py and merge_datasets.py."""
from __future__ import annotations

import logging
from pathlib import Path

from .taxonomy import CLASSES, NUM_CLASSES

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def check_split(split_dir: Path, num_classes: int = NUM_CLASSES) -> dict:
    """Run integrity checks on one YOLO split directory.

    Expected structure:
        split_dir/
            images/  *.jpg | *.png
            labels/  *.txt  (one per image, YOLO format)

    Returns a dict with counts and a list of warnings.
    """
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    warnings: list[str] = []
    result: dict = {
        "split": split_dir.name,
        "image_count": 0,
        "label_count": 0,
        "missing_labels": [],
        "empty_labels": [],
        "invalid_lines": [],
        "class_counts": {cls: 0 for cls in CLASSES},
        "unknown_class_ids": [],
        "warnings": warnings,
    }

    if not images_dir.exists():
        warnings.append(f"images/ directory missing in {split_dir}")
        return result
    if not labels_dir.exists():
        warnings.append(f"labels/ directory missing in {split_dir}")
        return result

    image_files = sorted(images_dir.glob("*.*"))
    image_stems = {f.stem for f in image_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    result["image_count"] = len(image_stems)

    label_files = sorted(labels_dir.glob("*.txt"))
    label_stems = {f.stem for f in label_files}
    result["label_count"] = len(label_stems)

    # Images without labels
    missing = sorted(image_stems - label_stems)
    if missing:
        result["missing_labels"] = missing[:20]  # cap at 20 for readability
        warnings.append(f"{len(missing)} images have no label file")

    # Parse labels
    for lf in label_files:
        text = lf.read_text().strip()
        if not text:
            result["empty_labels"].append(lf.name)
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) < 5:
                result["invalid_lines"].append(f"{lf.name}:{lineno}")
                continue
            try:
                cls_id = int(parts[0])
            except ValueError:
                result["invalid_lines"].append(f"{lf.name}:{lineno} (non-int class)")
                continue
            if cls_id < 0 or cls_id >= num_classes:
                result["unknown_class_ids"].append(cls_id)
            else:
                result["class_counts"][CLASSES[cls_id]] += 1

    if result["empty_labels"]:
        warnings.append(f"{len(result['empty_labels'])} label files are empty (background images)")
    if result["invalid_lines"]:
        warnings.append(f"{len(result['invalid_lines'])} malformed label lines")
    if result["unknown_class_ids"]:
        bad = sorted(set(result["unknown_class_ids"]))
        warnings.append(f"Unknown class IDs found: {bad} (expected 0–{num_classes - 1})")

    return result


def check_dataset(dataset_dir: Path) -> list[dict]:
    """Check all splits in a YOLO dataset directory."""
    results = []
    for split in SPLITS:
        split_dir = dataset_dir / split
        if split_dir.exists():
            results.append(check_split(split_dir))
        else:
            results.append({"split": split, "warnings": [f"Split '{split}' not found"]})
    return results


def class_imbalance_warnings(class_counts: dict[str, int], threshold: float = 10.0) -> list[str]:
    """Warn if any class has < 1/threshold of the max class count."""
    warnings = []
    counts = {k: v for k, v in class_counts.items() if v > 0}
    if not counts:
        return ["No labelled instances found"]
    max_count = max(counts.values())
    for cls, count in counts.items():
        if max_count / count > threshold:
            warnings.append(
                f"Class '{cls}' is severely underrepresented: {count} vs max {max_count} "
                f"({max_count / count:.0f}x imbalance)"
            )
    return warnings
