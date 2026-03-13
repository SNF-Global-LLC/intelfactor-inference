#!/usr/bin/env python3
"""Merge multiple YOLO-format source datasets into datasets/combined/.

Sources are read from datasets/raw/ (Roboflow, Kaggle, custom).
Class names are remapped to IntelFactor canonical taxonomy via taxonomy.py.
Images + labels are copied into train/val/test splits with configurable ratios.
A data.yaml is written at the end.

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --config config/dataset_config.yaml --dry-run
    python scripts/merge_datasets.py --source datasets/raw/roboflow/metal-defect-detection
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATASETS_DIR, CONFIG_DIR, COMBINED_DIR, ensure_dirs
from utils.taxonomy import CLASSES, NUM_CLASSES, resolve, yolo_names_block
from utils.yaml_io import load_yaml, save_yaml
from utils.dataset_checks import check_dataset

log = logging.getLogger(__name__)

RAW_DIR = DATASETS_DIR / "raw"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

def remap_label_file(src: Path, dst: Path, class_map: dict[int, int] | None) -> int:
    """Copy a YOLO label file with optional class ID remapping.

    Returns number of lines kept (skips lines with unknown class IDs).
    """
    lines_out = []
    for line in src.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if class_map is not None:
            if cls_id not in class_map:
                continue  # skip classes not in target taxonomy
            cls_id = class_map[cls_id]
        lines_out.append(f"{cls_id} " + " ".join(parts[1:]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + "\n" if lines_out else "")
    return len(lines_out)


def build_class_map(source_names: list[str]) -> dict[int, int]:
    """Map source class IDs → IntelFactor class IDs using taxonomy.resolve()."""
    mapping: dict[int, int] = {}
    for src_id, src_name in enumerate(source_names):
        canonical = resolve(src_name)
        if canonical is None:
            log.warning("  Skipping unknown class: %r (add to taxonomy.ALIASES if needed)", src_name)
            continue
        if canonical == "background":
            continue  # background images handled separately
        try:
            dst_id = CLASSES.index(canonical)
            mapping[src_id] = dst_id
        except ValueError:
            log.warning("  Canonical class %r not in CLASSES list", canonical)
    return mapping


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def find_sources(raw_dir: Path, extra_sources: list[Path]) -> list[Path]:
    """Return a list of YOLO dataset root directories to merge."""
    sources: list[Path] = []

    # Auto-discover under raw/
    for subdir in ["roboflow", "kaggle", "custom"]:
        parent = raw_dir / subdir
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir()):
            if child.is_dir() and (child / "train").exists():
                sources.append(child)

    sources.extend(extra_sources)

    if not sources:
        log.warning("No source datasets found under %s", raw_dir)
    else:
        log.info("Found %d source dataset(s):", len(sources))
        for s in sources:
            log.info("  %s", s)
    return sources


# ---------------------------------------------------------------------------
# Merging logic
# ---------------------------------------------------------------------------

def collect_samples(source_dir: Path) -> dict[str, list[tuple[Path, Path]]]:
    """Return {split: [(image_path, label_path), ...]} for a source dataset."""
    samples: dict[str, list[tuple[Path, Path]]] = {s: [] for s in SPLITS}

    # Some Roboflow exports use 'valid' instead of 'val'
    split_aliases = {"val": ["val", "valid"]}

    # Some Roboflow exports have a flat structure, others have train/val/test
    for split in SPLITS:
        candidates = split_aliases.get(split, [split])
        split_dir = None
        for candidate in candidates:
            d = source_dir / candidate
            if d.exists():
                split_dir = d
                break
        if split_dir is None:
            continue
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.glob("*.*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            samples[split].append((img, lbl if lbl.exists() else Path("/dev/null")))

    return samples


def merge_source(
    source_dir: Path,
    combined_dir: Path,
    split_ratios: dict[str, float],
    dry_run: bool,
) -> dict[str, int]:
    """Merge one source into combined_dir. Returns {split: count}."""
    # Load source data.yaml to get class names
    data_yaml = source_dir / "data.yaml"
    source_names: list[str] = []
    if data_yaml.exists():
        data = load_yaml(data_yaml)
        raw_names = data.get("names", [])
        if isinstance(raw_names, dict):
            source_names = [raw_names[i] for i in sorted(raw_names)]
        else:
            source_names = raw_names
    class_map = build_class_map(source_names) if source_names else None

    all_samples: list[tuple[Path, Path]] = []
    split_samples = collect_samples(source_dir)

    # Flatten all splits — we'll re-split to ensure consistent ratios
    for split_list in split_samples.values():
        all_samples.extend(split_list)

    random.shuffle(all_samples)
    n = len(all_samples)
    if n == 0:
        log.warning("  No samples found in %s", source_dir)
        return {}

    train_end = int(n * split_ratios.get("train", 0.75))
    val_end = train_end + int(n * split_ratios.get("val", 0.15))

    split_assignment: dict[str, list[tuple[Path, Path]]] = {
        "train": all_samples[:train_end],
        "val": all_samples[train_end:val_end],
        "test": all_samples[val_end:],
    }

    counts: dict[str, int] = {}
    prefix = source_dir.name.replace(" ", "_")

    for split, pairs in split_assignment.items():
        dst_img_dir = combined_dir / split / "images"
        dst_lbl_dir = combined_dir / split / "labels"
        if not dry_run:
            ensure_dirs(dst_img_dir, dst_lbl_dir)
        count = 0
        for img_src, lbl_src in pairs:
            stem = f"{prefix}__{img_src.stem}"
            dst_img = dst_img_dir / (stem + img_src.suffix)
            dst_lbl = dst_lbl_dir / (stem + ".txt")
            if not dry_run:
                shutil.copy2(img_src, dst_img)
                if lbl_src.exists() and str(lbl_src) != "/dev/null":
                    remap_label_file(lbl_src, dst_lbl, class_map)
                else:
                    dst_lbl.write_text("")  # empty = background image
            count += 1
        counts[split] = count
        log.info("  %s → %s: %d images", source_dir.name, split, count)

    return counts


def write_data_yaml(combined_dir: Path) -> None:
    """Write data.yaml for the merged combined dataset."""
    data = {
        "path": str(combined_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NUM_CLASSES,
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    out = combined_dir / "data.yaml"
    save_yaml(
        data,
        out,
        comment=(
            "IntelFactor combined dataset — auto-generated by merge_datasets.py\n"
            "Do not edit manually; re-run the script to regenerate."
        ),
    )
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge YOLO datasets into combined/")
    p.add_argument("--config", default=str(CONFIG_DIR / "dataset_config.yaml"))
    p.add_argument(
        "--source",
        action="append",
        dest="extra_sources",
        default=[],
        metavar="DIR",
        help="Additional source dataset directories (repeatable)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for split assignment")
    p.add_argument("--dry-run", action="store_true", help="Print plan without copying files")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    random.seed(args.seed)

    cfg = load_yaml(args.config)
    split_ratios = cfg.get("splits", {"train": 0.75, "val": 0.15, "test": 0.10})
    extra_sources = [Path(s) for s in args.extra_sources]

    sources = find_sources(RAW_DIR, extra_sources)
    if not sources:
        log.error("No sources to merge. Run download_datasets.py first.")
        sys.exit(1)

    if args.dry_run:
        log.info("DRY RUN — no files will be written.")
    else:
        # Clear combined dir before re-merging to avoid stale data
        if COMBINED_DIR.exists():
            log.info("Clearing existing combined dataset at %s", COMBINED_DIR)
            shutil.rmtree(COMBINED_DIR)
        ensure_dirs(COMBINED_DIR)

    total: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for source_dir in sources:
        log.info("Merging: %s", source_dir)
        counts = merge_source(source_dir, COMBINED_DIR, split_ratios, dry_run=args.dry_run)
        for split, count in counts.items():
            total[split] = total.get(split, 0) + count

    log.info("Merge complete. Total: train=%d val=%d test=%d", total["train"], total["val"], total["test"])

    if not args.dry_run:
        write_data_yaml(COMBINED_DIR)
        log.info("Run analyze_dataset.py to validate the merged dataset.")


if __name__ == "__main__":
    main()
