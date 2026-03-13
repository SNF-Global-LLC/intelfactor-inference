#!/usr/bin/env python3
"""Add hard-negative (background) images to the combined training set.

Hard negatives are defect-free images that teach the model not to fire
on clean metal surfaces.  They must have empty label files.

Usage:
    python scripts/add_hard_negatives.py --neg-dir /path/to/clean/images
    python scripts/add_hard_negatives.py --neg-dir datasets/raw/negatives --target-ratio 0.20
    python scripts/add_hard_negatives.py --neg-dir datasets/raw/negatives --augment --augment-factor 5
    python scripts/add_hard_negatives.py --combined-dir datasets/combined --target-ratio 0.15
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import DATASETS_DIR, COMBINED_DIR, ensure_dirs

log = logging.getLogger(__name__)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_existing(split_dir: Path) -> int:
    img_dir = split_dir / "images"
    if not img_dir.exists():
        return 0
    return sum(1 for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)


def collect_negatives(neg_dir: Path) -> list[Path]:
    """Recursively find all images in neg_dir."""
    return [
        p for p in sorted(neg_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]


def augment_images(sources: list[Path], target_count: int, out_dir: Path, seed: int) -> list[Path]:
    """Generate augmented variants of source images until target_count is reached.

    Augmentations applied (sequentially, cycling):
      1. Horizontal flip
      2. Vertical flip
      3. 90° clockwise rotation
      4. Brightness +30%
      5. Contrast +30%
      6. Brightness -20%
      7. Contrast -20%
      8. 180° rotation

    Returns list of generated file paths.
    """
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        log.error("Pillow not installed — cannot augment. Run: pip install pillow")
        sys.exit(1)

    ensure_dirs(out_dir)
    rng = random.Random(seed)
    generated: list[Path] = []

    # Build ordered list of (src, aug_index) pairs, shuffled, until we reach target
    aug_count = max(1, target_count - len(sources))  # how many new images to generate

    # All (src, aug_idx) combos, shuffled repeatably
    combos: list[tuple[Path, int]] = []
    factor = math.ceil(aug_count / len(sources)) + 1
    for aug_idx in range(1, factor + 1):
        for src in sources:
            combos.append((src, aug_idx))
    rng.shuffle(combos)
    combos = combos[:aug_count]

    def apply_aug(img: "Image.Image", aug_idx: int) -> "Image.Image":
        idx = aug_idx % 8
        if idx == 1:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        elif idx == 2:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        elif idx == 3:
            return img.transpose(Image.ROTATE_90)
        elif idx == 4:
            return ImageEnhance.Brightness(img).enhance(1.3)
        elif idx == 5:
            return ImageEnhance.Contrast(img).enhance(1.3)
        elif idx == 6:
            return ImageEnhance.Brightness(img).enhance(0.8)
        elif idx == 7:
            return ImageEnhance.Contrast(img).enhance(0.7)
        else:  # 0 → 180°
            return img.transpose(Image.ROTATE_180)

    for i, (src, aug_idx) in enumerate(combos):
        out_path = out_dir / f"aug_{i:05d}_{aug_idx}{src.suffix}"
        try:
            with Image.open(src) as img:
                aug = apply_aug(img.copy(), aug_idx)
                aug.save(out_path, quality=90)
            generated.append(out_path)
        except Exception as exc:
            log.warning("Augmentation failed for %s: %s", src, exc)

    log.info("Generated %d augmented images in %s", len(generated), out_dir)
    return generated


def add_to_split(
    neg_images: list[Path],
    split_dir: Path,
    count: int,
    dry_run: bool,
) -> int:
    """Copy `count` negatives into split_dir/images with empty label files."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not dry_run:
        ensure_dirs(img_dir, lbl_dir)

    sampled = random.sample(neg_images, min(count, len(neg_images)))
    for i, src in enumerate(sampled):
        stem = f"hardneg_{split_dir.name}_{i:05d}"
        dst_img = img_dir / (stem + src.suffix)
        dst_lbl = lbl_dir / (stem + ".txt")
        if not dry_run:
            shutil.copy2(src, dst_img)
            dst_lbl.write_text("")  # empty label = no defect
    return len(sampled)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add hard-negative images to training set")
    p.add_argument(
        "--neg-dir",
        required=True,
        help="Directory containing defect-free images (searched recursively)",
    )
    p.add_argument(
        "--combined-dir",
        default=str(COMBINED_DIR),
        help="Path to datasets/combined/",
    )
    p.add_argument(
        "--target-ratio",
        type=float,
        default=0.20,
        help="Target fraction of training images that should be backgrounds (default: 0.20)",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Generate synthetic augmented variants (flip, rotate, brightness/contrast) "
             "to multiply available negatives before adding to dataset",
    )
    p.add_argument(
        "--augment-factor",
        type=int,
        default=5,
        help="Target multiplier when --augment is set (default: 5x source images)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    random.seed(args.seed)

    neg_dir = Path(args.neg_dir)
    combined_dir = Path(args.combined_dir)

    if not neg_dir.exists():
        log.error("--neg-dir does not exist: %s", neg_dir)
        sys.exit(1)
    if not combined_dir.exists():
        log.error("Combined dataset not found at %s — run merge_datasets.py first", combined_dir)
        sys.exit(1)

    all_negs = collect_negatives(neg_dir)
    if not all_negs:
        log.error("No images found in %s", neg_dir)
        sys.exit(1)
    log.info("Found %d negative images in %s", len(all_negs), neg_dir)

    if args.augment:
        target_aug_count = len(all_negs) * args.augment_factor
        aug_cache = neg_dir.parent / "_aug_cache"
        log.info(
            "Augmenting: %d source images → %d total (%dx) in %s",
            len(all_negs), target_aug_count, args.augment_factor, aug_cache,
        )
        if not args.dry_run:
            aug_generated = augment_images(all_negs, target_aug_count, aug_cache, seed=args.seed)
            all_negs = all_negs + aug_generated
            log.info("Total negatives available (source + augmented): %d", len(all_negs))
        else:
            log.info("  [DRY RUN] Would generate %d augmented images", target_aug_count - len(all_negs))

    # Only add negatives to train (and proportionally val/test)
    split_fracs = {"train": 0.80, "val": 0.15, "test": 0.05}

    for split_name, frac in split_fracs.items():
        split_dir = combined_dir / split_name
        existing_count = count_existing(split_dir)
        if existing_count == 0:
            log.warning("Split '%s' is empty — skipping", split_name)
            continue

        # How many negatives do we need to hit target_ratio?
        # n_negs / (existing + n_negs) = ratio  →  n_negs = ratio * existing / (1 - ratio)
        ratio = args.target_ratio
        desired_negs = math.ceil(ratio * existing_count / (1.0 - ratio))
        # Scale by split fraction (most negatives go to train)
        desired_negs = max(1, int(desired_negs * frac / split_fracs["train"]))

        log.info(
            "Split '%s': %d existing images → adding %d negatives (target ratio %.0f%%)",
            split_name,
            existing_count,
            desired_negs,
            ratio * 100,
        )

        if args.dry_run:
            log.info("  [DRY RUN] Would copy %d images to %s", desired_negs, split_dir / "images")
        else:
            added = add_to_split(all_negs, split_dir, desired_negs, dry_run=False)
            log.info("  Added %d hard negatives to %s", added, split_name)

    if not args.dry_run:
        log.info("Done. Run analyze_dataset.py to verify class balance.")


if __name__ == "__main__":
    main()
