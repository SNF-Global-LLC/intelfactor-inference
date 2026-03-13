#!/usr/bin/env python3
"""
Merge metal defect datasets (NEU-DET, GC10-DET) with Wiko dataset.
Handles class remapping and unified output format.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# Class mapping configuration
CLASS_MAPPINGS = {
    # NEU-DET classes -> Wiko unified classes
    "neu-det": {
        "scratches": "blade_scratch",
        "inclusion": "inclusion",
        "patches": "surface_discolor",
        "pitted_surface": "surface_dent",
        "rolled-in_scale": "grinding_mark",
        "crazing": "surface_crack",
    },
    # GC10-DET classes -> Wiko unified classes
    "gc10-det": {
        "Punching_hole": "surface_dent",
        "Weld_line": "weld_defect",
        "Crescent_gap": "edge_burr",
        "Water_spot": "surface_discolor",
        "Oil_spot": "surface_discolor",
        "Silk_spot": "surface_discolor",
        "Inclusion": "inclusion",
        "Rolled_pit": "surface_dent",
        "Crease": "grinding_mark",
        "Waist_folding": "edge_burr",
    },
}

# Unified class list (Wiko taxonomy)
UNIFIED_CLASSES = [
    "blade_scratch",
    "grinding_mark",
    "surface_dent",
    "surface_crack",
    "weld_defect",
    "edge_burr",
    "edge_crack",
    "handle_defect",
    "bolster_gap",
    "etching_defect",
    "inclusion",
    "surface_discolor",
    "overgrind",
]


def load_yolo_labels(label_path: Path) -> List[Dict]:
    """Load YOLO format labels from file."""
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    "class_id": int(parts[0]),
                    "x_center": float(parts[1]),
                    "y_center": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                })
    return labels


def save_yolo_labels(label_path: Path, labels: List[Dict]):
    """Save YOLO format labels to file."""
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{label['class_id']} {label['x_center']} {label['y_center']} "
                   f"{label['width']} {label['height']}\n")


def get_class_names(dataset_path: Path, dataset_type: str) -> List[str]:
    """Get class names from dataset YAML."""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        return []
    
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    
    return data.get("names", [])


def remap_labels(
    labels: List[Dict],
    source_classes: List[str],
    class_mapping: Dict[str, str],
    unified_classes: List[str],
) -> List[Dict]:
    """Remap labels from source classes to unified classes."""
    remapped = []
    
    for label in labels:
        source_class = source_classes[label["class_id"]]
        target_class = class_mapping.get(source_class)
        
        if target_class and target_class in unified_classes:
            new_class_id = unified_classes.index(target_class)
            remapped.append({
                "class_id": new_class_id,
                "x_center": label["x_center"],
                "y_center": label["y_center"],
                "width": label["width"],
                "height": label["height"],
            })
    
    return remapped


def merge_dataset(
    source_path: Path,
    output_path: Path,
    dataset_type: str,
    splits: List[str],
    unified_classes: List[str],
) -> Tuple[int, int, int]:
    """Merge a dataset into the output directory.
    
    Returns:
        Tuple of (images_copied, labels_copied, labels_remapped)
    """
    if not source_path.exists():
        print(f"  Warning: {source_path} does not exist, skipping")
        return 0, 0, 0
    
    class_mapping = CLASS_MAPPINGS.get(dataset_type, {})
    source_classes = get_class_names(source_path, dataset_type)
    
    if not class_mapping:
        print(f"  Warning: No class mapping for {dataset_type}")
        return 0, 0, 0
    
    print(f"  Source classes: {source_classes}")
    print(f"  Class mapping: {class_mapping}")
    
    images_copied = 0
    labels_copied = 0
    labels_remapped = 0
    
    for split in splits:
        src_img_dir = source_path / split / "images"
        src_lbl_dir = source_path / split / "labels"
        
        dst_img_dir = output_path / split / "images"
        dst_lbl_dir = output_path / split / "labels"
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_img_dir.exists():
            continue
        
        for img_file in src_img_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            # Copy image
            shutil.copy2(img_file, dst_img_dir / img_file.name)
            images_copied += 1
            
            # Process labels
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            dst_lbl_file = dst_lbl_dir / f"{img_file.stem}.txt"
            
            if lbl_file.exists():
                labels = load_yolo_labels(lbl_file)
                if labels:
                    remapped = remap_labels(
                        labels, source_classes, class_mapping, unified_classes
                    )
                    if remapped:
                        save_yolo_labels(dst_lbl_file, remapped)
                        labels_copied += 1
                        labels_remapped += len(remapped)
    
    return images_copied, labels_copied, labels_remapped


def create_unified_yaml(output_path: Path, unified_classes: List[str]):
    """Create unified data.yaml file."""
    yaml_data = {
        "path": str(output_path.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(unified_classes),
        "names": unified_classes,
    }
    
    with open(output_path / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nCreated unified data.yaml with {len(unified_classes)} classes")


def main():
    parser = argparse.ArgumentParser(
        description="Merge metal defect datasets with Wiko"
    )
    parser.add_argument(
        "--wiko",
        type=Path,
        required=True,
        help="Path to Wiko dataset",
    )
    parser.add_argument(
        "--neu-det",
        type=Path,
        help="Path to NEU-DET dataset",
    )
    parser.add_argument(
        "--gc10-det",
        type=Path,
        help="Path to GC10-DET dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/merged_metal_defects"),
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to process",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("METAL DEFECT DATASET MERGER")
    print("=" * 60)
    print(f"\nUnified classes ({len(UNIFIED_CLASSES)}):")
    for i, cls in enumerate(UNIFIED_CLASSES):
        print(f"  {i}: {cls}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    total_stats = {
        "images": 0,
        "labeled_images": 0,
        "annotations": 0,
    }
    
    # Process Wiko dataset (copy as-is, already in unified format)
    print("\n" + "-" * 60)
    print("Processing Wiko dataset...")
    print("-" * 60)
    for split in args.splits:
        src_img_dir = args.wiko / split / "images"
        src_lbl_dir = args.wiko / split / "labels"
        
        dst_img_dir = args.output / split / "images"
        dst_lbl_dir = args.output / split / "labels"
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_img_dir.exists():
            continue
        
        for img_file in src_img_dir.iterdir():
            if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            shutil.copy2(img_file, dst_img_dir / img_file.name)
            total_stats["images"] += 1
            
            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)
                total_stats["labeled_images"] += 1
                # Count annotations
                with open(lbl_file) as f:
                    total_stats["annotations"] += len(f.readlines())
    
    print(f"  Copied {total_stats['images']} images")
    print(f"  Copied {total_stats['labeled_images']} labeled images")
    
    # Process NEU-DET
    if args.neu_det and args.neu_det.exists():
        print("\n" + "-" * 60)
        print("Processing NEU-DET dataset...")
        print("-" * 60)
        stats = merge_dataset(
            args.neu_det, args.output, "neu-det",
            args.splits, UNIFIED_CLASSES
        )
        print(f"  Copied {stats[0]} images")
        print(f"  Copied {stats[1]} labeled images")
        print(f"  Remapped {stats[2]} annotations")
        total_stats["images"] += stats[0]
        total_stats["labeled_images"] += stats[1]
        total_stats["annotations"] += stats[2]
    
    # Process GC10-DET
    if args.gc10_det and args.gc10_det.exists():
        print("\n" + "-" * 60)
        print("Processing GC10-DET dataset...")
        print("-" * 60)
        stats = merge_dataset(
            args.gc10_det, args.output, "gc10-det",
            args.splits, UNIFIED_CLASSES
        )
        print(f"  Copied {stats[0]} images")
        print(f"  Copied {stats[1]} labeled images")
        print(f"  Remapped {stats[2]} annotations")
        total_stats["images"] += stats[0]
        total_stats["labeled_images"] += stats[1]
        total_stats["annotations"] += stats[2]
    
    # Create unified YAML
    create_unified_yaml(args.output, UNIFIED_CLASSES)
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nTotal images: {total_stats['images']}")
    print(f"Total labeled images: {total_stats['labeled_images']}")
    print(f"Total annotations: {total_stats['annotations']}")
    print(f"\nOutput directory: {args.output}")
    print("\nNext steps:")
    print(f"  1. Train model: yolo detect train data={args.output}/data.yaml")
    print(f"  2. Export to TRT: yolo export model=best.pt format=engine")
    print(f"  3. Re-run Kyoto evaluation")


if __name__ == "__main__":
    main()
