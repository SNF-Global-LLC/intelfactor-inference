#!/usr/bin/env python3
"""
Merge Retrain v1 Sprint Dataset (700 images)
Combines Wiko, NEU-DET, GC10-DET (if available), and hard negatives.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# Class mappings
NEU_DET_MAPPING = {
    "scratches": "blade_scratch",
    "inclusion": "inclusion", 
    "pitted_surface": "surface_dent",
    "crazing": "surface_crack",
    "rolled-in_scale": "grinding_mark",
    # "patches": SKIP — too different
}

GC10_DET_MAPPING = {
    "Inclusion": "inclusion",
    "Rolled_pit": "surface_dent",
    "Weld_line": "weld_defect",
    "Punching_hole": "surface_dent",
    "Crease": "grinding_mark",
    "Crescent_gap": "edge_burr",
    "Waist_folding": "edge_burr",
    # Skip: Water_spot, Oil_spot, Silk_spot
}

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
    """Load YOLO format labels."""
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
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
    """Save YOLO format labels."""
    with open(label_path, "w") as f:
        for label in labels:
            f.write(f"{label['class_id']} {label['x_center']} {label['y_center']} "
                   f"{label['width']} {label['height']}\n")


def get_class_names(dataset_path: Path) -> List[str]:
    """Get class names from dataset YAML."""
    yaml_path = dataset_path / "data.yaml"
    if not yaml_path.exists():
        return []
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def remap_labels(labels: List[Dict], source_classes: List[str], 
                 mapping: Dict[str, str], unified_classes: List[str]) -> List[Dict]:
    """Remap labels to unified class space."""
    remapped = []
    for label in labels:
        source_class = source_classes[label["class_id"]]
        target_class = mapping.get(source_class)
        if target_class and target_class in unified_classes:
            new_id = unified_classes.index(target_class)
            remapped.append({
                "class_id": new_id,
                "x_center": label["x_center"],
                "y_center": label["y_center"],
                "width": label["width"],
                "height": label["height"],
            })
    return remapped


def select_neu_det_samples(neu_det_path: Path, target_count: int) -> List[Tuple[str, Path]]:
    """Select NEU-DET samples with per-class caps."""
    class_caps = {
        "scratches": 70,
        "pitted_surface": 50,
        "crazing": 30,
        "inclusion": 20,
        "rolled-in_scale": 10,
    }
    
    selected = []
    class_counts = {c: 0 for c in class_caps.keys()}
    
    # Collect all samples
    all_samples = []
    for split in ["train", "valid", "test"]:
        img_dir = neu_det_path / split / "images"
        lbl_dir = neu_det_path / split / "labels"
        if not img_dir.exists():
            continue
        for img_file in img_dir.glob("*.jpg"):
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                labels = load_yolo_labels(lbl_file)
                if labels:
                    # Get primary class (first or most confident)
                    class_id = labels[0]["class_id"]
                    all_samples.append((img_file, lbl_file, class_id))
    
    # Get class names
    class_names = get_class_names(neu_det_path)
    
    # Shuffle for random selection
    random.shuffle(all_samples)
    
    # Select with caps
    for img_file, lbl_file, class_id in all_samples:
        if len(selected) >= target_count:
            break
        
        class_name = class_names[class_id] if class_id < len(class_names) else ""
        if class_name in class_caps and class_counts[class_name] < class_caps[class_name]:
            selected.append((img_file, lbl_file))
            class_counts[class_name] += 1
    
    return selected


def copy_to_split(files: List[Tuple[Path, Path]], dst_img_dir: Path, dst_lbl_dir: Path,
                  class_mapping: Dict[str, str] = None, source_classes: List[str] = None,
                  unified_classes: List[str] = None, prefix: str = ""):
    """Copy files to destination with optional remapping."""
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file, lbl_file in files:
        # Copy image
        new_name = f"{prefix}{img_file.name}"
        shutil.copy2(img_file, dst_img_dir / new_name)
        
        # Process labels
        if class_mapping and source_classes and unified_classes:
            labels = load_yolo_labels(lbl_file)
            remapped = remap_labels(labels, source_classes, class_mapping, unified_classes)
            if remapped:
                save_yolo_labels(dst_lbl_dir / f"{Path(new_name).stem}.txt", remapped)
        else:
            # Direct copy (no remapping needed)
            shutil.copy2(lbl_file, dst_lbl_dir / f"{Path(new_name).stem}.txt")


def main():
    parser = argparse.ArgumentParser(description="Merge retrain v1 sprint dataset")
    parser.add_argument("--wiko", type=Path, required=True, help="Wiko dataset path")
    parser.add_argument("--neu-det", type=Path, help="NEU-DET dataset path")
    parser.add_argument("--gc10-det", type=Path, help="GC10-DET dataset path (optional)")
    parser.add_argument("--hard-negatives", type=Path, help="Hard negatives directory")
    parser.add_argument("--output", type=Path, default=Path("datasets/retrain_v1_sprint"))
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print("=" * 60)
    print("MERGE RETRAIN V1 SPRINT DATASET")
    print("=" * 60)
    
    # Create output structure
    train_img_dir = args.output / "train" / "images"
    train_lbl_dir = args.output / "train" / "labels"
    val_img_dir = args.output / "val" / "images"
    val_lbl_dir = args.output / "val" / "labels"
    
    # Track statistics
    stats = {"train": 0, "val": 0, "by_source": {}}
    
    # 1. Process Wiko/Kyoto (domain anchor, P0)
    print("\n[1/4] Processing Wiko/Kyoto positives...")
    if args.wiko.exists():
        wiko_files = []
        for split in ["train", "valid", "test"]:
            img_dir = args.wiko / split / "images"
            lbl_dir = args.wiko / split / "labels"
            if img_dir.exists():
                for img_file in img_dir.glob("*.jpg"):
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        wiko_files.append((img_file, lbl_file))
        
        # Split train/val
        random.shuffle(wiko_files)
        split_idx = int(len(wiko_files) * (1 - args.val_split))
        train_files = wiko_files[:split_idx]
        val_files = wiko_files[split_idx:]
        
        copy_to_split(train_files, train_img_dir, train_lbl_dir, prefix="wiko_")
        copy_to_split(val_files, val_img_dir, val_lbl_dir, prefix="wiko_")
        
        stats["train"] += len(train_files)
        stats["val"] += len(val_files)
        stats["by_source"]["wiko"] = len(wiko_files)
        print(f"  Added {len(wiko_files)} Wiko samples ({len(train_files)} train, {len(val_files)} val)")
    
    # 2. Process NEU-DET (broad priors, P1)
    print("\n[2/4] Processing NEU-DET...")
    if args.neu_det and args.neu_det.exists():
        neu_files = select_neu_det_samples(args.neu_det, target_count=180)
        neu_classes = get_class_names(args.neu_det)
        
        # Split train/val
        random.shuffle(neu_files)
        split_idx = int(len(neu_files) * (1 - args.val_split))
        train_files = neu_files[:split_idx]
        val_files = neu_files[split_idx:]
        
        copy_to_split(train_files, train_img_dir, train_lbl_dir, 
                     NEU_DET_MAPPING, neu_classes, UNIFIED_CLASSES, prefix="neu_")
        copy_to_split(val_files, val_img_dir, val_lbl_dir,
                     NEU_DET_MAPPING, neu_classes, UNIFIED_CLASSES, prefix="neu_")
        
        stats["train"] += len(train_files)
        stats["val"] += len(val_files)
        stats["by_source"]["neu_det"] = len(neu_files)
        print(f"  Added {len(neu_files)} NEU-DET samples ({len(train_files)} train, {len(val_files)} val)")
    
    # 3. Process GC10-DET (extended diversity, P2) - OPTIONAL
    print("\n[3/4] Processing GC10-DET...")
    if args.gc10_det and args.gc10_det.exists():
        gc10_files = []
        for split in ["train", "valid", "test"]:
            img_dir = args.gc10_det / split / "images"
            lbl_dir = args.gc10_det / split / "labels"
            if img_dir.exists():
                for img_file in img_dir.glob("*.jpg"):
                    lbl_file = lbl_dir / f"{img_file.stem}.txt"
                    if lbl_file.exists():
                        gc10_files.append((img_file, lbl_file))
        
        gc10_classes = get_class_names(args.gc10_det)
        
        # Limit to 120 samples
        random.shuffle(gc10_files)
        gc10_files = gc10_files[:120]
        
        # Split train/val
        split_idx = int(len(gc10_files) * (1 - args.val_split))
        train_files = gc10_files[:split_idx]
        val_files = gc10_files[split_idx:]
        
        copy_to_split(train_files, train_img_dir, train_lbl_dir,
                     GC10_DET_MAPPING, gc10_classes, UNIFIED_CLASSES, prefix="gc10_")
        copy_to_split(val_files, val_img_dir, val_lbl_dir,
                     GC10_DET_MAPPING, gc10_classes, UNIFIED_CLASSES, prefix="gc10_")
        
        stats["train"] += len(train_files)
        stats["val"] += len(val_files)
        stats["by_source"]["gc10_det"] = len(gc10_files)
        print(f"  Added {len(gc10_files)} GC10-DET samples ({len(train_files)} train, {len(val_files)} val)")
    else:
        print("  GC10-DET not found, skipping (optional)")
    
    # 4. Process Hard Negatives (THE FIX, P0)
    print("\n[4/4] Processing Hard Negatives...")
    if args.hard_negatives and args.hard_negatives.exists():
        neg_files = []
        for img_file in args.hard_negatives.glob("**/*.jpg"):
            # Hard negatives have empty label files
            lbl_file = args.hard_negatives / "labels" / f"{img_file.stem}.txt"
            neg_files.append((img_file, lbl_file))
        
        # Split train/val
        random.shuffle(neg_files)
        split_idx = int(len(neg_files) * (1 - args.val_split))
        train_files = neg_files[:split_idx]
        val_files = neg_files[split_idx:]
        
        copy_to_split(train_files, train_img_dir, train_lbl_dir, prefix="neg_")
        copy_to_split(val_files, val_img_dir, val_lbl_dir, prefix="neg_")
        
        stats["train"] += len(train_files)
        stats["val"] += len(val_files)
        stats["by_source"]["hard_negatives"] = len(neg_files)
        print(f"  Added {len(neg_files)} hard negative samples ({len(train_files)} train, {len(val_files)} val)")
    else:
        print("  Hard negatives not found, skipping")
    
    # Create data.yaml
    print("\n[5/5] Creating data.yaml...")
    yaml_data = {
        "path": str(args.output.absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES,
    }
    
    with open(args.output / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"\nTotal images: {stats['train'] + stats['val']}")
    print(f"  Train: {stats['train']} ({stats['train']/(stats['train']+stats['val'])*100:.1f}%)")
    print(f"  Val: {stats['val']} ({stats['val']/(stats['train']+stats['val'])*100:.1f}%)")
    print(f"\nBy source:")
    for source, count in stats["by_source"].items():
        print(f"  {source}: {count}")
    
    print(f"\nOutput: {args.output}")
    print(f"  data.yaml: {args.output / 'data.yaml'}")
    print("\nNext steps:")
    print("  1. Verify dataset structure")
    print("  2. Train: yolo detect train data=" + str(args.output / "data.yaml") + " model=yolov8n.pt epochs=100")
    print("  3. Evaluate: python run_kyoto_batch.py")


if __name__ == "__main__":
    main()
