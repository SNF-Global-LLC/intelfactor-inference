#!/usr/bin/env python3
"""
Download metal defect datasets for knife-blade inspection training.
Sources: NEU-DET, GC10-DET, and others from metal-defect-datasets repo.
"""

import argparse
import logging
import os
import zipfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "neu-det": {
        "name": "NEU-DET",
        "roboflow_url": "https://universe.roboflow.com/defectdatasets/neu-det-fquva/1",
        "roboflow_download": "https://universe.roboflow.com/defectdatasets/neu-det-fquva/1/download/yolov8",
        "description": "Surface metal defects: scratches, inclusions, pitted/dented regions",
        "classes": ["Scratches", "Inclusion", "Patches", "Pitted", "Rolled", "Crazing"],
        "recommended": True,
    },
    "gc10-det": {
        "name": "GC10-DET",
        "roboflow_url": "https://universe.roboflow.com/g-deepti-raj/gc10-det-latest/3",
        "roboflow_download": "https://universe.roboflow.com/g-deepti-raj/gc10-det-latest/3/download/yolov8",
        "description": "Multiple industrial metal-surface defect patterns",
        "classes": ["Punching_hole", "Weld_line", "Crescent_gap", "Water_spot", 
                    "Oil_spot", "Silk_spot", "Inclusion", "Rolled_pit", "Crease", "Waist_folding"],
        "note": "Class naming needs correction when downloading",
        "recommended": True,
    },
    "kolektorsdd2": {
        "name": "KolektorSDD2",
        "roboflow_url": "https://universe.roboflow.com/defectdatasets/kolektorsdd2-xnm8r/2",
        "roboflow_download": "https://universe.roboflow.com/defectdatasets/kolektorsdd2-xnm8r/2/download/yolov8",
        "description": "Synthetic defects on electrical commutators",
        "classes": ["defect"],
        "recommended": False,
    },
    "mtd": {
        "name": "Magnetic-tile-defect",
        "roboflow_url": "https://universe.roboflow.com/defectdatasets/magnatic-tile-defect/1",
        "roboflow_download": "https://universe.roboflow.com/defectdatasets/magnatic-tile-defect/1/download/yolov8",
        "description": "Magnetic tile surface defects",
        "classes": ["blowhole", "crack", "fray", "breakage", "uneven"],
        "recommended": False,
    },
    "nrsd-cr": {
        "name": "NRSD-CR",
        "roboflow_url": "https://universe.roboflow.com/defectdatasets/nrsd-cr/1",
        "roboflow_download": "https://universe.roboflow.com/defectdatasets/nrsd-cr/1/download/yolov8",
        "description": "Non-rotated surface defect detection for cold-rolled steel",
        "classes": ["defect"],
        "recommended": False,
    },
}


def download_from_roboflow(dataset_key, output_dir, api_key=None):
    """Download a dataset from Roboflow."""
    dataset = DATASETS[dataset_key]
    logger.info(f"Downloading {dataset['name']}...")
    logger.info(f"  Description: {dataset['description']}")
    
    # Create output directory
    dataset_dir = output_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to use roboflow package if available
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key or os.environ.get("ROBOFLOW_API_KEY"))
        
        # Parse workspace/project from URL
        # URL format: https://universe.roboflow.com/{workspace}/{project}/{version}
        url_parts = dataset["roboflow_url"].replace("https://universe.roboflow.com/", "").split("/")
        workspace = url_parts[0]
        project = url_parts[1]
        version = int(url_parts[2]) if len(url_parts) > 2 else 1
        
        logger.info(f"  Workspace: {workspace}, Project: {project}, Version: {version}")
        
        proj = rf.workspace(workspace).project(project)
        ver = proj.version(version)
        ver.download("yolov8", location=str(dataset_dir))
        
        logger.info(f"  ✓ Downloaded to {dataset_dir}")
        return True
        
    except ImportError:
        logger.error("  ✗ roboflow package not installed")
        logger.info("  Install with: pip install roboflow")
        return False
    except Exception as e:
        logger.error(f"  ✗ Download failed: {e}")
        return False


def download_direct_zip(dataset_key, output_dir):
    """Download dataset as ZIP if direct link is available."""
    dataset = DATASETS[dataset_key]
    dataset_dir = output_dir / dataset_key
    zip_path = output_dir / f"{dataset_key}.zip"
    
    # Note: Roboflow requires API key for downloads
    # This is a placeholder for direct download links
    logger.info(f"Direct download not available for {dataset['name']}")
    logger.info(f"  Please download manually from: {dataset['roboflow_url']}")
    logger.info(f"  Format: YOLOv8, extract to: {dataset_dir}")
    return False


def create_dataset_config(output_dir, selected_datasets):
    """Create a YAML config for merging datasets."""
    config_path = output_dir / "merged_dataset_config.yaml"
    
    lines = [
        "# Merged Metal Defect Dataset Configuration",
        f"# Generated for knife-blade inspection training",
        "",
        "path: ./merged",  # Will be created by merge script
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        "# Class names (unified across datasets)",
        "names:",
    ]
    
    # Map classes from all selected datasets
    all_classes = set()
    for key in selected_datasets:
        all_classes.update(DATASETS[key].get("classes", []))
    
    # Add our custom classes for knife blades
    unified_classes = [
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
    
    for i, cls in enumerate(unified_classes):
        lines.append(f"  {i}: {cls}")
    
    lines.extend([
        "",
        "# Source datasets",
        "sources:",
    ])
    
    for key in selected_datasets:
        ds = DATASETS[key]
        lines.extend([
            f"  - name: {ds['name']}",
            f"    path: {key}",
            f"    classes: {ds.get('classes', [])}",
        ])
    
    with open(config_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Created dataset config: {config_path}")


def analyze_datasets(output_dir, selected_datasets):
    """Analyze downloaded datasets."""
    logger.info("\n" + "=" * 60)
    logger.info("DATASET ANALYSIS")
    logger.info("=" * 60)
    
    for key in selected_datasets:
        dataset_dir = output_dir / key
        if not dataset_dir.exists():
            continue
        
        ds = DATASETS[key]
        logger.info(f"\n{ds['name']}:")
        
        # Count images
        for split in ["train", "valid", "test"]:
            img_dir = dataset_dir / split / "images"
            if img_dir.exists():
                count = len(list(img_dir.glob("*.*")))
                logger.info(f"  {split}: {count} images")
        
        # Show class mapping
        logger.info(f"  Classes: {', '.join(ds.get('classes', []))}")


def main():
    parser = argparse.ArgumentParser(
        description="Download metal defect datasets for knife-blade inspection"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("datasets/metal_defects"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        choices=list(DATASETS.keys()) + ["recommended", "all"],
        default=["recommended"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Metal Defect Datasets:")
        print("=" * 60)
        for key, ds in DATASETS.items():
            rec = "★ RECOMMENDED" if ds.get("recommended") else ""
            print(f"\n{key}: {ds['name']} {rec}")
            print(f"  {ds['description']}")
            print(f"  Classes: {', '.join(ds.get('classes', []))}")
            if ds.get("note"):
                print(f"  Note: {ds['note']}")
        print()
        return
    
    # Determine which datasets to download
    if "recommended" in args.datasets:
        selected = [k for k, v in DATASETS.items() if v.get("recommended")]
    elif "all" in args.datasets:
        selected = list(DATASETS.keys())
    else:
        selected = args.datasets
    
    logger.info(f"Selected datasets: {[DATASETS[k]['name'] for k in selected]}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Download each dataset
    successful = []
    for key in selected:
        if download_from_roboflow(key, args.output, args.api_key):
            successful.append(key)
        else:
            download_direct_zip(key, args.output)
    
    # Create config
    if successful:
        create_dataset_config(args.output, successful)
        analyze_datasets(args.output, successful)
        
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS")
        logger.info("=" * 60)
        logger.info("1. Review downloaded datasets")
        logger.info("2. Merge with existing Wiko dataset:")
        logger.info(f"   python training/scripts/merge_datasets.py --config {args.output}/merged_dataset_config.yaml")
        logger.info("3. Retrain model:")
        logger.info("   make train")
    else:
        logger.warning("No datasets were successfully downloaded")
        logger.info("\nTo download manually:")
        for key in selected:
            ds = DATASETS[key]
            logger.info(f"  {ds['name']}: {ds['roboflow_url']}")


if __name__ == "__main__":
    main()
