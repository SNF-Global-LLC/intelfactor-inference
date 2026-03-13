#!/usr/bin/env python3
"""
Kyoto Evaluation Batch Runner - 60 Inferences (20 images × 3 configs)
Uses Roboflow Python SDK
"""

import csv
import json
import time
from pathlib import Path

from roboflow import Roboflow

# Configuration
API_KEY = "REDACTED_KEY"
WORKSPACE = "intelfactor-aj2un"
PROJECT = "metal-surface-defects-rmbhy-szb6t"
VERSION = 6

INPUT_DIR = Path("kyoto_eval_set")
OUTPUT_DIR = Path("kyoto_raw_outputs")

# Threshold configs to test (confidence values are 0-100 for this API)
CONFIGS = [
    {"name": "config_a", "detector": 40, "classifier": 40},  # 0.4, 0.4
    {"name": "config_b", "detector": 50, "classifier": 50},  # 0.5, 0.5
    {"name": "config_c", "detector": 50, "classifier": 60},  # 0.5, 0.6
]


def run_inference(model, image_path, confidence):
    """Run inference on a single image."""
    try:
        predictions = model.predict(str(image_path), confidence=confidence)
        return {
            "success": True,
            "predictions": predictions.json(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def process_result(image_id, config, split, finish_type, expected_behavior, result):
    """Process raw result into structured row."""
    if not result["success"]:
        return {
            "image_id": image_id,
            "config": config["name"],
            "split": split,
            "finish_type": finish_type,
            "expected_behavior": expected_behavior,
            "detector_fired": "no",
            "detector_box_count": 0,
            "detector_top_class": "",
            "detector_top_confidence": 0.0,
            "detector_box_area_pct": 0.0,
            "classifier_ran": "no",
            "classifier_top_class": "",
            "classifier_top_confidence": 0.0,
            "error": result.get("error", ""),
        }
    
    preds_data = result["predictions"]
    predictions = preds_data.get("predictions", [])
    image_info = preds_data.get("image", {})
    img_width = float(image_info.get("width", 1))
    img_height = float(image_info.get("height", 1))
    
    # Determine if detector fired
    detector_fired = len(predictions) > 0
    
    # Get top prediction by confidence
    top_pred = None
    if predictions:
        top_pred = max(predictions, key=lambda p: p.get("confidence", 0))
    
    # Calculate box area percentage
    box_area_pct = 0.0
    if predictions and img_width > 0 and img_height > 0:
        total_area = sum(
            (p.get("width", 0) * p.get("height", 0)) / (img_width * img_height) * 100
            for p in predictions
        )
        box_area_pct = min(total_area, 100.0)
    
    # Check if classifier ran (based on classifier threshold)
    classifier_threshold = config["classifier"] / 100  # Convert 0-100 to 0-1
    classifier_candidates = [
        p for p in predictions
        if p.get("confidence", 0) >= classifier_threshold
    ]
    classifier_ran = len(classifier_candidates) > 0
    
    if classifier_candidates:
        top_classifier = max(classifier_candidates, key=lambda p: p.get("confidence", 0))
        classifier_class = top_classifier.get("class", "")
        classifier_confidence = top_classifier.get("confidence", 0)
    else:
        classifier_class = ""
        classifier_confidence = 0.0
    
    return {
        "image_id": image_id,
        "config": config["name"],
        "split": split,
        "finish_type": finish_type,
        "expected_behavior": expected_behavior,
        "detector_fired": "yes" if detector_fired else "no",
        "detector_box_count": len(predictions),
        "detector_top_class": top_pred.get("class", "") if top_pred else "",
        "detector_top_confidence": round(top_pred.get("confidence", 0), 3) if top_pred else 0.0,
        "detector_box_area_pct": round(box_area_pct, 2),
        "classifier_ran": "yes" if classifier_ran else "no",
        "classifier_top_class": classifier_class,
        "classifier_top_confidence": round(classifier_confidence, 3),
    }


def load_master_data():
    """Load metadata from kyoto_eval_master.csv."""
    master_path = INPUT_DIR / "kyoto_eval_master.csv"
    data = {}
    with open(master_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["image_id"]] = row
    return data


def main():
    print("=" * 60)
    print("KYOTO EVALUATION BATCH - 60 INFERENCES")
    print("=" * 60)
    print(f"Workspace: {WORKSPACE}")
    print(f"Project: {PROJECT}")
    print(f"Version: {VERSION}")
    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Roboflow
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    workspace = rf.workspace(WORKSPACE)
    project = workspace.project(PROJECT)
    version = project.version(VERSION)
    model = version.model
    print("✓ Connected")
    print()
    
    # Load master data
    master_data = load_master_data()
    
    # Get all images
    images = sorted(INPUT_DIR.glob("*.jpg"))
    print(f"Found {len(images)} images")
    print(f"Testing {len(CONFIGS)} threshold configs")
    print(f"Total inferences: {len(images) * len(CONFIGS)}")
    print()
    
    # Run all combinations
    all_results = []
    total = len(images) * len(CONFIGS)
    count = 0
    
    for image_path in images:
        image_id = image_path.stem
        meta = master_data.get(image_id, {})
        split = meta.get("split", "")
        finish_type = meta.get("finish_type", "")
        expected = meta.get("expected_behavior", "")
        
        for config in CONFIGS:
            count += 1
            print(f"[{count:3d}/{total}] {image_id} + {config['name']} "
                  f"(det={config['detector']/100:.1f}, cls={config['classifier']/100:.1f})")
            
            # Run inference
            result = run_inference(model, image_path, config["detector"])
            
            # Process result
            processed = process_result(
                image_id, config, split, finish_type, expected, result
            )
            all_results.append(processed)
            
            # Print brief result
            if processed.get("error"):
                print(f"      ✗ Error: {processed['error'][:50]}")
            else:
                fired = processed['detector_fired'] == 'yes'
                print(f"      {'✓' if fired else '○'} {processed['detector_box_count']} boxes, "
                      f"{processed['detector_top_class'] or 'none'} "
                      f"({processed['detector_top_confidence']:.2f})")
            
            # Rate limiting
            time.sleep(0.3)
    
    # Save raw results
    raw_file = OUTPUT_DIR / "raw_results.json"
    with open(raw_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Raw results saved: {raw_file}")
    
    # Create evaluation CSV
    csv_file = OUTPUT_DIR / "kyoto_eval_results.csv"
    with open(csv_file, "w", newline="") as f:
        fieldnames = [
            "image_id", "config", "split", "finish_type", "expected_behavior",
            "detector_fired", "detector_box_count", "detector_top_class",
            "detector_top_confidence", "detector_box_area_pct",
            "classifier_ran", "classifier_top_class", "classifier_top_confidence",
            "human_judgment", "likely_failure_mode", "final_decision", "comments"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Add empty human review columns
            row = {**result}
            row["human_judgment"] = ""
            row["likely_failure_mode"] = ""
            row["final_decision"] = ""
            row["comments"] = ""
            writer.writerow(row)
    
    print(f"✓ Evaluation CSV: {csv_file}")
    print()
    print("=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"\nNext: Fill human review columns in {csv_file}")
    print("  - human_judgment: TP, FP, FN, TN")
    print("  - likely_failure_mode: texture_confusion, lighting_artifact, etc.")
    print("  - final_decision: pass, review, fail")
    print("\nThen run:")
    print("  python \"Kimi_Agent_Workflow Evaluation Rubric/evaluate_workflow.py\" "
          f"{csv_file} --threshold-config B")


if __name__ == "__main__":
    main()
