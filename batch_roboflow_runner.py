#!/usr/bin/env python3
"""
Batch Roboflow Workflow Inference Runner

Executes 20 images × 3 configs = 60 total inferences against Roboflow Workflow API.

Usage:
    python batch_roboflow_runner.py kyoto_eval_set kyoto_eval_outputs
    
Environment:
    ROBOFLOW_API_KEY - Your Roboflow API key
    ROBOFLOW_WORKSPACE - Workspace name (default: intelfactor)
    ROBOFLOW_WORKFLOW - Workflow ID (default: quality-gate-v1)
"""

import argparse
import base64
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


def run_workflow_inference(
    api_key: str,
    workspace: str,
    workflow: str,
    image_path: Path,
    config: dict,
) -> dict:
    """Run a single image through Roboflow Workflow API."""
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Build workflow URL
    url = f"https://detect.roboflow.com/{workflow}"
    
    # Prepare payload with config thresholds
    payload = {
        "api_key": api_key,
        "image": image_b64,
        "confidence": config.get("detector", 0.5),
        "overlap": 0.3,
        "format": "json",
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        return {
            "success": True,
            "image_id": image_path.stem,
            "config_name": config.get("name", "unknown"),
            "detector_threshold": config.get("detector"),
            "classifier_threshold": config.get("classifier"),
            "raw_response": result,
            "predictions": result.get("predictions", []),
            "inference_time_ms": result.get("inference_time", 0),
        }
    except Exception as e:
        return {
            "success": False,
            "image_id": image_path.stem,
            "config_name": config.get("name", "unknown"),
            "error": str(e),
        }


def process_predictions(
    predictions: list,
    classifier_threshold: float,
) -> dict:
    """Process raw predictions into structured format."""
    if not predictions:
        return {
            "detector_fired": False,
            "detector_box_count": 0,
            "detector_top_class": "",
            "detector_top_confidence": 0.0,
            "detector_box_area_pct": 0.0,
            "classifier_ran": False,
            "classifier_top_class": "",
            "classifier_top_confidence": 0.0,
        }
    
    # Get top prediction by confidence
    top_pred = max(predictions, key=lambda p: p.get("confidence", 0))
    
    # Calculate total box area percentage
    total_area_pct = sum(
        p.get("width", 0) * p.get("height", 0) / 10000  # Assuming normalized coords
        for p in predictions
    )
    
    # Determine if classifier would run (any box above classifier threshold)
    classifier_candidates = [
        p for p in predictions 
        if p.get("confidence", 0) >= classifier_threshold
    ]
    classifier_ran = len(classifier_candidates) > 0
    
    if classifier_candidates:
        top_classifier = max(classifier_candidates, key=lambda p: p.get("confidence", 0))
        classifier_class = top_classifier.get("class", "")
        classifier_conf = top_classifier.get("confidence", 0)
    else:
        classifier_class = ""
        classifier_conf = 0.0
    
    return {
        "detector_fired": True,
        "detector_box_count": len(predictions),
        "detector_top_class": top_pred.get("class", ""),
        "detector_top_confidence": top_pred.get("confidence", 0),
        "detector_box_area_pct": min(total_area_pct, 100.0),
        "classifier_ran": classifier_ran,
        "classifier_top_class": classifier_class,
        "classifier_top_confidence": classifier_conf,
    }


def run_batch(
    input_dir: Path,
    output_dir: Path,
    api_key: str,
    workspace: str,
    workflow: str,
):
    """Run batch inference on all images with all configs."""
    
    # Load configs
    configs = [
        {"name": "config_a", "detector": 0.4, "classifier": 0.4},
        {"name": "config_b", "detector": 0.5, "classifier": 0.5},
        {"name": "config_c", "detector": 0.5, "classifier": 0.6},
    ]
    
    # Get all images
    images = sorted(input_dir.glob("*.jpg"))
    if not images:
        raise ValueError(f"No .jpg images found in {input_dir}")
    
    print(f"Found {len(images)} images")
    print(f"Running {len(images)} × {len(configs)} = {len(images) * len(configs)} inferences")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run all combinations
    all_results = []
    total = len(images) * len(configs)
    count = 0
    
    for image_path in images:
        for config in configs:
            count += 1
            print(f"\n[{count}/{total}] {image_path.name} + {config['name']}")
            
            # Run inference
            result = run_workflow_inference(
                api_key=api_key,
                workspace=workspace,
                workflow=workflow,
                image_path=image_path,
                config=config,
            )
            
            if result["success"]:
                # Process predictions
                processed = process_predictions(
                    result["predictions"],
                    config["classifier"],
                )
                result.update(processed)
                print(f"  ✓ Detector fired: {processed['detector_fired']}")
                if processed['detector_fired']:
                    print(f"    Boxes: {processed['detector_box_count']}, "
                          f"Top: {processed['detector_top_class']} "
                          f"({processed['detector_top_confidence']:.3f})")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown')}")
            
            all_results.append(result)
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Save results
    results_file = output_dir / "raw_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"✓ Batch complete: {len(all_results)} inferences")
    print(f"  Results saved: {results_file}")
    
    # Create summary
    successful = sum(1 for r in all_results if r["success"])
    detector_fired = sum(1 for r in all_results if r.get("detector_fired"))
    
    print(f"\nSummary:")
    print(f"  Successful: {successful}/{len(all_results)}")
    print(f"  Detector fired: {detector_fired}/{len(all_results)}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run batch Roboflow Workflow inference on Kyoto eval set"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing images (e.g., kyoto_eval_set)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for results (e.g., kyoto_eval_outputs)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("ROBOFLOW_WORKSPACE", "intelfactor"),
        help="Roboflow workspace name",
    )
    parser.add_argument(
        "--workflow",
        default=os.environ.get("ROBOFLOW_WORKFLOW", "quality-gate-v1"),
        help="Roboflow workflow ID",
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("Error: Roboflow API key required. Use --api-key or set ROBOFLOW_API_KEY.")
        return 1
    
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_key=args.api_key,
        workspace=args.workspace,
        workflow=args.workflow,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
