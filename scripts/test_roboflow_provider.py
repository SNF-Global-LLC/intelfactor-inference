#!/usr/bin/env python3
"""
Test script for RoboflowHostedVisionProvider.
Verifies the provider can load and make predictions.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from packages.inference.providers.vision_roboflow import RoboflowHostedVisionProvider
from packages.inference.schemas import InferenceBackend, ModelSpec

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def test_with_image(image_path: str, config: dict) -> None:
    """Test provider with a real image."""
    import cv2
    
    logger.info("Loading image: %s", image_path)
    frame = cv2.imread(image_path)
    if frame is None:
        logger.error("Failed to load image: %s", image_path)
        sys.exit(1)
    
    logger.info("Image shape: %s", frame.shape)
    
    # Create provider
    spec = ModelSpec(
        model_name="roboflow-hosted",
        model_path=config["roboflow_api_url"],
        quantization="FP16",
        backend=InferenceBackend.TENSORRT,
        expected_latency_ms=500,
    )
    
    provider = RoboflowHostedVisionProvider(spec, config)
    
    # Load
    logger.info("Loading provider...")
    provider.load()
    
    # Run inference
    logger.info("Running inference...")
    result = provider.detect(frame)
    
    # Print results
    logger.info("=" * 50)
    logger.info("Inference Results")
    logger.info("=" * 50)
    logger.info("Verdict: %s", result.verdict)
    logger.info("Confidence: %.3f", result.confidence)
    logger.info("Inference time: %.1f ms", result.inference_ms)
    logger.info("Model: %s (v%s)", result.model_name, result.model_version)
    logger.info("Endpoint: %s", config["roboflow_api_url"])
    provider_meta = result.provider_metadata or {}
    if provider_meta:
        logger.info(
            "Provider: %s%s",
            provider_meta.get("label", "unknown"),
            f" [{provider_meta.get('verdict_policy')}]" if provider_meta.get("verdict_policy") else "",
        )
        if provider_meta.get("experimental"):
            logger.info("Experimental mode: enabled")
    if config.get("roboflow_workflow_id"):
        logger.info(
            "Workflow: %s/%s",
            config["roboflow_workspace"],
            config["roboflow_workflow_id"],
        )
    else:
        logger.info("Model ID: %s/%s", config["roboflow_project"], config["roboflow_version"])
    logger.info("Detections: %d", len(result.detections))
    
    for i, det in enumerate(result.detections[:10]):  # Show first 10
        logger.info(
            "  %d. %s (%.2f) at [%.0f, %.0f, %.0f, %.0f] severity=%.2f",
            i + 1,
            det.defect_type,
            det.confidence,
            det.bbox.x,
            det.bbox.y,
            det.bbox.width,
            det.bbox.height,
            det.severity,
        )
    
    if len(result.detections) > 10:
        logger.info("  ... and %d more", len(result.detections) - 10)
    
    # Unload
    provider.unload()
    logger.info("Test completed successfully!")


def test_with_synthetic(config: dict) -> None:
    """Test provider with synthetic image (no file required)."""
    logger.info("Creating synthetic test image...")
    
    # Create a blank image with some "defects" (just shapes for testing)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Add some white rectangles to simulate defects
    frame[100:150, 100:200] = 255  # White rectangle
    frame[300:350, 400:500] = 200  # Gray rectangle
    frame[500:550, 200:300] = 150  # Another defect
    
    # Create provider
    spec = ModelSpec(
        model_name="roboflow-hosted",
        model_path=config["roboflow_api_url"],
        quantization="FP16",
        backend=InferenceBackend.TENSORRT,
        expected_latency_ms=500,
    )
    
    provider = RoboflowHostedVisionProvider(spec, config)
    
    # Load
    logger.info("Loading provider...")
    try:
        provider.load()
    except RuntimeError as exc:
        logger.error("Failed to load: %s", exc)
        logger.error("\nMake sure you have set ROBOFLOW_API_KEY environment variable!")
        sys.exit(1)
    
    # Run inference
    logger.info("Running inference on synthetic image...")
    result = provider.detect(frame)
    
    # Print results
    logger.info("=" * 50)
    logger.info("Inference Results (Synthetic Image)")
    logger.info("=" * 50)
    logger.info("Verdict: %s", result.verdict)
    logger.info("Inference time: %.1f ms", result.inference_ms)
    logger.info("Detections: %d", len(result.detections))
    
    for det in result.detections:
        logger.info(
            "  - %s (%.2f)",
            det.defect_type,
            det.confidence,
        )
    
    provider.unload()
    logger.info("Test completed!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test RoboflowHostedVisionProvider")
    parser.add_argument("--image", "-i", help="Path to test image")
    parser.add_argument("--api-key", "-k", help="Roboflow API key (or set ROBOFLOW_API_KEY env var, preferred)")
    
    # EASY MODE: Just specify model_id
    parser.add_argument("--model-id", "-m", help="Full model ID (e.g., 'metal-surface-defects-rmbhy-szb6t/1')")
    
    # LEGACY MODE: Individual components
    parser.add_argument("--workspace", "-w", default="", help="Roboflow workspace")
    parser.add_argument("--workflow-id", default="", help="Roboflow workflow ID")
    parser.add_argument("--project", "-p", default="", help="Roboflow project")
    parser.add_argument("--version", "-v", default="", help="Model version")
    parser.add_argument(
        "--api-url",
        default="https://detect.roboflow.com",
        help="Roboflow API base URL",
    )
    parser.add_argument("--synthetic", "-s", action="store_true", help="Use synthetic image")
    
    args = parser.parse_args()
    
    # Build config
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        logger.error("Missing Roboflow API key. Pass --api-key or export ROBOFLOW_API_KEY.")
        sys.exit(1)
    
    # Default model for easy testing
    model_id = args.model_id or "metal-surface-defects-rmbhy-szb6t/1"
    
    config = {
        "roboflow_api_key": api_key,
        "roboflow_model_id": model_id,  # EASY MODE
        "roboflow_workspace": args.workspace,
        "roboflow_workflow_id": args.workflow_id,
        "roboflow_project": args.project,
        "roboflow_version": args.version,
        "roboflow_api_url": args.api_url,
        "station_id": "test_station",
        "confidence_threshold": 0.25,
        "class_name_map": {},
        "raise_on_request_error": True,
    }
    
    logger.info("Testing with model: %s", model_id)
    
    if args.image:
        test_with_image(args.image, config)
    else:
        test_with_synthetic(config)


if __name__ == "__main__":
    main()
