"""
Model Bundle Loader and Validator

Standard model bundle contract:
  model_bundle/
    model.engine       # TensorRT engine
    labels.json        # Class name mapping
    thresholds.yaml    # Per-class confidence thresholds
    metadata.json      # Model version, info, etc.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ModelBundleError(Exception):
    """Raised when model bundle is invalid or missing."""
    pass


def load_labels(path: Path | str) -> dict[int, str]:
    """
    Load class labels from JSON.
    
    Expected format:
    {
      "0": "blade_scratch",
      "1": "grinding_mark",
      ...
    }
    
    Returns:
        Dict mapping class_id (int) to class_name (str)
    
    Raises:
        ModelBundleError: If file missing or invalid
    """
    path = Path(path)
    if not path.exists():
        raise ModelBundleError(f"Labels file not found: {path}")
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ModelBundleError(f"Invalid JSON in labels file: {e}")
    
    # Convert string keys to int
    labels = {}
    for key, value in data.items():
        try:
            class_id = int(key)
            labels[class_id] = value
        except (ValueError, TypeError):
            raise ModelBundleError(f"Invalid class ID in labels: {key}")
    
    logger.info(f"Loaded {len(labels)} classes from {path.name}")
    return labels


def load_thresholds(path: Path | str) -> dict[str, float]:
    """
    Load per-class thresholds from YAML.
    
    Expected format:
    thresholds:
      blade_scratch: 0.45
      grinding_mark: 0.30
      surface_dent: 0.35
      ...
    
    Returns:
        Dict mapping class_name to threshold value
    
    Raises:
        ModelBundleError: If file missing or invalid
    """
    path = Path(path)
    if not path.exists():
        raise ModelBundleError(f"Thresholds file not found: {path}")
    
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ModelBundleError(f"Invalid YAML in thresholds file: {e}")
    
    if not isinstance(data, dict) or "thresholds" not in data:
        raise ModelBundleError("Thresholds file must contain 'thresholds' key")
    
    thresholds = data["thresholds"]
    if not isinstance(thresholds, dict):
        raise ModelBundleError("'thresholds' must be a dict")
    
    # Validate all values are floats
    for class_name, value in thresholds.items():
        try:
            thresholds[class_name] = float(value)
        except (ValueError, TypeError):
            raise ModelBundleError(f"Invalid threshold for {class_name}: {value}")
    
    logger.info(f"Loaded {len(thresholds)} thresholds from {path.name}")
    return thresholds


def load_metadata(path: Path | str) -> dict[str, Any]:
    """
    Load model metadata from JSON.
    
    Expected format:
    {
      "model_name": "kyoto-yolo26n-v1",
      "model_version": "1.0.0",
      "training_date": "2026-03-15",
      "classes": 13,
      "input_shape": [1, 3, 640, 640],
      "precision": "fp16"
    }
    
    Returns:
        Dict with metadata fields
    
    Raises:
        ModelBundleError: If file missing or invalid
    """
    path = Path(path)
    if not path.exists():
        raise ModelBundleError(f"Metadata file not found: {path}")
    
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ModelBundleError(f"Invalid JSON in metadata file: {e}")
    
    if not isinstance(data, dict):
        raise ModelBundleError("Metadata must be a JSON object")
    
    # Validate required fields
    required = ["model_name", "model_version"]
    for field in required:
        if field not in data:
            raise ModelBundleError(f"Metadata missing required field: {field}")
    
    logger.info(f"Loaded metadata for model: {data['model_name']} v{data['model_version']}")
    return data


def validate_bundle(
    engine_path: Path | str,
    labels_path: Path | str,
    thresholds_path: Path | str,
    metadata_path: Path | str,
    expected_classes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate complete model bundle consistency.
    
    Args:
        engine_path: Path to TensorRT .engine file
        labels_path: Path to labels.json
        thresholds_path: Path to thresholds.yaml
        metadata_path: Path to metadata.json
        expected_classes: Optional list of expected class names (canonical taxonomy)
    
    Returns:
        Dict with loaded bundle components
    
    Raises:
        ModelBundleError: If any validation fails
    """
    engine_path = Path(engine_path)
    labels_path = Path(labels_path)
    thresholds_path = Path(thresholds_path)
    metadata_path = Path(metadata_path)
    
    # Check engine exists
    if not engine_path.exists():
        raise ModelBundleError(f"Engine file not found: {engine_path}")
    
    # Load all components
    labels = load_labels(labels_path)
    thresholds = load_thresholds(thresholds_path)
    metadata = load_metadata(metadata_path)
    
    # Validate consistency
    label_classes = set(labels.values())
    threshold_classes = set(thresholds.keys())
    
    # Check all labeled classes have thresholds
    missing_thresholds = label_classes - threshold_classes
    if missing_thresholds:
        raise ModelBundleError(
            f"Classes missing thresholds: {missing_thresholds}"
        )
    
    # Check all thresholds have labels
    extra_thresholds = threshold_classes - label_classes
    if extra_thresholds:
        logger.warning(f"Thresholds for unknown classes (will be ignored): {extra_thresholds}")
    
    # Validate against expected taxonomy if provided
    if expected_classes is not None:
        expected_set = set(expected_classes)
        label_set = set(labels.values())
        
        missing = expected_set - label_set
        if missing:
            raise ModelBundleError(f"Missing expected classes: {missing}")
        
        extra = label_set - expected_set
        if extra:
            logger.warning(f"Extra classes not in canonical taxonomy: {extra}")
    
    logger.info(
        f"Bundle validated: {len(labels)} classes, "
        f"{len(thresholds)} thresholds, model={metadata.get('model_name')}"
    )
    
    return {
        "labels": labels,
        "thresholds": thresholds,
        "metadata": metadata,
        "engine_path": engine_path,
    }


def get_model_version(bundle_dir: Path | str) -> str:
    """
    Quick helper to get model version from bundle directory.
    
    Returns:
        Model version string or "unknown" if metadata missing
    """
    metadata_path = Path(bundle_dir) / "metadata.json"
    try:
        metadata = load_metadata(metadata_path)
        return metadata.get("model_version", "unknown")
    except ModelBundleError:
        return "unknown"
