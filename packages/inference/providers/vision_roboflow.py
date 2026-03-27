"""
IntelFactor.ai — Roboflow Hosted Vision Provider
Temporary single-frame inference via Roboflow API.

This is a BRIDGE provider — use it to get real defect predictions immediately
while training your own local TensorRT model in parallel.

Architecture:
- Local capture (Jetson/workstation)
- Roboflow API for inference (this provider)
- Local evidence write, SQLite persistence, UI rendering, sync

Swap to local TensorRT later by changing vision_model config.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
import requests

from packages.inference.providers.base import VisionProvider
from packages.inference.schemas import (
    BoundingBox,
    Detection,
    DetectionResult,
    ModelSpec,
    Verdict,
)

logger = logging.getLogger(__name__)


class RoboflowHostedVisionProvider(VisionProvider):
    """
    Roboflow-hosted inference for temporary use.
    
    Uses the Roboflow direct model inference API.
    Sends captured frames to Roboflow and returns normalized DetectionResult objects.
    
    Config required:
      - roboflow_api_key: Your Roboflow API key
      - roboflow_model_id: Full model ID (e.g., "metal-surface-defects-rmbhy-szb6t/1")
    
    Optional:
      - confidence_threshold: Global detection threshold (default: 0.5)
      - fail_threshold: Verdict routing threshold (default: 0.85)
      - review_threshold: Verdict routing threshold (default: 0.50)
      - class_name_map: Dict mapping Roboflow class names to canonical names
    """

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        # Build a minimal ModelSpec if not provided
        if model_spec is None or not model_spec.model_name:
            model_spec = ModelSpec(
                model_name="roboflow-hosted",
                model_path="https://detect.roboflow.com",
                quantization="FP16",
                backend=None,
                expected_latency_ms=500,
            )
        super().__init__(model_spec, config)
        
        # Roboflow API configuration (required)
        self.api_key: str = (config or {}).get("roboflow_api_key", "")
        self.model_id: str = (config or {}).get("roboflow_model_id", "")
        
        # Inference parameters
        self.confidence_threshold: float = (config or {}).get("confidence_threshold", 0.5)
        self.station_id: str = (config or {}).get("station_id", "unknown")
        
        # Verdict routing thresholds
        self.fail_threshold: float = (config or {}).get("fail_threshold", 0.85)
        self.review_threshold: float = (config or {}).get("review_threshold", 0.50)
        
        # Taxonomy mapping: Roboflow class name -> canonical class name
        self.class_name_map: dict[str, str] = (config or {}).get("class_name_map", {})
        
        # Defect classes from config (for consistency with local providers)
        self.defect_classes: list[str] = (config or {}).get("defect_classes", [])
        
        # API endpoint
        self.api_url: str = "https://detect.roboflow.com"
        
        self._session: requests.Session | None = None

    def load(self) -> None:
        """Initialize API session."""
        if not self.api_key:
            raise RuntimeError(
                "Roboflow API key not configured. "
                "Set roboflow_api_key in station.yaml or ROBOFLOW_API_KEY env var"
            )
        if not self.model_id:
            raise RuntimeError(
                "Roboflow model_id not configured. "
                "Set roboflow_model_id in station.yaml (e.g., 'workspace/project/version')"
            )
        
        self._session = requests.Session()
        self._loaded = True
        
        logger.info(
            "RoboflowHostedVisionProvider loaded: %s",
            self.model_id
        )
        logger.warning(
            "Using TEMPORARY Roboflow hosted inference. "
            "Swap to local TensorRT for production by setting vision_model: yolov8n_trt"
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference via Roboflow API.
        
        Args:
            frame: BGR numpy array (H, W, 3), uint8
            
        Returns:
            DetectionResult with normalized detections
        """
        if not self._loaded:
            raise RuntimeError("Provider not loaded. Call load() first.")
        if self._session is None:
            raise RuntimeError("Session not initialized")
        
        t0 = time.perf_counter()
        
        # Encode frame as JPEG bytes
        encoded = self._encode_frame(frame)
        
        # Build API URL with model_id and api_key
        url = f"{self.api_url}/{self.model_id}"
        params = {"api_key": self.api_key}
        
        # Call Roboflow API with multipart form
        try:
            files = {"file": ("frame.jpg", encoded, "image/jpeg")}
            response = self._session.post(
                url,
                params=params,
                files=files,
                timeout=30.0,
            )
            response.raise_for_status()
            api_result = response.json()
            
        except requests.exceptions.RequestException as exc:
            logger.error("Roboflow API request failed: %s", exc)
            # Return empty result on API failure (graceful degradation)
            return DetectionResult(
                station_id=self.station_id,
                detections=[],
                verdict=Verdict.PASS,
                confidence=0.0,
                inference_ms=(time.perf_counter() - t0) * 1000,
                model_version=f"roboflow-{self.model_id}",
                model_name="roboflow-hosted",
            )
        
        # Parse Roboflow response
        detections = self._parse_roboflow_response(api_result, frame.shape)
        
        inference_ms = (time.perf_counter() - t0) * 1000
        
        # Apply verdict rules
        verdict, confidence = self._apply_rules(detections)
        
        return DetectionResult(
            station_id=self.station_id,
            detections=detections,
            verdict=verdict,
            confidence=confidence,
            inference_ms=inference_ms,
            model_version=f"roboflow-{self.model_id}",
            model_name="roboflow-hosted",
        )

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode numpy frame as JPEG bytes."""
        import cv2
        
        # Ensure BGR format
        if len(frame.shape) == 2:
            # Grayscale -> BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Encode to JPEG
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")
        
        return encoded.tobytes()

    def _parse_roboflow_response(
        self, api_result: dict, orig_shape: tuple[int, ...]
    ) -> list[Detection]:
        """
        Parse Roboflow API response into Detection objects.
        
        Roboflow response format:
        {
            "predictions": [
                {
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                    "confidence": 0.92,
                    "class": "scratch",
                    "class_id": 0
                }
            ]
        }
        """
        detections: list[Detection] = []
        
        predictions = api_result.get("predictions", [])
        img_h, img_w = orig_shape[:2]
        
        for pred in predictions:
            confidence = float(pred.get("confidence", 0))
            
            # Skip below threshold
            if confidence < self.confidence_threshold:
                continue
            
            # Get class name and map to canonical
            roboflow_class = pred.get("class", "unknown")
            defect_type = self.class_name_map.get(roboflow_class, roboflow_class)
            
            # Roboflow returns center x,y and width,height
            cx = float(pred.get("x", 0))
            cy = float(pred.get("y", 0))
            w = float(pred.get("width", 0))
            h = float(pred.get("height", 0))
            
            # Convert to top-left corner
            x = cx - w / 2
            y = cy - h / 2
            
            # Clamp to image bounds
            x = max(0, min(x, img_w))
            y = max(0, min(y, img_h))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            detection = Detection(
                defect_type=defect_type,
                confidence=confidence,
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                severity=self._estimate_severity(defect_type, confidence),
                threshold_used=self.confidence_threshold,
                model_version=f"roboflow-{self.model_id}",
            )
            detections.append(detection)
        
        logger.debug("Parsed %d detections from Roboflow response", len(detections))
        return detections

    def _estimate_severity(self, defect_type: str, confidence: float) -> float:
        """Estimate defect severity based on type and confidence."""
        severity_ranges: dict[str, tuple[float, float]] = {
            "blade_scratch":    (0.2, 0.9),
            "grinding_mark":    (0.2, 0.7),
            "surface_dent":     (0.3, 0.9),
            "surface_crack":    (0.7, 1.0),
            "weld_defect":      (0.3, 0.9),
            "edge_burr":        (0.5, 1.0),
            "edge_crack":       (0.7, 1.0),
            "handle_defect":    (0.5, 1.0),
            "bolster_gap":      (0.3, 0.8),
            "etching_defect":   (0.4, 0.9),
            "inclusion":        (0.2, 0.8),
            "surface_discolor": (0.2, 0.7),
            "overgrind":        (0.4, 1.0),
            # Workflow-specific mappings
            "scratch":          (0.2, 0.9),
            "dent":             (0.3, 0.9),
            "crack":            (0.7, 1.0),
            "discoloration":    (0.2, 0.7),
            "pitted_surface":   (0.3, 0.9),
            "rolled-in_scale":  (0.3, 0.8),
            "crazing":          (0.4, 0.9),
            "patches":          (0.3, 0.8),
        }
        
        if defect_type in severity_ranges:
            min_sev, max_sev = severity_ranges[defect_type]
            return min_sev + (max_sev - min_sev) * confidence
        
        return confidence

    def _apply_rules(self, detections: list[Detection]) -> tuple[Verdict, float]:
        """
        DefectIQ rules engine — verdict routing based on detection confidence.
        """
        if not detections:
            return Verdict.PASS, 1.0
        
        max_conf = max(d.confidence for d in detections)
        
        if max_conf >= self.fail_threshold:
            return Verdict.FAIL, max_conf
        elif max_conf >= self.review_threshold:
            return Verdict.REVIEW, max_conf
        else:
            return Verdict.PASS, 1.0 - max_conf

    def unload(self) -> None:
        """Close API session."""
        if self._session:
            self._session.close()
            self._session = None
        self._loaded = False
        logger.info("RoboflowHostedVisionProvider unloaded")
