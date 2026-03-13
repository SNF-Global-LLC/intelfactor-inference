"""
IntelFactor.ai — TensorRT Vision Provider
Runs YOLOv8/YOLO26 TensorRT engines on any NVIDIA GPU.
Jetson: via DeepStream or direct TensorRT.
Server: via TensorRT or Triton.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from packages.inference.providers.base import VisionProvider
from packages.inference.schemas import (
    BoundingBox,
    Detection,
    DetectionResult,
    ModelSpec,
    Verdict,
)

logger = logging.getLogger(__name__)


class TensorRTVisionProvider(VisionProvider):
    """
    TensorRT-based vision inference.
    Loads a .engine file, runs inference, returns DetectionResult.

    Works on:
    - Jetson (Orin Nano / NX / AGX / Thor) via JetPack TensorRT
    - Server GPUs (RTX / L4 / A10 / A100) via desktop TensorRT

    The engine file must be built for the target GPU architecture.
    Use trtexec or the model registry's build pipeline.
    """

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        super().__init__(model_spec, config)
        self.engine = None
        self.context = None
        self.input_shape: tuple[int, ...] = (1, 3, 640, 640)  # default YOLO
        # Global fallback detection threshold — used when no per-class entry exists.
        # Keep this low; per-class thresholds do the real filtering.
        self.confidence_threshold: float = config.get("confidence_threshold", 0.25) if config else 0.25
        self.nms_threshold: float = config.get("nms_threshold", 0.45) if config else 0.45
        self.station_id: str = config.get("station_id", "unknown") if config else "unknown"
        # Verdict routing thresholds (separate from detection filter threshold).
        # Detections above fail_threshold → FAIL; between review and fail → REVIEW.
        # Per CLAUDE.md: >0.85 auto-fail, 0.5–0.85 human verify.
        self.fail_threshold: float = config.get("fail_threshold", 0.85) if config else 0.85
        self.review_threshold: float = config.get("review_threshold", 0.50) if config else 0.50
        
        # Model bundle integration
        # Priority: 1. model_bundle labels/thresholds 2. config defect_classes 3. fallback
        self.model_version: str = config.get("model_version", "unknown") if config else "unknown"
        self.model_name: str = config.get("model_name", model_spec.model_name) if config else model_spec.model_name
        
        # Load labels from model bundle or config
        self.labels: dict[int, str] = config.get("labels", {}) if config else {}
        self.defect_classes: list[str] = self._resolve_defect_classes(config)
        
        # Load per-class thresholds from model bundle or config
        self.per_class_thresholds: dict[str, float] = self._load_per_class_thresholds(config)

    def _resolve_defect_classes(self, config: dict[str, Any] | None) -> list[str]:
        """Resolve defect class list from labels or config.
        
        Priority:
          1. model_bundle labels (sorted by class_id)
          2. config defect_classes
          3. empty list (fallback)
        """
        if not config:
            return []
        
        # If we have labels from model bundle, use them
        if self.labels:
            max_id = max(self.labels.keys())
            classes = [self.labels.get(i, f"defect_{i}") for i in range(max_id + 1)]
            logger.info("Loaded %d classes from model bundle labels", len(classes))
            return classes
        
        # Fallback to config
        classes = config.get("defect_classes", [])
        if classes:
            logger.debug("Using defect_classes from config (%d classes)", len(classes))
        return classes

    def _load_per_class_thresholds(self, config: dict[str, Any] | None) -> dict[str, float]:
        """Load per-class thresholds from config or model bundle.

        Priority:
          1. config["thresholds"] from model bundle — highest priority
          2. config["per_class_thresholds"] — inline dict
          3. config["thresholds_path"] — path to YAML file
          4. {} — fall back to self.confidence_threshold globally
        """
        if not config:
            return {}

        # Priority 1: thresholds from model bundle
        bundle_thresholds = config.get("thresholds")
        if isinstance(bundle_thresholds, dict) and bundle_thresholds:
            logger.info(
                "Loaded %d per-class thresholds from model bundle",
                len(bundle_thresholds),
            )
            return {k: float(v) for k, v in bundle_thresholds.items()}

        # Priority 2: inline per_class_thresholds
        inline = config.get("per_class_thresholds")
        if isinstance(inline, dict) and inline:
            logger.debug("Loaded %d per-class thresholds from config", len(inline))
            return {k: float(v) for k, v in inline.items()}

        # Priority 3: YAML file path
        thresholds_path = config.get("thresholds_path")
        if thresholds_path:
            path = Path(thresholds_path)
            if path.exists():
                try:
                    import yaml
                    with path.open() as f:
                        data = yaml.safe_load(f)
                    thresholds = data.get("thresholds", {})
                    logger.info(
                        "Loaded per-class thresholds from %s (%d classes)",
                        path.name, len(thresholds),
                    )
                    return {k: float(v) for k, v in thresholds.items()}
                except Exception as exc:
                    logger.warning("Failed to load thresholds from %s: %s", thresholds_path, exc)
            else:
                logger.warning("thresholds_path not found: %s", thresholds_path)

        return {}

    def _get_class_threshold(self, class_name: str) -> float:
        """Return the detection threshold for a given class name."""
        return self.per_class_thresholds.get(class_name, self.confidence_threshold)

    def load(self) -> None:
        engine_path = Path(self.model_spec.model_path)
        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        try:
            import tensorrt as trt

            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            if self.engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            self.context = self.engine.create_execution_context()
            self._loaded = True
            logger.info(
                "TensorRT engine loaded: %s (classes: %d)",
                engine_path.name,
                len(self.defect_classes),
            )

        except ImportError:
            # Fallback for development/testing without TensorRT
            logger.warning("TensorRT not available — using stub mode")
            self._loaded = True
            self._stub_mode = True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        t0 = time.perf_counter()

        if getattr(self, "_stub_mode", False):
            detections = self._stub_detect(frame)
        else:
            detections = self._trt_detect(frame)

        inference_ms = (time.perf_counter() - t0) * 1000

        # Apply DefectIQ rules engine
        verdict, confidence = self._apply_rules(detections)

        return DetectionResult(
            station_id=self.station_id,
            detections=detections,
            verdict=verdict,
            confidence=confidence,
            inference_ms=inference_ms,
            model_version=self.model_version,
            model_name=self.model_name,
        )

    def _trt_detect(self, frame: np.ndarray) -> list[Detection]:
        """Real TensorRT inference path."""
        import tensorrt as trt  # noqa: F811
        import pycuda.driver as cuda

        # Preprocess: resize, normalize, NCHW, float32
        blob = self._preprocess(frame)

        # Allocate buffers and run inference
        inputs, outputs, bindings, stream = self._allocate_buffers()
        np.copyto(inputs[0].host, blob.ravel())

        # Transfer input to GPU
        cuda.memcpy_htod_async(inputs[0].device, inputs[0].host, stream)
        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer output back
        cuda.memcpy_dtoh_async(outputs[0].host, outputs[0].device, stream)
        stream.synchronize()

        # Parse YOLO output
        return self._parse_yolo_output(outputs[0].host, frame.shape)

    def _stub_detect(self, frame: np.ndarray) -> list[Detection]:
        """Stub for development without GPU. Returns empty detections."""
        return []

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize frame for YOLO input."""
        import cv2

        h, w = self.input_shape[2], self.input_shape[3]
        resized = cv2.resize(frame, (w, h))
        blob = resized.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # add batch dim
        return np.ascontiguousarray(blob)

    def _parse_yolo_output(
        self, raw_output: np.ndarray, orig_shape: tuple[int, ...]
    ) -> list[Detection]:
        """
        Parse YOLOv8 output tensor into Detection objects.

        YOLOv8 output shape: [batch, 4 + num_classes, num_detections]
        - First 4 rows: x_center, y_center, width, height (normalized to input size)
        - Remaining rows: class confidence scores

        Args:
            raw_output: Flattened output from TensorRT engine
            orig_shape: Original frame shape (H, W, C)

        Returns:
            List of Detection objects after NMS filtering
        """
        num_classes = len(self.defect_classes) if self.defect_classes else 13
        num_outputs = 4 + num_classes  # bbox + class scores

        # Reshape output: YOLOv8 outputs [1, 4+classes, 8400] for 640x640 input
        # 8400 = (80*80 + 40*40 + 20*20) detection points
        expected_detections = 8400  # default for 640x640
        try:
            output = raw_output.reshape(1, num_outputs, -1)
            num_detections = output.shape[2]
        except ValueError:
            # Try to infer shape from raw output
            total_elements = raw_output.size
            num_detections = total_elements // num_outputs
            if num_detections * num_outputs != total_elements:
                logger.warning("Cannot parse YOLO output: size=%d", total_elements)
                return []
            output = raw_output.reshape(1, num_outputs, num_detections)

        # Transpose to [1, num_detections, 4+classes]
        output = np.transpose(output, (0, 2, 1))[0]  # [num_detections, 4+classes]

        # Extract boxes and scores
        boxes = output[:, :4]  # [num_detections, 4] - cx, cy, w, h
        scores = output[:, 4:]  # [num_detections, num_classes]

        # Get max class score and class index for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Apply per-class detection thresholds.
        # Each box is kept only if its confidence meets the threshold for its class.
        per_class_thresh = np.array([
            self._get_class_threshold(
                self.defect_classes[int(cid)] if int(cid) < len(self.defect_classes) else ""
            )
            for cid in class_ids
        ])
        mask = confidences >= per_class_thresh
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert cx, cy, w, h to x, y, w, h and scale to original image
        orig_h, orig_w = orig_shape[:2]
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h

        # Convert center format to corner format
        x_centers = boxes[:, 0] * scale_x
        y_centers = boxes[:, 1] * scale_y
        widths = boxes[:, 2] * scale_x
        heights = boxes[:, 3] * scale_y

        x1 = x_centers - widths / 2
        y1 = y_centers - heights / 2

        # Apply NMS
        indices = self._nms(x1, y1, widths, heights, confidences)

        # Build Detection objects
        detections = []
        for i in indices:
            class_id = int(class_ids[i])
            confidence = float(confidences[i])
            defect_type = self.defect_classes[class_id] if class_id < len(self.defect_classes) else f"defect_{class_id}"
            threshold_used = self._get_class_threshold(defect_type)

            detection = Detection(
                defect_type=defect_type,
                confidence=confidence,
                threshold_used=threshold_used,
                model_version=self.model_version,
                bbox=BoundingBox(
                    x=float(x1[i]),
                    y=float(y1[i]),
                    width=float(widths[i]),
                    height=float(heights[i]),
                ),
                severity=self._estimate_severity(defect_type, confidence),
            )
            detections.append(detection)

        logger.debug("Parsed %d detections from YOLO output", len(detections))
        return detections

    def _nms(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        h: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        """
        Non-Maximum Suppression.

        Args:
            x, y: Top-left corner coordinates
            w, h: Width and height
            scores: Confidence scores

        Returns:
            Indices of boxes to keep
        """
        if len(x) == 0:
            return []

        # Convert to x1, y1, x2, y2
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter_area = inter_w * inter_h

            iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-6)

            # Keep boxes with IoU below threshold
            mask = iou <= self.nms_threshold
            order = order[1:][mask]

        return keep

    def _estimate_severity(self, defect_type: str, confidence: float) -> float:
        """Estimate defect severity based on type and confidence.

        Ranges are derived from configs/wiko_taxonomy.yaml severity_range fields.
        Critical classes (AQL=0) have high floors; minor classes have low ceilings.
        """
        # Canonical KMG Kyoto taxonomy severity ranges — (min, max)
        # critical: edge_burr, edge_crack, surface_crack, etching_defect → floor ≥ 0.5
        # minor: surface_discolor → ceiling 0.7
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
        }

        if defect_type in severity_ranges:
            min_sev, max_sev = severity_ranges[defect_type]
            return min_sev + (max_sev - min_sev) * confidence

        # Default: use confidence as severity
        return confidence

    def _allocate_buffers(self):
        """Allocate GPU buffers for TensorRT inference."""
        import pycuda.driver as cuda

        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            size = abs(int(np.prod(shape)))

            host_mem = cuda.pagelocked_empty(size, np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            class MemHolder:
                def __init__(self, host, device):
                    self.host = host
                    self.device = device

            holder = MemHolder(host_mem, device_mem)
            mode = self.engine.get_tensor_mode(name)
            if mode == self.engine.get_tensor_mode(self.engine.get_tensor_name(0)):
                inputs.append(holder)
            else:
                outputs.append(holder)

        return inputs, outputs, bindings, stream

    def _apply_rules(self, detections: list[Detection]) -> tuple[Verdict, float]:
        """
        DefectIQ rules engine — verdict routing based on detection confidence.

        Routing (per CLAUDE.md):
          > fail_threshold (default 0.85)   → FAIL  (auto-reject, high confidence)
          > review_threshold (default 0.50) → REVIEW (human verify)
          otherwise                         → PASS

        Note: detections reaching this method have already passed per-class
        detection thresholds in _parse_yolo_output. The routing thresholds
        here are separate and intentionally higher.
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
        self.context = None
        self.engine = None
        self._loaded = False
        logger.info("TensorRT engine unloaded")
