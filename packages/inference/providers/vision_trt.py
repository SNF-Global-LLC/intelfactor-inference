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
        self.confidence_threshold: float = config.get("confidence_threshold", 0.5) if config else 0.5
        self.nms_threshold: float = config.get("nms_threshold", 0.45) if config else 0.45
        self.defect_classes: list[str] = config.get("defect_classes", []) if config else []
        self.station_id: str = config.get("station_id", "unknown") if config else "unknown"

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
            model_version=self.model_spec.model_name,
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

        # Filter by confidence threshold
        mask = confidences >= self.confidence_threshold
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
            defect_type = self.defect_classes[class_id] if class_id < len(self.defect_classes) else f"defect_{class_id}"

            detection = Detection(
                defect_type=defect_type,
                confidence=float(confidences[i]),
                bbox=BoundingBox(
                    x=float(x1[i]),
                    y=float(y1[i]),
                    width=float(widths[i]),
                    height=float(heights[i]),
                ),
                severity=self._estimate_severity(defect_type, float(confidences[i])),
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
        """
        Estimate defect severity based on type and confidence.
        Uses severity ranges from wiko_taxonomy.yaml.
        """
        # Severity ranges from taxonomy (hardcoded for performance)
        severity_ranges = {
            "scratch_surface": (0.2, 0.9),
            "scratch_edge": (0.4, 1.0),
            "burr": (0.3, 0.8),
            "pit_corrosion": (0.5, 1.0),
            "discoloration": (0.2, 0.7),
            "dent": (0.3, 0.9),
            "crack": (0.7, 1.0),
            "warp": (0.4, 1.0),
            "handle_gap": (0.3, 0.8),
            "handle_crack": (0.5, 1.0),
            "logo_defect": (0.2, 0.6),
            "dimension_out_of_spec": (0.4, 1.0),
            "foreign_material": (0.2, 0.7),
        }

        if defect_type in severity_ranges:
            min_sev, max_sev = severity_ranges[defect_type]
            # Scale confidence to severity range
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
        DefectIQ rules engine.
        Applies configurable thresholds per sop_criterion.
        """
        if not detections:
            return Verdict.PASS, 1.0

        max_conf = max(d.confidence for d in detections)

        # Escalation threshold from config
        escalation_limit = self.config.get("escalation_limit", 0.05)
        review_threshold = self.config.get("review_threshold", 0.7)

        if max_conf >= self.confidence_threshold:
            return Verdict.FAIL, max_conf
        elif max_conf >= review_threshold:
            return Verdict.REVIEW, max_conf
        else:
            return Verdict.PASS, 1.0 - max_conf

    def unload(self) -> None:
        self.context = None
        self.engine = None
        self._loaded = False
        logger.info("TensorRT engine unloaded")
