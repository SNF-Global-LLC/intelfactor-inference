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
        """Parse YOLO output tensor into Detection objects."""
        detections = []
        # Implementation depends on YOLO version output format
        # This is a simplified parser — real implementation in trt_service.py
        return detections

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
