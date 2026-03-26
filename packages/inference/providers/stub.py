"""
IntelFactor.ai — Stub Providers for Local Development
No GPU, no TensorRT, no llama.cpp required.

Use vision_model: stub and language_model: stub in your station config
to run the full inspection pipeline on a Mac or any CPU-only machine.

Vision stub: returns 0 detections (PASS) with 1ms latency.
Language stub: returns a static bilingual RCA explanation.

NOT for production use.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from packages.inference.providers.base import LanguageProvider, VisionProvider
from packages.inference.schemas import (
    DetectionResult,
    ModelSpec,
    RCAExplanation,
    Verdict,
)

logger = logging.getLogger(__name__)


class StubVisionProvider(VisionProvider):
    """
    CPU-safe stub for development without TensorRT.
    Always returns 0 detections and PASS verdict.
    """

    def __init__(self, model_spec: ModelSpec | None = None, config: dict[str, Any] | None = None):
        if model_spec is None:
            model_spec = ModelSpec(
                model_name="stub-vision",
                model_path="/dev/null",
                quantization="none",
                backend=None,  # type: ignore[arg-type]
            )
        super().__init__(model_spec, config)
        self.station_id: str = (config or {}).get("station_id", "dev")

    def load(self) -> None:
        self._loaded = True
        logger.warning(
            "StubVisionProvider loaded — no real inference will run. "
            "Set vision_model to a real engine for production."
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if not self._loaded:
            raise RuntimeError("StubVisionProvider not loaded")
        t0 = time.perf_counter()
        # Simulate minimal processing time
        _ = frame.shape
        inference_ms = (time.perf_counter() - t0) * 1000 + 1.0

        return DetectionResult(
            station_id=self.station_id,
            detections=[],
            verdict=Verdict.PASS,
            confidence=0.99,
            inference_ms=inference_ms,
            model_version="stub-0.0.0",
            model_name="stub-vision",
        )

    def unload(self) -> None:
        self._loaded = False


class StubLanguageProvider(LanguageProvider):
    """
    CPU-safe stub for development without llama.cpp or vLLM.
    Returns a static bilingual RCA explanation.
    """

    def __init__(self, model_spec: ModelSpec | None = None, config: dict[str, Any] | None = None):
        if model_spec is None:
            model_spec = ModelSpec(
                model_name="stub-language",
                model_path="/dev/null",
                quantization="none",
                backend=None,  # type: ignore[arg-type]
                max_tokens=256,
                context_window=4096,
            )
        super().__init__(model_spec, config)

    def load(self) -> None:
        self._loaded = True
        logger.warning(
            "StubLanguageProvider loaded — static RCA explanations only. "
            "Set language_model to a real GGUF or vLLM model for production."
        )

    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> RCAExplanation:
        if not self._loaded:
            raise RuntimeError("StubLanguageProvider not loaded")
        return RCAExplanation(
            explanation_zh="[开发模式] 检测到异常缺陷率上升，建议检查研磨轮磨损及工艺参数设置。",
            explanation_en="[DEV MODE] Elevated defect rate detected. Check grinding wheel wear and process parameter settings.",
            confidence=0.5,
            model_used="stub-language",
            generation_ms=5.0,
        )

    def unload(self) -> None:
        self._loaded = False
