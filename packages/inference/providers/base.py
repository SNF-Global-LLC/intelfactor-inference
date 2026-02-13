"""
IntelFactor.ai — Provider Interfaces
Abstract base classes for vision and language inference.
One interface, multiple backends. Swap via config, not code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from packages.inference.schemas import (
    DetectionResult,
    RCAExplanation,
    ModelSpec,
)


class VisionProvider(ABC):
    """
    Abstract vision inference provider.
    Implementations: TensorRT (Jetson), TensorRT/Triton (server).
    """

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        self.model_spec = model_spec
        self.config = config or {}
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model into GPU memory. Called once at startup."""
        ...

    @abstractmethod
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Run inference on a single frame.
        Args:
            frame: BGR numpy array (H, W, 3), uint8
        Returns:
            DetectionResult with detections, verdict, and timing.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release GPU resources."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()


class LanguageProvider(ABC):
    """
    Abstract language model provider for RCA explanation.
    Implementations: llama.cpp (Jetson), vLLM (server), TensorRT-LLM (server).
    """

    def __init__(self, model_spec: ModelSpec, config: dict[str, Any] | None = None):
        self.model_spec = model_spec
        self.config = config or {}
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model into GPU memory."""
        ...

    @abstractmethod
    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> RCAExplanation:
        """
        Generate RCA explanation from structured prompt.
        Args:
            prompt: Formatted prompt with defect + correlation data.
            context: Additional context (SOP refs, defect taxonomy, etc.)
        Returns:
            RCAExplanation with bilingual output and timing.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release GPU resources."""
        ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()
