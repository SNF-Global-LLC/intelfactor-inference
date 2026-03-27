"""
IntelFactor.ai — Inference Providers
Vision and language model provider implementations.
"""

from packages.inference.providers.base import LanguageProvider, VisionProvider
from packages.inference.providers.stub import StubLanguageProvider, StubVisionProvider
from packages.inference.providers.vision_trt import TensorRTVisionProvider
from packages.inference.providers.vision_roboflow import RoboflowHostedVisionProvider
from packages.inference.providers.language_llama import LlamaCppLanguageProvider
from packages.inference.providers.language_vllm import VLLMLanguageProvider
from packages.inference.providers.resolver import CapabilityResolver

__all__ = [
    # Base classes
    "VisionProvider",
    "LanguageProvider",
    # Vision providers
    "TensorRTVisionProvider",
    "RoboflowHostedVisionProvider",
    "StubVisionProvider",
    # Language providers
    "LlamaCppLanguageProvider",
    "VLLMLanguageProvider",
    "StubLanguageProvider",
    # Resolver
    "CapabilityResolver",
]
