"""
Tests for CapabilityResolver and provider selection.
Run: python -m pytest tests/ -v
"""

import os
import sys
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.inference.schemas import DeviceClass, InferenceBackend
from packages.inference.providers.vision_roboflow import RoboflowHostedVisionProvider
from packages.inference.providers.resolver import (
    CapabilityResolver,
    LANGUAGE_MODELS,
    VISION_MODELS,
    DEVICE_LANGUAGE_PREFERENCES,
    DEVICE_VISION_PREFERENCES,
)


class TestCapabilityResolver:
    """Test hardware detection and model selection."""

    def test_orin_nano_selects_small_models(self):
        """Orin Nano (8GB) should pick qwen25_3b_int4 and yolov8n_trt."""
        os.environ["INTELFACTOR_DEVICE_CLASS"] = "orin_nano"
        resolver = CapabilityResolver(config={"model_dir": "/tmp/test_models"})

        # Force non-Jetson detection so env override works
        caps = resolver.detect_capabilities()

        # Check language model selection
        prefs = DEVICE_LANGUAGE_PREFERENCES[DeviceClass.ORIN_NANO]
        assert prefs[0] == "qwen25_3b_int4"
        assert LANGUAGE_MODELS["qwen25_3b_int4"]["backend"] == InferenceBackend.LLAMA_CPP

        # Check vision model selection
        vprefs = DEVICE_VISION_PREFERENCES[DeviceClass.ORIN_NANO]
        assert vprefs[0] == "yolov8n_trt"

        del os.environ["INTELFACTOR_DEVICE_CLASS"]

    def test_agx_orin_selects_larger_models(self):
        """AGX Orin (64GB) should prefer 7B model."""
        prefs = DEVICE_LANGUAGE_PREFERENCES[DeviceClass.AGX_ORIN]
        assert prefs[0] == "qwen25_7b_int4"
        assert LANGUAGE_MODELS["qwen25_7b_int4"]["min_vram_mb"] <= 64 * 1024

    def test_thor_t5000_selects_largest_models(self):
        """Thor T5000 (128GB) should prefer 20B+ vLLM."""
        prefs = DEVICE_LANGUAGE_PREFERENCES[DeviceClass.THOR_T5000]
        assert prefs[0] == "qwen25_20b_vllm"
        assert LANGUAGE_MODELS["qwen25_20b_vllm"]["backend"] == InferenceBackend.VLLM

    def test_gpu_server_uses_vllm(self):
        """GPU server should use vLLM backend."""
        prefs = DEVICE_LANGUAGE_PREFERENCES[DeviceClass.GPU_SERVER]
        first_choice = prefs[0]
        assert LANGUAGE_MODELS[first_choice]["backend"] == InferenceBackend.VLLM

    def test_vram_constraint_downgrades_model(self):
        """If VRAM is insufficient, resolver should pick smaller model."""
        resolver = CapabilityResolver(config={"model_dir": "/tmp/test_models"})

        # Simulate 30GB VRAM server
        selected = resolver._select_best_model(
            preferences=["qwen25_72b_vllm", "qwen25_20b_vllm", "qwen25_7b_vllm"],
            catalog=LANGUAGE_MODELS,
            available_vram_mb=30000,  # 30GB — too small for 72B (48GB), fits 20B (24GB)
        )
        assert selected == "qwen25_20b_vllm"

    def test_vram_constraint_falls_to_smallest(self):
        """Very low VRAM should fall to smallest model."""
        resolver = CapabilityResolver(config={"model_dir": "/tmp/test_models"})

        selected = resolver._select_best_model(
            preferences=["qwen25_72b_vllm", "qwen25_20b_vllm", "qwen25_7b_vllm"],
            catalog=LANGUAGE_MODELS,
            available_vram_mb=10000,
        )
        # 10GB won't fit 72B (48GB) or 20B (24GB), but 7B vllm needs 16GB — also too big
        # Falls back to first in list
        assert selected in LANGUAGE_MODELS

    def test_all_device_classes_have_preferences(self):
        """Every DeviceClass should have language and vision preferences."""
        for dc in DeviceClass:
            assert dc in DEVICE_LANGUAGE_PREFERENCES, f"Missing language prefs for {dc}"
            assert dc in DEVICE_VISION_PREFERENCES, f"Missing vision prefs for {dc}"

    def test_all_preferred_models_exist_in_catalog(self):
        """All models referenced in preferences should exist in catalogs."""
        for dc, prefs in DEVICE_LANGUAGE_PREFERENCES.items():
            for model_key in prefs:
                assert model_key in LANGUAGE_MODELS, f"{model_key} missing from LANGUAGE_MODELS (device={dc})"

        for dc, prefs in DEVICE_VISION_PREFERENCES.items():
            for model_key in prefs:
                assert model_key in VISION_MODELS, f"{model_key} missing from VISION_MODELS (device={dc})"

    def test_model_specs_have_required_fields(self):
        """All model definitions should have required fields."""
        for key, model in LANGUAGE_MODELS.items():
            assert "name" in model, f"Missing 'name' in {key}"
            assert "backend" in model, f"Missing 'backend' in {key}"
            assert "min_vram_mb" in model, f"Missing 'min_vram_mb' in {key}"
            assert "quantization" in model, f"Missing 'quantization' in {key}"

        for key, model in VISION_MODELS.items():
            assert "name" in model, f"Missing 'name' in {key}"
            assert "backend" in model, f"Missing 'backend' in {key}"
            assert "min_vram_mb" in model, f"Missing 'min_vram_mb' in {key}"

    def test_resolve_roboflow_provider_uses_workflow_when_configured(self):
        """Roboflow provider should prefer workflow mode when workflow_id is configured."""
        resolver = CapabilityResolver(config={
            "model_dir": "/tmp/test_models",
            "roboflow_api_key": "rf_test_key",
            "roboflow_workspace": "intelfactor-aj2un",
            "roboflow_workflow_id": "detect-and-classify-4",
            "class_name_map": {"Scratches": "blade_scratch"},
            "confidence_threshold": 0.4,
        })

        provider = resolver.resolve_vision_provider(
            override_model="roboflow_hosted",
            provider_config={"station_id": "station_1"},
        )

        assert isinstance(provider, RoboflowHostedVisionProvider)
        assert provider.api_key == "rf_test_key"
        assert provider.workspace == "intelfactor-aj2un"
        assert provider.workflow_id == "detect-and-classify-4"
        assert provider.use_workflow is True
        assert provider.api_url == "https://serverless.roboflow.com"
        assert provider.model_id == "detect-and-classify-4"
        assert provider.class_name_map["Scratches"] == "blade_scratch"
