"""
IntelFactor.ai — Capability Resolver
Auto-detects GPU hardware and resolves the optimal model + backend + quantization.
Run once at startup. Returns provider instances ready to load().
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from packages.inference.providers.base import LanguageProvider, VisionProvider
from packages.inference.providers.language_llama import LlamaCppLanguageProvider
from packages.inference.providers.language_vllm import VLLMLanguageProvider
from packages.inference.providers.vision_trt import TensorRTVisionProvider
from packages.inference.schemas import (
    DeviceCapabilities,
    DeviceClass,
    InferenceBackend,
    ModelSpec,
)
from packages.inference.utils.model_bundle import (
    load_labels,
    load_thresholds,
    load_metadata,
    validate_bundle,
    ModelBundleError,
)

logger = logging.getLogger(__name__)

# Canonical defect class list — KMG Kyoto Specs (13 classes, IDs 0-12).
# Must match training/config/defect_taxonomy.yaml and the trained .engine file.
# Overridden at runtime by station.yaml defect_classes (passed via provider_config).
_CANONICAL_DEFECT_CLASSES: list[str] = [
    "blade_scratch",    # 0
    "grinding_mark",    # 1
    "surface_dent",     # 2
    "surface_crack",    # 3
    "weld_defect",      # 4
    "edge_burr",        # 5
    "edge_crack",       # 6
    "handle_defect",    # 7
    "bolster_gap",      # 8
    "etching_defect",   # 9
    "inclusion",        # 10
    "surface_discolor", # 11
    "overgrind",        # 12
]


# ── Model Catalog ──────────────────────────────────────────────────────
# Maps device class to recommended models.
# model_path is resolved at runtime from config or model registry.

VISION_MODELS: dict[str, dict[str, Any]] = {
    "yolov8n_trt": {
        "name": "yolov8n-cutlery-v3",
        "quantization": "FP16",
        "backend": InferenceBackend.TENSORRT,
        "expected_latency_ms": 15,
        "min_vram_mb": 512,
        "defect_classes": _CANONICAL_DEFECT_CLASSES,
    },
    "yolov8s_trt": {
        "name": "yolov8s-cutlery-v3",
        "quantization": "FP16",
        "backend": InferenceBackend.TENSORRT,
        "expected_latency_ms": 25,
        "min_vram_mb": 1024,
        "defect_classes": _CANONICAL_DEFECT_CLASSES,
    },
    "yolo26_trt": {
        "name": "yolo26-cutlery-v1",
        "quantization": "INT8",
        "backend": InferenceBackend.TENSORRT,
        "expected_latency_ms": 20,
        "min_vram_mb": 768,
        "defect_classes": _CANONICAL_DEFECT_CLASSES,
    },
}

LANGUAGE_MODELS: dict[str, dict[str, Any]] = {
    "qwen25_3b_int4": {
        "name": "Qwen2.5-3B-Instruct-GGUF",
        "quantization": "INT4",
        "backend": InferenceBackend.LLAMA_CPP,
        "max_tokens": 512,
        "context_window": 4096,
        "expected_latency_ms": 3000,
        "min_vram_mb": 2500,
    },
    "phi3_mini_int4": {
        "name": "Phi-3-mini-4k-instruct-GGUF",
        "quantization": "INT4",
        "backend": InferenceBackend.LLAMA_CPP,
        "max_tokens": 512,
        "context_window": 4096,
        "expected_latency_ms": 3500,
        "min_vram_mb": 2800,
    },
    "qwen25_7b_int4": {
        "name": "Qwen2.5-7B-Instruct-GGUF",
        "quantization": "INT4",
        "backend": InferenceBackend.LLAMA_CPP,
        "max_tokens": 1024,
        "context_window": 8192,
        "expected_latency_ms": 5000,
        "min_vram_mb": 5000,
    },
    "qwen25_7b_vllm": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": "FP16",
        "backend": InferenceBackend.VLLM,
        "max_tokens": 1024,
        "context_window": 8192,
        "expected_latency_ms": 2000,
        "min_vram_mb": 16000,
    },
    "qwen25_20b_vllm": {
        "name": "Qwen/Qwen2.5-20B-Instruct",
        "quantization": "INT4",
        "backend": InferenceBackend.VLLM,
        "max_tokens": 2048,
        "context_window": 16384,
        "expected_latency_ms": 4000,
        "min_vram_mb": 24000,
    },
    "qwen25_72b_vllm": {
        "name": "Qwen/Qwen2.5-72B-Instruct",
        "quantization": "INT4",
        "backend": InferenceBackend.VLLM,
        "max_tokens": 2048,
        "context_window": 32768,
        "expected_latency_ms": 8000,
        "min_vram_mb": 48000,
    },
}

# Device class → preferred language model (ordered by preference)
DEVICE_LANGUAGE_PREFERENCES: dict[DeviceClass, list[str]] = {
    DeviceClass.ORIN_NANO:   ["qwen25_3b_int4", "phi3_mini_int4"],
    DeviceClass.ORIN_NX:     ["qwen25_3b_int4", "phi3_mini_int4"],
    DeviceClass.AGX_ORIN:    ["qwen25_7b_int4", "qwen25_3b_int4"],
    DeviceClass.THOR_T4000:  ["qwen25_7b_vllm", "qwen25_20b_vllm", "qwen25_7b_int4"],
    DeviceClass.THOR_T5000:  ["qwen25_20b_vllm", "qwen25_72b_vllm", "qwen25_7b_vllm"],
    DeviceClass.GPU_SERVER:  ["qwen25_20b_vllm", "qwen25_7b_vllm", "qwen25_72b_vllm"],
}

# Device class → preferred vision model
DEVICE_VISION_PREFERENCES: dict[DeviceClass, list[str]] = {
    DeviceClass.ORIN_NANO:   ["yolov8n_trt"],
    DeviceClass.ORIN_NX:     ["yolov8s_trt", "yolo26_trt"],
    DeviceClass.AGX_ORIN:    ["yolo26_trt", "yolov8s_trt"],
    DeviceClass.THOR_T4000:  ["yolo26_trt", "yolov8s_trt"],
    DeviceClass.THOR_T5000:  ["yolo26_trt", "yolov8s_trt"],
    DeviceClass.GPU_SERVER:  ["yolo26_trt", "yolov8s_trt"],
}


class CapabilityResolver:
    """
    Detects hardware capabilities and resolves optimal providers.

    Usage:
        resolver = CapabilityResolver(config)
        caps = resolver.detect_capabilities()
        vision = resolver.resolve_vision_provider()
        language = resolver.resolve_language_provider()
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._capabilities: DeviceCapabilities | None = None
        self.model_dir = Path(self.config.get("model_dir", "/opt/intelfactor/models"))

    def detect_capabilities(self) -> DeviceCapabilities:
        """Detect GPU hardware capabilities."""
        if self._capabilities is not None:
            return self._capabilities

        caps = DeviceCapabilities()

        # Check if we're on Jetson
        caps.jetson = self._is_jetson()

        if caps.jetson:
            caps = self._detect_jetson(caps)
        else:
            caps = self._detect_server_gpu(caps)

        self._capabilities = caps
        logger.info(
            "Hardware detected: %s (GPU: %s, VRAM: %dMB, Jetson: %s)",
            caps.device_class.value, caps.gpu_name, caps.vram_mb, caps.jetson,
        )
        return caps

    def resolve_vision_provider(
        self,
        override_model: str | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> VisionProvider:
        """Resolve the optimal VisionProvider for detected hardware."""
        caps = self.detect_capabilities()

        # Dev/stub mode — no TRT hardware required
        model_key = override_model or self.config.get("vision_model")
        if model_key == "stub":
            from packages.inference.providers.stub import StubVisionProvider
            merged_config = {**self.config, **(provider_config or {})}
            return StubVisionProvider(config=merged_config)

        # Allow config override
        if model_key and model_key in VISION_MODELS:
            model_def = VISION_MODELS[model_key]
        else:
            # Auto-select based on device class
            model_key = self._select_best_model(
                DEVICE_VISION_PREFERENCES.get(caps.device_class, ["yolov8n_trt"]),
                VISION_MODELS,
                caps.vram_mb,
            )
            model_def = VISION_MODELS[model_key]

        # Load model bundle if configured
        merged_config = {**self.config, **(provider_config or {})}
        bundle_config = self.config.get("model_bundle", {})
        
        if bundle_config and bundle_config.get("path"):
            try:
                bundle = self._load_model_bundle(bundle_config)
                # Merge bundle data into provider config
                merged_config["model_bundle"] = bundle
                merged_config["labels"] = bundle["labels"]
                merged_config["thresholds"] = bundle["thresholds"]
                merged_config["model_version"] = bundle["metadata"].get("model_version", "unknown")
                merged_config["model_name"] = bundle["metadata"].get("model_name", "unknown")
                # Use bundle engine path if available
                engine_path = bundle["engine_path"]
                logger.info(
                    "Loaded model bundle: %s v%s",
                    bundle["metadata"].get("model_name"),
                    bundle["metadata"].get("model_version"),
                )
            except ModelBundleError as e:
                if bundle_config.get("validate_on_load", True):
                    raise RuntimeError(f"Model bundle validation failed: {e}") from e
                logger.error(f"Failed to load model bundle: {e}")
                # Fallback to legacy path resolution
                engine_path = self._find_engine_path(model_def["name"])
        else:
            # Legacy path resolution
            engine_path = self._find_engine_path(model_def["name"])

        # Build ModelSpec
        spec = ModelSpec(
            model_name=model_def["name"],
            model_path=str(engine_path),
            quantization=model_def["quantization"],
            backend=model_def["backend"],
            expected_latency_ms=model_def["expected_latency_ms"],
        )

        # Guard: defect_classes must be explicit — silent fallback produces defect_0/defect_1 labels
        if not merged_config.get("defect_classes"):
            model_defaults = model_def.get("defect_classes", [])
            if model_defaults:
                merged_config = {**merged_config, "defect_classes": model_defaults}
                logger.warning(
                    "defect_classes not provided via station.yaml for model %s — "
                    "using model registry defaults. Set defect_classes in station.yaml "
                    "to suppress this warning.",
                    model_key,
                )
            else:
                logger.warning(
                    "defect_classes is empty for model %s — TensorRTVisionProvider will "
                    "use generic fallback names (defect_0, defect_1, ...). "
                    "Check configs/wiko_taxonomy.yaml and station.yaml.",
                    model_key,
                )
        elif len(merged_config["defect_classes"]) != 13:
            logger.warning(
                "defect_classes has %d entries for model %s — expected 13. "
                "Check configs/wiko_taxonomy.yaml alignment.",
                len(merged_config["defect_classes"]),
                model_key,
            )

        return TensorRTVisionProvider(spec, merged_config)

    def _load_model_bundle(self, bundle_config: dict[str, Any]) -> dict[str, Any]:
        """Load and validate model bundle from config.
        
        Args:
            bundle_config: model_bundle section from station.yaml
            
        Returns:
            Dict with loaded bundle components
            
        Raises:
            ModelBundleError: If bundle is invalid or missing
        """
        bundle_path = Path(bundle_config["path"])
        if not bundle_path.is_absolute():
            bundle_path = self.model_dir / bundle_path
        
        # Resolve individual file paths
        engine_path = bundle_path / bundle_config.get("engine", "model.engine")
        labels_path = bundle_path / bundle_config.get("labels", "labels.json")
        thresholds_path = bundle_path / bundle_config.get("thresholds", "thresholds.yaml")
        metadata_path = bundle_path / bundle_config.get("metadata", "metadata.json")
        
        # Validate against canonical taxonomy if strict mode enabled
        expected_classes = None
        if bundle_config.get("strict_taxonomy", True):
            expected_classes = self.config.get("defect_classes", _CANONICAL_DEFECT_CLASSES)
        
        return validate_bundle(
            engine_path=engine_path,
            labels_path=labels_path,
            thresholds_path=thresholds_path,
            metadata_path=metadata_path,
            expected_classes=expected_classes,
        )

    def resolve_language_provider(
        self,
        override_model: str | None = None,
        provider_config: dict[str, Any] | None = None,
    ) -> LanguageProvider:
        """Resolve the optimal LanguageProvider for detected hardware."""
        caps = self.detect_capabilities()

        # Dev/stub mode — no GPU or GGUF required
        model_key = override_model or self.config.get("language_model")
        if model_key == "stub":
            from packages.inference.providers.stub import StubLanguageProvider
            merged_config = {**self.config, **(provider_config or {})}
            return StubLanguageProvider(config=merged_config)

        # Allow config override
        if model_key and model_key in LANGUAGE_MODELS:
            model_def = LANGUAGE_MODELS[model_key]
        else:
            model_key = self._select_best_model(
                DEVICE_LANGUAGE_PREFERENCES.get(caps.device_class, ["qwen25_3b_int4"]),
                LANGUAGE_MODELS,
                caps.vram_mb,
            )
            model_def = LANGUAGE_MODELS[model_key]

        # Build ModelSpec
        model_path = self._find_model_path(model_def["name"], model_def["backend"])
        spec = ModelSpec(
            model_name=model_def["name"],
            model_path=str(model_path),
            quantization=model_def["quantization"],
            backend=model_def["backend"],
            max_tokens=model_def["max_tokens"],
            context_window=model_def["context_window"],
            expected_latency_ms=model_def["expected_latency_ms"],
        )

        merged_config = {**self.config, **(provider_config or {})}

        # Select provider class based on backend
        if model_def["backend"] == InferenceBackend.LLAMA_CPP:
            return LlamaCppLanguageProvider(spec, merged_config)
        elif model_def["backend"] in (InferenceBackend.VLLM, InferenceBackend.TENSORRT_LLM):
            return VLLMLanguageProvider(spec, merged_config)
        else:
            raise ValueError(f"Unsupported language backend: {model_def['backend']}")

    # ── Hardware Detection ─────────────────────────────────────────────

    def _is_jetson(self) -> bool:
        """Detect if running on NVIDIA Jetson."""
        # Method 1: Check for Jetson-specific file
        if Path("/etc/nv_tegra_release").exists():
            return True
        # Method 2: Check for Jetson in device tree
        dt_model = Path("/proc/device-tree/model")
        if dt_model.exists():
            model = dt_model.read_text(errors="ignore").lower()
            if "jetson" in model or "orin" in model or "thor" in model:
                return True
        # Method 3: Environment variable override
        return os.environ.get("INTELFACTOR_DEVICE", "").startswith("jetson")

    def _detect_jetson(self, caps: DeviceCapabilities) -> DeviceCapabilities:
        """Detect Jetson model and capabilities."""
        caps.jetson = True

        # Read device tree model
        dt_model = Path("/proc/device-tree/model")
        model_str = ""
        if dt_model.exists():
            model_str = dt_model.read_text(errors="ignore").lower()

        # Read JetPack version
        jetpack_file = Path("/etc/nv_jetpack_release")
        if jetpack_file.exists():
            caps.jetpack_version = jetpack_file.read_text().strip()

        # Classify device
        if "thor" in model_str and "t5000" in model_str:
            caps.device_class = DeviceClass.THOR_T5000
            caps.gpu_name = "Jetson Thor T5000"
            caps.vram_mb = 128 * 1024  # 128GB unified
            caps.max_power_w = 130
        elif "thor" in model_str or "t4000" in model_str:
            caps.device_class = DeviceClass.THOR_T4000
            caps.gpu_name = "Jetson Thor T4000"
            caps.vram_mb = 64 * 1024
            caps.max_power_w = 70
        elif "agx" in model_str and "orin" in model_str:
            caps.device_class = DeviceClass.AGX_ORIN
            caps.gpu_name = "Jetson AGX Orin"
            caps.vram_mb = self._get_jetson_memory_mb(default=64 * 1024)
            caps.max_power_w = 60
        elif "orin" in model_str and "nx" in model_str:
            caps.device_class = DeviceClass.ORIN_NX
            caps.gpu_name = "Jetson Orin NX"
            caps.vram_mb = self._get_jetson_memory_mb(default=16 * 1024)
            caps.max_power_w = 40
        elif "orin" in model_str and "nano" in model_str:
            caps.device_class = DeviceClass.ORIN_NANO
            caps.gpu_name = "Jetson Orin Nano"
            caps.vram_mb = self._get_jetson_memory_mb(default=8 * 1024)
            caps.max_power_w = 25
        else:
            # Default to Orin Nano if unknown Jetson
            caps.device_class = DeviceClass.ORIN_NANO
            caps.gpu_name = f"Jetson (unknown: {model_str[:40]})"
            caps.vram_mb = self._get_jetson_memory_mb(default=8 * 1024)

        # Env var override for testing
        override = os.environ.get("INTELFACTOR_DEVICE_CLASS")
        if override:
            try:
                caps.device_class = DeviceClass(override)
                logger.info("Device class overridden to: %s", override)
            except ValueError:
                pass

        return caps

    def _detect_server_gpu(self, caps: DeviceCapabilities) -> DeviceCapabilities:
        """Detect server-class NVIDIA GPU."""
        caps.device_class = DeviceClass.GPU_SERVER

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]
                parts = line.split(",")
                caps.gpu_name = parts[0].strip()
                caps.vram_mb = int(float(parts[1].strip()))
        except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
            caps.gpu_name = "Unknown GPU"
            caps.vram_mb = 8 * 1024  # conservative default

        # Env var override
        override = os.environ.get("INTELFACTOR_DEVICE_CLASS")
        if override:
            try:
                caps.device_class = DeviceClass(override)
            except ValueError:
                pass

        return caps

    def _get_jetson_memory_mb(self, default: int = 8192) -> int:
        """Get total system memory on Jetson (unified memory = GPU memory)."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        except (FileNotFoundError, ValueError):
            pass
        return default

    # ── Model Selection ────────────────────────────────────────────────

    def _select_best_model(
        self,
        preferences: list[str],
        catalog: dict[str, dict[str, Any]],
        available_vram_mb: int,
    ) -> str:
        """Select the best model from preference list that fits in VRAM."""
        for model_key in preferences:
            model_def = catalog.get(model_key)
            if model_def and model_def.get("min_vram_mb", 0) <= available_vram_mb:
                return model_key

        # Fallback to first in catalog
        return preferences[0] if preferences else list(catalog.keys())[0]

    def _find_engine_path(self, model_name: str) -> Path:
        """Find TensorRT engine file in model directory."""
        candidates = [
            self.model_dir / f"{model_name}.engine",
            self.model_dir / model_name / "model.engine",
            Path(f"/opt/intelfactor/active/{model_name}.engine"),
        ]
        for p in candidates:
            if p.exists():
                return p
        # Return expected path even if not found (will fail at load time)
        return candidates[0]

    def _find_model_path(self, model_name: str, backend: InferenceBackend) -> Path:
        """Find model file/directory in model directory."""
        if backend == InferenceBackend.LLAMA_CPP:
            # GGUF files
            candidates = [
                self.model_dir / f"{model_name}.gguf",
                self.model_dir / model_name / "model.gguf",
            ]
            # Also check for any .gguf file matching the model name prefix
            if self.model_dir.exists():
                for f in self.model_dir.glob("*.gguf"):
                    if model_name.lower().replace("-", "").replace("_", "") in f.stem.lower().replace("-", "").replace("_", ""):
                        return f
        else:
            # HuggingFace model directory or name
            candidates = [
                self.model_dir / model_name,
            ]

        for p in candidates:
            if p.exists():
                return p

        # For vLLM, model_name is the HF repo ID — return as-is
        if backend in (InferenceBackend.VLLM, InferenceBackend.TENSORRT_LLM):
            return Path(model_name)

        return candidates[0] if candidates else Path(model_name)
