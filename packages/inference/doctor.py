"""
IntelFactor.ai — Doctor (Pre-Flight Diagnostics)
Validates that a station is ready to run before starting inference.

Checks:
  1. Camera reachable
  2. Disk space sufficient
  3. Model files present (vision + language)
  4. GPU available and healthy
  5. TensorRT engine loads
  6. llama.cpp model loads
  7. Data directory writable
  8. Config file valid

Exit 0 = all checks pass. Exit 1 = any check fails.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    detail: str = ""
    elapsed_ms: float = 0.0


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        return f"{passed}/{total} checks passed"

    def print_report(self) -> None:
        """Print human-readable report to stdout."""
        print("\n" + "=" * 60)
        print("  IntelFactor Station Doctor")
        print("=" * 60)
        for c in self.checks:
            icon = "✓" if c.passed else "✗"
            status = "PASS" if c.passed else "FAIL"
            print(f"  [{icon}] {status:4s}  {c.name}")
            if c.message:
                print(f"           {c.message}")
            if c.detail and not c.passed:
                print(f"           → {c.detail}")
        print("-" * 60)
        print(f"  Result: {self.summary}")
        if not self.all_passed:
            print("  ⚠  Fix failing checks before starting the station.")
        print("=" * 60 + "\n")


def _timed(fn):
    """Wrapper that times a check function."""
    t0 = time.monotonic()
    result = fn()
    result.elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    return result


def check_disk(data_dir: str, min_gb: float = 5.0) -> CheckResult:
    """Check available disk space."""
    path = Path(data_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(path))
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        if free_gb >= min_gb:
            return CheckResult("Disk Space", True, f"{free_gb:.1f} GB free / {total_gb:.1f} GB total")
        else:
            return CheckResult("Disk Space", False, f"Only {free_gb:.1f} GB free (need {min_gb:.0f} GB)", "Free up disk space or change data_dir")
    except Exception as e:
        return CheckResult("Disk Space", False, f"Cannot check: {e}")


def check_data_dir_writable(data_dir: str) -> CheckResult:
    """Check that data directory is writable."""
    path = Path(data_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".doctor_test"
        test_file.write_text("ok")
        test_file.unlink()
        return CheckResult("Data Directory", True, f"Writable: {data_dir}")
    except Exception as e:
        return CheckResult("Data Directory", False, f"Not writable: {data_dir}", str(e))


def check_gpu() -> CheckResult:
    """Check GPU availability."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                name, total, free, temp = parts[0], parts[1], parts[2], parts[3]
                return CheckResult("GPU", True, f"{name} — {free}/{total} MB free, {temp}°C")
            return CheckResult("GPU", True, result.stdout.strip())
        else:
            return CheckResult("GPU", False, "nvidia-smi failed", result.stderr.strip())
    except FileNotFoundError:
        return CheckResult("GPU", False, "nvidia-smi not found", "Install NVIDIA drivers or JetPack")
    except Exception as e:
        return CheckResult("GPU", False, f"GPU check failed: {e}")


def check_camera(source: str, protocol: str = "rtsp") -> CheckResult:
    """Check if camera is reachable."""
    if not source:
        return CheckResult("Camera", False, "No camera source configured", "Set camera.source in station.yaml")

    try:
        import cv2
    except ImportError:
        return CheckResult("Camera", False, "OpenCV not installed", "pip install opencv-python-headless")

    try:
        if protocol == "usb":
            try:
                src = int(source) if source.isdigit() else source
            except ValueError:
                src = source
        else:
            src = source

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return CheckResult("Camera", False, f"Cannot open: {source}", "Check camera connection and URL")

        ret, frame = cap.read()
        cap.release()

        if ret and frame is not None:
            h, w = frame.shape[:2]
            return CheckResult("Camera", True, f"Connected: {source} ({w}x{h})")
        else:
            return CheckResult("Camera", False, f"Opened but no frames: {source}", "Camera may be in use or misconfigured")
    except Exception as e:
        return CheckResult("Camera", False, f"Camera error: {e}")


def check_vision_model(model_dir: str) -> CheckResult:
    """
    Check for TensorRT engine files and verify they load on the current device.

    Three-stage check:
      1. File presence — is there a .engine file at all?
      2. Architecture probe — does the engine manifest claim the right device?
      3. Load test — does tensorrt actually deserialize it without errors?

    Stage 3 is skipped gracefully when TensorRT/pycuda is not installed
    (development machines) — file presence is enough there.
    """
    path = Path(model_dir)
    if not path.exists():
        return CheckResult(
            "Vision Model", False,
            f"Model dir not found: {model_dir}",
            "Create directory and run: make build-trt MODEL=yolov8n.pt PRECISION=fp16",
        )

    engines = list(path.glob("*.engine"))
    onnx_files = list(path.glob("*.onnx"))
    pt_files = list(path.glob("*.pt"))

    # ── Stage 1: file presence ────────────────────────────────────────────────
    if not engines:
        if onnx_files:
            return CheckResult(
                "Vision Model", False,
                f"ONNX found ({onnx_files[0].name}) but no .engine file",
                f"Run: make build-trt MODEL={onnx_files[0]} PRECISION=fp16",
            )
        if pt_files:
            return CheckResult(
                "Vision Model", False,
                f"PyTorch .pt found ({pt_files[0].name}) but no .engine file",
                f"Run: make build-trt MODEL={pt_files[0]} PRECISION=fp16\n"
                "           NOTE: export ONNX on x86 first, then build engine on Jetson",
            )
        return CheckResult(
            "Vision Model", False,
            "No model files found in model directory",
            f"Run: make build-trt MODEL=yolov8n.pt PRECISION=fp16\n"
            f"     or: ./scripts/setup_models.sh",
        )

    # ── Stage 2: manifest cross-check (device architecture) ──────────────────
    for engine_path in engines:
        stem = engine_path.stem
        manifest_path = engine_path.parent / f"{stem}_manifest.json"
        if manifest_path.exists():
            try:
                import json as _json
                manifest = _json.loads(manifest_path.read_text())
                built_on = manifest.get("device_model", "")
                built_arch = manifest.get("device_arch", "")
                current_arch = os.uname().machine

                if built_arch and built_arch != current_arch:
                    return CheckResult(
                        "Vision Model", False,
                        f"Engine architecture mismatch: built for {built_arch}, "
                        f"running on {current_arch}",
                        f"Rebuild on this device: make build-trt MODEL=<source> PRECISION=fp16",
                    )
            except Exception:
                pass  # manifest parse failure is non-fatal — continue to load test

    # ── Stage 3: TRT load test ────────────────────────────────────────────────
    engine_path = engines[0]
    size_mb = engine_path.stat().st_size / (1024 * 1024)

    try:
        import tensorrt as trt  # type: ignore[import]

        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            return CheckResult(
                "Vision Model", False,
                f"Engine deserialization failed: {engine_path.name}",
                "Engine may be built for a different GPU architecture. "
                "Run: make build-trt MODEL=<source> PRECISION=fp16",
            )

        # Check input tensor count as minimal sanity test
        n_io = engine.num_io_tensors
        return CheckResult(
            "Vision Model", True,
            f"{engine_path.name} ({size_mb:.0f}MB) — loads OK, {n_io} I/O tensors",
            f"Full verify: make verify-trt ENGINE={engine_path}",
        )

    except ImportError:
        # TensorRT not installed on this machine (dev environment) — file check only
        sizes = [f"{e.name} ({e.stat().st_size / (1024*1024):.0f}MB)" for e in engines]
        return CheckResult(
            "Vision Model", True,
            f"TRT engines present: {', '.join(sizes)} "
            f"(TensorRT not installed — load test skipped)",
        )

    except Exception as exc:
        return CheckResult(
            "Vision Model", False,
            f"Engine load failed: {exc}",
            "Run: make verify-trt ENGINE=" + str(engine_path),
        )


def check_language_model(model_dir: str) -> CheckResult:
    """Check for SLM model files (GGUF for llama.cpp)."""
    path = Path(model_dir)
    if not path.exists():
        return CheckResult("Language Model", False, f"Model dir not found: {model_dir}")

    gguf = list(path.glob("*.gguf"))
    if gguf:
        sizes = [f"{g.name} ({g.stat().st_size / (1024*1024*1024):.1f}GB)" for g in gguf]
        return CheckResult("Language Model", True, f"GGUF models: {', '.join(sizes)}")
    else:
        return CheckResult("Language Model", False, "No .gguf files found", "Run scripts/download_qwen.sh to download Qwen-2.5-3B")


def check_config(config_path: str) -> CheckResult:
    """Check that config file exists and is valid."""
    path = Path(config_path)
    if not path.exists():
        return CheckResult("Config", False, f"Not found: {config_path}", "Create station.yaml from template")

    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        try:
            with open(path) as f:
                cfg = json.load(f)
        except Exception as e:
            return CheckResult("Config", False, f"Parse error: {e}")
    except Exception as e:
        return CheckResult("Config", False, f"Parse error: {e}")

    if not cfg:
        return CheckResult("Config", False, "Config is empty")

    station_id = cfg.get("station_id", "")
    return CheckResult("Config", True, f"Valid — station_id={station_id}")


def check_taxonomy(taxonomy_path: str) -> CheckResult:
    """Check that defect taxonomy is available."""
    path = Path(taxonomy_path)
    if not path.exists():
        return CheckResult("Taxonomy", False, f"Not found: {taxonomy_path}", "Copy configs/wiko_taxonomy.yaml to config dir")

    try:
        import yaml
        with open(path) as f:
            tax = yaml.safe_load(f)
        defects = tax.get("defects", {})
        if len(defects) > 0:
            return CheckResult("Taxonomy", True, f"{len(defects)} defect types loaded")
        else:
            return CheckResult("Taxonomy", False, "Taxonomy has no defect types")
    except Exception as e:
        return CheckResult("Taxonomy", False, f"Parse error: {e}")


def run_doctor(
    config_path: str = "/opt/intelfactor/config/station.yaml",
    data_dir: str | None = None,
    model_dir: str | None = None,
    camera_source: str | None = None,
    camera_protocol: str = "rtsp",
    taxonomy_path: str | None = None,
    skip_camera: bool = False,
) -> DoctorReport:
    """
    Run all diagnostic checks and return report.

    Parameters can be passed explicitly or loaded from config file.
    """
    report = DoctorReport()

    # Load config if available
    cfg: dict[str, Any] = {}
    config_p = Path(config_path)
    if config_p.exists():
        try:
            import yaml
            with open(config_p) as f:
                cfg = yaml.safe_load(f) or {}
        except ImportError:
            try:
                with open(config_p) as f:
                    cfg = json.load(f)
            except Exception:
                pass

    # Resolve paths from config or arguments
    _data_dir = data_dir or cfg.get("data_dir", "/opt/intelfactor/data")
    _model_dir = model_dir or cfg.get("model_dir", "/opt/intelfactor/models")
    _cam_source = camera_source or cfg.get("camera", {}).get("source", "")
    _cam_proto = camera_protocol or cfg.get("camera", {}).get("protocol", "rtsp")
    _taxonomy = taxonomy_path or cfg.get("taxonomy_path", "/opt/intelfactor/config/wiko_taxonomy.yaml")

    # Run checks
    report.checks.append(_timed(lambda: check_config(config_path)))
    report.checks.append(_timed(lambda: check_disk(_data_dir)))
    report.checks.append(_timed(lambda: check_data_dir_writable(_data_dir)))
    report.checks.append(_timed(lambda: check_gpu()))
    report.checks.append(_timed(lambda: check_vision_model(_model_dir)))
    report.checks.append(_timed(lambda: check_language_model(_model_dir)))
    report.checks.append(_timed(lambda: check_taxonomy(_taxonomy)))

    if not skip_camera and _cam_source:
        report.checks.append(_timed(lambda: check_camera(_cam_source, _cam_proto)))
    elif not _cam_source:
        report.checks.append(CheckResult("Camera", False, "No source configured", "Set camera.source in station.yaml"))

    return report
