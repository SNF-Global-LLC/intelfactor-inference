#!/usr/bin/env python3
"""
IntelFactor.ai — TensorRT Engine Verifier

Loads a .engine file, allocates buffers, runs inference on a zeros tensor,
and prints timing + shape information. Used as a post-build sanity check
and called by doctor.py to verify engines are runnable on the current device.

Usage:
    python scripts/verify_trt_engine.py <engine_path>
    python scripts/verify_trt_engine.py /opt/intelfactor/models/vision/yolov8n_fp16.engine
    make verify-trt ENGINE=/opt/intelfactor/models/vision/yolov8n_fp16.engine

Exit codes:
    0 — engine loaded and ran successfully
    1 — engine not found, failed to load, or inference error
    2 — TensorRT / pycuda not installed (can't verify on this machine)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


# ── Colour helpers ────────────────────────────────────────────────────────────

def _col(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def ok(msg: str) -> None:
    print(f"  {_col('32', '[OK]')}    {msg}")

def info(msg: str) -> None:
    print(f"  {_col('36', '[INFO]')}  {msg}")

def warn(msg: str) -> None:
    print(f"  {_col('33', '[WARN]')}  {msg}")

def fail(msg: str) -> None:
    print(f"  {_col('31', '[FAIL]')}  {msg}")


# ── Core verification ─────────────────────────────────────────────────────────

def verify_engine(engine_path: str, verbose: bool = False) -> dict[str, Any]:
    """
    Load and run a TensorRT engine.

    Returns a dict with:
      success: bool
      load_ms: float
      first_inference_ms: float
      input_shapes: list[tuple]
      output_shapes: list[tuple]
      device_memory_mb: float
      error: str | None
    """
    result: dict[str, Any] = {
        "success": False,
        "load_ms": 0.0,
        "first_inference_ms": 0.0,
        "input_shapes": [],
        "output_shapes": [],
        "device_memory_mb": 0.0,
        "error": None,
    }

    path = Path(engine_path)
    if not path.exists():
        result["error"] = f"Engine file not found: {engine_path}"
        return result

    engine_size_mb = path.stat().st_size / (1024 * 1024)
    result["engine_size_mb"] = round(engine_size_mb, 1)

    # ── TensorRT import ───────────────────────────────────────────────────────
    try:
        import tensorrt as trt
    except ImportError:
        result["error"] = "tensorrt Python package not installed (TRT not available on this machine)"
        result["trt_missing"] = True
        return result

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises CUDA context
        import numpy as np
    except ImportError as exc:
        result["error"] = f"pycuda not installed: {exc}"
        result["trt_missing"] = True
        return result

    # ── Load engine ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()

    try:
        trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            result["error"] = (
                "deserialize_cuda_engine returned None — likely wrong GPU architecture. "
                "Build the engine on THIS device."
            )
            return result

    except Exception as exc:
        result["error"] = f"Engine load failed: {exc}"
        if "incompatible" in str(exc).lower() or "invalid" in str(exc).lower():
            result["error"] += (
                "\n  → This engine was likely built on a different GPU architecture. "
                "Rebuild on this device."
            )
        return result

    load_ms = (time.perf_counter() - t0) * 1000
    result["load_ms"] = round(load_ms, 1)

    # ── Inspect tensor shapes ─────────────────────────────────────────────────
    context = engine.create_execution_context()
    input_shapes = []
    output_shapes = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        mode = engine.get_tensor_mode(name)
        is_input = (mode == trt.TensorIOMode.INPUT)

        if is_input:
            input_shapes.append({"name": name, "shape": shape})
        else:
            output_shapes.append({"name": name, "shape": shape})

    result["input_shapes"] = input_shapes
    result["output_shapes"] = output_shapes

    if not input_shapes:
        result["error"] = "Engine has no input tensors — corrupt engine file?"
        return result

    # ── Allocate buffers ──────────────────────────────────────────────────────
    stream = cuda.Stream()
    bindings = []
    host_buffers = []
    device_buffers = []
    total_device_bytes = 0

    try:
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)

            # Replace dynamic dims (-1) with concrete values
            concrete_shape = tuple(max(1, d) for d in shape)
            size = int(np.prod(concrete_shape))

            np_dtype = trt.nptype(dtype)
            host_buf = cuda.pagelocked_empty(size, np_dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)

            host_buffers.append(host_buf)
            device_buffers.append(dev_buf)
            bindings.append(int(dev_buf))
            total_device_bytes += host_buf.nbytes

    except Exception as exc:
        result["error"] = f"Buffer allocation failed (OOM?): {exc}"
        return result

    result["device_memory_mb"] = round(total_device_bytes / (1024 * 1024), 1)

    # ── Run inference on zeros ────────────────────────────────────────────────
    t1 = time.perf_counter()

    try:
        # Copy zeros to first input
        host_buffers[0].fill(0)
        cuda.memcpy_htod_async(device_buffers[0], host_buffers[0], stream)

        # Execute
        ok_exec = context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        if not ok_exec:
            result["error"] = "execute_async_v2 returned False"
            return result

        # Copy output back
        cuda.memcpy_dtoh_async(host_buffers[-1], device_buffers[-1], stream)
        stream.synchronize()

    except Exception as exc:
        result["error"] = f"Inference failed: {exc}"
        return result

    first_inference_ms = (time.perf_counter() - t1) * 1000
    result["first_inference_ms"] = round(first_inference_ms, 1)
    result["success"] = True

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="IntelFactor TensorRT Engine Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_trt_engine.py /opt/intelfactor/models/vision/yolov8n_fp16.engine
  make verify-trt ENGINE=/opt/intelfactor/models/vision/yolov8n_fp16.engine
""",
    )
    parser.add_argument("engine_path", help="Path to .engine file")
    parser.add_argument("--verbose", action="store_true", help="Enable TensorRT verbose logging")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Compare against build manifest (looks for <engine>_manifest.json)",
    )
    args = parser.parse_args()

    if not args.json:
        print()
        print("  \033[1mIntelFactor TensorRT Engine Verifier\033[0m")
        print("  " + "─" * 38)
        info(f"Engine: {args.engine_path}")

    result = verify_engine(args.engine_path, verbose=args.verbose)

    # Optional manifest cross-check
    if args.manifest and result["success"]:
        engine_path = Path(args.engine_path)
        stem = engine_path.stem
        manifest_path = engine_path.parent / f"{stem}_manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            stored_sha = manifest.get("engine_sha256", "")
            import hashlib
            actual_sha = hashlib.sha256(engine_path.read_bytes()).hexdigest()
            if stored_sha and stored_sha != actual_sha:
                result["manifest_mismatch"] = True
                warn(f"SHA256 mismatch vs manifest (engine may have been modified)")
            else:
                result["manifest_match"] = True
        else:
            warn(f"Manifest not found at {manifest_path}")

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["success"] else 1

    if result.get("trt_missing"):
        warn("TensorRT / pycuda not available on this machine.")
        info("Verification is only possible on a CUDA-capable device with TensorRT installed.")
        info(f"Error: {result['error']}")
        return 2

    if not result["success"]:
        fail(f"Verification failed: {result['error']}")
        return 1

    # Print success details
    ok(f"Engine loaded in {result['load_ms']}ms")
    ok(f"First inference: {result['first_inference_ms']}ms (zeros tensor)")
    ok(f"Device memory allocated: {result['device_memory_mb']}MB")
    ok(f"Engine file size: {result.get('engine_size_mb', '?')}MB")

    print()
    info("Input tensors:")
    for t in result["input_shapes"]:
        print(f"    {t['name']:30s}  shape={t['shape']}")

    info("Output tensors:")
    for t in result["output_shapes"]:
        print(f"    {t['name']:30s}  shape={t['shape']}")

    print()
    ok("Engine is valid and runnable on this device.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
