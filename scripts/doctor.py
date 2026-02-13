#!/usr/bin/env python3
"""
IntelFactor.ai — System Doctor
Checks system health for edge-only or hybrid deployments.

Usage:
    python scripts/doctor.py
    python scripts/doctor.py --full
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path


def check_mark(ok: bool) -> str:
    return "[OK]" if ok else "[FAIL]"


def check_python():
    """Check Python version."""
    v = sys.version_info
    ok = v.major >= 3 and v.minor >= 10
    print(f"{check_mark(ok)} Python {v.major}.{v.minor}.{v.micro}")
    return ok


def check_gpu():
    """Check NVIDIA GPU availability."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"[OK] GPU: {gpu_info}")
            return True
        else:
            print("[WARN] nvidia-smi failed — no GPU detected")
            return False
    except FileNotFoundError:
        print("[WARN] nvidia-smi not found — no GPU detected")
        return False
    except Exception as e:
        print(f"[WARN] GPU check failed: {e}")
        return False


def check_camera(camera_uri: str):
    """Check camera connectivity."""
    try:
        import cv2
        cap = cv2.VideoCapture(camera_uri)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"[OK] Camera: {camera_uri} ({frame.shape[1]}x{frame.shape[0]})")
                return True
            else:
                print(f"[FAIL] Camera: {camera_uri} — could not read frame")
                return False
        else:
            print(f"[FAIL] Camera: {camera_uri} — could not open")
            return False
    except ImportError:
        print("[WARN] OpenCV not installed — camera check skipped")
        return False
    except Exception as e:
        print(f"[FAIL] Camera check failed: {e}")
        return False


def check_storage():
    """Check storage mode and database."""
    mode = os.environ.get("STORAGE_MODE", "local")
    print(f"[INFO] Storage mode: {mode}")

    if mode == "local":
        db_path = os.environ.get("SQLITE_DB_PATH", "/opt/intelfactor/data/local.db")
        path = Path(db_path)

        if path.exists():
            try:
                conn = sqlite3.connect(str(path))
                tables = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                conn.close()
                table_names = [t[0] for t in tables]
                print(f"[OK] SQLite: {db_path}")
                print(f"     Tables: {', '.join(table_names)}")
                return True
            except Exception as e:
                print(f"[FAIL] SQLite error: {e}")
                return False
        else:
            print(f"[INFO] SQLite: {db_path} (will be created on first run)")
            # Check if parent directory is writable
            if path.parent.exists() or path.parent.parent.exists():
                print("[OK] Database directory accessible")
                return True
            else:
                print("[WARN] Database directory may not be writable")
                return False

    return True


def check_evidence_dir():
    """Check evidence directory."""
    evidence_dir = Path(os.environ.get("EVIDENCE_DIR", "/opt/intelfactor/data/evidence"))

    if evidence_dir.exists():
        # Count files
        date_dirs = [d for d in evidence_dir.iterdir() if d.is_dir() and len(d.name) == 10]
        total_size = sum(
            f.stat().st_size
            for d in date_dirs
            for f in d.rglob("*") if f.is_file()
        )
        size_mb = total_size / (1024 * 1024)
        print(f"[OK] Evidence dir: {evidence_dir}")
        print(f"     {len(date_dirs)} date dirs, {size_mb:.1f} MB used")
        return True
    else:
        print(f"[INFO] Evidence dir: {evidence_dir} (will be created)")
        return True


def check_disk_space():
    """Check available disk space."""
    data_dir = Path(os.environ.get("EVIDENCE_DIR", "/opt/intelfactor/data/evidence")).parent
    if not data_dir.exists():
        data_dir = Path("/opt/intelfactor/data")
        if not data_dir.exists():
            data_dir = Path.home()

    total, used, free = shutil.disk_usage(data_dir)
    free_gb = free / (1024 ** 3)
    total_gb = total / (1024 ** 3)
    used_pct = (used / total) * 100

    ok = free_gb >= 10  # At least 10GB free
    print(f"{check_mark(ok)} Disk space: {free_gb:.1f} GB free of {total_gb:.1f} GB ({used_pct:.1f}% used)")
    return ok


def check_api_health():
    """Check if local API is running."""
    api_port = os.environ.get("API_PORT", "8080")
    url = f"http://localhost:{api_port}/health"

    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status == 200:
                print(f"[OK] API health: {url}")
                return True
    except Exception:
        pass

    print(f"[INFO] API not running (or not reachable at {url})")
    return False


def check_models():
    """Check if model files exist."""
    vision_path = Path(os.environ.get("VISION_MODEL_PATH", "/opt/intelfactor/models/vision"))
    llm_path = Path(os.environ.get("LLM_MODEL_PATH", "/opt/intelfactor/models/llm"))

    all_ok = True

    if vision_path.exists():
        files = list(vision_path.glob("*.engine")) + list(vision_path.glob("*.onnx"))
        if files:
            print(f"[OK] Vision model: {vision_path} ({len(files)} files)")
        else:
            print(f"[WARN] Vision model dir exists but no .engine/.onnx files: {vision_path}")
            all_ok = False
    else:
        print(f"[INFO] Vision model: {vision_path} (not found)")

    if llm_path.exists():
        files = list(llm_path.glob("*.gguf")) + list(llm_path.glob("*.bin"))
        if files:
            print(f"[OK] LLM model: {llm_path} ({len(files)} files)")
        else:
            print(f"[WARN] LLM model dir exists but no model files: {llm_path}")
            all_ok = False
    else:
        print(f"[INFO] LLM model: {llm_path} (not found)")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="IntelFactor System Doctor")
    parser.add_argument("--full", action="store_true", help="Run full checks including camera")
    parser.add_argument("--camera", type=str, help="Camera URI to test")
    args = parser.parse_args()

    print("=" * 60)
    print("IntelFactor.ai System Doctor")
    print("=" * 60)
    print()

    results = []

    # Basic checks
    results.append(("Python", check_python()))
    results.append(("GPU", check_gpu()))
    results.append(("Storage", check_storage()))
    results.append(("Evidence", check_evidence_dir()))
    results.append(("Disk", check_disk_space()))
    results.append(("API", check_api_health()))
    results.append(("Models", check_models()))

    # Camera check (optional)
    if args.full or args.camera:
        camera_uri = args.camera or os.environ.get("CAMERA_URI", "/dev/video0")
        results.append(("Camera", check_camera(camera_uri)))

    print()
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("System is ready for deployment.")
        sys.exit(0)
    else:
        print("Some checks failed. Review warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
