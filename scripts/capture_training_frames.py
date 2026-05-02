#!/usr/bin/env python3
"""
Capture raw camera frames for model-training datasets.

This is intentionally separate from EvidenceWriter. EvidenceWriter stores
FAIL/REVIEW inspection evidence after inference; this script samples raw frames
before inference so operators can build a training dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture raw frames for model training.")
    parser.add_argument(
        "--source",
        default="/dev/video0",
        help="Camera/video source: /dev/video0, RTSP URL, video file, image file, or FLIR serial.",
    )
    parser.add_argument(
        "--protocol",
        choices=("usb", "rtsp", "file", "pyspin", "webcam"),
        default="usb",
        help="Source protocol. Use pyspin for FLIR Blackfly S.",
    )
    parser.add_argument("--output-dir", default="/opt/intelfactor/data/training_frames", help="Directory for captured frames.")
    parser.add_argument("--station-id", default="station_01", help="Station identifier stored in manifest entries.")
    parser.add_argument("--label", default="unlabeled", help="Dataset label/class folder, e.g. scratch_surface or pass.")
    parser.add_argument("--interval-sec", type=float, default=1.0, help="Seconds between saved frames.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to save. Use 0 for unlimited.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Frames to skip after opening source.")
    parser.add_argument("--width", type=int, default=1920, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Requested capture height.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 1-100.")
    return parser.parse_args()


def _open_video_capture(source: str, protocol: str, width: int, height: int) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install opencv-python-headless or use the Jetson Docker image.") from exc

    capture_source: str | int = source
    if protocol == "usb" and source.startswith("/dev/video"):
        suffix = source.removeprefix("/dev/video")
        if suffix.isdigit():
            capture_source = int(suffix)

    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {protocol} source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _read_frame(cap: Any, protocol: str) -> tuple[bool, Any]:
    if protocol in {"pyspin", "webcam", "file"} and hasattr(cap, "capture_frame"):
        return True, cap.capture_frame()
    return cap.read()


def _release_capture(cap: Any) -> None:
    if hasattr(cap, "release"):
        cap.release()


def _open_capture(source: str, protocol: str, width: int, height: int, warmup_frames: int) -> Any:
    if protocol in {"pyspin", "webcam"}:
        from packages.inference.capture import get_capture

        config: dict[str, Any] = {
            "protocol": protocol,
            "width": width,
            "height": height,
            "warmup_frames": warmup_frames,
        }
        if protocol == "pyspin":
            config["serial_number"] = "" if source in {"", "auto"} else source
        else:
            config["device_index"] = int(source) if source.isdigit() else 0
        return get_capture(config)

    # If protocol=file and source is a still image, use FileCapture so repeated
    # captures produce a small labeled dataset for script validation.
    if protocol == "file" and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        from packages.inference.capture import get_capture

        return get_capture({"protocol": "file", "test_image": source, "width": width, "height": height})

    return _open_video_capture(source, protocol, width, height)


def _write_manifest_entry(manifest_path: Path, entry: dict[str, Any]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> int:
    args = _parse_args()

    if args.interval_sec < 0:
        raise SystemExit("--interval-sec must be >= 0")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    try:
        import cv2
    except ImportError:
        print("ERROR: OpenCV is required. Install opencv-python-headless or use the Jetson Docker image.", file=sys.stderr)
        return 2

    cap = _open_capture(args.source, args.protocol, args.width, args.height, args.warmup_frames)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label_dir = Path(args.output_dir) / args.label / run_id
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = label_dir / "manifest.jsonl"

    print(f"Capturing training frames from {args.source} ({args.protocol})")
    print(f"Output: {label_dir}")
    print("Press Ctrl+C to stop.")

    if args.protocol in {"usb", "rtsp"} or (args.protocol == "file" and hasattr(cap, "read")):
        for _ in range(max(args.warmup_frames, 0)):
            cap.read()

    saved = 0
    read_count = 0
    next_save_at = time.monotonic()

    try:
        while args.max_frames == 0 or saved < args.max_frames:
            ok, frame = _read_frame(cap, args.protocol)
            read_count += 1
            if not ok or frame is None:
                if args.protocol == "file":
                    print("End of file source.")
                    break
                print("WARNING: frame read failed; retrying...")
                time.sleep(0.25)
                continue

            now = time.monotonic()
            if now < next_save_at:
                continue

            timestamp = datetime.now(timezone.utc).isoformat()
            filename = f"{args.station_id}_{run_id}_{saved + 1:06d}.jpg"
            frame_path = label_dir / filename
            write_ok = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not write_ok:
                print(f"ERROR: failed to write {frame_path}", file=sys.stderr)
                return 1

            entry = {
                "station_id": args.station_id,
                "source": args.source,
                "protocol": args.protocol,
                "label": args.label,
                "timestamp": timestamp,
                "frame_index": read_count,
                "path": str(frame_path),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "jpeg_quality": args.jpeg_quality,
            }
            _write_manifest_entry(manifest_path, entry)
            saved += 1
            print(f"saved {saved}: {frame_path}")
            next_save_at = now + args.interval_sec
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        _release_capture(cap)

    print(f"Done. Saved {saved} frame(s). Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
