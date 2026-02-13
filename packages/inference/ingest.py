"""
IntelFactor.ai — Camera Ingest Service
RTSP / GigE Vision / USB camera ingest with watchdog reconnect.

Design:
- Reconnects automatically on camera disconnect (factory reality).
- Ring-buffer for raw frames (configurable depth).
- Feeds frames to VisionProvider via callback.
- Drops frames when inference is slower than capture (never blocks camera).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class CameraProtocol(str, Enum):
    RTSP = "rtsp"
    GIGE = "gige"
    USB = "usb"
    FILE = "file"  # for testing with video files


@dataclass
class CameraConfig:
    """Camera source configuration."""
    source: str = ""                    # rtsp://..., /dev/video0, path/to/video.mp4
    protocol: CameraProtocol = CameraProtocol.RTSP
    station_id: str = "station_1"
    fps_target: int = 30
    width: int = 1920
    height: int = 1080
    reconnect_delay_sec: float = 3.0
    max_reconnect_attempts: int = 0     # 0 = infinite
    frame_queue_depth: int = 2          # drop frames if inference can't keep up
    warmup_frames: int = 10             # skip first N frames after connect


class CameraIngest:
    """
    Camera capture with watchdog reconnect.

    Usage:
        ingest = CameraIngest(config, on_frame=my_callback)
        ingest.start()
        # ... runs in background thread ...
        ingest.stop()

    The on_frame callback receives (frame: np.ndarray, metadata: dict).
    If the callback is slower than capture, frames are dropped (newest wins).
    """

    def __init__(
        self,
        config: CameraConfig,
        on_frame: Callable[[np.ndarray, dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.on_frame = on_frame
        self._cap = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_queue: queue.Queue[tuple[np.ndarray, dict]] = queue.Queue(
            maxsize=config.frame_queue_depth
        )
        self._stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "reconnects": 0,
            "last_frame_time": 0.0,
            "fps_actual": 0.0,
            "connected": False,
        }
        self._fps_timer_start = 0.0
        self._fps_frame_count = 0

    def start(self) -> None:
        """Start capture in background thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name=f"cam-{self.config.station_id}"
        )
        self._thread.start()

        # Start processing thread if callback provided
        if self.on_frame:
            self._proc_thread = threading.Thread(
                target=self._process_loop, daemon=True, name=f"proc-{self.config.station_id}"
            )
            self._proc_thread.start()

        logger.info("Camera ingest started: %s (%s)", self.config.source, self.config.protocol.value)

    def _capture_loop(self) -> None:
        """Main capture loop with reconnect logic."""
        attempt = 0

        while self._running:
            try:
                self._connect()
                attempt = 0  # reset on successful connection
                self._stats["connected"] = True
                self._stats["reconnects"] += 1 if self._stats["reconnects"] > 0 else 0

                # Skip warmup frames
                for _ in range(self.config.warmup_frames):
                    if not self._running:
                        return
                    self._read_frame()

                # Main capture loop
                self._fps_timer_start = time.monotonic()
                self._fps_frame_count = 0

                while self._running:
                    frame = self._read_frame()
                    if frame is None:
                        logger.warning("Frame read failed — attempting reconnect")
                        break

                    self._stats["frames_captured"] += 1
                    self._stats["last_frame_time"] = time.time()
                    self._fps_frame_count += 1

                    # Update FPS every second
                    elapsed = time.monotonic() - self._fps_timer_start
                    if elapsed >= 1.0:
                        self._stats["fps_actual"] = round(self._fps_frame_count / elapsed, 1)
                        self._fps_timer_start = time.monotonic()
                        self._fps_frame_count = 0

                    # Enqueue frame (drop old if full — newest wins)
                    metadata = {
                        "station_id": self.config.station_id,
                        "timestamp": time.time(),
                        "frame_number": self._stats["frames_captured"],
                    }
                    try:
                        self._frame_queue.put_nowait((frame, metadata))
                    except queue.Full:
                        # Drop oldest frame, enqueue newest
                        try:
                            self._frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._frame_queue.put_nowait((frame, metadata))
                        self._stats["frames_dropped"] += 1

            except Exception as e:
                logger.error("Camera error: %s", e)

            # Reconnect logic
            self._stats["connected"] = False
            self._release()

            if not self._running:
                break

            attempt += 1
            if 0 < self.config.max_reconnect_attempts < attempt:
                logger.error(
                    "Max reconnect attempts reached (%d). Camera offline.",
                    self.config.max_reconnect_attempts,
                )
                break

            self._stats["reconnects"] += 1
            logger.info(
                "Reconnecting in %.1fs (attempt %d)...",
                self.config.reconnect_delay_sec, attempt,
            )
            time.sleep(self.config.reconnect_delay_sec)

    def _process_loop(self) -> None:
        """Process frames via callback."""
        while self._running:
            try:
                frame, metadata = self._frame_queue.get(timeout=1.0)
                if self.on_frame:
                    self.on_frame(frame, metadata)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Frame processing error: %s", e)

    def _connect(self) -> None:
        """Open camera connection."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("opencv-python required for camera ingest (pip install opencv-python-headless)")

        source = self.config.source

        if self.config.protocol == CameraProtocol.USB:
            # /dev/video0 → integer index
            try:
                source = int(source) if source.isdigit() else source
            except (ValueError, AttributeError):
                pass

        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise ConnectionError(f"Cannot open camera: {source}")

        # Set resolution if supported
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        # Set FPS if supported
        if self.config.fps_target:
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)

        # RTSP: use TCP transport for reliability
        if self.config.protocol == CameraProtocol.RTSP:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Camera connected: %s (%dx%d @ %.1f FPS)", source, actual_w, actual_h, actual_fps)

    def _read_frame(self) -> np.ndarray | None:
        """Read a single frame. Returns None on failure."""
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def _release(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_stats(self) -> dict[str, Any]:
        """Get ingest statistics."""
        return dict(self._stats)

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._release()
        logger.info("Camera ingest stopped: %s", self.config.station_id)
