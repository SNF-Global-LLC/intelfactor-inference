"""
IntelFactor.ai — Single-Frame Capture Backends
Trigger-based capture for manual QC inspection.

Two backends:
- FLIRCapture: PySpin/Spinnaker SDK for FLIR Blackfly S cameras
- FileCapture: Load test images from disk (dev/testing without hardware)

Lifecycle:
- Initialize once at app startup
- capture_frame() per inspection trigger (SingleFrame acquisition)
- release() once at app shutdown
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CaptureError(Exception):
    """Raised when frame capture fails."""
    pass


class FLIRCapture:
    """
    PySpin single-frame capture for FLIR Blackfly S cameras.
    Camera object persists for app lifetime — NOT torn down per capture.
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._system = None
        self._cam = None
        self._serial = config.get("serial_number", "")

        try:
            import PySpin
        except ImportError:
            raise CaptureError(
                "PySpin not installed. Install Spinnaker SDK from FLIR:\n"
                "  https://www.flir.com/products/spinnaker-sdk/\n"
                "  Then: pip install spinnaker_python-*.whl"
            )

        self._PySpin = PySpin
        self._system = PySpin.System.GetInstance()
        cam_list = self._system.GetCameras()

        if cam_list.GetSize() == 0:
            cam_list.Clear()
            self._system.ReleaseInstance()
            self._system = None
            raise CaptureError("No FLIR cameras found. Check USB3 connection.")

        # Find camera by serial number or use first available
        if self._serial:
            self._cam = None
            for i in range(cam_list.GetSize()):
                cam = cam_list[i]
                cam.Init()
                node_serial = PySpin.CStringPtr(
                    cam.GetTLDeviceNodeMap().GetNode("DeviceSerialNumber")
                )
                if PySpin.IsReadable(node_serial) and node_serial.GetValue() == self._serial:
                    self._cam = cam
                    break
                cam.DeInit()

            if self._cam is None:
                cam_list.Clear()
                self._system.ReleaseInstance()
                self._system = None
                raise CaptureError(f"FLIR camera serial {self._serial} not found")
        else:
            self._cam = cam_list[0]
            self._cam.Init()

        cam_list.Clear()

        # Configure camera settings
        self._configure()
        logger.info(
            "FLIRCapture ready: serial=%s, %dx%d %s",
            self._serial or "auto",
            config.get("width", 1920),
            config.get("height", 1200),
            config.get("pixel_format", "Mono8"),
        )

    def _configure(self) -> None:
        """Apply camera settings from config."""
        PySpin = self._PySpin
        nodemap = self._cam.GetNodeMap()

        # Pixel format
        pixel_fmt = self._config.get("pixel_format", "Mono8")
        node_pixel = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if PySpin.IsWritable(node_pixel):
            entry = node_pixel.GetEntryByName(pixel_fmt)
            if PySpin.IsReadable(entry):
                node_pixel.SetIntValue(entry.GetValue())

        # Width / Height
        width = self._config.get("width")
        if width:
            node_w = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
            if PySpin.IsWritable(node_w):
                node_w.SetValue(min(width, node_w.GetMax()))

        height = self._config.get("height")
        if height:
            node_h = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
            if PySpin.IsWritable(node_h):
                node_h.SetValue(min(height, node_h.GetMax()))

        # Exposure
        if self._config.get("exposure_auto", True):
            node_ea = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
            if PySpin.IsWritable(node_ea):
                entry = node_ea.GetEntryByName("Continuous")
                if PySpin.IsReadable(entry):
                    node_ea.SetIntValue(entry.GetValue())
        else:
            node_ea = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
            if PySpin.IsWritable(node_ea):
                entry = node_ea.GetEntryByName("Off")
                if PySpin.IsReadable(entry):
                    node_ea.SetIntValue(entry.GetValue())

        # Gain
        gain = self._config.get("gain")
        if gain is not None:
            node_ga = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
            if PySpin.IsWritable(node_ga):
                entry = node_ga.GetEntryByName("Off")
                if PySpin.IsReadable(entry):
                    node_ga.SetIntValue(entry.GetValue())
            node_g = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
            if PySpin.IsWritable(node_g):
                node_g.SetValue(min(float(gain), node_g.GetMax()))

        # Acquisition mode: SingleFrame
        node_acq = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsWritable(node_acq):
            entry = node_acq.GetEntryByName("SingleFrame")
            if PySpin.IsReadable(entry):
                node_acq.SetIntValue(entry.GetValue())

    def capture_frame(self) -> np.ndarray:
        """
        Capture ONE high-res frame using SingleFrame acquisition mode.
        Returns raw numpy array (H, W) uint8 for Mono8.
        """
        if self._cam is None:
            raise CaptureError("Camera not initialized")

        self._cam.BeginAcquisition()
        try:
            image_result = self._cam.GetNextImage(timeout=5000)

            if image_result.IsIncomplete():
                status = image_result.GetImageStatus()
                image_result.Release()
                raise CaptureError(f"Incomplete image, status={status}")

            # Copy to numpy before releasing the image buffer
            frame = image_result.GetNDArray().copy()
            image_result.Release()
        finally:
            self._cam.EndAcquisition()

        return frame

    def release(self) -> None:
        """Full PySpin shutdown. Called ONLY on app exit."""
        if self._cam is not None:
            try:
                self._cam.DeInit()
            except Exception:
                pass
            self._cam = None

        if self._system is not None:
            try:
                self._system.ReleaseInstance()
            except Exception:
                pass
            self._system = None

        logger.info("FLIRCapture released")


class FileCapture:
    """
    Load test images from disk. For development/testing without camera hardware.
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._test_image_path = config.get("test_image")
        if self._test_image_path:
            p = Path(self._test_image_path)
            if not p.exists():
                raise CaptureError(f"Test image not found: {p}")
        logger.info("FileCapture ready: %s", self._test_image_path or "(synthetic)")

    def capture_frame(self) -> np.ndarray:
        """Load and return image from disk, or generate synthetic frame."""
        if self._test_image_path:
            try:
                import cv2
            except ImportError:
                raise CaptureError("OpenCV required for FileCapture (pip install opencv-python-headless)")
            frame = cv2.imread(self._test_image_path, cv2.IMREAD_UNCHANGED)
            if frame is None:
                raise CaptureError(f"Failed to load image: {self._test_image_path}")
            return frame

        # Generate synthetic grayscale frame for testing
        width = self._config.get("width", 1920)
        height = self._config.get("height", 1200)
        return np.random.randint(0, 255, (height, width), dtype=np.uint8)

    def release(self) -> None:
        """No-op."""
        pass


class WebcamCapture:
    """
    OpenCV webcam capture for development on Mac or any laptop.
    Uses a persistent VideoCapture object — opened once, reused per capture.
    Tomorrow: swap protocol to "pyspin" for FLIR Blackfly. Everything else stays.
    """

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._device_index = config.get("device_index", 0)
        self._warmup_frames = config.get("warmup_frames", 3)

        try:
            import cv2
        except ImportError:
            raise CaptureError("OpenCV required for WebcamCapture (pip install opencv-python-headless)")

        self._cap = cv2.VideoCapture(self._device_index)
        if not self._cap.isOpened():
            raise CaptureError(f"Cannot open webcam device index {self._device_index}")

        # Set resolution if specified
        width = config.get("width")
        height = config.get("height")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Warm up: discard first few frames so auto-exposure settles
        for _ in range(self._warmup_frames):
            self._cap.read()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("WebcamCapture ready: device=%d %dx%d", self._device_index, actual_w, actual_h)

    def capture_frame(self) -> np.ndarray:
        """Grab one frame from the webcam."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CaptureError(f"Webcam read failed (device {self._device_index})")
        return frame

    def release(self) -> None:
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info("WebcamCapture released")


def get_capture(config: dict[str, Any]) -> FLIRCapture | FileCapture | WebcamCapture:
    """Factory: return the right capture backend based on config['protocol']."""
    protocol = config.get("protocol", "file")

    if protocol == "pyspin":
        return FLIRCapture(config)
    elif protocol == "webcam":
        return WebcamCapture(config)
    elif protocol == "file":
        return FileCapture(config)
    else:
        raise CaptureError(f"Unknown capture protocol: {protocol}")
