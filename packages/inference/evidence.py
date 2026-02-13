"""
IntelFactor.ai — Evidence Frame Writer
JPEG ring buffer for defect evidence capture.

Design:
- Writes JPEG frames for FAIL and REVIEW events only.
- FIFO deletion when disk quota exceeded.
- Metadata JSON sidecar per frame.
- Frame paths written back into DetectionResult.frame_ref.
- Fully local. No network. No cloud.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EvidenceWriter:
    """
    Writes evidence frames (JPEG) and metadata (JSON) to disk.

    Directory layout:
        {evidence_dir}/
            2026-02-13/
                evt_20260213_143022_a1b2c3.jpg
                evt_20260213_143022_a1b2c3.json
            2026-02-14/
                ...

    Disk management:
        - Tracks total evidence size.
        - When max_disk_gb exceeded, deletes oldest date directories first (FIFO).
        - Checks disk quota every N writes (not every frame).
    """

    def __init__(
        self,
        evidence_dir: str = "/opt/intelfactor/data/evidence",
        max_disk_gb: float = 50.0,
        jpeg_quality: int = 85,
        check_interval: int = 100,  # check disk every N writes
    ):
        self.evidence_dir = Path(evidence_dir)
        self.max_disk_bytes = int(max_disk_gb * 1024 * 1024 * 1024)
        self.jpeg_quality = jpeg_quality
        self.check_interval = check_interval

        self._write_count = 0
        self._total_bytes = 0

        # Create base directory
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # Calculate initial disk usage
        self._total_bytes = self._calculate_usage()
        logger.info(
            "EvidenceWriter ready: dir=%s max=%.1fGB current=%.1fMB",
            self.evidence_dir,
            max_disk_gb,
            self._total_bytes / (1024 * 1024),
        )

    def write(
        self,
        frame: np.ndarray,
        event_id: str,
        metadata: dict[str, Any] | None = None,
        bbox_list: list[dict] | None = None,
    ) -> str:
        """
        Write a frame and metadata to disk.

        Args:
            frame: BGR numpy array from OpenCV.
            event_id: Unique event identifier (used as filename).
            metadata: Additional metadata dict to store as JSON sidecar.
            bbox_list: List of bounding box dicts for overlay drawing.

        Returns:
            Relative path to the saved JPEG (used as frame_ref).
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV not available — evidence not saved for %s", event_id)
            return ""

        # Date-partitioned directory
        date_str = datetime.now().strftime("%Y-%m-%d")
        day_dir = self.evidence_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        jpg_path = day_dir / f"{event_id}.jpg"
        json_path = day_dir / f"{event_id}.json"

        # Draw bounding boxes on a copy if provided
        if bbox_list:
            frame_annotated = frame.copy()
            for bbox in bbox_list:
                x, y, w, h = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("width", 0)), int(bbox.get("height", 0))
                label = bbox.get("label", "defect")
                conf = bbox.get("confidence", 0.0)
                color = (0, 0, 255) if bbox.get("severity", 0) > 0.5 else (0, 165, 255)
                cv2.rectangle(frame_annotated, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame_annotated,
                    f"{label} {conf:.0%}",
                    (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )
            save_frame = frame_annotated
        else:
            save_frame = frame

        # Write JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success = cv2.imwrite(str(jpg_path), save_frame, encode_params)
        if not success:
            logger.error("Failed to write evidence JPEG: %s", jpg_path)
            return ""

        jpg_size = jpg_path.stat().st_size
        self._total_bytes += jpg_size

        # Write metadata sidecar
        meta = metadata or {}
        meta["event_id"] = event_id
        meta["frame_path"] = str(jpg_path.relative_to(self.evidence_dir))
        meta["timestamp"] = datetime.now().isoformat()
        meta["jpeg_size_bytes"] = jpg_size

        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self._total_bytes += json_path.stat().st_size

        # Periodic disk quota check
        self._write_count += 1
        if self._write_count % self.check_interval == 0:
            self._enforce_quota()

        # Return relative path for frame_ref field
        return str(jpg_path.relative_to(self.evidence_dir))

    def get_frame_path(self, event_id: str) -> Path | None:
        """Find the JPEG path for a given event_id by searching date directories."""
        # Try today first (most common case)
        today = datetime.now().strftime("%Y-%m-%d")
        today_path = self.evidence_dir / today / f"{event_id}.jpg"
        if today_path.exists():
            return today_path

        # Search all date directories (newest first)
        for day_dir in sorted(self.evidence_dir.iterdir(), reverse=True):
            if day_dir.is_dir() and len(day_dir.name) == 10:  # YYYY-MM-DD
                path = day_dir / f"{event_id}.jpg"
                if path.exists():
                    return path
        return None

    def get_metadata(self, event_id: str) -> dict[str, Any] | None:
        """Load metadata JSON for a given event_id."""
        for day_dir in sorted(self.evidence_dir.iterdir(), reverse=True):
            if day_dir.is_dir() and len(day_dir.name) == 10:
                path = day_dir / f"{event_id}.json"
                if path.exists():
                    with open(path) as f:
                        return json.load(f)
        return None

    def _enforce_quota(self) -> None:
        """Delete oldest date directories when over quota (FIFO)."""
        if self._total_bytes <= self.max_disk_bytes:
            return

        # List date directories oldest-first
        date_dirs = sorted(
            [d for d in self.evidence_dir.iterdir() if d.is_dir() and len(d.name) == 10]
        )

        deleted = 0
        while self._total_bytes > self.max_disk_bytes and date_dirs:
            oldest = date_dirs.pop(0)
            dir_size = sum(f.stat().st_size for f in oldest.rglob("*") if f.is_file())
            shutil.rmtree(oldest, ignore_errors=True)
            self._total_bytes -= dir_size
            deleted += 1
            logger.info("Evidence quota: deleted %s (%.1fMB freed)", oldest.name, dir_size / (1024 * 1024))

        if deleted:
            logger.info("Evidence quota enforced: deleted %d dirs, current=%.1fMB", deleted, self._total_bytes / (1024 * 1024))

    def _calculate_usage(self) -> int:
        """Calculate total disk usage of evidence directory."""
        total = 0
        if self.evidence_dir.exists():
            for f in self.evidence_dir.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total

    def get_stats(self) -> dict[str, Any]:
        """Get evidence writer statistics."""
        date_dirs = [d for d in self.evidence_dir.iterdir() if d.is_dir()] if self.evidence_dir.exists() else []
        return {
            "evidence_dir": str(self.evidence_dir),
            "total_writes": self._write_count,
            "total_bytes": self._total_bytes,
            "total_mb": round(self._total_bytes / (1024 * 1024), 1),
            "max_gb": round(self.max_disk_bytes / (1024 * 1024 * 1024), 1),
            "usage_pct": round(self._total_bytes / self.max_disk_bytes * 100, 1) if self.max_disk_bytes > 0 else 0,
            "date_dirs": len(date_dirs),
        }
