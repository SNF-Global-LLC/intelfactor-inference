"""
IntelFactor.ai — Inspection Sync Worker
Background worker that uploads inspection events from edge to cloud.

Flow (every 30–60 seconds):
    1. Query SQLite for inspections with sync_status='pending' or 'failed'
    2. For each inspection:
       a. Request presigned S3 URLs from backend
       b. Upload original + annotated images to S3
       c. POST inspection metadata to backend
       d. Update sync_status to 'synced' (or 'failed' with error)

Runs as a daemon thread inside StationRuntime.
Also usable standalone for testing / manual replay.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from packages.inference.schemas import InspectionEvent, SyncStatus
from packages.inference.storage.inspection_store import InspectionStore

logger = logging.getLogger(__name__)


class InspectionSyncWorker:
    """
    Background worker that syncs inspection events to cloud.

    Config via environment:
        CLOUD_API_URL: Base URL for IntelFactor backend (required for sync)
        CLOUD_API_KEY: API key for authentication
        SYNC_INTERVAL_SEC: Seconds between sync cycles (default: 30)
        SYNC_BATCH_SIZE: Max inspections per cycle (default: 20)
    """

    def __init__(
        self,
        inspection_store: InspectionStore,
        evidence_dir: str | Path,
        cloud_api_url: str = "",
        cloud_api_key: str = "",
        sync_interval_sec: int = 30,
        batch_size: int = 20,
    ):
        self.store = inspection_store
        self.evidence_dir = Path(evidence_dir)
        self.api_url = cloud_api_url.rstrip("/") if cloud_api_url else ""
        self.api_key = cloud_api_key
        self.sync_interval = sync_interval_sec
        self.batch_size = batch_size

        self._running = False
        self._thread: threading.Thread | None = None
        self._stats = {
            "cycles": 0,
            "synced": 0,
            "failed": 0,
            "last_cycle": None,
        }

    def start(self) -> None:
        """Start the sync worker as a daemon thread."""
        if not self.api_url:
            logger.info("Inspection sync disabled: CLOUD_API_URL not configured")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._sync_loop,
            daemon=True,
            name="inspection-sync",
        )
        self._thread.start()
        logger.info(
            "Inspection sync worker started: interval=%ds batch=%d api=%s",
            self.sync_interval, self.batch_size, self.api_url,
        )

    def stop(self) -> None:
        """Stop the sync worker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_stats(self) -> dict[str, Any]:
        """Return sync worker statistics."""
        return {**self._stats, "running": self._running, "api_url": self.api_url}

    def _sync_loop(self) -> None:
        """Main sync loop — runs until stopped."""
        while self._running:
            try:
                self._sync_cycle()
            except Exception as exc:
                logger.error("Sync cycle failed: %s", exc)

            time.sleep(self.sync_interval)

    def _sync_cycle(self) -> None:
        """One pass: find pending, upload, finalize."""
        pending = self.store.get_pending_sync(limit=self.batch_size)
        if not pending:
            return

        self._stats["cycles"] += 1
        self._stats["last_cycle"] = time.time()
        logger.info("Sync cycle: %d pending inspections", len(pending))

        for event in pending:
            try:
                self._sync_one(event)
                self._stats["synced"] += 1
            except Exception as exc:
                logger.error("Sync failed for %s: %s", event.inspection_id, exc)
                self.store.update_sync_status(
                    event.inspection_id,
                    SyncStatus.FAILED,
                    error=str(exc),
                )
                self._stats["failed"] += 1

    def _sync_one(self, event: InspectionEvent) -> None:
        """Upload one inspection event to cloud."""
        import httpx

        self.store.update_sync_status(event.inspection_id, SyncStatus.UPLOADING)

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        # Step A: Request presigned upload URLs
        upload_urls = self._get_upload_urls(event, headers)

        # Step B: Upload images to S3
        original_url = ""
        annotated_url = ""

        if upload_urls.get("original_url") and event.image_original_path:
            original_path = self.evidence_dir / event.image_original_path
            if original_path.exists():
                original_url = self._upload_file(
                    original_path,
                    upload_urls["original_url"],
                )

        if upload_urls.get("annotated_url") and event.image_annotated_path:
            annotated_path = self.evidence_dir / event.image_annotated_path
            if annotated_path.exists():
                annotated_url = self._upload_file(
                    annotated_path,
                    upload_urls["annotated_url"],
                )

        # Step C: Finalize — POST inspection metadata to backend
        payload = self._build_payload(event, original_url, annotated_url)

        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self.api_url}/api/v1/edge/inspections",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()

        # Step D: Mark synced
        self.store.update_sync_status(
            event.inspection_id,
            SyncStatus.SYNCED,
            image_original_url=original_url,
            image_annotated_url=annotated_url,
        )

        logger.info("Synced: %s", event.inspection_id)

    def _get_upload_urls(
        self,
        event: InspectionEvent,
        headers: dict[str, str],
    ) -> dict[str, str]:
        """Request presigned S3 upload URLs from backend."""
        import httpx

        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{self.api_url}/api/v1/edge/inspections/upload-urls",
                json={
                    "inspection_id": event.inspection_id,
                    "station_id": event.station_id,
                    "workspace_id": event.workspace_id,
                    "has_original": bool(event.image_original_path),
                    "has_annotated": bool(event.image_annotated_path),
                },
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    def _upload_file(self, local_path: Path, presigned_url: str) -> str:
        """Upload a file to S3 via presigned PUT URL. Returns the public URL."""
        import httpx

        content_type = "image/jpeg" if local_path.suffix == ".jpg" else "application/json"

        with open(local_path, "rb") as f:
            data = f.read()

        with httpx.Client(timeout=60) as client:
            resp = client.put(
                presigned_url,
                content=data,
                headers={"Content-Type": content_type},
            )
            resp.raise_for_status()

        # Return the URL without query params (the actual S3 object URL)
        return presigned_url.split("?")[0]

    def _build_payload(
        self,
        event: InspectionEvent,
        original_url: str,
        annotated_url: str,
    ) -> dict[str, Any]:
        """Build the JSON payload for the backend ingestion endpoint."""
        detections = []
        for d in event.detections:
            detections.append({
                "class": d.defect_type,
                "confidence": round(d.confidence, 4),
                "severity": round(d.severity, 4),
                "threshold_used": round(d.threshold_used, 4),
                "bbox": {
                    "x": round(d.bbox.x, 1),
                    "y": round(d.bbox.y, 1),
                    "width": round(d.bbox.width, 1),
                    "height": round(d.bbox.height, 1),
                },
            })

        return {
            "inspection_id": event.inspection_id,
            "timestamp": event.timestamp.isoformat() if event.timestamp else "",
            "station_id": event.station_id,
            "workspace_id": event.workspace_id,
            "product_id": event.product_id,
            "operator_id": event.operator_id,
            "decision": event.decision.value if hasattr(event.decision, "value") else event.decision,
            "confidence": round(event.confidence, 4),
            "detections": detections,
            "num_detections": event.num_detections,
            "image_original_url": original_url,
            "image_annotated_url": annotated_url,
            "model_version": event.model_version,
            "model_name": event.model_name,
            "timing": {
                "capture_ms": round(event.capture_ms, 1),
                "inference_ms": round(event.inference_ms, 1),
                "total_ms": round(event.total_ms, 1),
            },
            "accepted": event.accepted,
            "rejection_reason": event.rejection_reason,
            "notes": event.notes,
        }
