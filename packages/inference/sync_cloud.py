"""
IntelFactor.ai — Cloud Sync Agent
Syncs local SQLite data to cloud API and evidence to S3.

Designed for hybrid deployment where edge keeps running if cloud dies.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class CloudSyncAgent:
    """
    Syncs station data to cloud API.

    - Reads from local SQLite database
    - Posts events and triples to cloud API
    - Optionally uploads evidence to S3
    - Tracks sync watermarks to avoid reprocessing
    """

    def __init__(
        self,
        station_id: str,
        sqlite_db_path: str,
        evidence_dir: str,
        cloud_api_url: str,
        cloud_api_key: str,
        sync_interval_sec: int = 300,
        batch_size: int = 100,
        s3_bucket: str | None = None,
    ):
        self.station_id = station_id
        self.db_path = Path(sqlite_db_path)
        self.evidence_dir = Path(evidence_dir)
        self.api_url = cloud_api_url.rstrip("/")
        self.api_key = cloud_api_key
        self.sync_interval = sync_interval_sec
        self.batch_size = batch_size
        self.s3_bucket = s3_bucket

        self._watermarks: dict[str, str] = {}
        self._s3_client = None

    def start(self) -> None:
        """Start the sync loop."""
        self._load_watermarks()
        self._init_s3()

        logger.info(
            "Cloud sync agent started: station=%s interval=%ds api=%s",
            self.station_id, self.sync_interval, self.api_url,
        )

        while True:
            try:
                self._sync_cycle()
            except Exception as e:
                logger.error("Sync cycle failed: %s", e)

            time.sleep(self.sync_interval)

    def _load_watermarks(self) -> None:
        """Load watermarks from a local file."""
        watermark_file = self.db_path.parent / f"{self.station_id}_watermarks.json"
        if watermark_file.exists():
            with open(watermark_file) as f:
                self._watermarks = json.load(f)
            logger.info("Loaded watermarks: %s", self._watermarks)

    def _save_watermarks(self) -> None:
        """Save watermarks to a local file."""
        watermark_file = self.db_path.parent / f"{self.station_id}_watermarks.json"
        with open(watermark_file, "w") as f:
            json.dump(self._watermarks, f)

    def _init_s3(self) -> None:
        """Initialize S3 client if configured."""
        if not self.s3_bucket:
            return

        try:
            import boto3
            self._s3_client = boto3.client("s3")
            logger.info("S3 client initialized: bucket=%s", self.s3_bucket)
        except ImportError:
            logger.warning("boto3 not available — S3 upload disabled")

    def _sync_cycle(self) -> None:
        """Run one sync cycle."""
        if not self.db_path.exists():
            logger.warning("Database not found: %s", self.db_path)
            return

        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row

        try:
            self._sync_events(conn)
            self._sync_triples(conn)
            self._sync_evidence()
            self._send_heartbeat()
        finally:
            conn.close()

    def _sync_events(self, conn: sqlite3.Connection) -> None:
        """Sync new events to cloud."""
        watermark = self._watermarks.get("events", "")

        if watermark:
            rows = conn.execute(
                """SELECT * FROM defect_events
                   WHERE timestamp > ?
                   ORDER BY timestamp
                   LIMIT ?""",
                (watermark, self.batch_size),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM defect_events
                   ORDER BY timestamp
                   LIMIT ?""",
                (self.batch_size,),
            ).fetchall()

        if not rows:
            return

        events = [dict(row) for row in rows]
        synced = self._post_to_api("/api/v1/events/batch", {"events": events})

        if synced:
            self._watermarks["events"] = events[-1]["timestamp"]
            self._save_watermarks()
            logger.info("Synced %d events to cloud", len(events))
        else:
            logger.warning("Event sync deferred; %d local events remain pending", len(events))

    def _sync_triples(self, conn: sqlite3.Connection) -> None:
        """Sync new triples to cloud."""
        watermark = self._watermarks.get("triples", "")

        if watermark:
            rows = conn.execute(
                """SELECT * FROM causal_triples
                   WHERE timestamp > ?
                   ORDER BY timestamp
                   LIMIT ?""",
                (watermark, self.batch_size),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM causal_triples
                   ORDER BY timestamp
                   LIMIT ?""",
                (self.batch_size,),
            ).fetchall()

        if not rows:
            return

        triples = [dict(row) for row in rows]
        synced = self._post_to_api("/api/v1/triples/batch", {"triples": triples})

        if synced:
            self._watermarks["triples"] = triples[-1]["timestamp"]
            self._save_watermarks()
            logger.info("Synced %d triples to cloud", len(triples))
        else:
            logger.warning("Triple sync deferred; %d local triples remain pending", len(triples))

    def _sync_evidence(self) -> None:
        """Upload evidence files to S3."""
        if not self._s3_client or not self.evidence_dir.exists():
            return

        # Find evidence files not yet uploaded. Use a file-level watermark so
        # new same-day evidence can keep syncing without skipping partial dates.
        file_watermark = self._watermarks.get("evidence_file", "")
        date_watermark = self._watermarks.get("evidence_date", "")

        for date_dir in sorted(self.evidence_dir.iterdir()):
            if not date_dir.is_dir() or len(date_dir.name) != 10:
                continue

            if date_watermark and not file_watermark and date_dir.name < date_watermark:
                continue

            uploaded = 0
            last_uploaded_rel = ""
            failed_file: Path | None = None

            for jpg_file in sorted(date_dir.glob("*.jpg")):
                rel_path = f"{date_dir.name}/{jpg_file.name}"
                if file_watermark and rel_path <= file_watermark:
                    continue

                s3_key = f"{self.station_id}/{date_dir.name}/{jpg_file.name}"
                try:
                    self._s3_client.upload_file(
                        str(jpg_file),
                        self.s3_bucket,
                        s3_key,
                        ExtraArgs={"ContentType": "image/jpeg"},
                    )
                    uploaded += 1
                    last_uploaded_rel = rel_path
                except Exception as e:
                    failed_file = jpg_file
                    logger.error("S3 upload failed; local evidence kept for retry: %s - %s", jpg_file, e)
                    break

            if uploaded > 0:
                logger.info("Uploaded %d evidence files for %s", uploaded, date_dir.name)
                self._watermarks["evidence_file"] = last_uploaded_rel
                self._watermarks["evidence_date"] = date_dir.name
                self._save_watermarks()

            if failed_file is not None:
                logger.warning(
                    "Evidence sync paused at %s; watermark remains at %s",
                    failed_file,
                    self._watermarks.get("evidence_file", ""),
                )
                break

    def _send_heartbeat(self) -> None:
        """Send station heartbeat to cloud."""
        self._post_to_api("/api/v1/heartbeats", {
            "station_id": self.station_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "online",
        })

    def _post_to_api(self, endpoint: str, data: dict[str, Any]) -> bool:
        """Post data to cloud API with retry."""
        if not self.api_url or not self.api_key:
            logger.warning("Cloud API not configured; skipping %s and keeping local data pending", endpoint)
            return False

        url = f"{self.api_url}{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Station-ID": self.station_id,
        }

        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = requests.post(url, json=data, headers=headers, timeout=30)
                response.raise_for_status()
                return True
            except requests.RequestException as e:
                last_error = e
                logger.warning("API request failed: %s attempt=%d/3 error=%s", endpoint, attempt, e)
                if attempt < 3:
                    time.sleep(min(2 ** (attempt - 1), 10))

        logger.error(
            "API request exhausted retries: %s error=%s; local data remains pending",
            endpoint,
            last_error,
        )
        return False


def main():
    """Entry point for cloud sync agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [sync-cloud] %(levelname)s: %(message)s",
    )

    agent = CloudSyncAgent(
        station_id=os.environ.get("STATION_ID", "station_01"),
        sqlite_db_path=os.environ.get("SQLITE_DB_PATH", "/data/local.db"),
        evidence_dir=os.environ.get("EVIDENCE_DIR", "/data/evidence"),
        cloud_api_url=os.environ.get("CLOUD_API_URL", "https://api.intelfactor.ai"),
        cloud_api_key=os.environ.get("CLOUD_API_KEY", ""),
        sync_interval_sec=int(os.environ.get("SYNC_INTERVAL_SEC", "300")),
        batch_size=int(os.environ.get("SYNC_BATCH_SIZE", "100")),
        s3_bucket=os.environ.get("S3_BUCKET") or None,
    )
    agent.start()


if __name__ == "__main__":
    main()
