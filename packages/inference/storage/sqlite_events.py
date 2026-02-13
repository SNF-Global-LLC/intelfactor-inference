"""
IntelFactor.ai — SQLite Event Store
Local storage for defect events.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from .base import EventStore
from .sqlite_base import get_connection

logger = logging.getLogger(__name__)


class SQLiteEventStore(EventStore):
    """SQLite-backed event storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = get_connection(db_path)

    def insert(self, event: dict[str, Any]) -> str:
        """Insert a defect event."""
        event_id = event.get("event_id", "")
        if not event_id:
            raise ValueError("event_id is required")

        # Serialize detections list to JSON
        detections = event.get("detections", [])
        detections_json = json.dumps(detections, ensure_ascii=False) if detections else None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO defect_events
                (event_id, timestamp, station_id, defect_type, severity,
                 confidence, verdict, shift, sku, sop_criterion,
                 model_version, frame_ref, detections_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event.get("timestamp", datetime.now().isoformat()),
                event.get("station_id", ""),
                event.get("defect_type", ""),
                event.get("severity", 0),
                event.get("confidence", 0),
                event.get("verdict", "PASS"),
                event.get("shift", ""),
                event.get("sku", ""),
                event.get("sop_criterion", ""),
                event.get("model_version", ""),
                event.get("frame_ref", ""),
                detections_json,
            ),
        )
        self._conn.commit()
        return event_id

    def get(self, event_id: str) -> dict[str, Any] | None:
        """Get a single event by ID."""
        row = self._conn.execute(
            "SELECT * FROM defect_events WHERE event_id = ?", (event_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list(
        self,
        limit: int = 50,
        verdict: str | None = None,
        station_id: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """List events with optional filters."""
        query = "SELECT * FROM defect_events WHERE 1=1"
        params: list[Any] = []

        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def count(
        self,
        verdict: str | None = None,
        station_id: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Count events matching filters."""
        query = "SELECT COUNT(*) FROM defect_events WHERE 1=1"
        params: list[Any] = []

        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())

        result = self._conn.execute(query, params).fetchone()
        return result[0] if result else 0

    def _row_to_dict(self, row) -> dict[str, Any]:
        """Convert a sqlite Row to dict with parsed JSON fields."""
        d = dict(row)

        # Parse detections JSON
        if d.get("detections_json"):
            try:
                d["detections"] = json.loads(d["detections_json"])
            except json.JSONDecodeError:
                d["detections"] = []
        else:
            d["detections"] = []
        del d["detections_json"]

        return d
