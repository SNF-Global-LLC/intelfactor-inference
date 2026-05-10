"""
IntelFactor.ai — Inspection Store
SQLite persistence for inspection events with sync_status tracking.

Each inspection transaction gets one row. The sync worker reads rows
with sync_status='pending' and uploads them to the cloud backend.

Schema lives here, not in sqlite_base.py, because the inspection store
is separate from the continuous-pipeline storage.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.inference.schemas import (
    BoundingBox,
    Detection,
    InspectionEvent,
    SyncStatus,
    Verdict,
)

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    station_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL DEFAULT '',
    product_id TEXT DEFAULT '',
    operator_id TEXT DEFAULT '',
    decision TEXT NOT NULL,
    confidence REAL DEFAULT 0,
    detections_json TEXT DEFAULT '[]',
    num_detections INTEGER DEFAULT 0,
    image_original_path TEXT DEFAULT '',
    image_annotated_path TEXT DEFAULT '',
    report_path TEXT DEFAULT '',
    image_original_url TEXT DEFAULT '',
    image_annotated_url TEXT DEFAULT '',
    model_version TEXT DEFAULT '',
    model_name TEXT DEFAULT '',
    capture_ms REAL DEFAULT 0,
    inference_ms REAL DEFAULT 0,
    total_ms REAL DEFAULT 0,
    accepted INTEGER DEFAULT NULL,
    rejection_reason TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    sync_status TEXT DEFAULT 'pending',
    sync_error TEXT DEFAULT '',
    last_attempt_at TEXT DEFAULT NULL,
    synced_at TEXT DEFAULT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_insp_timestamp ON inspections(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_insp_station ON inspections(station_id);
CREATE INDEX IF NOT EXISTS idx_insp_workspace ON inspections(workspace_id);
CREATE INDEX IF NOT EXISTS idx_insp_sync ON inspections(sync_status);
CREATE INDEX IF NOT EXISTS idx_insp_decision ON inspections(decision);

CREATE TABLE IF NOT EXISTS operator_actions (
    action_id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    station_id TEXT DEFAULT '',
    workspace_id TEXT DEFAULT '',
    operator_id TEXT DEFAULT '',
    action TEXT NOT NULL,
    reason TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id)
);

CREATE INDEX IF NOT EXISTS idx_operator_actions_inspection ON operator_actions(inspection_id);
CREATE INDEX IF NOT EXISTS idx_operator_actions_timestamp ON operator_actions(timestamp DESC);
"""


class InspectionStore:
    """
    SQLite store for inspection events.
    Thread-safe. WAL mode. One DB file per station.
    Uses connection pooling for efficiency.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._local = threading.local()

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.executescript(_CREATE_SQL)
        self._ensure_schema(conn)
        conn.commit()
        conn.close()
        logger.info("InspectionStore ready: %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Apply small additive migrations for existing edge databases."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(inspections)")}
        if "last_attempt_at" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN last_attempt_at TEXT DEFAULT NULL")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection (pooled per thread)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._connect()
        return self._local.conn

    def save(self, event: InspectionEvent) -> None:
        """Insert or replace an inspection event."""
        detections_json = json.dumps([
            {
                "defect_type": d.defect_type,
                "confidence": d.confidence,
                "severity": d.severity,
                "threshold_used": d.threshold_used,
                "model_version": d.model_version,
                "bbox": {"x": d.bbox.x, "y": d.bbox.y, "width": d.bbox.width, "height": d.bbox.height},
            }
            for d in event.detections
        ], ensure_ascii=False)

        with self._lock:
            conn = self._get_conn()
            existing = conn.execute(
                """
                SELECT workspace_id, sync_status, sync_error, last_attempt_at, synced_at,
                       image_original_url, image_annotated_url
                FROM inspections
                WHERE inspection_id = ?
                """,
                (event.inspection_id,),
            ).fetchone()

            workspace_id = event.workspace_id
            sync_status = event.sync_status.value if isinstance(event.sync_status, SyncStatus) else event.sync_status
            sync_error = event.sync_error
            last_attempt_at = event.last_attempt_at.isoformat() if event.last_attempt_at else None
            synced_at = event.synced_at.isoformat() if event.synced_at else None
            image_original_url = event.image_original_url
            image_annotated_url = event.image_annotated_url

            if existing is not None:
                existing_workspace = existing["workspace_id"] or ""
                incoming_workspace = event.workspace_id or ""
                if existing_workspace and incoming_workspace and existing_workspace != incoming_workspace:
                    raise ValueError(
                        f"inspection_id {event.inspection_id} already belongs to workspace {existing_workspace}"
                    )
                if existing_workspace and not incoming_workspace:
                    workspace_id = existing_workspace

                default_pending = (
                    sync_status == SyncStatus.PENDING.value
                    and not sync_error
                    and last_attempt_at is None
                    and synced_at is None
                )
                if existing["sync_status"] == SyncStatus.SYNCED.value and default_pending:
                    sync_status = existing["sync_status"]
                    sync_error = existing["sync_error"] or ""
                    last_attempt_at = existing["last_attempt_at"]
                    synced_at = existing["synced_at"]
                    image_original_url = image_original_url or existing["image_original_url"] or ""
                    image_annotated_url = image_annotated_url or existing["image_annotated_url"] or ""

            conn.execute("""
                INSERT OR REPLACE INTO inspections (
                    inspection_id, timestamp, station_id, workspace_id,
                    product_id, operator_id, decision, confidence,
                    detections_json, num_detections,
                    image_original_path, image_annotated_path, report_path,
                    image_original_url, image_annotated_url,
                    model_version, model_name,
                    capture_ms, inference_ms, total_ms,
                    accepted, rejection_reason, notes,
                    sync_status, sync_error, last_attempt_at, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.inspection_id,
                event.timestamp.isoformat() if event.timestamp else datetime.now(tz=timezone.utc).isoformat(),
                event.station_id,
                workspace_id,
                event.product_id,
                event.operator_id,
                event.decision.value if isinstance(event.decision, Verdict) else event.decision,
                event.confidence,
                detections_json,
                event.num_detections,
                event.image_original_path,
                event.image_annotated_path,
                event.report_path,
                image_original_url,
                image_annotated_url,
                event.model_version,
                event.model_name,
                event.capture_ms,
                event.inference_ms,
                event.total_ms,
                1 if event.accepted is True else (0 if event.accepted is False else None),
                event.rejection_reason,
                event.notes,
                sync_status,
                sync_error,
                last_attempt_at,
                synced_at,
            ))
            conn.commit()

    def get(self, inspection_id: str) -> InspectionEvent | None:
        """Fetch one inspection by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?",
            (inspection_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def list_inspections(
        self,
        station_id: str | None = None,
        workspace_id: str | None = None,
        decision: str | None = None,
        sync_status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[InspectionEvent]:
        """List inspections with optional filters."""
        clauses = []
        params: list[Any] = []
        if station_id:
            clauses.append("station_id = ?")
            params.append(station_id)
        if workspace_id:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        if sync_status:
            clauses.append("sync_status = ?")
            params.append(sync_status)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM inspections WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_pending_sync(self, limit: int = 50) -> list[InspectionEvent]:
        """Get inspections waiting to be synced."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM inspections WHERE sync_status IN ('pending', 'failed') ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def update_sync_status(
        self,
        inspection_id: str,
        status: SyncStatus,
        error: str = "",
        image_original_url: str = "",
        image_annotated_url: str = "",
    ) -> None:
        """Update sync status after upload attempt."""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now(tz=timezone.utc)
            last_attempt_at = now.isoformat() if status in {SyncStatus.FAILED, SyncStatus.SYNCED} else ""
            synced_at = now.isoformat() if status == SyncStatus.SYNCED else None
            conn.execute("""
                UPDATE inspections
                SET sync_status = ?, sync_error = ?, last_attempt_at = CASE WHEN ? != '' THEN ? ELSE last_attempt_at END,
                    synced_at = ?,
                    image_original_url = CASE WHEN ? != '' THEN ? ELSE image_original_url END,
                    image_annotated_url = CASE WHEN ? != '' THEN ? ELSE image_annotated_url END
                WHERE inspection_id = ?
            """, (
                status.value, error, last_attempt_at, last_attempt_at, synced_at,
                image_original_url, image_original_url,
                image_annotated_url, image_annotated_url,
                inspection_id,
            ))
            conn.commit()

    def update_feedback(
        self,
        inspection_id: str,
        accepted: bool,
        operator_id: str = "",
        action: str | None = None,
        reason: str = "",
        notes: str = "",
    ) -> bool:
        """Record operator feedback on an inspection."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("""
                UPDATE inspections
                SET accepted = ?, rejection_reason = ?, notes = ?,
                    operator_id = CASE WHEN ? != '' THEN ? ELSE operator_id END,
                    sync_status = CASE WHEN sync_status = 'synced' THEN 'pending' ELSE sync_status END
                WHERE inspection_id = ?
            """, (
                1 if accepted else 0,
                reason, notes,
                operator_id, operator_id,
                inspection_id,
            ))
            if cur.rowcount <= 0:
                conn.commit()
                return False

            row = conn.execute(
                "SELECT station_id, workspace_id FROM inspections WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()
            action_name = action or ("confirm_defect" if accepted else "override_to_pass")
            now = datetime.now(tz=timezone.utc).isoformat()
            conn.execute("""
                INSERT INTO operator_actions (
                    action_id, inspection_id, timestamp, station_id, workspace_id,
                    operator_id, action, reason, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"act_{uuid.uuid4().hex}",
                inspection_id,
                now,
                row["station_id"] if row else "",
                row["workspace_id"] if row else "",
                operator_id,
                action_name,
                reason,
                notes,
            ))
            conn.commit()
            return True

    def list_operator_actions(self, inspection_id: str) -> list[dict[str, Any]]:
        """Return the local operator action trail for one inspection."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT action_id, inspection_id, timestamp, station_id, workspace_id,
                   operator_id, action, reason, notes
            FROM operator_actions
            WHERE inspection_id = ?
            ORDER BY timestamp ASC
            """,
            (inspection_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_stats(self, workspace_id: str | None = None) -> dict[str, Any]:
        """Get inspection store statistics."""
        conn = self._get_conn()
        where = "WHERE workspace_id = ?" if workspace_id else ""
        params = (workspace_id,) if workspace_id else ()
        total = conn.execute(f"SELECT COUNT(*) FROM inspections {where}", params).fetchone()[0]
        by_decision = {}
        for row in conn.execute(
            f"SELECT decision, COUNT(*) as cnt FROM inspections {where} GROUP BY decision",
            params,
        ):
            by_decision[row["decision"]] = row["cnt"]
        by_sync = {}
        for row in conn.execute(
            f"SELECT sync_status, COUNT(*) as cnt FROM inspections {where} GROUP BY sync_status",
            params,
        ):
            by_sync[row["sync_status"]] = row["cnt"]
        return {
            "total": total,
            "by_decision": by_decision,
            "by_sync_status": by_sync,
        }

    def _row_to_event(self, row: sqlite3.Row) -> InspectionEvent:
        """Convert a SQLite row to InspectionEvent."""
        detections = []
        raw_dets = json.loads(row["detections_json"] or "[]")
        for d in raw_dets:
            bbox_data = d.get("bbox", {})
            detections.append(Detection(
                defect_type=d["defect_type"],
                confidence=d["confidence"],
                severity=d.get("severity", 0.0),
                threshold_used=d.get("threshold_used", 0.0),
                model_version=d.get("model_version", ""),
                bbox=BoundingBox(
                    x=bbox_data.get("x", 0),
                    y=bbox_data.get("y", 0),
                    width=bbox_data.get("width", 0),
                    height=bbox_data.get("height", 0),
                ),
            ))

        accepted_raw = row["accepted"]
        accepted = None if accepted_raw is None else bool(accepted_raw)

        return InspectionEvent(
            inspection_id=row["inspection_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            station_id=row["station_id"],
            workspace_id=row["workspace_id"],
            product_id=row["product_id"] or "",
            operator_id=row["operator_id"] or "",
            decision=Verdict(row["decision"]) if row["decision"] in Verdict._value2member_map_ else Verdict.PASS,
            confidence=row["confidence"],
            detections=detections,
            num_detections=row["num_detections"],
            image_original_path=row["image_original_path"] or "",
            image_annotated_path=row["image_annotated_path"] or "",
            report_path=row["report_path"] or "",
            image_original_url=row["image_original_url"] or "",
            image_annotated_url=row["image_annotated_url"] or "",
            model_version=row["model_version"] or "",
            model_name=row["model_name"] or "",
            capture_ms=row["capture_ms"],
            inference_ms=row["inference_ms"],
            total_ms=row["total_ms"],
            accepted=accepted,
            rejection_reason=row["rejection_reason"] or "",
            notes=row["notes"] or "",
            sync_status=SyncStatus(row["sync_status"]) if row["sync_status"] in SyncStatus._value2member_map_ else SyncStatus.PENDING,
            sync_error=row["sync_error"] or "",
            last_attempt_at=datetime.fromisoformat(row["last_attempt_at"]) if "last_attempt_at" in row.keys() and row["last_attempt_at"] else None,
            synced_at=datetime.fromisoformat(row["synced_at"]) if row["synced_at"] else None,
        )
