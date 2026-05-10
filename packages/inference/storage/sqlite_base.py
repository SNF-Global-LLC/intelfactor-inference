"""
IntelFactor.ai — SQLite Base Utilities
Shared connection and migration logic for local storage.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Thread-local storage for connections
_local = threading.local()


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Get a thread-local SQLite connection.
    Creates database directory and initializes schema if needed.
    """
    if not hasattr(_local, "connections"):
        _local.connections = {}

    if db_path not in _local.connections:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")

        # Initialize schema
        _init_schema(conn)

        _local.connections[db_path] = conn
        logger.info("SQLite connection opened: %s", db_path)

    return _local.connections[db_path]


def _init_schema(conn: sqlite3.Connection) -> None:
    """Initialize database schema if not exists."""
    conn.executescript("""
        -- Defect events table
        CREATE TABLE IF NOT EXISTS defect_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            defect_type TEXT,
            severity REAL DEFAULT 0,
            confidence REAL DEFAULT 0,
            verdict TEXT NOT NULL,
            shift TEXT,
            sku TEXT,
            sop_criterion TEXT,
            model_version TEXT,
            frame_ref TEXT,
            detections_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON defect_events(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_events_station ON defect_events(station_id);
        CREATE INDEX IF NOT EXISTS idx_events_verdict ON defect_events(verdict);

        -- Evidence index table
        CREATE TABLE IF NOT EXISTS evidence_index (
            event_id TEXT PRIMARY KEY,
            date_dir TEXT NOT NULL,
            image_path TEXT,
            thumb_path TEXT,
            json_path TEXT,
            file_size_bytes INTEGER DEFAULT 0,
            metadata_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_evidence_date ON evidence_index(date_dir);

        -- Causal triples table
        CREATE TABLE IF NOT EXISTS causal_triples (
            triple_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            defect_event_id TEXT,
            defect_type TEXT,
            defect_severity REAL DEFAULT 0,
            cause_parameter TEXT,
            cause_value REAL,
            cause_target REAL,
            cause_drift_pct REAL,
            cause_confidence REAL,
            cause_explanation_zh TEXT,
            cause_explanation_en TEXT,
            recommendation_id TEXT,
            operator_action TEXT DEFAULT 'pending',
            operator_id TEXT,
            outcome_measured TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_triples_timestamp ON causal_triples(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_triples_station ON causal_triples(station_id);
        CREATE INDEX IF NOT EXISTS idx_triples_status ON causal_triples(status);

        -- Anomaly alerts table
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            alert_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            defect_type TEXT,
            current_rate REAL,
            baseline_rate REAL,
            z_score REAL,
            window_hours REAL,
            event_ids_json TEXT,
            acknowledged INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON anomaly_alerts(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_alerts_ack ON anomaly_alerts(acknowledged);

        -- Device heartbeats table (optional)
        CREATE TABLE IF NOT EXISTS device_heartbeats (
            heartbeat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL,
            device_class TEXT,
            gpu_name TEXT,
            vram_mb INTEGER,
            status TEXT,
            metrics_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_heartbeats_station ON device_heartbeats(station_id, timestamp DESC);
    """)
    conn.commit()
    logger.debug("SQLite schema initialized")
