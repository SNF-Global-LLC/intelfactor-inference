"""
IntelFactor.ai — Defect Pattern Accumulator (RCA Layer 1)
SQLite WAL-mode daemon that detects anomalous defect rate spikes.

Runs on ANY device. No GPU required. ~50MB RAM.
30-day rolling window. Z-score anomaly detection on 4-hour windows.
"""

from __future__ import annotations

import logging
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from packages.inference.schemas import AnomalyAlert, DetectionResult, Verdict

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS defect_events (
    event_id       TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    station_id     TEXT NOT NULL,
    defect_type    TEXT NOT NULL,
    severity       REAL NOT NULL,
    confidence     REAL NOT NULL,
    shift          TEXT DEFAULT '',
    sku            TEXT DEFAULT '',
    sop_criterion  TEXT DEFAULT '',
    model_version  TEXT DEFAULT '',
    frame_ref      TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_station_type_ts
    ON defect_events(station_id, defect_type, timestamp);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON defect_events(timestamp);

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    alert_id       TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    station_id     TEXT NOT NULL,
    defect_type    TEXT NOT NULL,
    current_rate   REAL NOT NULL,
    baseline_rate  REAL NOT NULL,
    z_score        REAL NOT NULL,
    window_hours   REAL NOT NULL,
    acknowledged   INTEGER DEFAULT 0
);
"""

RETENTION_DAYS = 30
WINDOW_HOURS = 4.0
Z_SCORE_THRESHOLD = 2.5
MIN_EVENTS_FOR_BASELINE = 20


class DefectAccumulator:
    """
    Accumulates defect events in SQLite and detects anomalous rate spikes.

    Design:
    - WAL mode for concurrent read/write (inference writes, RCA reads).
    - Rolling 30-day window with automatic pruning.
    - Z-score anomaly detection on 4-hour windows per station per defect type.
    - No external dependencies. Runs within 50MB RAM alongside TensorRT.
    """

    def __init__(self, db_path: str | Path = "/opt/intelfactor/data/accumulator.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def start(self) -> None:
        """Initialize database and enable WAL mode."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # faster writes, safe with WAL
        self._conn.execute("PRAGMA cache_size=-8000")     # 8MB cache
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()
        logger.info("Accumulator started: %s", self.db_path)

    def record_event(self, result: DetectionResult) -> None:
        """Record a detection event. Only stores FAIL and REVIEW verdicts."""
        if result.verdict == Verdict.PASS:
            return

        if self._conn is None:
            raise RuntimeError("Accumulator not started. Call start() first.")

        for detection in result.detections:
            self._conn.execute(
                """INSERT OR IGNORE INTO defect_events
                   (event_id, timestamp, station_id, defect_type, severity,
                    confidence, shift, sku, model_version, frame_ref)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.event_id,
                    result.timestamp.isoformat(),
                    result.station_id,
                    detection.defect_type,
                    detection.severity,
                    detection.confidence,
                    result.shift,
                    result.sku,
                    result.model_version,
                    result.frame_ref,
                ),
            )
        self._conn.commit()

    def check_anomalies(
        self,
        station_id: str | None = None,
        window_hours: float = WINDOW_HOURS,
        z_threshold: float = Z_SCORE_THRESHOLD,
    ) -> list[AnomalyAlert]:
        """
        Check for anomalous defect rate spikes.

        Compares the current window rate against 30-day baseline.
        Returns alerts for any station/defect_type combos exceeding z_threshold.
        """
        if self._conn is None:
            raise RuntimeError("Accumulator not started.")

        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        window_start = now - timedelta(hours=window_hours)
        baseline_start = now - timedelta(days=RETENTION_DAYS)

        # Get all active station/defect_type combos
        station_filter = "AND station_id = ?" if station_id else ""
        params: list[Any] = [baseline_start.isoformat()]
        if station_id:
            params.append(station_id)

        combos = self._conn.execute(
            f"""SELECT DISTINCT station_id, defect_type
                FROM defect_events
                WHERE timestamp > ? {station_filter}""",
            params,
        ).fetchall()

        alerts: list[AnomalyAlert] = []

        for sid, dtype in combos:
            alert = self._check_combo(sid, dtype, now, window_start, baseline_start, window_hours, z_threshold)
            if alert is not None:
                alerts.append(alert)
                self._store_alert(alert)

        return alerts

    def _check_combo(
        self,
        station_id: str,
        defect_type: str,
        now: datetime,
        window_start: datetime,
        baseline_start: datetime,
        window_hours: float,
        z_threshold: float,
    ) -> AnomalyAlert | None:
        """Check a single station/defect_type combo for anomaly."""
        assert self._conn is not None

        # Current window count
        current_count = self._conn.execute(
            """SELECT COUNT(*) FROM defect_events
               WHERE station_id = ? AND defect_type = ? AND timestamp > ?""",
            (station_id, defect_type, window_start.isoformat()),
        ).fetchone()[0]

        # Baseline: get counts per window_hours block over RETENTION_DAYS
        # We calculate the rate per window for each historical window
        baseline_rates = self._get_baseline_rates(
            station_id, defect_type, baseline_start, window_start, window_hours
        )

        if len(baseline_rates) < MIN_EVENTS_FOR_BASELINE:
            return None  # Not enough data for meaningful baseline

        mean_rate = statistics.mean(baseline_rates)
        if mean_rate == 0:
            # If baseline is zero and current is nonzero, that's anomalous
            if current_count > 0:
                return AnomalyAlert(
                    station_id=station_id,
                    defect_type=defect_type,
                    current_rate=float(current_count),
                    baseline_rate=0.0,
                    z_score=999.0,  # effectively infinite
                    window_hours=window_hours,
                )
            return None

        stdev = statistics.stdev(baseline_rates) if len(baseline_rates) > 1 else 0.0
        if stdev == 0:
            stdev = mean_rate * 0.1  # avoid division by zero; use 10% of mean

        z_score = (current_count - mean_rate) / stdev

        if z_score >= z_threshold:
            # Get event IDs in the anomalous window
            event_ids = [
                row[0]
                for row in self._conn.execute(
                    """SELECT event_id FROM defect_events
                       WHERE station_id = ? AND defect_type = ? AND timestamp > ?""",
                    (station_id, defect_type, window_start.isoformat()),
                ).fetchall()
            ]

            return AnomalyAlert(
                station_id=station_id,
                defect_type=defect_type,
                current_rate=float(current_count),
                baseline_rate=round(mean_rate, 2),
                z_score=round(z_score, 2),
                window_hours=window_hours,
                event_ids=event_ids,
            )

        return None

    def _get_baseline_rates(
        self,
        station_id: str,
        defect_type: str,
        baseline_start: datetime,
        window_end: datetime,
        window_hours: float,
    ) -> list[float]:
        """Get historical defect counts per window for baseline calculation."""
        assert self._conn is not None

        rates: list[float] = []
        cursor = baseline_start
        window_delta = timedelta(hours=window_hours)

        while cursor + window_delta <= window_end:
            count = self._conn.execute(
                """SELECT COUNT(*) FROM defect_events
                   WHERE station_id = ? AND defect_type = ?
                   AND timestamp >= ? AND timestamp < ?""",
                (station_id, defect_type, cursor.isoformat(), (cursor + window_delta).isoformat()),
            ).fetchone()[0]
            rates.append(float(count))
            cursor += window_delta

        return rates

    def _store_alert(self, alert: AnomalyAlert) -> None:
        """Persist anomaly alert to database."""
        assert self._conn is not None
        self._conn.execute(
            """INSERT OR IGNORE INTO anomaly_alerts
               (alert_id, timestamp, station_id, defect_type,
                current_rate, baseline_rate, z_score, window_hours)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.station_id,
                alert.defect_type,
                alert.current_rate,
                alert.baseline_rate,
                alert.z_score,
                alert.window_hours,
            ),
        )
        self._conn.commit()

    def prune(self) -> int:
        """Remove events older than RETENTION_DAYS. Returns count deleted."""
        if self._conn is None:
            raise RuntimeError("Accumulator not started.")

        cutoff = (datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=RETENTION_DAYS)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM defect_events WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("Pruned %d events older than %d days", deleted, RETENTION_DAYS)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get accumulator statistics for monitoring."""
        if self._conn is None:
            return {"status": "not_started"}

        total = self._conn.execute("SELECT COUNT(*) FROM defect_events").fetchone()[0]
        stations = self._conn.execute("SELECT COUNT(DISTINCT station_id) FROM defect_events").fetchone()[0]
        types = self._conn.execute("SELECT COUNT(DISTINCT defect_type) FROM defect_events").fetchone()[0]
        alerts = self._conn.execute("SELECT COUNT(*) FROM anomaly_alerts WHERE acknowledged = 0").fetchone()[0]

        return {
            "status": "running",
            "total_events": total,
            "active_stations": stations,
            "defect_types": types,
            "unacknowledged_alerts": alerts,
            "db_path": str(self.db_path),
        }

    def stop(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Accumulator stopped")
