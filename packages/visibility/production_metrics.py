"""
IntelFactor Production Metrics Engine
=====================================
Sits on top of the existing events table in SQLite.
Called by the station API whenever a new detection event arrives.

This module does NOT touch DeepStream or the bridge.
It reads from the same SQLite that the station already writes to.

Deploy to: /opt/intelfactor/packages/visibility/production_metrics.py
"""

import sqlite3
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger("intelfactor.visibility")

# ---------------------------------------------------------------------------
# Shift Configuration (matches Wiko 3-shift pattern, configurable)
# ---------------------------------------------------------------------------
DEFAULT_SHIFTS = {
    "shift_1": {"start": "06:00", "end": "14:00"},
    "shift_2": {"start": "14:00", "end": "22:00"},
    "shift_3": {"start": "22:00", "end": "06:00"},  # crosses midnight
}

# Station goes idle after this many seconds without a detection
IDLE_THRESHOLD_SECONDS = 120  # 2 minutes — tune per station


class ProductionMetrics:
    """
    Core metrics engine. One instance per station.

    Usage:
        metrics = ProductionMetrics(db_path="/opt/intelfactor/data/local.db",
                                     station_id="SNF-Vision-1")
        # Called from station API on every new event:
        metrics.on_event(event_dict)

        # Called by metrics API:
        metrics.get_throughput(hours=8)
        metrics.get_cycle_times(hours=1)
        metrics.get_utilization(hours=8)
    """

    def __init__(
        self,
        db_path: str = "/opt/intelfactor/data/local.db",
        station_id: str = "SNF-Vision-1",
        shifts: Optional[Dict] = None,
        idle_threshold: float = IDLE_THRESHOLD_SECONDS,
    ):
        self.db_path = db_path
        self.station_id = station_id
        self.shifts = shifts or DEFAULT_SHIFTS
        self.idle_threshold = idle_threshold

        # In-memory state for fast cycle time calculation
        self._last_detection_time: Optional[float] = None
        self._last_inspection_id: Optional[str] = None
        self._last_class: Optional[str] = None

        # Utilization state
        self._current_state: str = "idle"  # 'active' or 'idle'
        self._state_since: float = time.time()

        self._ensure_tables()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------
    def _ensure_tables(self):
        """Run migration if tables don't exist."""
        migration_sql = Path(__file__).parent.parent.parent / "migrations" / "001_production_metrics.sql"

        conn = self._connect()
        try:
            # Check if tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='production_counts'"
            )
            if cursor.fetchone() is None:
                # Try migration file first, fall back to inline
                if migration_sql.exists():
                    conn.executescript(migration_sql.read_text())
                else:
                    self._create_tables_inline(conn)
                conn.commit()
                logger.info("Production metrics tables created")
        finally:
            conn.close()

    def _create_tables_inline(self, conn: sqlite3.Connection):
        """Inline table creation if migration file not found."""
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS production_counts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                hour_bucket TEXT NOT NULL,
                shift_id TEXT,
                object_class TEXT NOT NULL,
                unit_count INTEGER NOT NULL DEFAULT 0,
                first_seen TEXT,
                last_seen TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                synced INTEGER DEFAULT 0,
                UNIQUE(station_id, hour_bucket, object_class)
            );

            CREATE TABLE IF NOT EXISTS cycle_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                inspection_id TEXT NOT NULL,
                prev_inspection_id TEXT,
                cycle_seconds REAL NOT NULL,
                object_class TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                shift_id TEXT,
                synced INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS station_utilization (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                state TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                duration_seconds REAL,
                shift_id TEXT,
                synced INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS shift_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                station_id TEXT NOT NULL,
                shift_id TEXT NOT NULL,
                shift_date TEXT NOT NULL,
                shift_start TEXT NOT NULL,
                shift_end TEXT NOT NULL,
                total_units INTEGER DEFAULT 0,
                avg_cycle_seconds REAL,
                min_cycle_seconds REAL,
                max_cycle_seconds REAL,
                cycle_stddev REAL,
                utilization_pct REAL,
                total_active_seconds REAL,
                total_idle_seconds REAL,
                longest_idle_seconds REAL,
                longest_idle_start TEXT,
                defect_count INTEGER DEFAULT 0,
                review_count INTEGER DEFAULT 0,
                pass_count INTEGER DEFAULT 0,
                narrative TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                synced INTEGER DEFAULT 0,
                UNIQUE(station_id, shift_id, shift_date)
            );
        """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ------------------------------------------------------------------
    # Shift identification
    # ------------------------------------------------------------------
    def _get_shift_id(self, ts: Optional[datetime] = None) -> str:
        """Determine which shift a timestamp falls in."""
        ts = ts or datetime.now()
        t = ts.strftime("%H:%M")

        for shift_id, window in self.shifts.items():
            start, end = window["start"], window["end"]
            if start < end:
                # Normal shift (e.g., 06:00-14:00)
                if start <= t < end:
                    return shift_id
            else:
                # Overnight shift (e.g., 22:00-06:00)
                if t >= start or t < end:
                    return shift_id

        return "unknown"

    def _get_hour_bucket(self, ts: Optional[datetime] = None) -> str:
        """Truncate timestamp to hour for bucketing."""
        ts = ts or datetime.now()
        return ts.replace(minute=0, second=0, microsecond=0).isoformat()

    # ------------------------------------------------------------------
    # Event processing (called on every detection)
    # ------------------------------------------------------------------
    def on_event(self, event: Dict[str, Any]):
        """
        Process a new detection event. Called from the station API
        after storing the event in the events table.

        Expected event dict:
        {
            "inspection_id": "abc123",
            "timestamp": "2026-02-16T14:23:45.123",
            "class": "knife",
            "confidence": 0.92,
            "verdict": "PASS",
            ...
        }
        """
        try:
            ts_str = event.get("timestamp", datetime.now().isoformat())
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if isinstance(ts_str, str) else ts_str
            now = time.time()

            inspection_id = event.get("inspection_id", "")
            obj_class = event.get("class", "unknown")

            shift_id = self._get_shift_id(ts)
            hour_bucket = self._get_hour_bucket(ts)

            conn = self._connect()
            try:
                # 1. Update production count
                self._update_count(conn, hour_bucket, shift_id, obj_class, ts_str)

                # 2. Calculate and store cycle time
                if self._last_detection_time is not None:
                    cycle_seconds = now - self._last_detection_time
                    # Only record if reasonable (< 10 min gap, otherwise it's downtime)
                    if cycle_seconds < 600:
                        self._store_cycle_time(
                            conn, inspection_id, self._last_inspection_id,
                            cycle_seconds, obj_class, ts_str, shift_id
                        )

                # 3. Update utilization state
                self._update_utilization(conn, now, shift_id)

                conn.commit()
            finally:
                conn.close()

            # Update in-memory state
            self._last_detection_time = now
            self._last_inspection_id = inspection_id
            self._last_class = obj_class

        except Exception as e:
            logger.error(f"Metrics processing error: {e}", exc_info=True)

    def _update_count(self, conn, hour_bucket, shift_id, obj_class, ts_str):
        """Upsert production count for this hour bucket."""
        conn.execute("""
            INSERT INTO production_counts (station_id, hour_bucket, shift_id,
                                           object_class, unit_count, first_seen, last_seen)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(station_id, hour_bucket, object_class) DO UPDATE SET
                unit_count = unit_count + 1,
                last_seen = excluded.last_seen,
                updated_at = datetime('now'),
                synced = 0
        """, (self.station_id, hour_bucket, shift_id, obj_class, ts_str, ts_str))

    def _store_cycle_time(self, conn, inspection_id, prev_id, cycle_secs, obj_class, ts_str, shift_id):
        """Store a cycle time measurement."""
        conn.execute("""
            INSERT INTO cycle_times (station_id, inspection_id, prev_inspection_id,
                                     cycle_seconds, object_class, timestamp, shift_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (self.station_id, inspection_id, prev_id, cycle_secs, obj_class, ts_str, shift_id))

    def _update_utilization(self, conn, now: float, shift_id: str):
        """Track state transitions between active and idle."""
        if self._current_state == "idle":
            # Transition: idle -> active
            idle_duration = now - self._state_since
            ts_now = datetime.now().isoformat()

            # Close the idle period
            conn.execute("""
                UPDATE station_utilization
                SET ended_at = ?, duration_seconds = ?
                WHERE station_id = ? AND state = 'idle' AND ended_at IS NULL
            """, (ts_now, idle_duration, self.station_id))

            # Open active period
            conn.execute("""
                INSERT INTO station_utilization (station_id, state, started_at, shift_id)
                VALUES (?, 'active', ?, ?)
            """, (self.station_id, ts_now, shift_id))

            self._current_state = "active"
            self._state_since = now

    def check_idle(self):
        """
        Called periodically (e.g., every 30s) to detect idle state.
        Wire this into a background timer in the station service.
        """
        if self._current_state == "active":
            elapsed = time.time() - (self._last_detection_time or self._state_since)
            if elapsed > self.idle_threshold:
                now = time.time()
                active_duration = now - self._state_since
                ts_now = datetime.now().isoformat()
                shift_id = self._get_shift_id()

                conn = self._connect()
                try:
                    # Close active period
                    conn.execute("""
                        UPDATE station_utilization
                        SET ended_at = ?, duration_seconds = ?
                        WHERE station_id = ? AND state = 'active' AND ended_at IS NULL
                    """, (ts_now, active_duration, self.station_id))

                    # Open idle period
                    conn.execute("""
                        INSERT INTO station_utilization (station_id, state, started_at, shift_id)
                        VALUES (?, 'idle', ?, ?)
                    """, (self.station_id, ts_now, shift_id))

                    conn.commit()
                finally:
                    conn.close()

                self._current_state = "idle"
                self._state_since = now

    # ------------------------------------------------------------------
    # Query methods (called by metrics API)
    # ------------------------------------------------------------------
    def get_throughput(self, hours: int = 8) -> Dict[str, Any]:
        """Get production counts for the last N hours."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn = self._connect()
        try:
            # Hourly breakdown
            rows = conn.execute("""
                SELECT hour_bucket, object_class, SUM(unit_count) as count
                FROM production_counts
                WHERE station_id = ? AND hour_bucket >= ?
                GROUP BY hour_bucket, object_class
                ORDER BY hour_bucket ASC
            """, (self.station_id, since)).fetchall()

            hourly = [dict(r) for r in rows]

            # Total
            total_row = conn.execute("""
                SELECT SUM(unit_count) as total
                FROM production_counts
                WHERE station_id = ? AND hour_bucket >= ?
            """, (self.station_id, since)).fetchone()

            # Current hour rate
            current_hour = self._get_hour_bucket()
            current_row = conn.execute("""
                SELECT SUM(unit_count) as count
                FROM production_counts
                WHERE station_id = ? AND hour_bucket = ?
            """, (self.station_id, current_hour)).fetchone()

            return {
                "station_id": self.station_id,
                "period_hours": hours,
                "since": since,
                "total_units": total_row["total"] or 0,
                "current_hour_units": current_row["count"] or 0,
                "hourly_breakdown": hourly,
            }
        finally:
            conn.close()

    def get_cycle_times(self, hours: int = 1) -> Dict[str, Any]:
        """Get cycle time statistics for the last N hours."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT cycle_seconds FROM cycle_times
                WHERE station_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            """, (self.station_id, since)).fetchall()

            cycles = [r["cycle_seconds"] for r in rows]

            if not cycles:
                return {
                    "station_id": self.station_id,
                    "period_hours": hours,
                    "count": 0,
                    "avg": None, "min": None, "max": None, "stddev": None,
                    "values": [],
                }

            return {
                "station_id": self.station_id,
                "period_hours": hours,
                "count": len(cycles),
                "avg": round(statistics.mean(cycles), 2),
                "min": round(min(cycles), 2),
                "max": round(max(cycles), 2),
                "stddev": round(statistics.stdev(cycles), 2) if len(cycles) > 1 else 0.0,
                "values": [round(c, 2) for c in cycles[-100:]],  # last 100 for charting
            }
        finally:
            conn.close()

    def get_utilization(self, hours: int = 8) -> Dict[str, Any]:
        """Get station utilization for the last N hours."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn = self._connect()
        try:
            rows = conn.execute("""
                SELECT state, duration_seconds
                FROM station_utilization
                WHERE station_id = ? AND started_at >= ? AND duration_seconds IS NOT NULL
            """, (self.station_id, since)).fetchall()

            total_active = sum(r["duration_seconds"] for r in rows if r["state"] == "active")
            total_idle = sum(r["duration_seconds"] for r in rows if r["state"] == "idle")
            total = total_active + total_idle

            # Find longest idle period
            idle_rows = conn.execute("""
                SELECT started_at, duration_seconds
                FROM station_utilization
                WHERE station_id = ? AND started_at >= ?
                  AND state = 'idle' AND duration_seconds IS NOT NULL
                ORDER BY duration_seconds DESC
                LIMIT 1
            """, (self.station_id, since)).fetchone()

            return {
                "station_id": self.station_id,
                "period_hours": hours,
                "utilization_pct": round((total_active / total * 100), 1) if total > 0 else 0.0,
                "active_seconds": round(total_active, 1),
                "idle_seconds": round(total_idle, 1),
                "longest_idle_seconds": round(idle_rows["duration_seconds"], 1) if idle_rows else 0.0,
                "longest_idle_start": idle_rows["started_at"] if idle_rows else None,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Shift summary generation
    # ------------------------------------------------------------------
    def generate_shift_summary(self, shift_id: Optional[str] = None,
                                shift_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a shift summary. Can be called at shift boundary
        or on-demand. Returns the summary dict and stores it in SQLite.
        """
        shift_id = shift_id or self._get_shift_id()
        shift_date = shift_date or datetime.now().strftime("%Y-%m-%d")

        shift_config = self.shifts.get(shift_id, {"start": "00:00", "end": "08:00"})
        shift_start = f"{shift_date}T{shift_config['start']}:00"

        # Handle overnight shifts
        if shift_config["start"] > shift_config["end"]:
            next_date = (datetime.strptime(shift_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            shift_end = f"{next_date}T{shift_config['end']}:00"
        else:
            shift_end = f"{shift_date}T{shift_config['end']}:00"

        conn = self._connect()
        try:
            # Total units
            count_row = conn.execute("""
                SELECT SUM(unit_count) as total
                FROM production_counts
                WHERE station_id = ? AND shift_id = ?
                  AND hour_bucket >= ? AND hour_bucket < ?
            """, (self.station_id, shift_id, shift_start, shift_end)).fetchone()

            total_units = count_row["total"] or 0

            # Cycle times for this shift
            cycle_rows = conn.execute("""
                SELECT cycle_seconds FROM cycle_times
                WHERE station_id = ? AND shift_id = ?
                  AND timestamp >= ? AND timestamp < ?
            """, (self.station_id, shift_id, shift_start, shift_end)).fetchall()

            cycles = [r["cycle_seconds"] for r in cycle_rows]

            # Utilization
            util = self.get_utilization(hours=8)  # approximate

            # Verdict counts from events table (existing)
            verdict_rows = conn.execute("""
                SELECT verdict, COUNT(*) as cnt
                FROM defect_events
                WHERE station_id = ? AND timestamp >= ? AND timestamp < ?
                GROUP BY verdict
            """, (self.station_id, shift_start, shift_end)).fetchall()

            verdict_counts = {r["verdict"]: r["cnt"] for r in verdict_rows}

            summary = {
                "station_id": self.station_id,
                "shift_id": shift_id,
                "shift_date": shift_date,
                "shift_start": shift_start,
                "shift_end": shift_end,
                "total_units": total_units,
                "avg_cycle_seconds": round(statistics.mean(cycles), 2) if cycles else None,
                "min_cycle_seconds": round(min(cycles), 2) if cycles else None,
                "max_cycle_seconds": round(max(cycles), 2) if cycles else None,
                "cycle_stddev": round(statistics.stdev(cycles), 2) if len(cycles) > 1 else None,
                "utilization_pct": util["utilization_pct"],
                "total_active_seconds": util["active_seconds"],
                "total_idle_seconds": util["idle_seconds"],
                "longest_idle_seconds": util["longest_idle_seconds"],
                "longest_idle_start": util["longest_idle_start"],
                "pass_count": verdict_counts.get("PASS", 0),
                "defect_count": verdict_counts.get("FAIL", 0),
                "review_count": verdict_counts.get("REVIEW", 0),
            }

            # Store in SQLite
            conn.execute("""
                INSERT INTO shift_summaries (
                    station_id, shift_id, shift_date, shift_start, shift_end,
                    total_units, avg_cycle_seconds, min_cycle_seconds, max_cycle_seconds,
                    cycle_stddev, utilization_pct, total_active_seconds, total_idle_seconds,
                    longest_idle_seconds, longest_idle_start,
                    pass_count, defect_count, review_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(station_id, shift_id, shift_date) DO UPDATE SET
                    total_units = excluded.total_units,
                    avg_cycle_seconds = excluded.avg_cycle_seconds,
                    min_cycle_seconds = excluded.min_cycle_seconds,
                    max_cycle_seconds = excluded.max_cycle_seconds,
                    cycle_stddev = excluded.cycle_stddev,
                    utilization_pct = excluded.utilization_pct,
                    total_active_seconds = excluded.total_active_seconds,
                    total_idle_seconds = excluded.total_idle_seconds,
                    longest_idle_seconds = excluded.longest_idle_seconds,
                    longest_idle_start = excluded.longest_idle_start,
                    pass_count = excluded.pass_count,
                    defect_count = excluded.defect_count,
                    review_count = excluded.review_count,
                    synced = 0
            """, (
                summary["station_id"], summary["shift_id"], summary["shift_date"],
                summary["shift_start"], summary["shift_end"],
                summary["total_units"], summary["avg_cycle_seconds"],
                summary["min_cycle_seconds"], summary["max_cycle_seconds"],
                summary["cycle_stddev"], summary["utilization_pct"],
                summary["total_active_seconds"], summary["total_idle_seconds"],
                summary["longest_idle_seconds"], summary["longest_idle_start"],
                summary["pass_count"], summary["defect_count"], summary["review_count"],
            ))
            conn.commit()

            return summary
        finally:
            conn.close()

    def get_live_snapshot(self) -> Dict[str, Any]:
        """
        Real-time snapshot for the dashboard live ticker.
        Combines current throughput, cycle time, and utilization.
        """
        throughput = self.get_throughput(hours=1)
        cycle = self.get_cycle_times(hours=1)
        util = self.get_utilization(hours=1)

        return {
            "station_id": self.station_id,
            "timestamp": datetime.now().isoformat(),
            "shift": self._get_shift_id(),
            "current_hour_units": throughput["current_hour_units"],
            "units_per_minute": round(throughput["current_hour_units"] / max(1, datetime.now().minute or 1), 1),
            "avg_cycle_seconds": cycle["avg"],
            "cycle_stddev": cycle["stddev"],
            "utilization_pct": util["utilization_pct"],
            "state": self._current_state,
            "seconds_since_last": round(time.time() - self._last_detection_time, 1) if self._last_detection_time else None,
        }
