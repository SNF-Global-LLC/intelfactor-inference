"""
IntelFactor.ai — Machine Health Copilot: Sensor Ingestion Service
MQTT subscriber for vibration, current, and acoustic sensor streams.

Design mirrors DefectAccumulator:
- SQLite WAL-mode for concurrent read/write.
- Rolling 4-hour baseline windows, 30-day retention.
- Z-score anomaly scoring per (machine_id, sensor_type) pair.
- No GPU required. ~30MB RAM. Runs alongside the vision pipeline.

Sensor topics (configurable via edge.yaml or constructor):
  sensors/vibration  → {"machine_id": "...", "x": 0.12, "y": 0.08, "z": 1.02, "rms": 0.61}
  sensors/current    → {"machine_id": "...", "amps": 4.7}
  sensors/acoustic   → {"machine_id": "...", "db": 72.3, "peak_hz": 1200}
"""

from __future__ import annotations

import json
import logging
import sqlite3
import statistics
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import paho.mqtt.client as mqtt

    _PAHO_AVAILABLE = True
except ImportError:  # paho-mqtt not installed in test environment
    _PAHO_AVAILABLE = False

from packages.ingestion.schemas import (
    BaselineProfile,
    HealthVerdict,
    SensorEvent,
    SensorReading,
    SensorType,
)

logger = logging.getLogger(__name__)

# ── Schema ──────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sensor_events (
    event_id         TEXT PRIMARY KEY,
    timestamp        TEXT NOT NULL,
    station_id       TEXT NOT NULL,
    machine_id       TEXT NOT NULL,
    sensor_type      TEXT NOT NULL,
    raw_values       TEXT NOT NULL,       -- JSON
    anomaly_score    REAL DEFAULT 0.0,
    confidence       REAL DEFAULT 0.0,
    edge_verdict     TEXT DEFAULT 'HEALTHY',
    operator_action  TEXT DEFAULT 'pending',
    operator_id      TEXT DEFAULT '',
    rejection_reason TEXT DEFAULT '',
    comment          TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_sensor_events_machine_type_ts
    ON sensor_events(machine_id, sensor_type, timestamp);

CREATE INDEX IF NOT EXISTS idx_sensor_events_timestamp
    ON sensor_events(timestamp);

CREATE TABLE IF NOT EXISTS baseline_profiles (
    profile_id      TEXT PRIMARY KEY,
    machine_id      TEXT NOT NULL,
    sensor_type     TEXT NOT NULL,
    shift           TEXT DEFAULT '',
    mean            REAL NOT NULL,
    std             REAL NOT NULL,
    p95             REAL NOT NULL,
    p99             REAL NOT NULL,
    sample_count    INTEGER NOT NULL,
    computed_at     TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_baseline_machine_type_shift
    ON baseline_profiles(machine_id, sensor_type, shift);
"""

# ── Constants ─────────────────────────────────────────────────────────

RETENTION_DAYS = 30
BASELINE_WINDOW_HOURS = 4.0
MIN_SAMPLES_FOR_BASELINE = 10
BASELINE_RECOMPUTE_INTERVAL_SEC = 3600   # recompute baselines every hour

DEFAULT_TOPICS: dict[str, SensorType] = {
    "sensors/vibration": SensorType.VIBRATION,
    "sensors/current": SensorType.CURRENT,
    "sensors/acoustic": SensorType.ACOUSTIC,
}


def _extract_scalar(raw_values: dict[str, float], sensor_type: SensorType) -> float:
    """
    Extract the primary scalar value from a sensor payload.

    Convention:
      vibration → rms (falls back to magnitude of x/y/z)
      current   → amps
      acoustic  → db
    """
    if sensor_type == SensorType.VIBRATION:
        if "rms" in raw_values:
            return raw_values["rms"]
        # Compute RMS from axes if available
        axes = [raw_values.get(k, 0.0) for k in ("x", "y", "z") if k in raw_values]
        if axes:
            return (sum(v**2 for v in axes) / len(axes)) ** 0.5
        return next(iter(raw_values.values()), 0.0)

    if sensor_type == SensorType.CURRENT:
        return raw_values.get("amps", next(iter(raw_values.values()), 0.0))

    if sensor_type == SensorType.ACOUSTIC:
        return raw_values.get("db", next(iter(raw_values.values()), 0.0))

    return next(iter(raw_values.values()), 0.0)


# ── SensorService ──────────────────────────────────────────────────────


class SensorService:
    """
    Subscribes to MQTT sensor topics and persists scored events to SQLite.

    Usage:
        service = SensorService(station_id="station_1", db_path="/data/sensors.db")
        service.start()          # connects MQTT, initialises DB
        ...
        service.stop()           # clean disconnect + DB close

    In test environments without a broker, call ingest_reading() directly
    instead of relying on MQTT callbacks.

    Scoring:
        Each reading is scored by looking up the rolling baseline for that
        (machine_id, sensor_type) pair. If baseline is thin (<MIN_SAMPLES),
        the event is stored with confidence=0 and verdict=HEALTHY (unknown).
        Otherwise z_score = (value − mean) / std, mapped to HealthVerdict.

    Baseline recomputation:
        A background thread recomputes baseline_profiles every
        BASELINE_RECOMPUTE_INTERVAL_SEC from raw sensor_events. Baselines
        are always built on the [now − RETENTION_DAYS, now − BASELINE_WINDOW_HOURS]
        window so the most recent window is excluded (it may be anomalous).
    """

    def __init__(
        self,
        station_id: str,
        db_path: str | Path = "/opt/intelfactor/data/sensors.db",
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        mqtt_topics: dict[str, SensorType] | None = None,
        on_event: Callable[[SensorEvent], None] | None = None,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.5,
        mqtt_username: str = "",
        mqtt_password: str = "",
        mqtt_tls_ca: str = "",
        mqtt_tls_cert: str = "",
        mqtt_tls_key: str = "",
    ):
        self.station_id = station_id
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_username = mqtt_username
        self.mqtt_password = mqtt_password
        self.mqtt_tls_ca = mqtt_tls_ca
        self.mqtt_tls_cert = mqtt_tls_cert
        self.mqtt_tls_key = mqtt_tls_key
        self.topics = mqtt_topics or DEFAULT_TOPICS
        self.on_event = on_event  # optional upstream callback (e.g. feed MaintenanceIQ)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self._conn: sqlite3.Connection | None = None
        self._mqtt: Any | None = None   # paho Client or None in stub mode
        self._baseline_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise database, connect MQTT, start baseline thread."""
        self._init_db()
        self._start_mqtt()
        self._start_baseline_thread()
        logger.info(
            "SensorService started: station=%s topics=%s",
            self.station_id,
            list(self.topics.keys()),
        )

    def stop(self) -> None:
        """Gracefully disconnect and close resources."""
        self._stop_event.set()

        if self._mqtt is not None and _PAHO_AVAILABLE:
            try:
                self._mqtt.loop_stop()
                self._mqtt.disconnect()
            except Exception:
                pass

        if self._baseline_thread is not None:
            self._baseline_thread.join(timeout=5)

        if self._conn is not None:
            self._conn.close()
            self._conn = None

        logger.info("SensorService stopped: station=%s", self.station_id)

    # ── Public API ─────────────────────────────────────────────────────

    def ingest_reading(self, reading: SensorReading) -> SensorEvent:
        """
        Score and persist a sensor reading.

        This is the hot path. Called either from the MQTT callback or
        directly in tests / non-MQTT deployments.
        """
        if self._conn is None:
            raise RuntimeError("SensorService not started. Call start() first.")

        scalar = _extract_scalar(reading.raw_values, reading.sensor_type)
        z_score, confidence, verdict = self._score(
            reading.machine_id, reading.sensor_type, scalar,
            self.warning_threshold, self.critical_threshold,
        )

        event = SensorEvent(
            timestamp=reading.timestamp,
            station_id=reading.station_id or self.station_id,
            machine_id=reading.machine_id,
            sensor_type=reading.sensor_type,
            raw_values=reading.raw_values,
            anomaly_score=z_score,
            confidence=confidence,
            edge_verdict=verdict,
        )

        self._store_event(event)

        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception as exc:
                logger.warning("on_event callback raised: %s", exc)

        return event

    def get_latest_events(
        self,
        machine_id: str,
        sensor_type: SensorType | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve recent events for a machine, newest first."""
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        query = "SELECT * FROM sensor_events WHERE machine_id = ?"
        params: list[Any] = [machine_id]
        if sensor_type is not None:
            query += " AND sensor_type = ?"
            params.append(sensor_type.value)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["raw_values"] = json.loads(d["raw_values"])
            result.append(d)
        return result

    def get_baseline(
        self, machine_id: str, sensor_type: SensorType, shift: str = ""
    ) -> BaselineProfile | None:
        """Return the current stored baseline for a (machine, sensor, shift) triple."""
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        row = self._conn.execute(
            """SELECT * FROM baseline_profiles
               WHERE machine_id = ? AND sensor_type = ? AND shift = ?""",
            (machine_id, sensor_type.value, shift),
        ).fetchone()

        if row is None:
            return None

        cols = [d[0] for d in self._conn.execute(
            "SELECT * FROM baseline_profiles LIMIT 0"
        ).description]
        data = dict(zip(cols, row))

        return BaselineProfile(
            profile_id=data["profile_id"],
            machine_id=data["machine_id"],
            sensor_type=SensorType(data["sensor_type"]),
            shift=data["shift"],
            mean=data["mean"],
            std=data["std"],
            p95=data["p95"],
            p99=data["p99"],
            sample_count=data["sample_count"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
        )

    def recompute_baselines(self, shift: str = "") -> list[BaselineProfile]:
        """
        Recompute rolling baselines for all (machine_id, sensor_type) combos.

        Excludes the most recent BASELINE_WINDOW_HOURS to avoid contaminating
        the baseline with a live anomaly. Called automatically by the background
        thread and can also be triggered manually after model swap or shift change.
        """
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        window_end = now - timedelta(hours=BASELINE_WINDOW_HOURS)
        window_start = now - timedelta(days=RETENTION_DAYS)

        combos = self._conn.execute(
            """SELECT DISTINCT machine_id, sensor_type
               FROM sensor_events
               WHERE timestamp > ?""",
            (window_start.isoformat(),),
        ).fetchall()

        profiles: list[BaselineProfile] = []
        for machine_id, sensor_type_str in combos:
            sensor_type = SensorType(sensor_type_str)
            profile = self._compute_baseline(
                machine_id, sensor_type, shift, window_start, window_end
            )
            if profile is not None:
                self._upsert_baseline(profile)
                profiles.append(profile)

        logger.info("Recomputed %d baseline profiles", len(profiles))
        return profiles

    def prune(self) -> int:
        """Remove events older than RETENTION_DAYS. Returns row count deleted."""
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        cutoff = (
            datetime.now(tz=timezone.utc).replace(tzinfo=None)
            - timedelta(days=RETENTION_DAYS)
        ).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM sensor_events WHERE timestamp < ?", (cutoff,)
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("Pruned %d sensor events older than %d days", deleted, RETENTION_DAYS)
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Stats for /health and Prometheus scraping."""
        if self._conn is None:
            return {"status": "not_started"}

        total = self._conn.execute("SELECT COUNT(*) FROM sensor_events").fetchone()[0]
        machines = self._conn.execute(
            "SELECT COUNT(DISTINCT machine_id) FROM sensor_events"
        ).fetchone()[0]
        baselines = self._conn.execute("SELECT COUNT(*) FROM baseline_profiles").fetchone()[0]
        critical = self._conn.execute(
            "SELECT COUNT(*) FROM sensor_events WHERE edge_verdict = 'CRITICAL'"
        ).fetchone()[0]
        warnings = self._conn.execute(
            "SELECT COUNT(*) FROM sensor_events WHERE edge_verdict = 'WARNING'"
        ).fetchone()[0]

        return {
            "status": "running",
            "total_events": total,
            "active_machines": machines,
            "baseline_profiles": baselines,
            "critical_events": critical,
            "warning_events": warnings,
            "db_path": str(self.db_path),
        }

    def list_events(
        self,
        machine_id: str | None = None,
        sensor_type: SensorType | None = None,
        verdict: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query sensor events with optional filters, newest first."""
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        query = "SELECT * FROM sensor_events WHERE 1=1"
        params: list[Any] = []

        if machine_id is not None:
            query += " AND machine_id = ?"
            params.append(machine_id)
        if sensor_type is not None:
            query += " AND sensor_type = ?"
            params.append(sensor_type.value if hasattr(sensor_type, "value") else sensor_type)
        if verdict is not None:
            query += " AND edge_verdict = ?"
            params.append(verdict)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(query, params)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d["raw_values"] = json.loads(d["raw_values"])
            result.append(d)
        return result

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Return a single sensor event by ID, or None if not found."""
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        cursor = self._conn.execute(
            "SELECT * FROM sensor_events WHERE event_id = ?", (event_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cursor.description]
        d = dict(zip(cols, row))
        d["raw_values"] = json.loads(d["raw_values"])
        return d

    def list_baselines(self) -> list[dict[str, Any]]:
        """Return all computed baseline profiles as dicts."""
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        cursor = self._conn.execute(
            "SELECT * FROM baseline_profiles ORDER BY machine_id, sensor_type"
        )
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    def get_incidents(
        self, machine_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Return WARNING and CRITICAL events grouped by machine_id.

        Each entry contains: machine_id, severity (worst verdict in group),
        event_count, first_seen, last_seen, contributing_factors (sensor types).
        """
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        where = "WHERE edge_verdict IN ('WARNING', 'CRITICAL')"
        params: list[Any] = []
        if machine_id is not None:
            where += " AND machine_id = ?"
            params.append(machine_id)

        cursor = self._conn.execute(
            f"""
            SELECT
                machine_id,
                CASE WHEN SUM(CASE WHEN edge_verdict='CRITICAL' THEN 1 ELSE 0 END) > 0
                     THEN 'CRITICAL' ELSE 'WARNING' END AS severity,
                COUNT(*) AS event_count,
                MIN(timestamp) AS first_seen,
                MAX(timestamp) AS last_seen
            FROM sensor_events
            {where}
            GROUP BY machine_id
            ORDER BY severity DESC, event_count DESC
            LIMIT ?
            """,
            params + [limit],
        )
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        incidents = []
        for row in rows:
            incident = dict(zip(cols, row))
            cfactors = self._conn.execute(
                """SELECT DISTINCT sensor_type FROM sensor_events
                   WHERE machine_id = ? AND edge_verdict IN ('WARNING', 'CRITICAL')""",
                (incident["machine_id"],),
            ).fetchall()
            incident["contributing_factors"] = [r[0] for r in cfactors]
            incidents.append(incident)
        return incidents

    def get_recent_events_for_machine(
        self, machine_id: str, limit: int = 20
    ) -> list[SensorEvent]:
        """
        Return the most recent SensorEvent objects for a machine.

        Returns typed SensorEvent instances suitable for passing directly to
        MaintenanceIQ.evaluate(). Use list_events() when you need dicts for JSON.
        """
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        cursor = self._conn.execute(
            """SELECT * FROM sensor_events WHERE machine_id = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (machine_id, limit),
        )
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        events: list[SensorEvent] = []
        for row in rows:
            d = dict(zip(cols, row))
            events.append(
                SensorEvent(
                    event_id=d["event_id"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                    station_id=d["station_id"],
                    machine_id=d["machine_id"],
                    sensor_type=SensorType(d["sensor_type"]),
                    raw_values=json.loads(d["raw_values"]),
                    anomaly_score=d.get("anomaly_score", 0.0),
                    confidence=d.get("confidence", 0.0),
                    edge_verdict=HealthVerdict(d.get("edge_verdict", "HEALTHY")),
                )
            )
        return events

    def update_event_feedback(
        self,
        event_id: str,
        operator_action: str,
        operator_id: str = "",
        rejection_reason: str = "",
        comment: str = "",
    ) -> bool:
        """
        Write operator feedback back onto a sensor event.

        Returns True if the event was found and updated, False if not found.
        operator_action must be one of: confirm, reject, uncertain.
        """
        if self._conn is None:
            raise RuntimeError("SensorService not started.")

        allowed = {"confirm", "reject", "uncertain"}
        if operator_action not in allowed:
            raise ValueError(f"operator_action must be one of {allowed}, got {operator_action!r}")

        cursor = self._conn.execute(
            """UPDATE sensor_events
               SET operator_action = ?, operator_id = ?,
                   rejection_reason = ?, comment = ?
               WHERE event_id = ?""",
            (operator_action, operator_id, rejection_reason, comment, event_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Internal: scoring ─────────────────────────────────────────────

    def _score(
        self,
        machine_id: str,
        sensor_type: SensorType,
        value: float,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.5,
    ) -> tuple[float, float, HealthVerdict]:
        """
        Return (z_score, confidence, verdict) for a scalar sensor value.

        warning_threshold and critical_threshold are passed by the caller so that
        per-machine overrides from station.yaml flow through without a second
        config lookup. Single source of truth: __init__ → ingest_reading → _score.

        confidence reflects baseline quality:
          0.0   — no baseline (< MIN_SAMPLES)
          0.5   — thin baseline (MIN_SAMPLES <= n < 3×MIN_SAMPLES)
          1.0   — mature baseline
        """
        assert self._conn is not None

        baseline = self.get_baseline(machine_id, sensor_type)
        if baseline is None or baseline.sample_count < MIN_SAMPLES_FOR_BASELINE:
            return 0.0, 0.0, HealthVerdict.HEALTHY

        confidence = min(1.0, baseline.sample_count / (MIN_SAMPLES_FOR_BASELINE * 3))

        std = baseline.std if baseline.std > 0 else baseline.mean * 0.05 or 1e-6
        z_score = (value - baseline.mean) / std

        if z_score >= critical_threshold:
            verdict = HealthVerdict.CRITICAL
        elif z_score >= warning_threshold:
            verdict = HealthVerdict.WARNING
        else:
            verdict = HealthVerdict.HEALTHY

        return round(z_score, 3), round(confidence, 3), verdict

    # ── Internal: storage ─────────────────────────────────────────────

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-8000")
        self._conn.executescript(SCHEMA_SQL)
        # Migrate: add feedback columns that may not exist in older databases.
        for col_sql in (
            "ALTER TABLE sensor_events ADD COLUMN operator_action TEXT DEFAULT 'pending'",
            "ALTER TABLE sensor_events ADD COLUMN operator_id TEXT DEFAULT ''",
            "ALTER TABLE sensor_events ADD COLUMN rejection_reason TEXT DEFAULT ''",
            "ALTER TABLE sensor_events ADD COLUMN comment TEXT DEFAULT ''",
        ):
            try:
                self._conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass  # Column already exists — safe to ignore
        self._conn.commit()
        logger.info("SensorService DB initialised: %s", self.db_path)

    def _store_event(self, event: SensorEvent) -> None:
        assert self._conn is not None
        self._conn.execute(
            """INSERT OR IGNORE INTO sensor_events
               (event_id, timestamp, station_id, machine_id, sensor_type,
                raw_values, anomaly_score, confidence, edge_verdict)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.timestamp.isoformat(),
                event.station_id,
                event.machine_id,
                event.sensor_type.value,
                json.dumps(event.raw_values),
                event.anomaly_score,
                event.confidence,
                event.edge_verdict.value,
            ),
        )
        self._conn.commit()

    def _compute_baseline(
        self,
        machine_id: str,
        sensor_type: SensorType,
        shift: str,
        window_start: datetime,
        window_end: datetime,
    ) -> BaselineProfile | None:
        """Compute statistics from raw scalar values in the given time window."""
        assert self._conn is not None

        rows = self._conn.execute(
            """SELECT raw_values FROM sensor_events
               WHERE machine_id = ? AND sensor_type = ?
               AND timestamp >= ? AND timestamp < ?
               ORDER BY timestamp""",
            (
                machine_id,
                sensor_type.value,
                window_start.isoformat(),
                window_end.isoformat(),
            ),
        ).fetchall()

        scalars = [
            _extract_scalar(json.loads(r[0]), sensor_type) for r in rows
        ]

        if len(scalars) < MIN_SAMPLES_FOR_BASELINE:
            return None

        sorted_vals = sorted(scalars)
        n = len(sorted_vals)
        mean = statistics.mean(scalars)
        std = statistics.stdev(scalars) if n > 1 else 0.0
        p95 = sorted_vals[int(n * 0.95)]
        p99 = sorted_vals[int(n * 0.99)]

        return BaselineProfile(
            machine_id=machine_id,
            sensor_type=sensor_type,
            shift=shift,
            mean=round(mean, 6),
            std=round(std, 6),
            p95=round(p95, 6),
            p99=round(p99, 6),
            sample_count=n,
        )

    def _upsert_baseline(self, profile: BaselineProfile) -> None:
        assert self._conn is not None
        self._conn.execute(
            """INSERT INTO baseline_profiles
               (profile_id, machine_id, sensor_type, shift,
                mean, std, p95, p99, sample_count, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(machine_id, sensor_type, shift)
               DO UPDATE SET
                   mean=excluded.mean, std=excluded.std,
                   p95=excluded.p95, p99=excluded.p99,
                   sample_count=excluded.sample_count,
                   computed_at=excluded.computed_at""",
            (
                profile.profile_id,
                profile.machine_id,
                profile.sensor_type.value,
                profile.shift,
                profile.mean,
                profile.std,
                profile.p95,
                profile.p99,
                profile.sample_count,
                profile.computed_at.isoformat(),
            ),
        )
        self._conn.commit()

    # ── Internal: MQTT ────────────────────────────────────────────────

    def _start_mqtt(self) -> None:
        if not _PAHO_AVAILABLE:
            logger.warning(
                "paho-mqtt not installed — MQTT disabled. Use ingest_reading() directly."
            )
            return

        client = mqtt.Client()
        client.on_connect = self._on_mqtt_connect
        client.on_message = self._on_mqtt_message
        client.on_disconnect = self._on_mqtt_disconnect

        # TLS configuration
        if self.mqtt_tls_ca:
            client.tls_set(
                ca_certs=self.mqtt_tls_ca,
                certfile=self.mqtt_tls_cert or None,
                keyfile=self.mqtt_tls_key or None,
            )
            logger.info("MQTT TLS enabled (ca=%s)", self.mqtt_tls_ca)

        # Authentication
        if self.mqtt_username:
            client.username_pw_set(self.mqtt_username, self.mqtt_password)

        try:
            client.connect(self.mqtt_host, self.mqtt_port, keepalive=60)
            client.loop_start()
            self._mqtt = client
            logger.info("MQTT connected: %s:%d", self.mqtt_host, self.mqtt_port)
        except Exception as exc:
            logger.warning("MQTT connection failed (%s) — running without MQTT", exc)

    def _on_mqtt_connect(self, client: Any, userdata: Any, flags: Any, rc: int) -> None:
        if rc != 0:
            logger.error("MQTT connect failed: rc=%d", rc)
            return
        for topic in self.topics:
            client.subscribe(topic)
            logger.info("Subscribed: %s", topic)

    def _on_mqtt_disconnect(self, client: Any, userdata: Any, rc: int) -> None:
        if rc != 0:
            logger.warning("MQTT unexpected disconnect: rc=%d — will auto-reconnect", rc)

    def _on_mqtt_message(self, client: Any, userdata: Any, msg: Any) -> None:
        """Parse MQTT payload and hand off to ingest_reading()."""
        topic = msg.topic
        sensor_type = self.topics.get(topic)
        if sensor_type is None:
            return

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Malformed MQTT payload on %s: %s", topic, exc)
            return

        machine_id = payload.pop("machine_id", "unknown")
        reading = SensorReading(
            station_id=self.station_id,
            machine_id=machine_id,
            sensor_type=sensor_type,
            raw_values={k: float(v) for k, v in payload.items() if isinstance(v, (int, float))},
            mqtt_topic=topic,
        )

        try:
            self.ingest_reading(reading)
        except Exception as exc:
            logger.error("Failed to ingest reading from %s: %s", topic, exc)

    # ── Internal: background thread ───────────────────────────────────

    def _start_baseline_thread(self) -> None:
        self._baseline_thread = threading.Thread(
            target=self._baseline_loop,
            name="sensor-baseline",
            daemon=True,
        )
        self._baseline_thread.start()

    def _baseline_loop(self) -> None:
        """Periodically recompute baselines and prune old events."""
        while not self._stop_event.is_set():
            try:
                self.recompute_baselines()
                self.prune()
            except Exception as exc:
                logger.error("Baseline recompute failed: %s", exc)
            self._stop_event.wait(BASELINE_RECOMPUTE_INTERVAL_SEC)
