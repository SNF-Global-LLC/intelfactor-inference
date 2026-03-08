"""
IntelFactor.ai — Machine Health Copilot: Ingestion Schemas
Shared types for sensor ingestion, baseline computation, and maintenance verdicts.

No external dependencies beyond pydantic. Runs on Jetson or cloud.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────


class SensorType(str, Enum):
    VIBRATION = "vibration"
    CURRENT = "current"
    ACOUSTIC = "acoustic"


class HealthVerdict(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class MaintenanceActionType(str, Enum):
    MONITOR = "monitor"           # Increase polling frequency — no physical action
    INSPECT = "inspect"           # Operator visual / tactile check required
    SERVICE = "service"           # Scheduled maintenance task
    STOP = "stop"                 # Safe immediate shutdown requested by operator


# ── Sensor Reading (in-memory, pre-storage) ────────────────────────────


@dataclass
class SensorReading:
    """
    A single raw reading from one sensor channel, as received over MQTT.

    raw_values holds the full payload: for vibration this may be
    {"x": 0.12, "y": 0.08, "z": 1.02, "rms": 0.61}, for current a
    single {"amps": 4.7}, etc. The schema is intentionally open so
    the service layer is agnostic to sensor vendor/protocol.
    """

    reading_id: str = field(
        default_factory=lambda: f"rng_{datetime.now(tz=timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None)
    )
    station_id: str = ""
    machine_id: str = ""
    sensor_type: SensorType = SensorType.VIBRATION
    raw_values: dict[str, float] = field(default_factory=dict)
    mqtt_topic: str = ""


# ── Persisted Sensor Event (post-scoring) ─────────────────────────────


@dataclass
class SensorEvent:
    """
    A sensor reading after anomaly scoring has been applied.
    This is what gets written to the sensor_events SQLite table.
    """

    event_id: str = field(
        default_factory=lambda: f"se_{datetime.now(tz=timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None)
    )
    station_id: str = ""
    machine_id: str = ""
    sensor_type: SensorType = SensorType.VIBRATION
    raw_values: dict[str, float] = field(default_factory=dict)

    # Scoring outputs (populated by MaintenanceIQ after baseline lookup)
    anomaly_score: float = 0.0      # z-score against rolling baseline
    confidence: float = 0.0         # 0–1; low when baseline is thin
    edge_verdict: HealthVerdict = HealthVerdict.HEALTHY


# ── Baseline Profile ───────────────────────────────────────────────────


@dataclass
class BaselineProfile:
    """
    Statistical baseline for one (machine_id, sensor_type, shift) combination.
    Recomputed every 4-hour window from sensor_events.

    Shift is optional context — allows baselines to differ by day/night shift
    when machine behaviour is known to vary (e.g. startup thermals).
    """

    profile_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    machine_id: str = ""
    sensor_type: SensorType = SensorType.VIBRATION
    shift: str = ""                 # "" | "day" | "night" — empty means all-hours
    mean: float = 0.0
    std: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    sample_count: int = 0
    computed_at: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None)
    )


# ── Maintenance Verdict ────────────────────────────────────────────────


@dataclass
class MaintenanceVerdict:
    """
    Output from MaintenanceIQ.evaluate().

    contributing_factors is an ordered list of {sensor_type, z_score, value}
    dicts — the sensors that pushed the verdict above threshold, ranked
    by z_score descending. Kept as plain dicts so it serialises to JSON
    without extra work.
    """

    verdict_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None)
    )
    machine_id: str = ""
    station_id: str = ""

    verdict: HealthVerdict = HealthVerdict.HEALTHY
    z_score: float = 0.0            # highest z_score across all sensor types
    confidence: float = 0.0
    contributing_factors: list[dict[str, Any]] = field(default_factory=list)

    # Thresholds used at evaluation time (for auditability)
    warning_threshold: float = 2.0
    critical_threshold: float = 3.5


# ── Maintenance Action ────────────────────────────────────────────────


@dataclass
class MaintenanceAction:
    """
    A NOTIFY-ONLY recommended action for an operator.

    NON-NEGOTIABLE: This system never executes. Operator must confirm.
    urgency mirrors DefectIQ: normal | high | critical.
    """

    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None)
    )
    machine_id: str = ""
    station_id: str = ""

    verdict: MaintenanceVerdict = field(default_factory=MaintenanceVerdict)
    action_type: MaintenanceActionType = MaintenanceActionType.MONITOR

    # Bilingual display text
    action_en: str = ""
    action_zh: str = ""

    sop_section: str = ""
    urgency: str = "normal"         # normal | high | critical

    # Operator feedback (written back after response)
    operator_action: str = "pending"    # pending | accepted | rejected | modified
    operator_id: str = ""
    rejection_reason: str = ""
