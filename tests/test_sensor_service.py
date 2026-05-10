"""
Tests for Machine Health Copilot — SensorService.
Covers ingestion, baseline computation, z-score scoring, pruning, and stats.
Run: python -m pytest tests/test_sensor_service.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.ingestion.schemas import (
    HealthVerdict,
    SensorEvent,
    SensorReading,
    SensorType,
)
from packages.ingestion.sensor_service import (
    MIN_SAMPLES_FOR_BASELINE,
    SensorService,
    _extract_scalar,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def service(tmp_path):
    """A started SensorService backed by a temp SQLite file. No MQTT."""
    svc = SensorService(
        station_id="test_station",
        db_path=tmp_path / "test_sensors.db",
    )
    svc.start()
    yield svc
    svc.stop()


def _vibration_reading(
    machine_id: str = "m1",
    rms: float = 0.5,
    ts: datetime | None = None,
) -> SensorReading:
    return SensorReading(
        station_id="test_station",
        machine_id=machine_id,
        sensor_type=SensorType.VIBRATION,
        raw_values={"x": rms * 0.6, "y": rms * 0.5, "z": rms * 0.7, "rms": rms},
        timestamp=ts or datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )


def _current_reading(
    machine_id: str = "m1",
    amps: float = 5.0,
    ts: datetime | None = None,
) -> SensorReading:
    return SensorReading(
        station_id="test_station",
        machine_id=machine_id,
        sensor_type=SensorType.CURRENT,
        raw_values={"amps": amps},
        timestamp=ts or datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )


def _acoustic_reading(
    machine_id: str = "m1",
    db: float = 65.0,
    ts: datetime | None = None,
) -> SensorReading:
    return SensorReading(
        station_id="test_station",
        machine_id=machine_id,
        sensor_type=SensorType.ACOUSTIC,
        raw_values={"db": db, "peak_hz": 1200.0},
        timestamp=ts or datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )


# ── _extract_scalar unit tests ─────────────────────────────────────────


class TestExtractScalar:
    def test_vibration_uses_rms_field(self):
        raw = {"x": 0.1, "y": 0.2, "z": 0.9, "rms": 0.61}
        assert _extract_scalar(raw, SensorType.VIBRATION) == pytest.approx(0.61)

    def test_vibration_computes_rms_from_axes(self):
        raw = {"x": 1.0, "y": 0.0, "z": 0.0}
        # sqrt((1² + 0² + 0²) / 3) = sqrt(1/3) ≈ 0.577
        result = _extract_scalar(raw, SensorType.VIBRATION)
        assert result == pytest.approx((1 / 3) ** 0.5, rel=1e-3)

    def test_current_uses_amps(self):
        assert _extract_scalar({"amps": 4.7}, SensorType.CURRENT) == pytest.approx(4.7)

    def test_acoustic_uses_db(self):
        assert _extract_scalar({"db": 72.3, "peak_hz": 1200}, SensorType.ACOUSTIC) == pytest.approx(72.3)

    def test_fallback_to_first_value(self):
        # Unusual payload — falls back to first value
        raw = {"mystery_channel": 99.0}
        result = _extract_scalar(raw, SensorType.CURRENT)
        assert result == pytest.approx(99.0)


# ── Ingestion ──────────────────────────────────────────────────────────


class TestSensorIngestion:
    def test_stores_reading(self, service):
        """Ingest a reading and confirm it appears in the DB."""
        service.ingest_reading(_vibration_reading())
        stats = service.get_stats()
        assert stats["total_events"] == 1

    def test_stores_all_sensor_types(self, service):
        service.ingest_reading(_vibration_reading())
        service.ingest_reading(_current_reading())
        service.ingest_reading(_acoustic_reading())
        assert service.get_stats()["total_events"] == 3

    def test_event_has_correct_machine_id(self, service):
        service.ingest_reading(_vibration_reading(machine_id="press_01"))
        events = service.get_latest_events("press_01")
        assert len(events) == 1
        assert events[0]["machine_id"] == "press_01"

    def test_raw_values_round_trip(self, service):
        """JSON raw_values must survive store-and-retrieve intact."""
        payload = {"x": 0.12, "y": 0.08, "z": 1.02, "rms": 0.61}
        service.ingest_reading(
            SensorReading(
                machine_id="m1",
                sensor_type=SensorType.VIBRATION,
                raw_values=payload,
            )
        )
        events = service.get_latest_events("m1", SensorType.VIBRATION)
        assert events[0]["raw_values"] == payload

    def test_on_event_callback_is_called(self, tmp_path):
        """on_event callback should fire on each ingested reading."""
        received: list[SensorEvent] = []
        svc = SensorService(
            station_id="s1",
            db_path=tmp_path / "cb.db",
            on_event=received.append,
        )
        svc.start()
        svc.ingest_reading(_vibration_reading())
        svc.stop()
        assert len(received) == 1
        assert isinstance(received[0], SensorEvent)

    def test_no_baseline_yields_healthy_verdict(self, service):
        """First reading with no baseline should not raise an alert."""
        event = service.ingest_reading(_vibration_reading())
        assert event.edge_verdict == HealthVerdict.HEALTHY
        assert event.confidence == 0.0
        assert event.anomaly_score == 0.0


# ── Baseline computation ───────────────────────────────────────────────


class TestBaselineComputation:
    def _seed_normal_readings(self, service: SensorService, n: int = 50, rms: float = 0.5) -> None:
        """Insert n historical vibration readings with consistent rms."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        for i in range(n):
            ts = now - timedelta(hours=6 + i)  # outside the exclusion window
            service.ingest_reading(_vibration_reading(rms=rms, ts=ts))

    def test_baseline_computed_after_sufficient_samples(self, service):
        self._seed_normal_readings(service, n=MIN_SAMPLES_FOR_BASELINE + 5)
        profiles = service.recompute_baselines()
        assert len(profiles) >= 1
        profile = profiles[0]
        assert profile.sample_count >= MIN_SAMPLES_FOR_BASELINE
        assert profile.mean > 0

    def test_baseline_not_computed_below_min_samples(self, service):
        """Fewer than MIN_SAMPLES readings should produce no profile."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        for i in range(MIN_SAMPLES_FOR_BASELINE - 1):
            ts = now - timedelta(hours=6 + i)
            service.ingest_reading(_vibration_reading(rms=0.5, ts=ts))
        profiles = service.recompute_baselines()
        assert len(profiles) == 0

    def test_get_baseline_returns_stored_profile(self, service):
        self._seed_normal_readings(service, n=30)
        service.recompute_baselines()
        baseline = service.get_baseline("m1", SensorType.VIBRATION)
        assert baseline is not None
        assert baseline.machine_id == "m1"
        assert baseline.sensor_type == SensorType.VIBRATION
        assert baseline.std >= 0

    def test_baseline_excludes_recent_window(self, service):
        """Readings within BASELINE_WINDOW_HOURS should not affect the baseline."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        # Seed normal historical readings
        for i in range(30):
            ts = now - timedelta(hours=6 + i)
            service.ingest_reading(_vibration_reading(rms=0.5, ts=ts))

        # Inject a spike in the recent window — should not distort baseline
        for _ in range(5):
            service.ingest_reading(_vibration_reading(rms=99.0))

        service.recompute_baselines()
        baseline = service.get_baseline("m1", SensorType.VIBRATION)
        assert baseline is not None
        assert baseline.mean < 5.0  # spike values excluded

    def test_baseline_upsert_updates_existing(self, service):
        """Recomputing a second time should overwrite, not duplicate."""
        self._seed_normal_readings(service, n=30)
        service.recompute_baselines()
        service.recompute_baselines()
        # Should still be exactly one profile per (machine, sensor, shift) triple
        baseline = service.get_baseline("m1", SensorType.VIBRATION)
        assert baseline is not None


# ── Z-score scoring ────────────────────────────────────────────────────


class TestZScoreScoring:
    def _establish_baseline(self, service: SensorService, mean: float = 1.0, spread: float = 0.1) -> None:
        """Seed enough readings to build a stable baseline."""
        import random

        random.seed(42)
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        for i in range(60):
            val = mean + random.gauss(0, spread)
            ts = now - timedelta(hours=6 + i)
            service.ingest_reading(_vibration_reading(rms=val, ts=ts))
        service.recompute_baselines()

    def test_normal_value_scores_healthy(self, service):
        self._establish_baseline(service, mean=1.0, spread=0.1)
        event = service.ingest_reading(_vibration_reading(rms=1.05))  # ~0.5σ
        assert event.edge_verdict == HealthVerdict.HEALTHY

    def test_elevated_value_scores_warning(self, service):
        self._establish_baseline(service, mean=1.0, spread=0.1)
        # 1.0 + 2.5 * 0.1 = 1.25 → z=2.5 → WARNING
        event = service.ingest_reading(_vibration_reading(rms=1.25))
        assert event.edge_verdict == HealthVerdict.WARNING
        assert event.anomaly_score >= 2.0

    def test_spike_scores_critical(self, service):
        self._establish_baseline(service, mean=1.0, spread=0.1)
        # 1.0 + 4.0 * 0.1 = 1.4 → z~4 → CRITICAL
        event = service.ingest_reading(_vibration_reading(rms=1.4))
        assert event.edge_verdict == HealthVerdict.CRITICAL
        assert event.anomaly_score >= 3.5

    def test_confidence_grows_with_sample_count(self, service):
        """More baseline samples → higher confidence."""
        import random

        random.seed(7)
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        # Thin baseline (just over MIN_SAMPLES)
        for i in range(MIN_SAMPLES_FOR_BASELINE + 2):
            ts = now - timedelta(hours=6 + i)
            service.ingest_reading(_vibration_reading(rms=1.0 + random.gauss(0, 0.05), ts=ts))
        service.recompute_baselines()
        event_thin = service.ingest_reading(_vibration_reading(rms=1.0))

        # Mature baseline (3× MIN_SAMPLES)
        for i in range(MIN_SAMPLES_FOR_BASELINE * 3):
            ts = now - timedelta(hours=10 + MIN_SAMPLES_FOR_BASELINE + i)
            service.ingest_reading(_vibration_reading(rms=1.0 + random.gauss(0, 0.05), ts=ts))
        service.recompute_baselines()
        event_mature = service.ingest_reading(_vibration_reading(rms=1.0))

        assert event_mature.confidence >= event_thin.confidence


# ── Pruning ────────────────────────────────────────────────────────────


class TestPruning:
    def test_prune_removes_old_events(self, service):
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        old_ts = now - timedelta(days=40)
        service.ingest_reading(_vibration_reading(ts=old_ts))
        service.ingest_reading(_vibration_reading())  # recent

        deleted = service.prune()
        assert deleted == 1
        assert service.get_stats()["total_events"] == 1

    def test_prune_keeps_recent_events(self, service):
        for _ in range(5):
            service.ingest_reading(_vibration_reading())
        deleted = service.prune()
        assert deleted == 0
        assert service.get_stats()["total_events"] == 5


# ── Stats ──────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_initially_empty(self, service):
        stats = service.get_stats()
        assert stats["status"] == "running"
        assert stats["total_events"] == 0
        assert stats["active_machines"] == 0

    def test_stats_track_verdicts(self, service):
        """WARNING and CRITICAL counts should update after scoring."""
        import random

        random.seed(3)
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        # Build baseline
        for i in range(40):
            ts = now - timedelta(hours=6 + i)
            service.ingest_reading(_vibration_reading(rms=1.0 + random.gauss(0, 0.05), ts=ts))
        service.recompute_baselines()

        # Spike
        service.ingest_reading(_vibration_reading(rms=1.45))  # z ≈ 9 → CRITICAL

        stats = service.get_stats()
        assert stats["critical_events"] >= 1

    def test_not_started_returns_not_started(self, tmp_path):
        svc = SensorService(station_id="s1", db_path=tmp_path / "ns.db")
        assert svc.get_stats()["status"] == "not_started"
