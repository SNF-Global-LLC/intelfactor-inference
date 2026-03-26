"""
Tests for the 10 maintenance API endpoints added to api_v2.py.

Fixture pattern mirrors test_api_v2.py: create_app() with a real SensorService
backed by a tmp SQLite file and a real MaintenanceIQ instance.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

# ── Machine assets config (mirrors configs/station.yaml machine_health block) ──

MACHINE_HEALTH_CONFIG = {
    "enabled": True,
    "assets": [
        {
            "machine_id": "vacuum-chamber-01",
            "asset_type": "vacuum-chamber",
            "display_name": "Vacuum Chamber #1",
            "sensors": ["current"],
        },
        {
            "machine_id": "grinding-spindle-01",
            "asset_type": "grinding-spindle",
            "display_name": "Grinding Spindle #1",
            "sensors": ["vibration", "current"],
        },
        {
            "machine_id": "injection-moulder-01",
            "asset_type": "injection-moulder",
            "display_name": "Injection Moulder #1",
            "sensors": ["current", "vibration"],
        },
    ],
    "thresholds": {"warning": 2.0, "critical": 3.5},
    "sop_map": {
        "vibration": "SOP 4.2.3",
        "current": "SOP 4.3.1",
        "default_section": "SOP (machine health manual)",
    },
}


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def app(tmp_path):
    """
    Create a test Flask app with a real SensorService and MaintenanceIQ.

    Defect storage uses a fresh SQLite in tmp_path.
    Sensor storage uses a separate SQLite in tmp_path.
    """
    sensor_db = tmp_path / "sensors.db"
    defect_db = tmp_path / "defect.db"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()

    os.environ["STORAGE_MODE"] = "local"
    os.environ["SQLITE_DB_PATH"] = str(defect_db)
    os.environ["DB_PATH"] = str(defect_db)
    os.environ["EVIDENCE_DIR"] = str(evidence_dir)

    # Reset storage factory singletons so they pick up the tmp paths
    import packages.inference.storage.factory as factory
    factory._event_store = None
    factory._evidence_store = None
    factory._triple_store = None

    from packages.ingestion.sensor_service import SensorService
    from packages.policy.maintenance_iq import MaintenanceIQ
    from packages.inference.api_v2 import create_app

    svc = SensorService(station_id="test_station", db_path=sensor_db)
    svc.start()

    iq = MaintenanceIQ(
        sop_map=MACHINE_HEALTH_CONFIG["sop_map"],
        warning_threshold=2.0,
        critical_threshold=3.5,
    )

    flask_app = create_app(
        runtime=None,
        sensor_service=svc,
        maintenance_iq=iq,
        machine_health_config=MACHINE_HEALTH_CONFIG,
    )
    flask_app.config["TESTING"] = True

    yield flask_app

    svc.stop()
    os.environ.pop("SQLITE_DB_PATH", None)
    os.environ.pop("DB_PATH", None)
    os.environ.pop("EVIDENCE_DIR", None)


@pytest.fixture
def client(app):
    """Test client for the maintenance API app."""
    return app.test_client()


@pytest.fixture
def svc(app) -> "SensorService":  # type: ignore[name-defined]
    """Direct reference to the SensorService attached to the test app."""
    return app.sensor_service


def _make_reading(machine_id: str, sensor_type: str, raw_values: dict) -> dict:
    """Return a minimal sensor-event POST body."""
    return {
        "station_id": "test_station",
        "machine_id": machine_id,
        "sensor_type": sensor_type,
        "raw_values": raw_values,
    }


def _post_reading(client, machine_id: str, sensor_type: str, raw_values: dict):
    """Helper: POST a single sensor reading and return the response."""
    return client.post(
        "/api/maintenance/sensor-events",
        data=json.dumps(_make_reading(machine_id, sensor_type, raw_values)),
        content_type="application/json",
    )


# ── TestHealthEndpoint ──────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_all_three_machines(self, client):
        """Health endpoint lists every asset in MACHINE_HEALTH_CONFIG."""
        resp = client.get("/api/maintenance/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3
        machine_ids = {m["machine_id"] for m in data["machines"]}
        assert machine_ids == {
            "vacuum-chamber-01",
            "grinding-spindle-01",
            "injection-moulder-01",
        }

    def test_health_all_healthy_when_no_sensor_data(self, client):
        """With no sensor data, every machine reports HEALTHY and confidence=0."""
        resp = client.get("/api/maintenance/health")
        data = resp.get_json()
        for machine in data["machines"]:
            assert machine["verdict"] == "HEALTHY"
            assert machine["confidence"] == 0.0
            assert machine["last_reading_at"] is None

    def test_health_score_is_100_for_healthy_machine(self, client):
        """health_score field equals 100.0 when no anomaly detected."""
        resp = client.get("/api/maintenance/health")
        data = resp.get_json()
        for machine in data["machines"]:
            assert machine["health_score"] == 100.0


# ── TestSensorEventPost ─────────────────────────────────────────────────────


class TestSensorEventPost:
    def test_post_returns_202(self, client):
        resp = _post_reading(
            client, "grinding-spindle-01", "vibration", {"rms": 0.5}
        )
        assert resp.status_code == 202

    def test_post_returns_sensor_event_with_verdict(self, client):
        resp = _post_reading(
            client, "grinding-spindle-01", "vibration", {"rms": 0.5}
        )
        data = resp.get_json()
        assert data["status"] == "accepted"
        evt = data["event"]
        assert evt["machine_id"] == "grinding-spindle-01"
        assert evt["sensor_type"] == "vibration"
        assert "edge_verdict" in evt
        assert "anomaly_score" in evt
        assert "confidence" in evt

    def test_post_missing_required_field_returns_400(self, client):
        body = {"machine_id": "grinding-spindle-01", "sensor_type": "vibration"}
        resp = client.post(
            "/api/maintenance/sensor-events",
            data=json.dumps(body),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_post_unknown_sensor_type_returns_400(self, client):
        body = _make_reading("m1", "ultrasonic", {"db": 5.0})
        resp = client.post(
            "/api/maintenance/sensor-events",
            data=json.dumps(body),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ── TestBatchPost ───────────────────────────────────────────────────────────


class TestBatchPost:
    def test_batch_post_five_readings(self, client):
        readings = [
            _make_reading("grinding-spindle-01", "vibration", {"rms": float(i) * 0.1})
            for i in range(5)
        ]
        resp = client.post(
            "/api/maintenance/sensor-events/batch",
            data=json.dumps({"readings": readings}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["events_processed"] == 5

    def test_batch_returns_warning_and_critical_counts(self, client, svc):
        """After establishing a baseline, batch-injecting a spike shows it in summary."""
        from packages.ingestion.schemas import SensorReading, SensorType
        from datetime import datetime, timezone, timedelta

        # Seed historical baseline so scoring is active
        base_ts = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=6)
        for i in range(30):
            r = SensorReading(
                station_id="test_station",
                machine_id="grinding-spindle-01",
                sensor_type=SensorType.VIBRATION,
                raw_values={"rms": 1.0 + i * 0.01},
                timestamp=base_ts + timedelta(minutes=i),
            )
            svc.ingest_reading(r)
        svc.recompute_baselines()

        # Batch with one spike well above baseline (should be CRITICAL)
        readings = [
            _make_reading("grinding-spindle-01", "vibration", {"rms": 1.05}),
            _make_reading("grinding-spindle-01", "vibration", {"rms": 50.0}),  # spike
        ]
        resp = client.post(
            "/api/maintenance/sensor-events/batch",
            data=json.dumps({"readings": readings}),
            content_type="application/json",
        )
        assert resp.status_code == 202
        data = resp.get_json()
        assert data["events_processed"] == 2
        assert data["criticals"] >= 1

    def test_batch_exceeds_500_returns_400(self, client):
        readings = [
            _make_reading("m1", "vibration", {"rms": 0.5}) for _ in range(501)
        ]
        resp = client.post(
            "/api/maintenance/sensor-events/batch",
            data=json.dumps({"readings": readings}),
            content_type="application/json",
        )
        assert resp.status_code == 400


# ── TestEventsGet ───────────────────────────────────────────────────────────


class TestEventsGet:
    def test_list_events_empty(self, client):
        resp = client.get("/api/maintenance/events")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["events"] == []
        assert data["count"] == 0

    def test_list_events_filter_by_machine_id(self, client):
        _post_reading(client, "vacuum-chamber-01", "current", {"amps": 3.5})
        _post_reading(client, "grinding-spindle-01", "vibration", {"rms": 0.5})

        resp = client.get("/api/maintenance/events?machine_id=vacuum-chamber-01")
        data = resp.get_json()
        assert data["count"] == 1
        assert data["events"][0]["machine_id"] == "vacuum-chamber-01"

    def test_list_events_filter_by_sensor_type(self, client):
        _post_reading(client, "vacuum-chamber-01", "current", {"amps": 3.5})
        _post_reading(client, "grinding-spindle-01", "vibration", {"rms": 0.5})

        resp = client.get("/api/maintenance/events?sensor_type=current")
        data = resp.get_json()
        assert data["count"] == 1
        assert data["events"][0]["sensor_type"] == "current"

    def test_list_events_filter_by_verdict(self, client, svc):
        """After scoring, events with HEALTHY verdict appear in filtered list."""
        _post_reading(client, "grinding-spindle-01", "vibration", {"rms": 0.5})

        resp = client.get("/api/maintenance/events?verdict=HEALTHY")
        data = resp.get_json()
        assert data["count"] >= 1
        for evt in data["events"]:
            assert evt["edge_verdict"] == "HEALTHY"

    def test_list_events_unknown_sensor_type_returns_400(self, client):
        resp = client.get("/api/maintenance/events?sensor_type=ultrasonic")
        assert resp.status_code == 400


# ── TestGetEventById ────────────────────────────────────────────────────────


class TestGetEventById:
    def test_get_existing_event(self, client):
        post_resp = _post_reading(
            client, "vacuum-chamber-01", "current", {"amps": 4.2}
        )
        event_id = post_resp.get_json()["event"]["event_id"]

        resp = client.get(f"/api/maintenance/events/{event_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["event_id"] == event_id
        assert data["machine_id"] == "vacuum-chamber-01"

    def test_get_nonexistent_event_returns_404(self, client):
        resp = client.get("/api/maintenance/events/no-such-event")
        assert resp.status_code == 404


# ── TestBaselines ───────────────────────────────────────────────────────────


class TestBaselines:
    def test_baselines_empty_initially(self, client):
        resp = client.get("/api/maintenance/baselines")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["baselines"] == []

    def test_baselines_populated_after_enough_data(self, client, svc):
        from packages.ingestion.schemas import SensorReading, SensorType

        base_ts = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=6)
        for i in range(20):
            r = SensorReading(
                station_id="test_station",
                machine_id="grinding-spindle-01",
                sensor_type=SensorType.VIBRATION,
                raw_values={"rms": 1.0 + i * 0.01},
                timestamp=base_ts + timedelta(minutes=i),
            )
            svc.ingest_reading(r)
        svc.recompute_baselines()

        resp = client.get("/api/maintenance/baselines")
        data = resp.get_json()
        assert data["count"] >= 1
        profile = data["baselines"][0]
        assert "mean" in profile
        assert "std" in profile
        assert profile["machine_id"] == "grinding-spindle-01"


# ── TestIncidents ───────────────────────────────────────────────────────────


class TestIncidents:
    def test_incidents_empty_when_all_healthy(self, client):
        resp = client.get("/api/maintenance/incidents")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["incidents"] == []

    def test_incidents_groups_warning_events_by_machine(self, client, svc):
        """Seed WARNING events and confirm they appear grouped in incidents."""
        import sqlite3

        conn = svc._conn
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        for i in range(3):
            conn.execute(
                """INSERT OR IGNORE INTO sensor_events
                   (event_id, timestamp, station_id, machine_id, sensor_type,
                    raw_values, anomaly_score, confidence, edge_verdict)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"se_warn_{i}",
                    (now + timedelta(seconds=i)).isoformat(),
                    "test_station",
                    "grinding-spindle-01",
                    "vibration",
                    '{"rms": 5.0}',
                    2.5,
                    0.9,
                    "WARNING",
                ),
            )
        conn.commit()

        resp = client.get("/api/maintenance/incidents")
        data = resp.get_json()
        assert data["count"] >= 1
        incident = next(
            i for i in data["incidents"] if i["machine_id"] == "grinding-spindle-01"
        )
        assert incident["event_count"] == 3
        assert incident["severity"] == "WARNING"
        assert "vibration" in incident["contributing_factors"]


# ── TestRecommendations ─────────────────────────────────────────────────────


class TestRecommendations:
    def test_recommendations_empty_when_all_healthy(self, client):
        resp = client.get("/api/maintenance/recommendations")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["recommendations"] == []

    def test_recommendations_returned_for_warning_machine(self, client, svc):
        """Seed a WARNING-scored event, then check a recommendation is emitted."""
        import sqlite3
        from packages.ingestion.schemas import SensorReading, SensorType

        # Establish baseline so that the next reading scores as anomalous
        base_ts = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=6)
        for i in range(30):
            r = SensorReading(
                station_id="test_station",
                machine_id="grinding-spindle-01",
                sensor_type=SensorType.VIBRATION,
                raw_values={"rms": 1.0 + i * 0.01},
                timestamp=base_ts + timedelta(minutes=i),
            )
            svc.ingest_reading(r)
        svc.recompute_baselines()

        # Inject a spike — should score CRITICAL
        _post_reading(client, "grinding-spindle-01", "vibration", {"rms": 100.0})

        resp = client.get("/api/maintenance/recommendations")
        data = resp.get_json()
        assert data["count"] >= 1
        rec = data["recommendations"][0]
        assert rec["machine_id"] == "grinding-spindle-01"
        assert rec["action_type"] in ("service", "inspect")
        assert rec["action_en"]
        assert rec["action_zh"]
        assert rec["operator_action"] == "pending"


# ── TestFeedback ────────────────────────────────────────────────────────────


class TestFeedback:
    def test_feedback_updates_event(self, client, svc):
        post_resp = _post_reading(
            client, "vacuum-chamber-01", "current", {"amps": 4.0}
        )
        event_id = post_resp.get_json()["event"]["event_id"]

        resp = client.post(
            "/api/maintenance/feedback",
            data=json.dumps({
                "event_id": event_id,
                "operator_action": "confirm",
                "operator_id": "op_001",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "recorded"
        assert data["event_id"] == event_id

        # Verify the event was updated in the DB via the svc fixture
        event = svc.get_event(event_id)
        assert event["operator_action"] == "confirm"
        assert event["operator_id"] == "op_001"

    def test_feedback_invalid_action_returns_400(self, client):
        post_resp = _post_reading(
            client, "vacuum-chamber-01", "current", {"amps": 4.0}
        )
        event_id = post_resp.get_json()["event"]["event_id"]

        resp = client.post(
            "/api/maintenance/feedback",
            data=json.dumps({
                "event_id": event_id,
                "operator_action": "approved",  # not a valid action
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_feedback_nonexistent_event_returns_404(self, client):
        resp = client.post(
            "/api/maintenance/feedback",
            data=json.dumps({
                "event_id": "no-such-event",
                "operator_action": "confirm",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_feedback_rejection_reason_stored(self, client, svc):
        post_resp = _post_reading(
            client, "vacuum-chamber-01", "current", {"amps": 4.0}
        )
        event_id = post_resp.get_json()["event"]["event_id"]

        client.post(
            "/api/maintenance/feedback",
            data=json.dumps({
                "event_id": event_id,
                "operator_action": "reject",
                "rejection_reason": "False alarm — sensor miscalibrated",
            }),
            content_type="application/json",
        )

        event = svc.get_event(event_id)
        assert event["rejection_reason"] == "False alarm — sensor miscalibrated"


# ── TestStats ────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_initially_zero(self, client):
        resp = client.get("/api/maintenance/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_events"] == 0
        assert data["baseline_count"] == 0
        assert data["last_reading_at"] is None

    def test_stats_reflect_ingested_events(self, client):
        _post_reading(client, "vacuum-chamber-01", "current", {"amps": 3.5})
        _post_reading(client, "grinding-spindle-01", "vibration", {"rms": 0.5})

        resp = client.get("/api/maintenance/stats")
        data = resp.get_json()
        assert data["total_events"] == 2
        assert data["events_by_machine"].get("vacuum-chamber-01", 0) == 1
        assert data["events_by_machine"].get("grinding-spindle-01", 0) == 1
        assert data["events_by_sensor_type"].get("current", 0) == 1
        assert data["events_by_sensor_type"].get("vibration", 0) == 1
        assert data["last_reading_at"] is not None
