"""
Tests for the API v2 endpoints.
"""

import json
import os
import tempfile
from datetime import datetime

import pytest

AUTH_HEADERS = {"X-Edge-Api-Key": "test-secret-key"}


@pytest.fixture
def app():
    """Create a test Flask app."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_station_api_key = os.environ.get("STATION_API_KEY")
        db_path = os.path.join(tmpdir, "test.db")
        evidence_dir = os.path.join(tmpdir, "evidence")
        os.makedirs(evidence_dir)

        # Set environment variables
        os.environ["STORAGE_MODE"] = "local"
        os.environ["SQLITE_DB_PATH"] = db_path
        os.environ["DB_PATH"] = db_path
        os.environ["EVIDENCE_DIR"] = evidence_dir
        os.environ["STATION_API_KEY"] = "test-secret-key"

        # Reset singletons
        import packages.inference.storage.factory as factory
        factory._event_store = None
        factory._evidence_store = None
        factory._triple_store = None

        from packages.inference.api_v2 import create_app
        app = create_app(runtime=None)
        app.config["TESTING"] = True

        yield app

        # Cleanup
        os.environ.pop("SQLITE_DB_PATH", None)
        os.environ.pop("DB_PATH", None)
        os.environ.pop("EVIDENCE_DIR", None)
        if old_station_api_key is None:
            os.environ.pop("STATION_API_KEY", None)
        else:
            os.environ["STATION_API_KEY"] = old_station_api_key


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert "station" in data


class TestEventEndpoints:
    def test_list_events_empty(self, client):
        response = client.get("/api/events", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert data["events"] == []
        assert data["count"] == 0

    def test_create_and_get_event(self, client):
        # Create event
        event = {
            "event_id": "evt_api_test_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "defect_type": "scratch_surface",
            "severity": 0.8,
            "confidence": 0.95,
            "verdict": "FAIL",
        }
        response = client.post(
            "/api/events",
            data=json.dumps(event),
            content_type="application/json",
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 201
        assert response.get_json()["event_id"] == "evt_api_test_001"

        # Get event
        response = client.get("/api/events/evt_api_test_001", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert data["event_id"] == "evt_api_test_001"
        assert data["defect_type"] == "scratch_surface"

    def test_list_events_with_filter(self, client):
        # Create events
        for i in range(3):
            client.post(
                "/api/events",
                data=json.dumps({
                    "event_id": f"evt_filter_{i}",
                    "timestamp": datetime.now().isoformat(),
                    "station_id": "station_01",
                    "verdict": "FAIL",
                }),
                content_type="application/json",
                headers=AUTH_HEADERS,
            )

        # List with filter
        response = client.get("/api/events?verdict=FAIL&limit=10", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert len(data["events"]) == 3


class TestTripleEndpoints:
    def test_list_triples_empty(self, client):
        response = client.get("/api/triples", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert data["triples"] == []

    def test_feedback_workflow(self, client):
        # First insert a triple via storage layer
        from packages.inference.storage import get_triple_store
        store = get_triple_store()
        store.insert({
            "triple_id": "triple_feedback_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "defect_type": "scratch",
            "status": "pending",
        })

        # Record feedback via API
        response = client.post(
            "/api/feedback",
            data=json.dumps({
                "triple_id": "triple_feedback_001",
                "action": "accepted",
                "operator_id": "op_001",
            }),
            content_type="application/json",
            headers=AUTH_HEADERS,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "recorded"

        # Verify update
        response = client.get("/api/triples/triple_feedback_001", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert data["operator_action"] == "accepted"


class TestApiKeyAuth:
    """API key authentication tests for mutating endpoints."""

    @pytest.fixture
    def authed_app(self):
        """Create a test app with STATION_API_KEY set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_station_api_key = os.environ.get("STATION_API_KEY")
            db_path = os.path.join(tmpdir, "auth_test.db")
            evidence_dir = os.path.join(tmpdir, "evidence")
            os.makedirs(evidence_dir)

            os.environ["STORAGE_MODE"] = "local"
            os.environ["SQLITE_DB_PATH"] = db_path
            os.environ["DB_PATH"] = db_path
            os.environ["EVIDENCE_DIR"] = evidence_dir
            os.environ["STATION_API_KEY"] = "test-secret-key"

            import packages.inference.storage.factory as factory
            factory._event_store = None
            factory._evidence_store = None
            factory._triple_store = None

            from packages.inference.api_v2 import create_app
            app = create_app(runtime=None)
            app.config["TESTING"] = True

            yield app.test_client()

            os.environ.pop("SQLITE_DB_PATH", None)
            os.environ.pop("DB_PATH", None)
            os.environ.pop("EVIDENCE_DIR", None)
            if old_station_api_key is None:
                os.environ.pop("STATION_API_KEY", None)
            else:
                os.environ["STATION_API_KEY"] = old_station_api_key

    def test_health_always_open(self, authed_app):
        """GET /health requires no key."""
        response = authed_app.get("/health")
        assert response.status_code == 200

    def test_post_without_key_rejected(self, authed_app):
        response = authed_app.post(
            "/api/events",
            data=json.dumps({"event_id": "x"}),
            content_type="application/json",
        )
        assert response.status_code == 401
        assert response.get_json()["error"] == "Unauthorized"

    def test_post_wrong_key_rejected(self, authed_app):
        response = authed_app.post(
            "/api/events",
            data=json.dumps({"event_id": "x"}),
            content_type="application/json",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401

    def test_post_correct_x_api_key_accepted(self, authed_app):
        event = {
            "event_id": "evt_auth_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "verdict": "PASS",
        }
        response = authed_app.post(
            "/api/events",
            data=json.dumps(event),
            content_type="application/json",
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 201

    def test_post_correct_x_edge_api_key_accepted(self, authed_app):
        event = {
            "event_id": "evt_auth_edge_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "verdict": "PASS",
        }
        response = authed_app.post(
            "/api/events",
            data=json.dumps(event),
            content_type="application/json",
            headers={"X-Edge-Api-Key": "test-secret-key"},
        )
        assert response.status_code == 201

    def test_post_correct_bearer_token_accepted(self, authed_app):
        event = {
            "event_id": "evt_auth_002",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "verdict": "PASS",
        }
        response = authed_app.post(
            "/api/events",
            data=json.dumps(event),
            content_type="application/json",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert response.status_code == 201

    def test_get_events_without_key_rejected(self, authed_app):
        """GET API endpoints require edge auth."""
        response = authed_app.get("/api/events")
        assert response.status_code == 401


class TestEvidenceEndpoints:
    def test_evidence_manifest_empty(self, client):
        response = client.get("/api/v1/evidence/manifest?date=2026-02-13", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert data["date"] == "2026-02-13"
        assert data["entries"] == []

    def test_evidence_metadata_not_found(self, client):
        response = client.get("/api/v1/evidence/nonexistent", headers=AUTH_HEADERS)
        assert response.status_code == 404

    def test_evidence_image_not_found(self, client):
        response = client.get("/api/v1/evidence/nonexistent/image.jpg", headers=AUTH_HEADERS)
        assert response.status_code == 404

    def test_evidence_stats(self, client):
        response = client.get("/api/evidence/stats", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert "total_bytes" in data
        assert "date_dirs" in data


class TestApiFailClosed:
    def test_api_rejects_when_key_not_configured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STORAGE_MODE", "local")
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
        monkeypatch.delenv("STATION_API_KEY", raising=False)
        monkeypatch.delenv("EDGE_API_KEY", raising=False)
        (tmp_path / "evidence").mkdir()

        import packages.inference.storage.factory as factory
        factory._event_store = None
        factory._evidence_store = None
        factory._triple_store = None

        from packages.inference.api_v2 import create_app
        app = create_app(runtime=None)
        app.config["TESTING"] = True

        assert app.test_client().get("/health").status_code == 200
        response = app.test_client().get("/api/status")
        assert response.status_code == 503


class TestInspectionWorkspaceIsolation:
    @pytest.fixture
    def isolated_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STORAGE_MODE", "local")
        monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
        monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
        monkeypatch.setenv("STATION_API_KEY", "test-secret-key")
        monkeypatch.setenv("WORKSPACE_ID", "ws_a")
        (tmp_path / "evidence").mkdir()

        from packages.inference.schemas import InspectionEvent, Verdict
        from packages.inference.storage.inspection_store import InspectionStore

        store = InspectionStore(tmp_path / "inspections.db")
        store.save(InspectionEvent(
            inspection_id="insp_ws_a",
            timestamp=datetime.now(),
            station_id="station_01",
            workspace_id="ws_a",
            decision=Verdict.PASS,
            image_original_path="2026-05-10/insp_ws_a.jpg",
            image_original_url="https://public-bucket.example/insp_ws_a.jpg",
        ))
        store.save(InspectionEvent(
            inspection_id="insp_ws_b",
            timestamp=datetime.now(),
            station_id="station_01",
            workspace_id="ws_b",
            decision=Verdict.FAIL,
        ))
        store.save(InspectionEvent(
            inspection_id="insp_unscoped",
            timestamp=datetime.now(),
            station_id="station_01",
            workspace_id="",
            decision=Verdict.PASS,
        ))

        runtime = type("Runtime", (), {"_inspection_store": store})()

        from packages.inference.api_v2 import create_app
        app = create_app(runtime=runtime)
        app.config["TESTING"] = True
        return app.test_client()

    def test_list_inspections_scoped_to_authenticated_workspace(self, isolated_client):
        response = isolated_client.get("/api/inspections", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert [row["inspection_id"] for row in data["inspections"]] == ["insp_ws_a"]

    def test_cross_workspace_inspection_read_returns_403(self, isolated_client):
        response = isolated_client.get("/api/inspections/insp_ws_b", headers=AUTH_HEADERS)
        assert response.status_code == 403

    def test_unscoped_inspection_read_returns_403_when_workspace_is_configured(self, isolated_client):
        response = isolated_client.get("/api/inspections/insp_unscoped", headers=AUTH_HEADERS)
        assert response.status_code == 403

    def test_inspection_detail_does_not_return_public_evidence_url(self, isolated_client):
        response = isolated_client.get("/api/inspections/insp_ws_a", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.get_json()
        assert data["image_original_url"] == "/api/inspections/insp_ws_a/original.jpg"
        assert "public-bucket.example" not in json.dumps(data)
