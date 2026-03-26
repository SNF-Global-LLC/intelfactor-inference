"""
Tests for the API v2 endpoints.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def app():
    """Create a test Flask app."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        evidence_dir = os.path.join(tmpdir, "evidence")
        os.makedirs(evidence_dir)

        # Set environment variables
        os.environ["STORAGE_MODE"] = "local"
        os.environ["SQLITE_DB_PATH"] = db_path
        os.environ["DB_PATH"] = db_path
        os.environ["EVIDENCE_DIR"] = evidence_dir

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
        response = client.get("/api/events")
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
        )
        assert response.status_code == 201
        assert response.get_json()["event_id"] == "evt_api_test_001"

        # Get event
        response = client.get("/api/events/evt_api_test_001")
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
            )

        # List with filter
        response = client.get("/api/events?verdict=FAIL&limit=10")
        assert response.status_code == 200
        data = response.get_json()
        assert len(data["events"]) == 3


class TestTripleEndpoints:
    def test_list_triples_empty(self, client):
        response = client.get("/api/triples")
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
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "recorded"

        # Verify update
        response = client.get("/api/triples/triple_feedback_001")
        assert response.status_code == 200
        data = response.get_json()
        assert data["operator_action"] == "accepted"


class TestApiKeyAuth:
    """API key authentication tests for mutating endpoints."""

    @pytest.fixture
    def authed_app(self):
        """Create a test app with STATION_API_KEY set."""
        with tempfile.TemporaryDirectory() as tmpdir:
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
            os.environ.pop("STATION_API_KEY", None)

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

    def test_get_events_open_without_key(self, authed_app):
        """GET endpoints remain open — operator dashboard needs unauthenticated read."""
        response = authed_app.get("/api/events")
        assert response.status_code == 200


class TestEvidenceEndpoints:
    def test_evidence_manifest_empty(self, client):
        response = client.get("/api/v1/evidence/manifest?date=2026-02-13")
        assert response.status_code == 200
        data = response.get_json()
        assert data["date"] == "2026-02-13"
        assert data["entries"] == []

    def test_evidence_metadata_not_found(self, client):
        response = client.get("/api/v1/evidence/nonexistent")
        assert response.status_code == 404

    def test_evidence_image_not_found(self, client):
        response = client.get("/api/v1/evidence/nonexistent/image.jpg")
        assert response.status_code == 404

    def test_evidence_stats(self, client):
        response = client.get("/api/evidence/stats")
        assert response.status_code == 200
        data = response.get_json()
        assert "total_bytes" in data
        assert "date_dirs" in data
