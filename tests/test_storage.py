"""
Tests for the storage abstraction layer.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        evidence_dir = os.path.join(tmpdir, "evidence")
        os.makedirs(evidence_dir)

        # Set environment variables
        os.environ["STORAGE_MODE"] = "local"
        os.environ["SQLITE_DB_PATH"] = db_path
        os.environ["EVIDENCE_DIR"] = evidence_dir

        yield {"db_path": db_path, "evidence_dir": evidence_dir}

        # Cleanup environment
        os.environ.pop("SQLITE_DB_PATH", None)
        os.environ.pop("EVIDENCE_DIR", None)


class TestEventStore:
    def test_insert_and_get(self, temp_db):
        # Reset singleton
        import packages.inference.storage.factory as factory
        factory._event_store = None

        from packages.inference.storage import get_event_store

        store = get_event_store()

        event = {
            "event_id": "evt_test_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "defect_type": "scratch_surface",
            "severity": 0.8,
            "confidence": 0.95,
            "verdict": "FAIL",
            "shift": "A",
            "sku": "WKS-123",
        }

        # Insert
        event_id = store.insert(event)
        assert event_id == "evt_test_001"

        # Get
        retrieved = store.get("evt_test_001")
        assert retrieved is not None
        assert retrieved["event_id"] == "evt_test_001"
        assert retrieved["defect_type"] == "scratch_surface"
        assert retrieved["verdict"] == "FAIL"

    def test_list_events(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._event_store = None

        from packages.inference.storage import get_event_store

        store = get_event_store()

        # Insert multiple events
        for i in range(5):
            store.insert({
                "event_id": f"evt_list_{i}",
                "timestamp": datetime.now().isoformat(),
                "station_id": "station_01",
                "verdict": "FAIL" if i % 2 == 0 else "REVIEW",
            })

        # List all
        events = store.list(limit=10)
        assert len(events) == 5

        # Filter by verdict
        fail_events = store.list(verdict="FAIL")
        assert len(fail_events) == 3

    def test_count_events(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._event_store = None

        from packages.inference.storage import get_event_store

        store = get_event_store()

        for i in range(3):
            store.insert({
                "event_id": f"evt_count_{i}",
                "timestamp": datetime.now().isoformat(),
                "station_id": "station_01",
                "verdict": "FAIL",
            })

        count = store.count()
        assert count == 3


class TestTripleStore:
    def test_insert_and_get(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._triple_store = None

        from packages.inference.storage import get_triple_store

        store = get_triple_store()

        triple = {
            "triple_id": "triple_test_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "defect_event_id": "evt_001",
            "defect_type": "scratch_surface",
            "cause_parameter": "grinding_rpm",
            "cause_value": 3100,
            "cause_target": 3000,
            "cause_drift_pct": 3.3,
            "status": "pending",
        }

        # Insert
        triple_id = store.insert(triple)
        assert triple_id == "triple_test_001"

        # Get
        retrieved = store.get("triple_test_001")
        assert retrieved is not None
        assert retrieved["defect_type"] == "scratch_surface"
        assert retrieved["status"] == "pending"

    def test_update_triple(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._triple_store = None

        from packages.inference.storage import get_triple_store

        store = get_triple_store()

        store.insert({
            "triple_id": "triple_update_001",
            "timestamp": datetime.now().isoformat(),
            "station_id": "station_01",
            "status": "pending",
        })

        # Update
        updated = store.update("triple_update_001", {
            "operator_action": "accepted",
            "status": "verified",
        })
        assert updated is True

        # Verify
        retrieved = store.get("triple_update_001")
        assert retrieved["operator_action"] == "accepted"
        assert retrieved["status"] == "verified"


class TestEvidenceStore:
    def test_index_and_get(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._evidence_store = None

        from packages.inference.storage import get_evidence_store

        store = get_evidence_store()

        # Create test evidence file
        evidence_dir = Path(temp_db["evidence_dir"])
        date_dir = evidence_dir / "2026-02-13"
        date_dir.mkdir(parents=True, exist_ok=True)

        test_jpg = date_dir / "evt_evidence_001.jpg"
        test_jpg.write_bytes(b"fake jpeg data")

        test_json = date_dir / "evt_evidence_001.json"
        test_json.write_text('{"event_id": "evt_evidence_001", "defect": "scratch"}')

        # Index
        store.index("evt_evidence_001", {
            "date_dir": "2026-02-13",
            "image_path": "2026-02-13/evt_evidence_001.jpg",
            "file_size_bytes": len(b"fake jpeg data"),
        })

        # Get metadata
        meta = store.get_metadata("evt_evidence_001")
        assert meta is not None
        assert meta["date_dir"] == "2026-02-13"

        # Get image path
        img_path = store.get_image_path("evt_evidence_001")
        assert img_path is not None
        assert img_path.exists()

    def test_list_by_date(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._evidence_store = None

        from packages.inference.storage import get_evidence_store

        store = get_evidence_store()

        # Create test evidence
        evidence_dir = Path(temp_db["evidence_dir"])
        date_dir = evidence_dir / "2026-02-14"
        date_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            (date_dir / f"evt_date_{i}.json").write_text(
                f'{{"event_id": "evt_date_{i}", "index": {i}}}'
            )

        # List
        entries = store.list_by_date("2026-02-14")
        assert len(entries) == 3


class TestStorageFactory:
    def test_storage_mode(self, temp_db):
        from packages.inference.storage import get_storage_mode
        assert get_storage_mode() == "local"

    def test_singleton_behavior(self, temp_db):
        import packages.inference.storage.factory as factory
        factory._event_store = None

        from packages.inference.storage import get_event_store

        store1 = get_event_store()
        store2 = get_event_store()
        assert store1 is store2
