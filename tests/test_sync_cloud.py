"""
Tests for hybrid cloud sync failure handling.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import requests

from packages.inference.sync_cloud import CloudSyncAgent


def _create_agent(tmp_path: Path) -> CloudSyncAgent:
    return CloudSyncAgent(
        station_id="factory-line01-station01",
        sqlite_db_path=str(tmp_path / "local.db"),
        evidence_dir=str(tmp_path / "evidence"),
        cloud_api_url="https://api.intelfactor.ai",
        cloud_api_key="ifk_test",
        sync_interval_sec=300,
        batch_size=100,
    )


def _seed_event(db_path: Path, timestamp: str) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE defect_events (
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
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE causal_triples (
            triple_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            station_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO defect_events
            (event_id, timestamp, station_id, verdict)
        VALUES (?, ?, ?, ?)
        """,
        ("evt_001", timestamp, "factory-line01-station01", "FAIL"),
    )
    conn.commit()
    conn.close()


def test_api_retry_failure_does_not_advance_event_watermark(tmp_path, monkeypatch):
    agent = _create_agent(tmp_path)
    timestamp = datetime.now().isoformat()
    _seed_event(agent.db_path, timestamp)

    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("cloud offline")

    monkeypatch.setattr("packages.inference.sync_cloud.requests.post", raise_connection_error)
    monkeypatch.setattr("packages.inference.sync_cloud.time.sleep", lambda *_args, **_kwargs: None)

    agent._sync_cycle()

    assert "events" not in agent._watermarks
    assert not (agent.db_path.parent / f"{agent.station_id}_watermarks.json").exists()


def test_event_watermark_advances_after_successful_upload(tmp_path, monkeypatch):
    agent = _create_agent(tmp_path)
    timestamp = datetime.now().isoformat()
    _seed_event(agent.db_path, timestamp)

    class Response:
        def raise_for_status(self):
            return None

    monkeypatch.setattr("packages.inference.sync_cloud.requests.post", lambda *args, **kwargs: Response())

    agent._sync_cycle()

    assert agent._watermarks["events"] == timestamp


def test_evidence_watermark_pauses_on_failed_upload(tmp_path):
    agent = _create_agent(tmp_path)
    date_dir = agent.evidence_dir / "2026-05-05"
    date_dir.mkdir(parents=True)
    first = date_dir / "evt_001.jpg"
    second = date_dir / "evt_002.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    class FakeS3:
        def __init__(self):
            self.calls: list[str] = []

        def upload_file(self, filename, bucket, key, ExtraArgs=None):
            self.calls.append(Path(filename).name)
            if filename.endswith("evt_002.jpg"):
                raise RuntimeError("s3 offline")

    fake_s3 = FakeS3()
    agent._s3_client = fake_s3
    agent.s3_bucket = "intelfactor-evidence-prod"

    agent._sync_evidence()

    assert fake_s3.calls == ["evt_001.jpg", "evt_002.jpg"]
    assert agent._watermarks["evidence_file"] == "2026-05-05/evt_001.jpg"
    assert second.exists()
