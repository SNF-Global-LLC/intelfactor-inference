"""
Tests for edge inspection sync durability.
"""

import time
from datetime import datetime, timedelta

from packages.inference.schemas import InspectionEvent, SyncStatus, Verdict
from packages.inference.storage.inspection_store import InspectionStore
from packages.inference.sync_inspections import InspectionSyncWorker


def _event(inspection_id: str, workspace_id: str = "ws_a", image_path: str = "") -> InspectionEvent:
    return InspectionEvent(
        inspection_id=inspection_id,
        timestamp=datetime.now(),
        station_id="station_01",
        workspace_id=workspace_id,
        decision=Verdict.PASS,
        image_original_path=image_path,
    )


def test_inspection_save_is_idempotent(tmp_path):
    store = InspectionStore(tmp_path / "inspections.db")

    store.save(_event("insp_dup", workspace_id="ws_a"))
    replacement = _event("insp_dup", workspace_id="ws_a")
    replacement.decision = Verdict.FAIL
    store.save(replacement)

    rows = store.list_inspections(limit=10)
    assert len(rows) == 1
    assert rows[0].inspection_id == "insp_dup"
    assert rows[0].decision == Verdict.FAIL


def test_failed_sync_remains_retryable_with_error_and_attempt_time(tmp_path):
    store = InspectionStore(tmp_path / "inspections.db")
    store.save(_event("insp_retry"))

    worker = InspectionSyncWorker(store, tmp_path / "evidence", cloud_api_url="https://cloud.example")

    def fail_sync(_event):
        raise RuntimeError("cloud offline")

    worker._sync_one = fail_sync
    worker._sync_cycle()

    saved = store.get("insp_retry")
    assert saved.sync_status == SyncStatus.FAILED
    assert saved.sync_error == "cloud offline"
    assert saved.last_attempt_at is not None
    assert store.get_pending_sync(limit=10)[0].inspection_id == "insp_retry"


def test_missing_expected_evidence_fails_before_metadata_post(tmp_path):
    store = InspectionStore(tmp_path / "inspections.db")
    store.save(_event("insp_missing_evidence", image_path="2026-05-10/missing.jpg"))
    worker = InspectionSyncWorker(store, tmp_path / "evidence", cloud_api_url="https://cloud.example")

    worker._get_upload_urls = lambda _event, _headers: {"original_url": "https://s3.example/object?sig=1"}
    worker._upload_file = lambda *_args, **_kwargs: "should-not-run"

    worker._sync_cycle()

    saved = store.get("insp_missing_evidence")
    assert saved.sync_status == SyncStatus.FAILED
    assert "original evidence not found" in saved.sync_error
    assert saved.synced_at is None


def test_sync_payload_prefers_evidence_object_keys_over_public_urls(tmp_path):
    evidence_dir = tmp_path / "evidence"
    image_path = evidence_dir / "2026-05-10" / "insp_keyed.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake jpeg")

    store = InspectionStore(tmp_path / "inspections.db")
    store.save(_event("insp_keyed", image_path="2026-05-10/insp_keyed.jpg"))
    worker = InspectionSyncWorker(store, evidence_dir, cloud_api_url="https://cloud.example")

    worker._get_upload_urls = lambda _event, _headers: {
        "original_url": "https://s3.example/public-object?sig=1",
        "original_key": "evidence/ws_a/insp_keyed/original.jpg",
    }
    worker._upload_file = lambda *_args, **_kwargs: "https://s3.example/public-object"

    posts = []

    class FakeResponse:
        is_success = True
        status_code = 201
        text = ""

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, json, headers):
            posts.append({"url": url, "json": json, "headers": headers})
            return FakeResponse()

    import httpx

    original_client = httpx.Client
    httpx.Client = FakeClient
    try:
        worker._sync_cycle()
    finally:
        httpx.Client = original_client

    assert posts[0]["json"]["image_original_url"] == "evidence/ws_a/insp_keyed/original.jpg"
    assert not posts[0]["json"]["image_original_url"].startswith("https://")
    assert store.get("insp_keyed").image_original_url == "evidence/ws_a/insp_keyed/original.jpg"
    assert image_path.exists()


def test_sync_fails_when_upload_response_omits_object_key(tmp_path):
    evidence_dir = tmp_path / "evidence"
    image_path = evidence_dir / "2026-05-10" / "insp_no_key.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake jpeg")

    store = InspectionStore(tmp_path / "inspections.db")
    store.save(_event("insp_no_key", image_path="2026-05-10/insp_no_key.jpg"))
    worker = InspectionSyncWorker(store, evidence_dir, cloud_api_url="https://cloud.example")

    metadata_posts = {"count": 0}
    worker._get_upload_urls = lambda _event, _headers: {
        "original_url": "https://s3.example/public-object?sig=1",
    }
    worker._upload_file = lambda *_args, **_kwargs: "https://s3.example/public-object"
    worker._build_payload = lambda *_args, **_kwargs: metadata_posts.__setitem__("count", 1)

    worker._sync_cycle()

    saved = store.get("insp_no_key")
    assert saved.sync_status == SyncStatus.FAILED
    assert "original_key missing" in saved.sync_error
    assert metadata_posts["count"] == 0
    assert saved.image_original_url == ""
    assert image_path.exists()


def test_sync_worker_sends_edge_auth_and_workspace_headers(tmp_path):
    store = InspectionStore(tmp_path / "inspections.db")
    event = _event("insp_headers", workspace_id="ws_headers")
    worker = InspectionSyncWorker(
        store,
        tmp_path / "evidence",
        cloud_api_url="https://cloud.example",
        cloud_api_key="edge-secret",
    )

    headers = worker._auth_headers(event)

    assert headers["X-Edge-Api-Key"] == "edge-secret"
    assert headers["Authorization"] == "Bearer edge-secret"
    assert headers["X-Workspace-Id"] == "ws_headers"


def test_synced_inspection_is_not_reposted_on_duplicate_cycle(tmp_path):
    evidence_dir = tmp_path / "evidence"
    image_path = evidence_dir / "2026-05-10" / "insp_once.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake jpeg")

    store = InspectionStore(tmp_path / "inspections.db")
    store.save(_event("insp_once", image_path="2026-05-10/insp_once.jpg"))
    worker = InspectionSyncWorker(store, evidence_dir, cloud_api_url="https://cloud.example")

    calls = {"count": 0}

    def sync_success(event):
        calls["count"] += 1
        store.update_sync_status(
            event.inspection_id,
            SyncStatus.SYNCED,
            image_original_url="https://s3.example/insp_once.jpg",
        )

    worker._sync_one = sync_success
    worker._sync_cycle()
    worker._sync_cycle()

    assert calls["count"] == 1
    assert store.get("insp_once").sync_status == SyncStatus.SYNCED
    assert image_path.exists()


def test_ready_inspections_are_not_starved_by_backoff_rows(tmp_path):
    store = InspectionStore(tmp_path / "inspections.db")
    base = datetime.now() - timedelta(minutes=10)
    backoff_ids = []

    for index in range(3):
        inspection_id = f"insp_backoff_{index}"
        event = _event(inspection_id)
        event.timestamp = base + timedelta(seconds=index)
        store.save(event)
        backoff_ids.append(inspection_id)

    ready = _event("insp_ready")
    ready.timestamp = base + timedelta(seconds=10)
    store.save(ready)

    worker = InspectionSyncWorker(
        store,
        tmp_path / "evidence",
        cloud_api_url="https://cloud.example",
        batch_size=3,
    )
    worker._next_retry_at = {inspection_id: time.time() + 60 for inspection_id in backoff_ids}
    worker._retry_counts = {inspection_id: 1 for inspection_id in backoff_ids}

    calls = []

    def sync_success(event):
        calls.append(event.inspection_id)
        store.update_sync_status(event.inspection_id, SyncStatus.SYNCED)

    worker._sync_one = sync_success
    worker._sync_cycle()

    assert calls == ["insp_ready"]
    assert store.get("insp_ready").sync_status == SyncStatus.SYNCED
    assert all(store.get(inspection_id).sync_status == SyncStatus.PENDING for inspection_id in backoff_ids)
