from datetime import datetime
from pathlib import Path


AUTH_HEADERS = {"X-Edge-Api-Key": "test-secret-key"}


def _reset_storage_singletons():
    import packages.inference.storage.factory as factory

    factory._event_store = None
    factory._evidence_store = None
    factory._triple_store = None


def test_inspect_page_is_edge_operator_console(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("STATION_API_KEY", "test-secret-key")
    (tmp_path / "evidence").mkdir()
    _reset_storage_singletons()

    from packages.inference.api_v2 import create_app

    app = create_app(runtime=None)
    app.config["TESTING"] = True

    response = app.test_client().get("/inspect")
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Live Inspection" in html
    assert "Review Queue" in html
    assert "Inspection History" in html
    assert "Local System Status" in html
    assert "Station Config" in html
    assert "Confirm defect" in html
    assert "Override to pass" in html
    assert 'id="filterDate"' in html
    assert 'id="filterDefect"' in html
    assert 'id="filterBatch"' in html
    assert 'id="filterStation"' in html
    assert "SQLite healthy" in html
    assert "Sync queue" in html
    assert "edge-detail-image" in html
    assert "edge-action-trail" in html
    assert "approve" not in html.lower()
    assert "reject" not in html.lower()


def test_inspection_feedback_supports_operator_action_names(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "events.db"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("STATION_API_KEY", "test-secret-key")
    (tmp_path / "evidence").mkdir()
    _reset_storage_singletons()

    from packages.inference.api_v2 import create_app
    from packages.inference.schemas import InspectionEvent, Verdict
    from packages.inference.storage.inspection_store import InspectionStore

    store = InspectionStore(tmp_path / "inspections.db")
    store.save(
        InspectionEvent(
            inspection_id="insp_actions_001",
            timestamp=datetime.now(),
            station_id="station_edge",
            decision=Verdict.FAIL,
            confidence=0.92,
        )
    )
    runtime = type("Runtime", (), {"_inspection_store": store})()
    app = create_app(runtime=runtime)
    app.config["TESTING"] = True
    client = app.test_client()

    response = client.post(
        "/api/inspect/insp_actions_001/feedback",
        json={"action": "confirm_defect", "operator_id": "op_1"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    assert store.get("insp_actions_001").accepted is True
    actions = store.list_operator_actions("insp_actions_001")
    assert len(actions) == 1
    assert actions[0]["action"] == "confirm_defect"
    assert actions[0]["operator_id"] == "op_1"

    response = client.post(
        "/api/inspect/insp_actions_001/feedback",
        json={"action": "override_to_pass", "operator_id": "op_1"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    event = store.get("insp_actions_001")
    assert event.accepted is False
    assert event.rejection_reason == "override_to_pass"
    actions = store.list_operator_actions("insp_actions_001")
    assert [row["action"] for row in actions] == ["confirm_defect", "override_to_pass"]

    response = client.get("/api/inspections/insp_actions_001", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.get_json()
    assert [row["action"] for row in data["operator_actions"]] == [
        "confirm_defect",
        "override_to_pass",
    ]


def test_inspect_page_does_not_store_inspections_or_actions_in_browser(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_MODE", "local")
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("STATION_API_KEY", "test-secret-key")
    (tmp_path / "evidence").mkdir()
    _reset_storage_singletons()

    from packages.inference.api_v2 import create_app

    app = create_app(runtime=None)
    app.config["TESTING"] = True

    html = app.test_client().get("/inspect").get_data(as_text=True)

    assert "localStorage.setItem('edgeApiKey'" in html
    assert "localStorage.setItem('inspections'" not in html
    assert "localStorage.setItem('operator_actions'" not in html
    assert "localStorage.setItem('operatorActions'" not in html


def test_seed_operator_console_creates_local_rows_and_evidence(tmp_path):
    from packages.inference.storage.inspection_store import InspectionStore
    from scripts.seed_operator_console import seed_operator_console

    db_path = tmp_path / "operator-console.db"
    evidence_dir = tmp_path / "evidence"

    first = seed_operator_console(
        db_path=db_path,
        evidence_dir=evidence_dir,
        station_id="station_seed",
    )
    second = seed_operator_console(
        db_path=db_path,
        evidence_dir=evidence_dir,
        station_id="station_seed",
    )

    store = InspectionStore(db_path)
    rows = store.list_inspections(limit=10)
    decisions = {row.decision.value for row in rows}

    assert len(first) == 3
    assert len(second) == 3
    assert len(rows) == 3
    assert decisions == {"PASS", "REVIEW", "FAIL"}
    assert store.get("insp_seed_review_001").product_id == "batch-A17-part-0002"
    assert (evidence_dir / "operator-console-seed" / "insp_seed_review_001_original.jpg").exists()
    assert (evidence_dir / "operator-console-seed" / "insp_seed_defect_001_annotated.jpg").exists()


def test_operator_console_docs_link_seed_and_jetson_validation():
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text()
    validation_doc = (repo_root / "docs" / "JETSON_OPERATOR_CONSOLE_VALIDATION.md").read_text()

    assert "scripts/seed_operator_console.py" in readme
    assert "docs/JETSON_OPERATOR_CONSOLE_VALIDATION.md" in readme
    assert "FLIR" in validation_doc
    assert "TensorRT" in validation_doc
    assert "SQLite Persistence" in validation_doc
    assert "confirm_defect" in validation_doc
    assert "override_to_pass" in validation_doc
