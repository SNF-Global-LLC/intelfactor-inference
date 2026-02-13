"""
Tests for evidence writer and doctor diagnostics.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ━━━ Evidence Writer Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEvidenceWriter:
    """Test JPEG ring buffer + metadata writer."""

    def test_write_and_retrieve(self, tmp_path):
        """Basic write + read back."""
        from packages.inference.evidence import EvidenceWriter

        writer = EvidenceWriter(evidence_dir=str(tmp_path / "evidence"), max_disk_gb=1.0)

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ref = writer.write(
            frame=frame,
            event_id="evt_test_001",
            metadata={"station_id": "station_1", "verdict": "FAIL"},
        )

        assert ref != ""
        assert "evt_test_001.jpg" in ref

        # Retrieve
        found = writer.get_frame_path("evt_test_001")
        assert found is not None
        assert found.exists()
        assert found.stat().st_size > 0

    def test_metadata_sidecar(self, tmp_path):
        """Verify JSON sidecar written alongside JPEG."""
        from packages.inference.evidence import EvidenceWriter

        writer = EvidenceWriter(evidence_dir=str(tmp_path / "evidence"))

        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        writer.write(
            frame=frame,
            event_id="evt_meta_001",
            metadata={"station_id": "s1", "defect_type": "scratch_surface"},
        )

        meta = writer.get_metadata("evt_meta_001")
        assert meta is not None
        assert meta["event_id"] == "evt_meta_001"
        assert meta["station_id"] == "s1"
        assert "timestamp" in meta
        assert "jpeg_size_bytes" in meta

    def test_bbox_annotation(self, tmp_path):
        """Verify bounding box drawing doesn't crash."""
        from packages.inference.evidence import EvidenceWriter

        writer = EvidenceWriter(evidence_dir=str(tmp_path / "evidence"))
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        bbox_list = [
            {"x": 10, "y": 20, "width": 100, "height": 80, "label": "scratch", "confidence": 0.92, "severity": 0.7},
            {"x": 200, "y": 100, "width": 50, "height": 50, "label": "burr", "confidence": 0.85, "severity": 0.3},
        ]

        ref = writer.write(frame=frame, event_id="evt_bbox_001", bbox_list=bbox_list)
        assert ref != ""

        # Annotated frame should be larger than blank due to drawing
        found = writer.get_frame_path("evt_bbox_001")
        assert found is not None

    def test_date_partitioning(self, tmp_path):
        """Verify evidence stored in date-partitioned directories."""
        from packages.inference.evidence import EvidenceWriter
        from datetime import datetime

        writer = EvidenceWriter(evidence_dir=str(tmp_path / "evidence"))
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        writer.write(frame=frame, event_id="evt_date_001")

        date_str = datetime.now().strftime("%Y-%m-%d")
        day_dir = tmp_path / "evidence" / date_str
        assert day_dir.exists()
        assert (day_dir / "evt_date_001.jpg").exists()

    def test_disk_quota_enforcement(self, tmp_path):
        """Verify FIFO deletion when quota exceeded."""
        from packages.inference.evidence import EvidenceWriter

        # Very small quota: 0.001 GB = ~1MB
        writer = EvidenceWriter(
            evidence_dir=str(tmp_path / "evidence"),
            max_disk_gb=0.001,
            check_interval=5,
        )

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Write enough frames to exceed 1MB
        for i in range(20):
            writer.write(frame=frame, event_id=f"evt_quota_{i:04d}")

        # Some old files should have been cleaned up
        stats = writer.get_stats()
        assert stats["total_bytes"] <= writer.max_disk_bytes * 1.5  # allow some overshoot

    def test_missing_event_returns_none(self, tmp_path):
        """Non-existent event returns None."""
        from packages.inference.evidence import EvidenceWriter

        writer = EvidenceWriter(evidence_dir=str(tmp_path / "evidence"))
        assert writer.get_frame_path("nonexistent") is None
        assert writer.get_metadata("nonexistent") is None

    def test_stats(self, tmp_path):
        """Verify stats output."""
        from packages.inference.evidence import EvidenceWriter

        writer = EvidenceWriter(evidence_dir=str(tmp_path / "evidence"), max_disk_gb=0.001)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        for i in range(3):
            writer.write(frame=frame, event_id=f"evt_stats_{i}")

        stats = writer.get_stats()
        assert stats["total_writes"] == 3
        assert stats["total_bytes"] > 0
        assert stats["usage_pct"] > 0


# ━━━ Doctor Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDoctor:
    """Test pre-flight diagnostics."""

    def test_check_disk_passes(self, tmp_path):
        """Disk check passes when space available."""
        from packages.inference.doctor import check_disk

        result = check_disk(str(tmp_path), min_gb=0.001)
        assert result.passed is True
        assert "free" in result.message

    def test_check_disk_fails_unreasonable_threshold(self, tmp_path):
        """Disk check fails with unreasonably high threshold."""
        from packages.inference.doctor import check_disk

        result = check_disk(str(tmp_path), min_gb=999999)
        assert result.passed is False

    def test_check_data_dir_writable(self, tmp_path):
        """Data dir writable check passes on tmp."""
        from packages.inference.doctor import check_data_dir_writable

        result = check_data_dir_writable(str(tmp_path / "new_dir"))
        assert result.passed is True

    def test_check_config_valid(self, tmp_path):
        """Config check passes with valid YAML."""
        config = tmp_path / "station.yaml"
        config.write_text("station_id: test_1\ndata_dir: /tmp/data\n")

        from packages.inference.doctor import check_config

        result = check_config(str(config))
        assert result.passed is True
        assert "test_1" in result.message

    def test_check_config_missing(self):
        """Config check fails when file doesn't exist."""
        from packages.inference.doctor import check_config

        result = check_config("/nonexistent/station.yaml")
        assert result.passed is False

    def test_check_vision_model_missing(self, tmp_path):
        """Vision model check fails with empty directory."""
        from packages.inference.doctor import check_vision_model

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        result = check_vision_model(str(model_dir))
        assert result.passed is False

    def test_check_vision_model_with_engine(self, tmp_path):
        """Vision model check passes with .engine file."""
        from packages.inference.doctor import check_vision_model

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "yolov8n_fp16.engine").write_bytes(b"\x00" * 1024)

        result = check_vision_model(str(model_dir))
        assert result.passed is True
        assert "TRT engines" in result.message

    def test_check_language_model_missing(self, tmp_path):
        """Language model check fails with no GGUF."""
        from packages.inference.doctor import check_language_model

        model_dir = tmp_path / "models"
        model_dir.mkdir()

        result = check_language_model(str(model_dir))
        assert result.passed is False

    def test_check_language_model_with_gguf(self, tmp_path):
        """Language model check passes with .gguf file."""
        from packages.inference.doctor import check_language_model

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "qwen2.5-3b-instruct-q4_k_m.gguf").write_bytes(b"\x00" * 2048)

        result = check_language_model(str(model_dir))
        assert result.passed is True

    def test_check_taxonomy_valid(self, tmp_path):
        """Taxonomy check passes with valid file."""
        from packages.inference.doctor import check_taxonomy

        tax = tmp_path / "taxonomy.yaml"
        tax.write_text("defects:\n  scratch: {description_zh: test}\n  burr: {description_zh: test2}\n")

        result = check_taxonomy(str(tax))
        assert result.passed is True
        assert "2 defect types" in result.message

    def test_check_camera_no_source(self):
        """Camera check fails with empty source."""
        from packages.inference.doctor import check_camera

        result = check_camera("")
        assert result.passed is False

    def test_run_doctor_full(self, tmp_path):
        """Full doctor run returns structured report."""
        from packages.inference.doctor import run_doctor

        config = tmp_path / "station.yaml"
        config.write_text(f"station_id: test\ndata_dir: {tmp_path}/data\nmodel_dir: {tmp_path}/models\n")

        (tmp_path / "data").mkdir()
        (tmp_path / "models").mkdir()

        report = run_doctor(
            config_path=str(config),
            data_dir=str(tmp_path / "data"),
            model_dir=str(tmp_path / "models"),
            skip_camera=True,
        )

        assert len(report.checks) >= 5
        assert isinstance(report.all_passed, bool)
        assert isinstance(report.summary, str)

    def test_doctor_report_print(self, tmp_path, capsys):
        """Doctor print_report produces readable output."""
        from packages.inference.doctor import DoctorReport, CheckResult

        report = DoctorReport(checks=[
            CheckResult("Test 1", True, "All good"),
            CheckResult("Test 2", False, "Problem", "Fix this"),
        ])

        report.print_report()
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "FAIL" in captured.out
        assert "1/2 checks passed" in captured.out


# ━━━ API Evidence Endpoint Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAPIEvidence:
    """Test the /api/evidence/ endpoint."""

    def test_evidence_endpoint_404_without_writer(self):
        """Evidence endpoint returns 404 when no writer configured."""
        from packages.inference.api import create_app

        app = create_app(runtime=None)
        client = app.test_client()

        resp = client.get("/api/evidence/evt_test")
        assert resp.status_code == 404

    def test_dashboard_serves_html(self):
        """Root path serves the operator dashboard."""
        from packages.inference.api import create_app

        app = create_app(runtime=None)
        client = app.test_client()

        resp = client.get("/")
        assert resp.status_code == 200
        assert b"IntelFactor" in resp.data


# ━━━ Taxonomy Loading Tests ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTaxonomy:
    """Test Wiko taxonomy YAML structure."""

    def test_taxonomy_loads(self):
        """Taxonomy file loads and has expected structure."""
        import yaml
        tax_path = Path(__file__).parent.parent / "configs" / "wiko_taxonomy.yaml"
        if not tax_path.exists():
            pytest.skip("Taxonomy file not in expected location")

        with open(tax_path) as f:
            tax = yaml.safe_load(f)

        assert "defects" in tax
        assert len(tax["defects"]) == 13

        # Each defect has required fields
        for name, defect in tax["defects"].items():
            assert "description_zh" in defect, f"{name} missing description_zh"
            assert "description_en" in defect, f"{name} missing description_en"
            assert "common_causes" in defect, f"{name} missing common_causes"
            assert "severity_range" in defect, f"{name} missing severity_range"
            assert "sop_section" in defect, f"{name} missing sop_section"
            assert len(defect["severity_range"]) == 2

    def test_sop_sections_present(self):
        """Taxonomy includes SOP section reference."""
        import yaml
        tax_path = Path(__file__).parent.parent / "configs" / "wiko_taxonomy.yaml"
        if not tax_path.exists():
            pytest.skip("Taxonomy file not in expected location")

        with open(tax_path) as f:
            tax = yaml.safe_load(f)

        assert "sop_sections" in tax
        assert len(tax["sop_sections"]) >= 10

    def test_process_parameters_defined(self):
        """Taxonomy includes process parameters."""
        import yaml
        tax_path = Path(__file__).parent.parent / "configs" / "wiko_taxonomy.yaml"
        if not tax_path.exists():
            pytest.skip("Taxonomy file not in expected location")

        with open(tax_path) as f:
            tax = yaml.safe_load(f)

        assert "process_parameters" in tax
        for name, param in tax["process_parameters"].items():
            assert "unit" in param, f"{name} missing unit"
            assert "target" in param, f"{name} missing target"
            assert "tolerance" in param, f"{name} missing tolerance"
            assert "description_zh" in param, f"{name} missing description_zh"
