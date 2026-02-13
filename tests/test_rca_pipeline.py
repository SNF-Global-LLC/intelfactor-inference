"""
Tests for the RCA pipeline services.
Run: python -m pytest tests/ -v
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.inference.schemas import (
    AnomalyAlert,
    BoundingBox,
    CausalTriple,
    Detection,
    DetectionResult,
    OperatorAction,
    ProcessCorrelation,
    RCAExplanation,
    TripleStatus,
    Verdict,
)
from packages.inference.rca.accumulator import DefectAccumulator
from packages.inference.rca.correlator import (
    ParameterReading,
    ProcessCorrelator,
    ProcessParameter,
)
from packages.inference.rca.recommender import ActionRecommender


class TestDefectAccumulator:
    """Test Layer 1: defect event recording and anomaly detection."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test_accumulator.db")
        self.acc = DefectAccumulator(db_path=self.db_path)
        self.acc.start()

    def teardown_method(self):
        self.acc.stop()

    def _make_result(self, station: str = "s1", defect_type: str = "scratch", severity: float = 0.8) -> DetectionResult:
        return DetectionResult(
            station_id=station,
            verdict=Verdict.FAIL,
            confidence=severity,
            detections=[
                Detection(
                    defect_type=defect_type,
                    confidence=severity,
                    bbox=BoundingBox(x=10, y=20, width=50, height=30),
                    severity=severity,
                )
            ],
        )

    def test_records_fail_events(self):
        """FAIL events should be recorded."""
        result = self._make_result()
        self.acc.record_event(result)
        stats = self.acc.get_stats()
        assert stats["total_events"] == 1

    def test_ignores_pass_events(self):
        """PASS events should NOT be recorded."""
        result = DetectionResult(verdict=Verdict.PASS, station_id="s1")
        self.acc.record_event(result)
        stats = self.acc.get_stats()
        assert stats["total_events"] == 0

    def test_records_review_events(self):
        """REVIEW events should be recorded."""
        result = self._make_result()
        result.verdict = Verdict.REVIEW
        self.acc.record_event(result)
        stats = self.acc.get_stats()
        assert stats["total_events"] == 1

    def test_anomaly_detection_with_spike(self):
        """A sudden spike should trigger an anomaly alert."""
        # Create 30 days of baseline: ~2 events per 4-hour window
        base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=15)
        for day in range(15):
            for window in range(6):  # 6 four-hour windows per day
                ts = base + timedelta(days=day, hours=window * 4 + 1)
                for _ in range(2):  # 2 events per window (baseline)
                    result = self._make_result()
                    result.timestamp = ts
                    self.acc.record_event(result)

        # Now create a spike: 20 events in the last 4 hours
        recent = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        for i in range(20):
            result = self._make_result()
            result.timestamp = recent + timedelta(minutes=i)
            self.acc.record_event(result)

        alerts = self.acc.check_anomalies(station_id="s1")
        assert len(alerts) > 0
        assert alerts[0].z_score > 2.0

    def test_no_anomaly_under_baseline(self):
        """Normal rates should not trigger alerts."""
        # Create consistent baseline
        base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=10)
        for day in range(10):
            for window in range(6):
                ts = base + timedelta(days=day, hours=window * 4 + 1)
                for _ in range(3):
                    result = self._make_result()
                    result.timestamp = ts
                    self.acc.record_event(result)

        alerts = self.acc.check_anomalies(station_id="s1")
        # Should be 0 or very few alerts since current window matches baseline
        # (last window is also ~3 events)
        high_alerts = [a for a in alerts if a.z_score > 2.5]
        assert len(high_alerts) == 0

    def test_prune_removes_old_events(self):
        """Pruning should remove events older than retention period."""
        # Add an old event
        result = self._make_result()
        result.timestamp = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=60)
        self.acc.record_event(result)

        # Add a recent event
        result2 = self._make_result()
        self.acc.record_event(result2)

        deleted = self.acc.prune()
        assert deleted == 1
        assert self.acc.get_stats()["total_events"] == 1


class TestProcessCorrelator:
    """Test Layer 2: process parameter correlation."""

    def setup_method(self):
        self.correlator = ProcessCorrelator(parameters=[
            ProcessParameter(
                name="grinding_rpm",
                unit="RPM",
                target=3000,
                tolerance=50,
            ),
            ProcessParameter(
                name="grinding_temp",
                unit="°C",
                target=45,
                tolerance=5,
            ),
        ])

    def test_detects_drift(self):
        """Should detect parameter drift outside tolerance."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        # Record readings showing drift
        for i in range(10):
            self.correlator.record_reading(ParameterReading(
                parameter_name="grinding_rpm",
                value=3000 - i * 20,  # drifting down: 3000, 2980, 2960...
                timestamp=now - timedelta(minutes=30 - i * 3),
                station_id="s1",
            ))

        drifts = self.correlator.check_drift(station_id="s1")
        # Last value is 2820, target 3000 ±50 → drift of 180 > 50
        assert len(drifts) > 0
        assert drifts[0]["parameter"] == "grinding_rpm"
        assert drifts[0]["drift"] > 50

    def test_no_drift_within_tolerance(self):
        """Parameters within tolerance should not be flagged."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        for i in range(5):
            self.correlator.record_reading(ParameterReading(
                parameter_name="grinding_rpm",
                value=3010,  # within ±50
                timestamp=now - timedelta(minutes=i),
                station_id="s1",
            ))

        drifts = self.correlator.check_drift(station_id="s1")
        assert len(drifts) == 0

    def test_correlation_with_anomaly(self):
        """Should correlate drifting parameter with anomaly alert."""
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        # Create readings showing clear drift
        for i in range(10):
            self.correlator.record_reading(ParameterReading(
                parameter_name="grinding_rpm",
                value=3000 - i * 25,  # strong downward drift
                timestamp=now - timedelta(minutes=30 - i * 3),
                station_id="s1",
            ))

        alert = AnomalyAlert(
            timestamp=now,
            station_id="s1",
            defect_type="scratch_surface",
            current_rate=15.0,
            baseline_rate=3.0,
            z_score=3.5,
        )

        correlations = self.correlator.correlate(alert, min_correlation=0.3)
        # Should find grinding_rpm as correlated (drifted beyond tolerance)
        assert len(correlations) > 0

    def test_from_edge_yaml(self):
        """Should load parameters from edge.yaml format."""
        edge_yaml = {
            "process_parameters": {
                "belt_speed": {"unit": "m/min", "target": 12, "tolerance": 1, "data_source": "mqtt"},
                "coolant_flow": {"unit": "L/min", "target": 5, "tolerance": 0.5, "data_source": "manual"},
            }
        }
        correlator = ProcessCorrelator.from_edge_yaml(edge_yaml)
        assert len(correlator.parameters) == 2
        assert "belt_speed" in correlator.parameters


class TestActionRecommender:
    """Test Layer 4: recommendations and triple storage."""

    def setup_method(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "test_triples.db")
        self.rec = ActionRecommender(
            sop_map={
                "parameter_sops": {"grinding_rpm": "SOP 4.2.3"},
                "defect_sops": {"scratch_surface": "SOP 4.2.3"},
                "default_section": "SOP (general)",
            },
            db_path=self.db_path,
        )
        self.rec.start()

    def teardown_method(self):
        self.rec.stop()

    def _make_alert(self) -> AnomalyAlert:
        return AnomalyAlert(
            station_id="s1",
            defect_type="scratch_surface",
            current_rate=15.0,
            baseline_rate=3.0,
            z_score=3.5,
            event_ids=["evt_001", "evt_002"],
        )

    def _make_correlation(self) -> ProcessCorrelation:
        return ProcessCorrelation(
            parameter_name="grinding_rpm",
            current_value=2847,
            target_value=3000,
            tolerance=50,
            drift_pct=5.1,
            pearson_r=0.87,
            confidence=0.87,
        )

    def _make_explanation(self) -> RCAExplanation:
        return RCAExplanation(
            explanation_zh="研磨轮转速偏离目标5.1%",
            explanation_en="Grinding wheel RPM drifted 5.1% from target",
            confidence=0.85,
            model_used="qwen25_3b_int4",
            generation_ms=2500,
        )

    def test_generates_recommendation(self):
        """Should produce SOP-linked recommendation."""
        alert = self._make_alert()
        corr = self._make_correlation()
        expl = self._make_explanation()

        rec = self.rec.recommend(alert, [corr], expl)
        assert "SOP 4.2.3" in rec.sop_section
        assert "2847" in rec.action_en
        assert rec.urgency in ("normal", "high", "critical")

    def test_creates_triple(self):
        """Should create and store a causal triple."""
        alert = self._make_alert()
        corr = self._make_correlation()
        expl = self._make_explanation()
        rec = self.rec.recommend(alert, [corr], expl)

        triple = self.rec.create_triple(alert, [corr], expl, rec)
        assert triple.status == TripleStatus.PENDING
        assert triple.defect_type == "scratch_surface"
        assert triple.cause_parameter == "grinding_rpm"

    def test_operator_feedback_verifies_triple(self):
        """Accepting a recommendation should verify the triple."""
        alert = self._make_alert()
        corr = self._make_correlation()
        expl = self._make_explanation()
        rec = self.rec.recommend(alert, [corr], expl)
        triple = self.rec.create_triple(alert, [corr], expl, rec)

        updated = self.rec.record_operator_feedback(
            triple_id=triple.triple_id,
            action=OperatorAction.ACCEPTED,
            operator_id="op_zhang",
            outcome={"scratch_rate_after": 0.8, "time_to_baseline_min": 45},
        )

        assert updated is not None
        assert updated.status == TripleStatus.VERIFIED
        assert updated.operator_action == OperatorAction.ACCEPTED
        assert updated.operator_id == "op_zhang"

    def test_operator_rejection_disputes_triple(self):
        """Rejecting a recommendation should dispute the triple."""
        alert = self._make_alert()
        corr = self._make_correlation()
        expl = self._make_explanation()
        rec = self.rec.recommend(alert, [corr], expl)
        triple = self.rec.create_triple(alert, [corr], expl, rec)

        updated = self.rec.record_operator_feedback(
            triple_id=triple.triple_id,
            action=OperatorAction.REJECTED,
            operator_id="op_li",
            rejection_reason="RPM was already recalibrated 30 min ago",
        )

        assert updated is not None
        assert updated.status == TripleStatus.DISPUTED
        assert "rejection_reason" in updated.outcome_measured

    def test_triple_stats(self):
        """Should track triple statistics."""
        alert = self._make_alert()
        corr = self._make_correlation()
        expl = self._make_explanation()
        rec = self.rec.recommend(alert, [corr], expl)
        self.rec.create_triple(alert, [corr], expl, rec)

        stats = self.rec.get_triple_stats()
        assert stats["total_triples"] == 1
        assert stats["pending"] == 1
        assert stats["verified"] == 0

    def test_recommendation_without_correlations(self):
        """Should still produce a recommendation when no correlations found."""
        alert = self._make_alert()
        expl = self._make_explanation()

        rec = self.rec.recommend(alert, [], expl)
        assert rec.sop_section != ""
        assert "anomaly" in rec.action_en.lower()
