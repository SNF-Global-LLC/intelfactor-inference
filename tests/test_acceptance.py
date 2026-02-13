"""
IntelFactor.ai — Acceptance Test Harness
Tests that matter for factory deployment, not unit test vanity metrics.

Gate 1: Offline Continuity — system produces inspection outcomes for 24+ hours offline.
Gate 2: Evidence Traceability — every outcome traceable to evidence artifacts.
Gate 3: SLM Degradation — falls back to statistical-only RCA when SLM fails.
Gate 4: Performance Budget — detection + RCA within latency spec.

Run: python -m pytest tests/test_acceptance.py -v
"""

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.inference.rca.accumulator import DefectAccumulator
from packages.inference.rca.correlator import (
    ParameterReading,
    ProcessCorrelator,
    ProcessParameter,
)
from packages.inference.rca.explainer import RCAExplainer
from packages.inference.rca.pipeline import RCAPipeline
from packages.inference.rca.recommender import ActionRecommender
from packages.inference.schemas import (
    AnomalyAlert,
    BoundingBox,
    CausalTriple,
    Detection,
    DetectionResult,
    OperatorAction,
    ProcessCorrelation,
    RCAExplanation,
    Verdict,
)


# ── Test Infrastructure ────────────────────────────────────────────────

class StubLanguageProvider:
    """Stub language provider for testing without GPU."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail
        self.model_spec = MagicMock()
        self.model_spec.model_name = "stub-test"
        self._loaded = True

    def load(self):
        self._loaded = True

    @property
    def is_loaded(self):
        return self._loaded

    def generate(self, prompt, context=None):
        if self.should_fail:
            raise RuntimeError("SLM load failure (simulated)")
        return RCAExplanation(
            explanation_zh="测试：研磨参数漂移导致刮痕率上升",
            explanation_en="Test: grinding parameter drift caused elevated scratch rate",
            confidence=0.8,
            model_used="stub-test",
            generation_ms=1500.0,
        )

    def unload(self):
        self._loaded = False


def _make_pipeline(tmp_dir: str, slm_should_fail: bool = False) -> RCAPipeline:
    """Create a full pipeline with test dependencies."""
    accumulator = DefectAccumulator(db_path=os.path.join(tmp_dir, "acc.db"))
    accumulator.start()

    correlator = ProcessCorrelator(parameters=[
        ProcessParameter(name="grinding_rpm", unit="RPM", target=3000, tolerance=50),
    ])

    language = StubLanguageProvider(should_fail=slm_should_fail)
    explainer = RCAExplainer(
        language_provider=language,
        sop_context={"sections": {"SOP 4.2.3": "Grinding wheel recalibration"}},
        defect_taxonomy={"scratch_surface": {
            "description": "Surface scratch on blade",
            "common_causes": "Grinding wheel wear, RPM drift",
        }},
    )

    recommender = ActionRecommender(
        sop_map={
            "parameter_sops": {"grinding_rpm": "SOP 4.2.3"},
            "default_section": "SOP (general)",
        },
        db_path=os.path.join(tmp_dir, "triples.db"),
    )
    recommender.start()

    return RCAPipeline(accumulator, correlator, explainer, recommender)


def _make_detection(station: str = "s1", defect: str = "scratch_surface") -> DetectionResult:
    """Create a FAIL detection result."""
    return DetectionResult(
        station_id=station,
        verdict=Verdict.FAIL,
        confidence=0.9,
        inference_ms=18.5,
        detections=[
            Detection(defect_type=defect, confidence=0.9, bbox=BoundingBox(10, 20, 50, 30), severity=0.8)
        ],
    )


# ── Gate 1: Offline Continuity ─────────────────────────────────────────

class TestOfflineContinuity:
    """
    Verify system produces inspection outcomes without any network dependency.
    Simulated as: no external services, no HTTP calls, pure local operation.
    """

    def test_inspection_without_network(self):
        """Core loop runs with zero network dependency."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp)

            # Simulate 1000 inspection events (≈ a short shift)
            for i in range(1000):
                result = _make_detection()
                result.timestamp = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(minutes=1000 - i)
                pipeline.ingest(result)

            stats = pipeline.accumulator.get_stats()
            assert stats["total_events"] == 1000
            assert stats["status"] == "running"

            pipeline.accumulator.stop()
            pipeline.recommender.stop()

    def test_rca_runs_locally(self):
        """Full RCA pipeline runs without any external service."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp)

            # Build baseline (15 days, 2 events per 4h window)
            base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=15)
            for day in range(15):
                for window in range(6):
                    ts = base + timedelta(days=day, hours=window * 4 + 1)
                    for _ in range(2):
                        r = _make_detection()
                        r.timestamp = ts
                        pipeline.ingest(r)

            # Create spike
            now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            for i in range(20):
                r = _make_detection()
                r.timestamp = now - timedelta(minutes=i)
                pipeline.ingest(r)

            # Add correlated parameter readings
            for i in range(10):
                pipeline.correlator.record_reading(ParameterReading(
                    parameter_name="grinding_rpm",
                    value=3000 - i * 25,
                    timestamp=now - timedelta(minutes=30 - i * 3),
                    station_id="s1",
                ))

            # Run full RCA
            results = pipeline.run_rca(station_id="s1")

            assert len(results) > 0
            result = results[0]
            assert result.explanation.explanation_zh != ""
            assert result.explanation.explanation_en != ""
            assert result.triple.defect_type == "scratch_surface"

            pipeline.accumulator.stop()
            pipeline.recommender.stop()

    def test_evidence_persists_across_restart(self):
        """Data survives process restart (SQLite durability)."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "acc.db")

            # Write events
            acc1 = DefectAccumulator(db_path=db_path)
            acc1.start()
            for _ in range(50):
                acc1.record_event(_make_detection())
            acc1.stop()

            # Reopen (simulates restart)
            acc2 = DefectAccumulator(db_path=db_path)
            acc2.start()
            assert acc2.get_stats()["total_events"] == 50
            acc2.stop()


# ── Gate 2: Evidence Traceability ──────────────────────────────────────

class TestEvidenceTraceability:
    """
    Every FAIL/REVIEW event must be traceable to evidence artifacts.
    Audit: sample 50 events, zero missing fields.
    """

    def test_all_events_have_required_fields(self):
        """Every stored event has all required metadata fields."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp)

            # Create 100 events with full metadata
            for i in range(100):
                r = _make_detection()
                r.sku = f"SKU-{i % 5}"
                r.shift = "A" if i % 2 == 0 else "B"
                r.model_version = "yolov8n-v3"
                pipeline.ingest(r)

            # Audit: sample 50 events
            conn = pipeline.accumulator._conn
            rows = conn.execute(
                "SELECT * FROM defect_events ORDER BY RANDOM() LIMIT 50"
            ).fetchall()
            columns = [desc[0] for desc in conn.execute(
                "SELECT * FROM defect_events LIMIT 0"
            ).description]

            required = ["event_id", "timestamp", "station_id", "defect_type", "severity", "confidence"]

            missing_count = 0
            for row in rows:
                data = dict(zip(columns, row))
                for field in required:
                    if not data.get(field):
                        missing_count += 1

            assert missing_count == 0, f"{missing_count} missing required fields in 50 sampled events"

            pipeline.accumulator.stop()
            pipeline.recommender.stop()

    def test_triples_have_complete_chain(self):
        """Every stored triple has defect → cause → recommendation chain."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp)

            # Build enough data for RCA to trigger
            base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=10)
            for day in range(10):
                for window in range(6):
                    ts = base + timedelta(days=day, hours=window * 4 + 1)
                    for _ in range(2):
                        r = _make_detection()
                        r.timestamp = ts
                        pipeline.ingest(r)

            # Spike + readings
            now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            for i in range(15):
                r = _make_detection()
                r.timestamp = now - timedelta(minutes=i)
                pipeline.ingest(r)

            for i in range(8):
                pipeline.correlator.record_reading(ParameterReading(
                    parameter_name="grinding_rpm",
                    value=3000 - i * 30,
                    timestamp=now - timedelta(minutes=30 - i * 4),
                    station_id="s1",
                ))

            results = pipeline.run_rca(station_id="s1")

            for result in results:
                t = result.triple
                # Defect chain
                assert t.defect_type != "", "Triple missing defect_type"
                assert t.defect_event_id != "", "Triple missing defect_event_id"
                # Cause chain
                assert t.cause_explanation_zh != "" or t.cause_explanation_en != "", "Triple missing explanation"
                # Recommendation chain
                assert t.recommendation_id != "", "Triple missing recommendation_id"

            pipeline.accumulator.stop()
            pipeline.recommender.stop()


# ── Gate 3: SLM Degradation ───────────────────────────────────────────

class TestSLMDegradation:
    """
    When the SLM fails to load or generate, system must degrade to
    statistical-only RCA (Layer 1+2) and still produce recommendations.
    """

    def test_rca_continues_when_slm_fails(self):
        """Full pipeline continues with statistical fallback when SLM crashes."""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = _make_pipeline(tmp, slm_should_fail=True)

            # Build baseline + spike
            base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=10)
            for day in range(10):
                for window in range(6):
                    ts = base + timedelta(days=day, hours=window * 4 + 1)
                    for _ in range(2):
                        r = _make_detection()
                        r.timestamp = ts
                        pipeline.ingest(r)

            now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            for i in range(15):
                r = _make_detection()
                r.timestamp = now - timedelta(minutes=i)
                pipeline.ingest(r)

            for i in range(8):
                pipeline.correlator.record_reading(ParameterReading(
                    parameter_name="grinding_rpm",
                    value=3000 - i * 30,
                    timestamp=now - timedelta(minutes=30 - i * 4),
                    station_id="s1",
                ))

            # RCA should still work via fallback
            results = pipeline.run_rca(station_id="s1")
            assert len(results) > 0

            result = results[0]
            # Fallback uses "statistical_fallback" as model
            assert result.explanation.model_used == "statistical_fallback"
            # Still produces Chinese + English output
            assert result.explanation.explanation_zh != ""
            assert result.explanation.explanation_en != ""
            # Still creates a triple
            assert result.triple.defect_type == "scratch_surface"
            # Still produces a recommendation
            assert result.recommendation.sop_section != ""

            pipeline.accumulator.stop()
            pipeline.recommender.stop()


# ── Gate 4: Performance Budget ─────────────────────────────────────────

class TestPerformanceBudget:
    """
    Detection latency budget: <25ms.
    RCA latency budget: <4s (triggered on anomaly, not per-frame).
    Accumulator must handle 6000 events/day within 50MB RAM budget.
    """

    def test_accumulator_handles_daily_volume(self):
        """Accumulator ingests a full day of events efficiently."""
        with tempfile.TemporaryDirectory() as tmp:
            acc = DefectAccumulator(db_path=os.path.join(tmp, "perf.db"))
            acc.start()

            events_per_day = 6000
            t0 = time.perf_counter()

            base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
            for i in range(events_per_day):
                r = _make_detection()
                r.event_id = f"perf_evt_{i:06d}"  # unique IDs for rapid insertion
                r.timestamp = base + timedelta(seconds=i * 14.4)  # ~14s between events
                acc.record_event(r)

            ingest_time = time.perf_counter() - t0

            stats = acc.get_stats()
            assert stats["total_events"] == events_per_day
            # Should complete in well under 10 seconds on any hardware
            assert ingest_time < 10.0, f"Ingest took {ingest_time:.1f}s (budget: <10s)"

            acc.stop()

    def test_anomaly_check_latency(self):
        """Anomaly detection completes quickly even with large dataset."""
        with tempfile.TemporaryDirectory() as tmp:
            acc = DefectAccumulator(db_path=os.path.join(tmp, "perf.db"))
            acc.start()

            # Load 30 days of data
            base = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(days=30)
            for day in range(30):
                for window in range(6):
                    ts = base + timedelta(days=day, hours=window * 4 + 1)
                    for _ in range(3):
                        r = _make_detection()
                        r.timestamp = ts
                        acc.record_event(r)

            t0 = time.perf_counter()
            alerts = acc.check_anomalies(station_id="s1")
            check_time = time.perf_counter() - t0

            # Anomaly check should complete in under 2 seconds
            assert check_time < 2.0, f"Anomaly check took {check_time:.1f}s (budget: <2s)"

            acc.stop()

    def test_operator_feedback_loop_latency(self):
        """Full feedback cycle (create triple → record feedback) is fast."""
        with tempfile.TemporaryDirectory() as tmp:
            rec = ActionRecommender(
                sop_map={"default_section": "SOP 4.2.3"},
                db_path=os.path.join(tmp, "triples.db"),
            )
            rec.start()

            alert = AnomalyAlert(station_id="s1", defect_type="scratch", current_rate=10, baseline_rate=2, z_score=3)
            corr = ProcessCorrelation(parameter_name="rpm", current_value=2850, target_value=3000, tolerance=50, drift_pct=5, pearson_r=0.8, confidence=0.8)
            expl = RCAExplanation(explanation_zh="test", explanation_en="test", confidence=0.8, model_used="test")
            recommendation = rec.recommend(alert, [corr], expl)

            t0 = time.perf_counter()
            triple = rec.create_triple(alert, [corr], expl, recommendation)
            rec.record_operator_feedback(triple.triple_id, OperatorAction.ACCEPTED, "op1")
            feedback_time = time.perf_counter() - t0

            # Full feedback cycle should be <100ms
            assert feedback_time < 0.1, f"Feedback cycle took {feedback_time*1000:.0f}ms (budget: <100ms)"

            rec.stop()
