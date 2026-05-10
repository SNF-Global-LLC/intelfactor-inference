"""
Tests for Machine Health Copilot — MaintenanceIQ rules engine.
Covers threshold evaluation, verdict generation, edge cases, and edge.yaml loading.
Run: python -m pytest tests/test_maintenance_iq.py -v
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages.ingestion.schemas import (
    HealthVerdict,
    MaintenanceActionType,
    SensorEvent,
    SensorType,
)
from packages.policy.maintenance_iq import (
    DEFAULT_CRITICAL_THRESHOLD,
    DEFAULT_WARNING_THRESHOLD,
    MachineThresholds,
    MaintenanceIQ,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_event(
    machine_id: str = "press_01",
    sensor_type: SensorType = SensorType.VIBRATION,
    z_score: float = 0.0,
    confidence: float = 1.0,
    verdict: HealthVerdict = HealthVerdict.HEALTHY,
) -> SensorEvent:
    return SensorEvent(
        machine_id=machine_id,
        station_id="s1",
        sensor_type=sensor_type,
        raw_values={"rms": 0.5},
        anomaly_score=z_score,
        confidence=confidence,
        edge_verdict=verdict,
    )


@pytest.fixture()
def iq() -> MaintenanceIQ:
    return MaintenanceIQ(
        sop_map={
            "vibration": "SOP M.3.1",
            "current": "SOP M.2.4",
            "acoustic": "SOP M.4.2",
            "default_section": "SOP (machine health manual)",
        }
    )


# ── Threshold evaluation ───────────────────────────────────────────────


class TestThresholdEvaluation:
    def test_below_warning_threshold_is_healthy(self, iq):
        events = [_make_event(z_score=1.5)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.HEALTHY

    def test_at_warning_threshold_is_warning(self, iq):
        events = [_make_event(z_score=DEFAULT_WARNING_THRESHOLD)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.WARNING

    def test_above_warning_below_critical_is_warning(self, iq):
        events = [_make_event(z_score=3.0)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.WARNING

    def test_at_critical_threshold_is_critical(self, iq):
        events = [_make_event(z_score=DEFAULT_CRITICAL_THRESHOLD)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.CRITICAL

    def test_well_above_critical_is_critical(self, iq):
        events = [_make_event(z_score=10.0)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.CRITICAL

    def test_verdict_carries_correct_z_score(self, iq):
        events = [_make_event(z_score=4.2)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.z_score == pytest.approx(4.2)

    def test_verdict_carries_thresholds_used(self, iq):
        """Thresholds should be recorded on the verdict for auditability."""
        v = iq.evaluate("press_01", "s1", [_make_event(z_score=1.0)])
        assert v.warning_threshold == DEFAULT_WARNING_THRESHOLD
        assert v.critical_threshold == DEFAULT_CRITICAL_THRESHOLD


# ── Per-machine threshold overrides ──────────────────────────────────


class TestPerMachineThresholds:
    def test_per_machine_warning_override(self):
        iq = MaintenanceIQ(
            machine_thresholds=[
                MachineThresholds(machine_id="press_01", warning=3.0, critical=5.0)
            ]
        )
        # z=2.5 is above global default (2.0) but below override (3.0) → HEALTHY
        events = [_make_event(z_score=2.5)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.HEALTHY

    def test_per_machine_critical_override(self):
        iq = MaintenanceIQ(
            machine_thresholds=[
                MachineThresholds(machine_id="press_01", warning=2.0, critical=5.0)
            ]
        )
        # z=4.0 is above global critical (3.5) but below override (5.0) → WARNING
        events = [_make_event(z_score=4.0)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.WARNING

    def test_other_machine_uses_defaults(self):
        iq = MaintenanceIQ(
            machine_thresholds=[
                MachineThresholds(machine_id="press_01", warning=10.0, critical=20.0)
            ]
        )
        # press_02 has no override → uses defaults → z=3.5 is CRITICAL
        events = [_make_event(machine_id="press_02", z_score=3.5)]
        v = iq.evaluate("press_02", "s1", events)
        assert v.verdict == HealthVerdict.CRITICAL

    def test_from_edge_yaml_loads_per_machine(self):
        config = {
            "machine_health": {
                "thresholds": {
                    "default": {"warning": 2.0, "critical": 3.5},
                    "grinder_01": {"warning": 2.5, "critical": 4.5},
                },
                "sop_map": {"vibration": "SOP M.3.1"},
            }
        }
        iq = MaintenanceIQ.from_edge_yaml(config)

        # grinder_01 at z=3.0 → above 2.5 warning → WARNING (not at CRITICAL 4.5)
        events = [_make_event(machine_id="grinder_01", z_score=3.0)]
        v = iq.evaluate("grinder_01", "s1", events)
        assert v.verdict == HealthVerdict.WARNING

    def test_from_edge_yaml_missing_block_uses_defaults(self):
        iq = MaintenanceIQ.from_edge_yaml({})
        events = [_make_event(z_score=DEFAULT_CRITICAL_THRESHOLD)]
        v = iq.evaluate("m1", "s1", events)
        assert v.verdict == HealthVerdict.CRITICAL


# ── Multi-sensor aggregation ───────────────────────────────────────────


class TestMultiSensorAggregation:
    def test_worst_sensor_drives_verdict(self, iq):
        """Critical on one sensor → overall CRITICAL, regardless of others."""
        events = [
            _make_event(sensor_type=SensorType.VIBRATION, z_score=1.0),
            _make_event(sensor_type=SensorType.CURRENT, z_score=1.5),
            _make_event(sensor_type=SensorType.ACOUSTIC, z_score=4.0),  # CRITICAL
        ]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.CRITICAL
        assert v.z_score == pytest.approx(4.0)

    def test_contributing_factors_sorted_by_z(self, iq):
        events = [
            _make_event(sensor_type=SensorType.VIBRATION, z_score=1.0),
            _make_event(sensor_type=SensorType.CURRENT, z_score=3.8),
            _make_event(sensor_type=SensorType.ACOUSTIC, z_score=2.1),
        ]
        v = iq.evaluate("press_01", "s1", events)
        z_scores = [f["z_score"] for f in v.contributing_factors]
        assert z_scores == sorted(z_scores, reverse=True)

    def test_multiple_same_type_uses_highest_confidence(self, iq):
        """When multiple events of the same sensor_type arrive, pick highest confidence."""
        events = [
            _make_event(sensor_type=SensorType.VIBRATION, z_score=4.0, confidence=0.3),
            _make_event(sensor_type=SensorType.VIBRATION, z_score=1.5, confidence=0.9),
        ]
        v = iq.evaluate("press_01", "s1", events)
        # Highest confidence event (z=1.5) wins → should be WARNING or HEALTHY
        vib_factor = next(f for f in v.contributing_factors if f["sensor_type"] == "vibration")
        assert vib_factor["z_score"] == pytest.approx(1.5)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_events_returns_healthy(self, iq):
        """No events should produce a HEALTHY verdict with zero z_score."""
        v = iq.evaluate("press_01", "s1", [])
        assert v.verdict == HealthVerdict.HEALTHY
        assert v.z_score == 0.0
        assert v.confidence == 0.0
        assert v.contributing_factors == []

    def test_zero_baseline_confidence_treated_as_unknown(self, iq):
        """Events with confidence=0 (no baseline) must not trigger alerts."""
        events = [_make_event(z_score=99.0, confidence=0.0)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.HEALTHY

    def test_first_reading_no_baseline(self, iq):
        """Single reading with no history → HEALTHY (unknown, not anomaly)."""
        event = _make_event(z_score=0.0, confidence=0.0, verdict=HealthVerdict.HEALTHY)
        v = iq.evaluate("press_01", "s1", [event])
        assert v.verdict == HealthVerdict.HEALTHY

    def test_sustained_warning_produces_warning_verdict(self, iq):
        """Multiple sequential WARNING events should compound into WARNING verdict."""
        events = [
            _make_event(z_score=2.2, verdict=HealthVerdict.WARNING),
            _make_event(z_score=2.5, verdict=HealthVerdict.WARNING),
            _make_event(z_score=2.8, verdict=HealthVerdict.WARNING),
        ]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.WARNING
        # Peak z should be the highest
        assert v.z_score == pytest.approx(2.8)

    def test_single_critical_among_healthy(self, iq):
        """One critical reading should override several healthy ones."""
        events = [
            _make_event(sensor_type=SensorType.VIBRATION, z_score=0.5),
            _make_event(sensor_type=SensorType.CURRENT, z_score=3.6),  # CRITICAL
        ]
        v = iq.evaluate("press_01", "s1", events)
        assert v.verdict == HealthVerdict.CRITICAL

    def test_machine_id_isolation(self, iq):
        """Events for machine_a should not affect evaluation of machine_b."""
        v_b = iq.evaluate("b", "s1", [_make_event(machine_id="b", z_score=0.5)])
        assert v_b.verdict == HealthVerdict.HEALTHY

    def test_verdict_fields_are_populated(self, iq):
        events = [_make_event(z_score=2.5)]
        v = iq.evaluate("press_01", "s1", events)
        assert v.machine_id == "press_01"
        assert v.station_id == "s1"
        assert v.verdict_id != ""
        assert isinstance(v.timestamp, datetime)


# ── Recommendation generation ──────────────────────────────────────────


class TestRecommendation:
    def test_critical_verdict_produces_service_action(self, iq):
        events = [_make_event(z_score=4.0, verdict=HealthVerdict.CRITICAL)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert action.action_type == MaintenanceActionType.SERVICE
        assert action.urgency == "critical"

    def test_high_warning_produces_service_action(self, iq):
        events = [_make_event(z_score=3.2, verdict=HealthVerdict.WARNING)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert action.action_type == MaintenanceActionType.SERVICE
        assert action.urgency == "high"

    def test_low_warning_produces_inspect_action(self, iq):
        events = [_make_event(z_score=2.3, verdict=HealthVerdict.WARNING)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert action.action_type == MaintenanceActionType.INSPECT
        assert action.urgency == "normal"

    def test_healthy_verdict_produces_monitor_action(self, iq):
        events = [_make_event(z_score=1.0)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert action.action_type == MaintenanceActionType.MONITOR

    def test_recommendation_has_bilingual_text(self, iq):
        events = [_make_event(z_score=3.6, verdict=HealthVerdict.CRITICAL)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert len(action.action_en) > 0
        assert len(action.action_zh) > 0
        # Both must reference the machine
        assert "press_01" in action.action_en
        assert "press_01" in action.action_zh

    def test_recommendation_references_sop(self, iq):
        events = [_make_event(sensor_type=SensorType.VIBRATION, z_score=4.0)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert "SOP M.3.1" in action.sop_section

    def test_recommendation_falls_back_to_default_sop(self):
        iq_no_map = MaintenanceIQ()
        events = [_make_event(z_score=4.0)]
        v = iq_no_map.evaluate("press_01", "s1", events)
        action = iq_no_map.recommend(v)
        assert "SOP" in action.sop_section

    def test_recommendation_operator_action_defaults_pending(self, iq):
        events = [_make_event(z_score=4.0)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        assert action.operator_action == "pending"
        assert action.operator_id == ""

    def test_notify_only_no_execution_field(self, iq):
        """MaintenanceAction must not carry any execution command."""
        events = [_make_event(z_score=5.0)]
        v = iq.evaluate("press_01", "s1", events)
        action = iq.recommend(v)
        # Confirm there is no "execute" or "command" attribute
        assert not hasattr(action, "execute")
        assert not hasattr(action, "command")
        assert not hasattr(action, "plc_command")
