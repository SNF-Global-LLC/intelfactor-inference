"""
IntelFactor.ai — Machine Health Copilot: MaintenanceIQ Rules Engine
Threshold-based verdict engine for machine health signals.

Mirrors the DefectIQ enforcer pattern:
- Configurable thresholds per machine_id from edge.yaml.
- Z-score evaluation: HEALTHY / WARNING / CRITICAL.
- Aggregates across all sensor types for a single machine verdict.
- NOTIFY-ONLY: emits MaintenanceAction recommendations, never executes.

Edge.yaml machine_health block (optional per-machine overrides):
  machine_health:
    thresholds:
      default:
        warning: 2.0
        critical: 3.5
      machine_id: press_01
        warning: 2.5
        critical: 4.0
    sop_map:
      vibration: "SOP M.3.1"
      current: "SOP M.2.4"
      acoustic: "SOP M.4.2"
      default_section: "SOP (machine health manual)"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from packages.ingestion.schemas import (
    HealthVerdict,
    MaintenanceAction,
    MaintenanceActionType,
    MaintenanceVerdict,
    SensorEvent,
    SensorType,
)

logger = logging.getLogger(__name__)


# ── Threshold configuration ─────────────────────────────────────────────


@dataclass
class MachineThresholds:
    """
    Z-score thresholds for a single machine.

    Decoupled from the SensorService so thresholds can be changed at
    runtime (e.g. after a machine upgrade) without touching stored data.
    """

    machine_id: str = ""
    warning: float = 2.0
    critical: float = 3.5


# ── Default thresholds ──────────────────────────────────────────────────

DEFAULT_WARNING_THRESHOLD = 2.0
DEFAULT_CRITICAL_THRESHOLD = 3.5


# ── MaintenanceIQ ────────────────────────────────────────────────────────


class MaintenanceIQ:
    """
    Stateless rules engine. Evaluates a batch of SensorEvents for one machine
    and returns a structured MaintenanceVerdict.

    Usage:
        iq = MaintenanceIQ.from_edge_yaml(config)
        verdict = iq.evaluate(machine_id="press_01", events=[...])
        if verdict.verdict != HealthVerdict.HEALTHY:
            action = iq.recommend(verdict)

    Threshold resolution order (most specific wins):
        1. Per-machine override in machine_thresholds
        2. Global defaults passed at construction
        3. Module-level DEFAULT_WARNING_THRESHOLD / DEFAULT_CRITICAL_THRESHOLD

    Multi-sensor aggregation:
        Each SensorType produces its own z_score. The overall verdict is the
        worst (highest) verdict across all types. The MaintenanceVerdict carries
        contributing_factors sorted by z_score descending so the operator sees
        the primary signal first.
    """

    def __init__(
        self,
        machine_thresholds: list[MachineThresholds] | None = None,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
        sop_map: dict[str, Any] | None = None,
    ):
        # Index by machine_id for O(1) lookup
        self._thresholds: dict[str, MachineThresholds] = {
            t.machine_id: t for t in (machine_thresholds or [])
        }
        self._default_warning = warning_threshold
        self._default_critical = critical_threshold
        self.sop_map: dict[str, Any] = sop_map or {}

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def from_edge_yaml(cls, config: dict[str, Any]) -> "MaintenanceIQ":
        """
        Build a MaintenanceIQ from the machine_health block of edge.yaml.

        Expected shape:
          machine_health:
            thresholds:
              default:
                warning: 2.0
                critical: 3.5
              <machine_id>:
                warning: 2.5
                critical: 4.0
            sop_map:
              vibration: "SOP M.3.1"
              ...
        """
        mh_cfg = config.get("machine_health", {})
        threshold_cfg = mh_cfg.get("thresholds", {})
        sop_map = mh_cfg.get("sop_map", {})

        default_cfg = threshold_cfg.get("default", {})
        global_warning = float(default_cfg.get("warning", DEFAULT_WARNING_THRESHOLD))
        global_critical = float(default_cfg.get("critical", DEFAULT_CRITICAL_THRESHOLD))

        per_machine: list[MachineThresholds] = []
        for machine_id, values in threshold_cfg.items():
            if machine_id == "default":
                continue
            if not isinstance(values, dict):
                continue
            per_machine.append(
                MachineThresholds(
                    machine_id=machine_id,
                    warning=float(values.get("warning", global_warning)),
                    critical=float(values.get("critical", global_critical)),
                )
            )

        return cls(
            machine_thresholds=per_machine,
            warning_threshold=global_warning,
            critical_threshold=global_critical,
            sop_map=sop_map,
        )

    # ── Core evaluation ───────────────────────────────────────────────

    def evaluate(
        self,
        machine_id: str,
        station_id: str,
        events: list[SensorEvent],
    ) -> MaintenanceVerdict:
        """
        Evaluate a list of SensorEvents for one machine.

        Typically called with the N most-recent events (e.g. last 5 minutes)
        so the caller controls the time window.

        Returns HEALTHY immediately if events is empty or all events scored
        confidence=0 (no baseline yet).
        """
        warning_t, critical_t = self._get_thresholds(machine_id)

        if not events:
            return MaintenanceVerdict(
                machine_id=machine_id,
                station_id=station_id,
                verdict=HealthVerdict.HEALTHY,
                z_score=0.0,
                confidence=0.0,
                warning_threshold=warning_t,
                critical_threshold=critical_t,
            )

        # Group events by sensor_type. Priority: (1) highest confidence, then
        # (2) highest z_score as a tie-breaker — conservative / fail-safe.
        per_type: dict[SensorType, SensorEvent] = {}
        for evt in events:
            existing = per_type.get(evt.sensor_type)
            if existing is None:
                per_type[evt.sensor_type] = evt
            elif evt.confidence > existing.confidence:
                per_type[evt.sensor_type] = evt
            elif evt.confidence == existing.confidence and evt.anomaly_score > existing.anomaly_score:
                per_type[evt.sensor_type] = evt

        contributing_factors: list[dict[str, Any]] = []
        for sensor_type, evt in per_type.items():
            if evt.confidence == 0.0:
                continue  # baseline not ready — skip
            contributing_factors.append(
                {
                    "sensor_type": sensor_type.value,
                    "z_score": evt.anomaly_score,
                    "confidence": evt.confidence,
                    "raw_values": evt.raw_values,
                    "verdict": evt.edge_verdict.value,
                }
            )

        if not contributing_factors:
            # All sensors have no baseline — verdict is unknown, treat as HEALTHY
            return MaintenanceVerdict(
                machine_id=machine_id,
                station_id=station_id,
                verdict=HealthVerdict.HEALTHY,
                z_score=0.0,
                confidence=0.0,
                warning_threshold=warning_t,
                critical_threshold=critical_t,
            )

        contributing_factors.sort(key=lambda f: f["z_score"], reverse=True)
        peak_z = contributing_factors[0]["z_score"]
        avg_confidence = sum(f["confidence"] for f in contributing_factors) / len(
            contributing_factors
        )

        verdict = self._map_verdict(peak_z, warning_t, critical_t)

        logger.info(
            "MaintenanceIQ.evaluate: machine=%s verdict=%s z=%.2f confidence=%.2f",
            machine_id,
            verdict.value,
            peak_z,
            avg_confidence,
        )

        return MaintenanceVerdict(
            machine_id=machine_id,
            station_id=station_id,
            verdict=verdict,
            z_score=round(peak_z, 3),
            confidence=round(avg_confidence, 3),
            contributing_factors=contributing_factors,
            warning_threshold=warning_t,
            critical_threshold=critical_t,
        )

    def recommend(
        self,
        verdict: MaintenanceVerdict,
    ) -> MaintenanceAction:
        """
        Generate a NOTIFY-ONLY maintenance action from a verdict.

        NON-NEGOTIABLE: This method returns a recommendation. It never
        sends commands to PLCs, actuators, or control systems.
        """
        action_type, urgency = self._action_from_verdict(verdict.verdict, verdict.z_score)
        sop_section = self._find_sop_section(verdict.contributing_factors)

        top_sensor = (
            verdict.contributing_factors[0]["sensor_type"]
            if verdict.contributing_factors
            else "sensor"
        )
        top_z = verdict.z_score

        action_en = (
            f"Machine {verdict.machine_id} [{verdict.station_id}]: "
            f"{top_sensor} anomaly detected (z={top_z:.2f}, {verdict.verdict.value}). "
            f"Inspect per {sop_section}."
        )
        action_zh = (
            f"{verdict.station_id}站机器{verdict.machine_id}：{top_sensor}传感器异常"
            f"（z={top_z:.2f}，{verdict.verdict.value}）。"
            f"请按{sop_section}进行检查。"
        )

        return MaintenanceAction(
            machine_id=verdict.machine_id,
            station_id=verdict.station_id,
            verdict=verdict,
            action_type=action_type,
            action_en=action_en,
            action_zh=action_zh,
            sop_section=sop_section,
            urgency=urgency,
        )

    # ── Threshold helpers ─────────────────────────────────────────────

    def _get_thresholds(self, machine_id: str) -> tuple[float, float]:
        """Return (warning, critical) thresholds for a given machine."""
        override = self._thresholds.get(machine_id)
        if override is not None:
            return override.warning, override.critical
        return self._default_warning, self._default_critical

    @staticmethod
    def _map_verdict(
        z_score: float, warning: float, critical: float
    ) -> HealthVerdict:
        """Map a z-score to a HealthVerdict given thresholds."""
        if z_score >= critical:
            return HealthVerdict.CRITICAL
        if z_score >= warning:
            return HealthVerdict.WARNING
        return HealthVerdict.HEALTHY

    def _action_from_verdict(
        self, verdict: HealthVerdict, z_score: float
    ) -> tuple[MaintenanceActionType, str]:
        """
        Determine action type and urgency from verdict and z_score.

        z_score is passed in addition to verdict so we can differentiate
        within the WARNING band (e.g. 2.0–2.9 → INSPECT, 3.0–3.49 → SERVICE).
        """
        if verdict == HealthVerdict.CRITICAL:
            return MaintenanceActionType.SERVICE, "critical"
        if verdict == HealthVerdict.WARNING:
            if z_score >= 3.0:
                return MaintenanceActionType.SERVICE, "high"
            return MaintenanceActionType.INSPECT, "normal"
        return MaintenanceActionType.MONITOR, "normal"

    def _find_sop_section(self, contributing_factors: list[dict[str, Any]]) -> str:
        """Find the most relevant SOP section from the top contributing sensor."""
        if not contributing_factors:
            return self.sop_map.get("default_section", "SOP (machine health manual)")

        top_sensor = contributing_factors[0].get("sensor_type", "")
        if top_sensor in self.sop_map:
            return self.sop_map[top_sensor]

        return self.sop_map.get("default_section", "SOP (machine health manual)")
