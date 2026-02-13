"""
IntelFactor.ai — Process Parameter Correlator (RCA Layer 2)
Cross-references defect anomalies with process parameter drift.

Statistical, not ML. Pearson correlation on time-aligned windows.
Explainable and auditable. No GPU required.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from packages.inference.schemas import AnomalyAlert, ProcessCorrelation

logger = logging.getLogger(__name__)


@dataclass
class ProcessParameter:
    """A single process parameter definition from edge.yaml."""
    name: str
    unit: str = ""
    target: float = 0.0
    tolerance: float = 0.0          # ± from target
    data_source: str = "manual"     # manual | mqtt | opcua
    mqtt_topic: str = ""
    opcua_node: str = ""


@dataclass
class ParameterReading:
    """A single timestamped reading of a process parameter."""
    parameter_name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None))
    station_id: str = ""


class ProcessCorrelator:
    """
    Correlates defect anomalies with process parameter drift.

    Design:
    - Loads process parameters from edge.yaml config bundle.
    - Accepts readings from MQTT, OPC UA, or manual operator input.
    - When an anomaly alert fires (from Accumulator), checks if any
      parameter drifted outside tolerance in the same time window.
    - Uses Pearson coefficient on 30-minute time-aligned windows.
    - Output is structured JSON, not natural language (that's Layer 3's job).
    """

    def __init__(self, parameters: list[ProcessParameter] | None = None):
        self.parameters: dict[str, ProcessParameter] = {}
        self._readings: list[ParameterReading] = []
        self._max_readings = 50_000  # ~1 day of per-minute readings across 10 params

        if parameters:
            for p in parameters:
                self.parameters[p.name] = p

    def load_parameters(self, edge_yaml: dict[str, Any]) -> None:
        """Load process parameters from edge.yaml config."""
        params = edge_yaml.get("process_parameters", {})
        for name, spec in params.items():
            self.parameters[name] = ProcessParameter(
                name=name,
                unit=spec.get("unit", ""),
                target=float(spec.get("target", 0)),
                tolerance=float(spec.get("tolerance", 0)),
                data_source=spec.get("data_source", "manual"),
                mqtt_topic=spec.get("mqtt_topic", ""),
                opcua_node=spec.get("opcua_node", ""),
            )
        logger.info("Loaded %d process parameters", len(self.parameters))

    def record_reading(self, reading: ParameterReading) -> None:
        """Record a process parameter reading."""
        self._readings.append(reading)

        # Trim old readings to prevent unbounded memory growth
        if len(self._readings) > self._max_readings:
            cutoff = datetime.now(tz=timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
            self._readings = [r for r in self._readings if r.timestamp > cutoff]

    def record_readings_batch(self, readings: list[ParameterReading]) -> None:
        """Record multiple readings at once."""
        for r in readings:
            self.record_reading(r)

    def correlate(
        self,
        alert: AnomalyAlert,
        window_minutes: int = 30,
        min_correlation: float = 0.5,
    ) -> list[ProcessCorrelation]:
        """
        Find process parameters that correlate with a defect anomaly.

        For each parameter:
        1. Get readings in the anomaly time window.
        2. Check if the parameter drifted outside tolerance.
        3. Compute Pearson r between parameter values and defect timing.
        4. Return correlations above min_correlation threshold.
        """
        results: list[ProcessCorrelation] = []

        window_end = alert.timestamp
        window_start = window_end - timedelta(minutes=window_minutes)

        for param_name, param_spec in self.parameters.items():
            correlation = self._correlate_parameter(
                alert, param_spec, window_start, window_end, min_correlation
            )
            if correlation is not None:
                results.append(correlation)

        # Sort by confidence (abs Pearson r) descending
        results.sort(key=lambda c: abs(c.pearson_r), reverse=True)
        return results

    def check_drift(self, station_id: str | None = None) -> list[dict[str, Any]]:
        """
        Check all parameters for current drift from target.
        Returns list of drifted parameters with current values.
        Useful for operator dashboard display independent of anomaly alerts.
        """
        now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        lookback = now - timedelta(minutes=5)
        drifts: list[dict[str, Any]] = []

        for param_name, param_spec in self.parameters.items():
            if param_spec.target == 0 and param_spec.tolerance == 0:
                continue  # no target defined

            recent = [
                r for r in self._readings
                if r.parameter_name == param_name
                and r.timestamp > lookback
                and (station_id is None or r.station_id == station_id)
            ]

            if not recent:
                continue

            current = recent[-1].value
            drift = abs(current - param_spec.target)

            if drift > param_spec.tolerance:
                drift_pct = (drift / param_spec.target * 100) if param_spec.target != 0 else 0
                drifts.append({
                    "parameter": param_name,
                    "current_value": current,
                    "target": param_spec.target,
                    "tolerance": param_spec.tolerance,
                    "drift": round(drift, 3),
                    "drift_pct": round(drift_pct, 1),
                    "unit": param_spec.unit,
                    "station_id": station_id or "all",
                })

        return drifts

    def _correlate_parameter(
        self,
        alert: AnomalyAlert,
        param: ProcessParameter,
        window_start: datetime,
        window_end: datetime,
        min_correlation: float,
    ) -> ProcessCorrelation | None:
        """Correlate a single parameter with the anomaly."""

        # Get readings in window for this parameter and station
        readings = [
            r for r in self._readings
            if r.parameter_name == param.name
            and r.station_id == alert.station_id
            and window_start <= r.timestamp <= window_end
        ]

        if len(readings) < 3:
            return None  # need at least 3 points for meaningful correlation

        current_value = readings[-1].value
        drift = abs(current_value - param.target)
        drift_pct = (drift / param.target * 100) if param.target != 0 else 0

        # Check if drifted beyond tolerance
        if param.tolerance > 0 and drift <= param.tolerance:
            return None  # within tolerance, not interesting

        # Compute Pearson r between parameter deviation and time
        # (proxy for "parameter drifting correlates with defect timing")
        values = [r.value for r in readings]
        timestamps = [(r.timestamp - window_start).total_seconds() for r in readings]

        pearson_r = self._pearson(timestamps, values)
        if pearson_r is None or abs(pearson_r) < min_correlation:
            return None

        confidence = min(abs(pearson_r), 0.99)

        return ProcessCorrelation(
            anomaly_alert_id=alert.alert_id,
            parameter_name=param.name,
            current_value=round(current_value, 3),
            target_value=param.target,
            tolerance=param.tolerance,
            drift_pct=round(drift_pct, 1),
            pearson_r=round(pearson_r, 3),
            confidence=round(confidence, 3),
            time_window_minutes=int((window_end - window_start).total_seconds() / 60),
        )

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float | None:
        """Compute Pearson correlation coefficient."""
        n = len(x)
        if n < 3 or n != len(y):
            return None

        try:
            mean_x = statistics.mean(x)
            mean_y = statistics.mean(y)
            stdev_x = statistics.stdev(x)
            stdev_y = statistics.stdev(y)

            if stdev_x == 0 or stdev_y == 0:
                return None

            covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
            return covariance / (stdev_x * stdev_y)
        except (ZeroDivisionError, statistics.StatisticsError):
            return None

    @classmethod
    def from_edge_yaml(cls, edge_yaml: dict[str, Any]) -> ProcessCorrelator:
        """Factory: create correlator from edge.yaml config."""
        instance = cls()
        instance.load_parameters(edge_yaml)
        return instance
