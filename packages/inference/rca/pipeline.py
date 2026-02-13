"""
IntelFactor.ai — RCA Pipeline Orchestrator
Ties all 4 RCA layers into a single pipeline.

    Detection → Accumulator → Correlator → Explainer → Recommender → Triple

This is the core loop. Every anomaly triggers the full chain.
Every operator interaction generates a causal triple.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from packages.inference.providers.base import LanguageProvider
from packages.inference.rca.accumulator import DefectAccumulator
from packages.inference.rca.correlator import ProcessCorrelator
from packages.inference.rca.explainer import RCAExplainer
from packages.inference.rca.recommender import ActionRecommender
from packages.inference.schemas import (
    ActionRecommendation,
    AnomalyAlert,
    CausalTriple,
    DetectionResult,
    ProcessCorrelation,
    RCAExplanation,
)

logger = logging.getLogger(__name__)


@dataclass
class RCAPipelineResult:
    """Full result of an RCA pipeline run."""
    alert: AnomalyAlert
    correlations: list[ProcessCorrelation]
    explanation: RCAExplanation
    recommendation: ActionRecommendation
    triple: CausalTriple


class RCAPipeline:
    """
    Orchestrates the 4-layer RCA stack.

    Usage:
        pipeline = RCAPipeline(accumulator, correlator, explainer, recommender)
        pipeline.ingest(detection_result)  # called on every detection
        results = pipeline.run_rca()       # check for anomalies + full RCA
    """

    def __init__(
        self,
        accumulator: DefectAccumulator,
        correlator: ProcessCorrelator,
        explainer: RCAExplainer,
        recommender: ActionRecommender,
    ):
        self.accumulator = accumulator
        self.correlator = correlator
        self.explainer = explainer
        self.recommender = recommender

    def ingest(self, result: DetectionResult) -> None:
        """
        Ingest a detection result into the accumulator.
        Called on every frame that has FAIL or REVIEW verdict.
        This is the hot path — must be fast.
        """
        self.accumulator.record_event(result)

    def run_rca(
        self,
        station_id: str | None = None,
        z_threshold: float = 2.5,
    ) -> list[RCAPipelineResult]:
        """
        Run the full RCA pipeline.

        1. Check accumulator for anomalies.
        2. For each anomaly, correlate with process parameters.
        3. Generate SLM explanation.
        4. Produce SOP-linked recommendation.
        5. Create causal triple (pending operator feedback).

        This is NOT called per-frame. It runs on a schedule (e.g., every 5 min)
        or when the accumulator detects a rate spike.
        """
        # Layer 1: Check for anomalies
        alerts = self.accumulator.check_anomalies(
            station_id=station_id,
            z_threshold=z_threshold,
        )

        if not alerts:
            return []

        results: list[RCAPipelineResult] = []

        for alert in alerts:
            try:
                result = self._process_alert(alert)
                results.append(result)
            except Exception as e:
                logger.error("RCA pipeline failed for alert %s: %s", alert.alert_id, e)

        return results

    def _process_alert(self, alert: AnomalyAlert) -> RCAPipelineResult:
        """Process a single anomaly alert through the full RCA chain."""

        # Layer 2: Correlate with process parameters
        correlations = self.correlator.correlate(alert)
        logger.info(
            "Alert %s: %d correlations found", alert.alert_id, len(correlations),
        )

        # Layer 3: Generate explanation
        explanation = self.explainer.explain(alert, correlations)

        # Layer 4: Generate recommendation + store triple
        recommendation = self.recommender.recommend(alert, correlations, explanation)
        triple = self.recommender.create_triple(alert, correlations, explanation, recommendation)

        logger.info(
            "RCA complete: alert=%s triple=%s confidence=%.2f",
            alert.alert_id, triple.triple_id, explanation.confidence,
        )

        return RCAPipelineResult(
            alert=alert,
            correlations=correlations,
            explanation=explanation,
            recommendation=recommendation,
            triple=triple,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline-wide statistics."""
        return {
            "accumulator": self.accumulator.get_stats(),
            "triples": self.recommender.get_triple_stats(),
        }
