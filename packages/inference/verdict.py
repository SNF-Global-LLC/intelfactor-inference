"""
IntelFactor.ai — Verdict Logic
Deterministic PASS / REVIEW / FAIL evaluation for single-frame inspection.

Rules:
- FAIL: any detection >= fail_threshold
- REVIEW: any detection >= review_threshold (but below fail)
- PASS: no detections above review_threshold
- Confidence reported = max detection confidence (or 1.0 for clean PASS)
"""

from __future__ import annotations

import logging
from typing import Any

from packages.inference.schemas import Detection, Verdict

logger = logging.getLogger(__name__)

# Severity ranges per defect type (from wiko_taxonomy.yaml)
SEVERITY_RANGES: dict[str, tuple[float, float]] = {
    "scratch_surface": (0.2, 0.9),
    "scratch_edge": (0.4, 1.0),
    "burr": (0.3, 0.8),
    "pit_corrosion": (0.5, 1.0),
    "discoloration": (0.2, 0.7),
    "dent": (0.3, 0.9),
    "crack": (0.7, 1.0),
    "warp": (0.4, 1.0),
    "handle_gap": (0.3, 0.8),
    "handle_crack": (0.5, 1.0),
    "logo_defect": (0.2, 0.6),
    "dimension_out_of_spec": (0.4, 1.0),
    "foreign_material": (0.2, 0.7),
}


def estimate_severity(defect_type: str, confidence: float) -> float:
    """
    Map (defect_type, confidence) to a severity score in [0, 1].
    Uses linear interpolation within the defect's severity range.
    Falls back to raw confidence for unknown types.
    """
    if defect_type in SEVERITY_RANGES:
        lo, hi = SEVERITY_RANGES[defect_type]
        return lo + (hi - lo) * confidence
    return confidence


def evaluate_verdict(
    detections: list[Detection],
    fail_threshold: float = 0.5,
    review_threshold: float = 0.3,
) -> tuple[Verdict, float, list[Detection]]:
    """
    Evaluate inspection verdict from a list of detections.

    Args:
        detections: Raw detections from vision provider.
        fail_threshold: Confidence >= this triggers FAIL.
        review_threshold: Confidence >= this (but < fail) triggers REVIEW.

    Returns:
        (verdict, confidence, enriched_detections)
        - enriched_detections have severity populated
    """
    if not detections:
        return Verdict.PASS, 1.0, []

    # Enrich with severity
    enriched = []
    for d in detections:
        severity = estimate_severity(d.defect_type, d.confidence)
        enriched.append(Detection(
            defect_type=d.defect_type,
            confidence=d.confidence,
            bbox=d.bbox,
            severity=severity,
        ))

    max_conf = max(d.confidence for d in enriched)

    if max_conf >= fail_threshold:
        return Verdict.FAIL, max_conf, enriched
    elif max_conf >= review_threshold:
        return Verdict.REVIEW, max_conf, enriched
    else:
        return Verdict.PASS, 1.0 - max_conf, enriched
