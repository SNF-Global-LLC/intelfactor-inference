"""
IntelFactor.ai — Verdict Logic
Deterministic PASS / REVIEW / FAIL evaluation for single-frame inspection.

Rules:
- FAIL: any detection >= fail_threshold
- REVIEW: any detection >= review_threshold (but below fail)
- PASS: no detections above review_threshold
- Confidence reported = max detection confidence (or 1.0 for clean PASS)

Severity ranges are loaded from configs/wiko_taxonomy.yaml (canonical source).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from packages.inference.schemas import Detection, Verdict

logger = logging.getLogger(__name__)


def _load_severity_ranges() -> dict[str, tuple[float, float]]:
    """Load severity ranges from canonical taxonomy config."""
    ranges: dict[str, tuple[float, float]] = {}
    config_paths = [
        Path("/opt/intelfactor/config/wiko_taxonomy.yaml"),
        Path(__file__).parent.parent.parent / "configs" / "wiko_taxonomy.yaml",
    ]
    
    for path in config_paths:
        if path.exists():
            try:
                import yaml
                data = yaml.safe_load(path.read_text())
                for defect_name, defect_info in data.get("defects", {}).items():
                    if "severity_range" in defect_info:
                        sr = defect_info["severity_range"]
                        ranges[defect_name] = (sr[0], sr[1])
                logger.info("Loaded severity ranges for %d defects from %s", len(ranges), path)
                return ranges
            except Exception as exc:
                logger.warning("Failed to load taxonomy from %s: %s", path, exc)
                continue
    
    logger.warning("No taxonomy config found, using fallback severity mapping")
    return {}


# Lazy-loaded on first use
_SEVERITY_RANGES: dict[str, tuple[float, float]] | None = None


def _get_severity_ranges() -> dict[str, tuple[float, float]]:
    """Get cached severity ranges, loading if needed."""
    global _SEVERITY_RANGES
    if _SEVERITY_RANGES is None:
        _SEVERITY_RANGES = _load_severity_ranges()
    return _SEVERITY_RANGES


def estimate_severity(defect_type: str, confidence: float) -> float:
    """
    Map (defect_type, confidence) to a severity score in [0, 1].
    Uses linear interpolation within the defect's severity range from taxonomy.
    Falls back to raw confidence for unknown types.
    """
    ranges = _get_severity_ranges()
    if defect_type in ranges:
        lo, hi = ranges[defect_type]
        return lo + (hi - lo) * confidence
    return confidence


def evaluate_verdict(
    detections: list[Detection],
    fail_threshold: float = 0.85,
    review_threshold: float = 0.50,
) -> tuple[Verdict, float, list[Detection]]:
    """
    Evaluate inspection verdict from a list of detections.

    Args:
        detections: Raw detections from vision provider.
        fail_threshold: Confidence >= this triggers FAIL (default: 0.85 per CLAUDE.md).
        review_threshold: Confidence >= this (but < fail) triggers REVIEW (default: 0.50).

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
