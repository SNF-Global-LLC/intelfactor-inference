"""
IntelFactor.ai — Frame Annotation
Draw bounding boxes, labels, and verdict banner on inspection frames.

Colors:
- Green: PASS / low severity
- Orange: REVIEW / medium severity
- Red: FAIL / high severity

The annotated frame is a COPY — the original capture is never modified.
"""

from __future__ import annotations

import logging

import numpy as np

from packages.inference.schemas import Detection, Verdict

logger = logging.getLogger(__name__)

# BGR colors for OpenCV
COLOR_GREEN = (0, 200, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

VERDICT_COLORS = {
    Verdict.PASS: COLOR_GREEN,
    Verdict.REVIEW: COLOR_ORANGE,
    Verdict.FAIL: COLOR_RED,
}


def _severity_color(severity: float) -> tuple[int, int, int]:
    """Map severity [0,1] to a BGR color."""
    if severity >= 0.7:
        return COLOR_RED
    elif severity >= 0.4:
        return COLOR_ORANGE
    return COLOR_GREEN


def annotate_frame(
    frame: np.ndarray,
    detections: list[Detection],
    verdict: Verdict,
    confidence: float,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, and verdict banner on a frame copy.

    Args:
        frame: Original capture frame (Mono8 or BGR). NOT modified.
        detections: Enriched detections with severity.
        verdict: PASS / REVIEW / FAIL.
        confidence: Overall verdict confidence.

    Returns:
        Annotated BGR frame (always 3-channel).
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available — returning unannotated frame")
        return frame.copy()

    # Ensure 3-channel BGR for drawing
    if len(frame.shape) == 2:
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 1:
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        annotated = frame.copy()

    h, w = annotated.shape[:2]

    # Draw bounding boxes
    for det in detections:
        bx = int(det.bbox.x)
        by = int(det.bbox.y)
        bw = int(det.bbox.width)
        bh = int(det.bbox.height)
        color = _severity_color(det.severity)

        # Box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)

        # Label background
        label = f"{det.defect_type} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (bx, max(by - th - 6, 0)), (bx + tw + 4, by), color, -1)
        cv2.putText(
            annotated, label,
            (bx + 2, max(by - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1,
        )

    # Verdict banner at top
    banner_h = 40
    banner_color = VERDICT_COLORS.get(verdict, COLOR_GREEN)
    cv2.rectangle(annotated, (0, 0), (w, banner_h), banner_color, -1)

    banner_text = f"{verdict.value}  ({confidence:.0%})"
    if detections:
        banner_text += f"  |  {len(detections)} defect(s)"
    cv2.putText(
        annotated, banner_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2,
    )

    return annotated
