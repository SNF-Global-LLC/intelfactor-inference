"""
IntelFactor.ai — Inspection Transaction Service
Orchestrates a single discrete inspection: capture → infer → verdict → annotate → save.

This is the core of Phase 1: Manual QC Station.
One knife placed under camera → one button click → one verdict.

NOT part of the continuous runtime loop. Called on-demand via POST /api/inspect.

After local completion, the inspection event is persisted to SQLite
with sync_status='pending'. The background sync worker picks it up later.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from packages.inference.annotate import annotate_frame
from packages.inference.schemas import (
    Detection,
    DetectionResult,
    InspectionEvent,
    SyncStatus,
    Verdict,
)
from packages.inference.verdict import evaluate_verdict

logger = logging.getLogger(__name__)


def run_inspection(
    runtime: Any,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute one full inspection transaction.

    Flow:
        1. Capture frame (from runtime._capture)
        2. Run TRT inference (runtime.vision.detect)
        3. Evaluate verdict (deterministic rules)
        4. Annotate frame copy (bounding boxes + banner)
        5. Save evidence (original + annotated + JSON report)
        6. Persist InspectionEvent to SQLite (sync_status=pending)
        7. Feed RCA pipeline (FAIL/REVIEW only)
        8. Return structured JSON result

    Args:
        runtime: StationRuntime instance (provides .vision, ._capture, .evidence_writer, .config)
        metadata: Optional dict with product_id, operator_id, workspace_id, etc.

    Returns:
        Dict with inspection result (JSON-serializable).
    """
    metadata = metadata or {}
    t_start = time.perf_counter()

    station_id = runtime.config.station_id
    workspace_id = metadata.get("workspace_id", "")
    now = datetime.now(tz=timezone.utc)
    inspection_id = f"{station_id}-{now.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # ── 1. Capture ──────────────────────────────────────────────────
    capture = getattr(runtime, "_capture", None)
    if capture is None:
        return _error_result(inspection_id, station_id, "No capture backend. Check camera config.")

    try:
        t_cap = time.perf_counter()
        frame = capture.capture_frame()
        capture_ms = (time.perf_counter() - t_cap) * 1000
        logger.info(
            "Captured frame: shape=%s dtype=%s (%.1fms)",
            frame.shape, frame.dtype, capture_ms,
        )
    except Exception as exc:
        logger.error("Capture failed: %s", exc)
        return _error_result(inspection_id, station_id, f"Capture failed: {exc}")

    # Keep original frame for evidence (before any preprocessing)
    original_frame = frame.copy()

    # ── 2. Inference ────────────────────────────────────────────────
    vision = runtime.vision
    if vision is None or not getattr(vision, "_loaded", False):
        return _error_result(inspection_id, station_id, "Vision model not loaded.")

    try:
        detection_result: DetectionResult = vision.detect(frame)
        raw_detections = detection_result.detections
        inference_ms = detection_result.inference_ms
        logger.info(
            "Inference: %d raw detections in %.1fms (model=%s)",
            len(raw_detections), inference_ms, detection_result.model_version,
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return _error_result(inspection_id, station_id, f"Inference failed: {exc}")

    # ── 3. Verdict ──────────────────────────────────────────────────
    fail_threshold = getattr(
        runtime.config,
        "fail_threshold",
        getattr(runtime.config, "confidence_threshold", 0.5),
    )
    review_threshold = getattr(runtime.config, "review_threshold", fail_threshold * 0.6)

    verdict, confidence, enriched = evaluate_verdict(
        raw_detections,
        fail_threshold=fail_threshold,
        review_threshold=review_threshold,
    )
    verdict, confidence = _apply_provider_policy(detection_result, verdict, confidence, enriched)

    # ── 4. Annotate ─────────────────────────────────────────────────
    annotated_frame = annotate_frame(original_frame, enriched, verdict, confidence)

    # ── 5. Save evidence ────────────────────────────────────────────
    evidence_writer = runtime.evidence_writer
    image_original_path = ""
    image_annotated_path = ""
    report_path = ""

    if evidence_writer is not None:
        try:
            # Save original capture (untouched)
            image_original_path = evidence_writer.write(
                frame=_ensure_bgr(original_frame),
                event_id=inspection_id,
                metadata={
                    "verdict": verdict.value,
                    "confidence": confidence,
                    "num_detections": len(enriched),
                    "product_id": metadata.get("product_id", ""),
                    "operator_id": metadata.get("operator_id", ""),
                    "station_id": station_id,
                    "capture_ms": capture_ms,
                    "inference_ms": inference_ms,
                },
            )

            # Save annotated frame
            annotated_id = f"{inspection_id}_annotated"
            image_annotated_path = evidence_writer.write(
                frame=annotated_frame,
                event_id=annotated_id,
            )

            # Save structured JSON report alongside evidence
            report_path = _save_report(
                evidence_writer.evidence_dir,
                inspection_id, now, station_id, workspace_id,
                metadata, verdict, confidence, enriched,
                detection_result, capture_ms, inference_ms,
                image_original_path, image_annotated_path,
            )
        except Exception as exc:
            logger.error("Evidence save failed: %s", exc)

    total_ms = (time.perf_counter() - t_start) * 1000

    # ── 6. Persist InspectionEvent to SQLite ────────────────────────
    event = InspectionEvent(
        inspection_id=inspection_id,
        timestamp=now,
        station_id=station_id,
        workspace_id=workspace_id,
        product_id=metadata.get("product_id", ""),
        operator_id=metadata.get("operator_id", ""),
        decision=verdict,
        confidence=confidence,
        detections=enriched,
        num_detections=len(enriched),
        image_original_path=image_original_path,
        image_annotated_path=image_annotated_path,
        report_path=report_path,
        model_version=detection_result.model_version,
        model_name=detection_result.model_name,
        capture_ms=capture_ms,
        inference_ms=inference_ms,
        total_ms=total_ms,
        sync_status=SyncStatus.PENDING,
    )

    inspection_store = getattr(runtime, "_inspection_store", None)
    if inspection_store is not None:
        try:
            inspection_store.save(event)
            logger.info("Inspection persisted: %s (sync_status=pending)", inspection_id)
        except Exception as exc:
            logger.error("Inspection store save failed (non-blocking): %s", exc)

    # ── 7. Feed RCA pipeline (FAIL/REVIEW only) ────────────────────
    if verdict in (Verdict.FAIL, Verdict.REVIEW) and runtime.pipeline:
        try:
            result_for_rca = DetectionResult(
                event_id=inspection_id,
                station_id=station_id,
                detections=enriched,
                verdict=verdict,
                confidence=confidence,
                inference_ms=inference_ms,
                model_version=detection_result.model_version,
                frame_ref=image_original_path,
            )
            runtime.pipeline.ingest(result_for_rca)
        except Exception as exc:
            logger.warning("RCA ingest failed (non-blocking): %s", exc)

    # ── 8. Build response ───────────────────────────────────────────
    detection_dicts = []
    for d in enriched:
        detection_dicts.append({
            "defect_type": d.defect_type,
            "confidence": round(d.confidence, 4),
            "severity": round(d.severity, 4),
            "bbox": {
                "x": round(d.bbox.x, 1),
                "y": round(d.bbox.y, 1),
                "width": round(d.bbox.width, 1),
                "height": round(d.bbox.height, 1),
            },
        })

    result = {
        "inspection_id": inspection_id,
        "timestamp": now.isoformat(),
        "station_id": station_id,
        "workspace_id": workspace_id,
        "verdict": verdict.value,
        "confidence": round(confidence, 4),
        "detections": detection_dicts,
        "num_detections": len(enriched),
        "image_original_path": image_original_path,
        "image_annotated_path": image_annotated_path,
        "report_path": report_path,
        "product_id": metadata.get("product_id", ""),
        "operator_id": metadata.get("operator_id", ""),
        "timing": {
            "capture_ms": round(capture_ms, 1),
            "inference_ms": round(inference_ms, 1),
            "total_ms": round(total_ms, 1),
        },
        "model_version": detection_result.model_version,
        "model_name": detection_result.model_name,
        "provider": _provider_summary(detection_result),
        "sync_status": "pending",
    }

    logger.info(
        "Inspection complete: %s verdict=%s conf=%.2f detections=%d total=%.0fms",
        inspection_id, verdict.value, confidence, len(enriched), total_ms,
    )

    return result


def _save_report(
    evidence_dir: Path,
    inspection_id: str,
    timestamp: datetime,
    station_id: str,
    workspace_id: str,
    metadata: dict[str, Any],
    verdict: Verdict,
    confidence: float,
    detections: list[Detection],
    detection_result: DetectionResult,
    capture_ms: float,
    inference_ms: float,
    image_original_path: str,
    image_annotated_path: str,
) -> str:
    """Save a structured JSON report alongside evidence. Returns relative path."""
    date_str = timestamp.strftime("%Y-%m-%d")
    report_dir = evidence_dir / date_str
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"{inspection_id}_report.json"

    report = {
        "inspection_id": inspection_id,
        "timestamp": timestamp.isoformat(),
        "station_id": station_id,
        "workspace_id": workspace_id,
        "product_id": metadata.get("product_id", ""),
        "operator_id": metadata.get("operator_id", ""),
        "decision": verdict.value,
        "confidence": round(confidence, 4),
        "detections": [
            {
                "class": d.defect_type,
                "confidence": round(d.confidence, 4),
                "severity": round(d.severity, 4),
                "bbox": {
                    "x": round(d.bbox.x, 1),
                    "y": round(d.bbox.y, 1),
                    "width": round(d.bbox.width, 1),
                    "height": round(d.bbox.height, 1),
                },
            }
            for d in detections
        ],
        "image_original_path": image_original_path,
        "image_annotated_path": image_annotated_path,
        "model_version": detection_result.model_version,
        "model_name": detection_result.model_name,
        "provider": _provider_summary(detection_result, include_raw_output=True),
        "timing": {
            "capture_ms": round(capture_ms, 1),
            "inference_ms": round(inference_ms, 1),
        },
        "accepted": None,
        "notes": "",
    }

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return str(report_file.relative_to(evidence_dir))


def _apply_provider_policy(
    detection_result: DetectionResult,
    verdict: Verdict,
    confidence: float,
    detections: list[Detection],
) -> tuple[Verdict, float]:
    """Apply provider-specific safety policy after deterministic verdicting."""
    metadata = detection_result.provider_metadata or {}
    if not metadata.get("experimental"):
        return verdict, confidence
    if metadata.get("verdict_policy") != "review_only":
        return verdict, confidence
    if not detections or verdict != Verdict.FAIL:
        return verdict, confidence

    policy_adjustments = metadata.setdefault("policy_adjustments", [])
    policy_adjustments.append(
        {
            "source": "inspection",
            "policy": "review_only",
            "from": Verdict.FAIL.value,
            "to": Verdict.REVIEW.value,
            "reason": "experimental_hosted_model",
        }
    )
    logger.info(
        "Inspection verdict clamped from FAIL to REVIEW for experimental hosted provider (model=%s)",
        detection_result.model_version,
    )
    return Verdict.REVIEW, confidence


def _provider_summary(
    detection_result: DetectionResult,
    include_raw_output: bool = False,
) -> dict[str, Any]:
    metadata = dict(detection_result.provider_metadata or {})
    summary = {
        "provider": metadata.get("provider", detection_result.model_name or "unknown"),
        "label": metadata.get("label", detection_result.model_name or "Unknown"),
        "detail": metadata.get("detail", detection_result.model_version),
        "mode": metadata.get("mode", ""),
        "experimental": bool(metadata.get("experimental", False)),
        "verdict_policy": metadata.get("verdict_policy", "default"),
        "policy_adjustments": metadata.get("policy_adjustments", []),
    }
    if include_raw_output and "raw_output" in metadata:
        summary["raw_output"] = metadata["raw_output"]
    return summary


def _error_result(inspection_id: str, station_id: str, error: str) -> dict[str, Any]:
    """Build an error response dict."""
    return {
        "inspection_id": inspection_id,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "station_id": station_id,
        "verdict": "ERROR",
        "confidence": 0.0,
        "detections": [],
        "num_detections": 0,
        "error": error,
        "image_original_path": "",
        "image_annotated_path": "",
        "timing": {},
    }


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is 3-channel BGR for cv2.imwrite."""
    if len(frame.shape) == 2:
        try:
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except ImportError:
            return np.stack([frame, frame, frame], axis=-1)
    if frame.shape[2] == 1:
        try:
            import cv2
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        except ImportError:
            return np.concatenate([frame, frame, frame], axis=-1)
    return frame
