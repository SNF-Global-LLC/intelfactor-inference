#!/usr/bin/env python3
"""Seed local inspection rows for the edge operator console."""

from __future__ import annotations

import argparse
import base64
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.inference.schemas import BoundingBox, Detection, InspectionEvent, Verdict  # noqa: E402
from packages.inference.storage.inspection_store import InspectionStore  # noqa: E402

PLACEHOLDER_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/"
    "9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAA"
    "AAAAAAAAA/9oACAEBAAE/Iqf/2gAMAwEAAgADAAAAEP/EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQMBAT8QH//EABQRAQAAAAAAAAAAAAAAAAAAABD/"
    "2gAIAQIBAT8QH//EABQQAQAAAAAAAAAAAAAAAAAAABD/2gAIAQEAAT8QH//Z"
)


@dataclass(frozen=True)
class SeedInspection:
    inspection_id: str
    verdict: Verdict
    product_id: str
    defect_type: str
    confidence: float
    minutes_ago: int


SEED_ROWS = (
    SeedInspection(
        inspection_id="insp_seed_pass_001",
        verdict=Verdict.PASS,
        product_id="batch-A17-part-0001",
        defect_type="none",
        confidence=0.98,
        minutes_ago=12,
    ),
    SeedInspection(
        inspection_id="insp_seed_review_001",
        verdict=Verdict.REVIEW,
        product_id="batch-A17-part-0002",
        defect_type="surface_crack",
        confidence=0.79,
        minutes_ago=8,
    ),
    SeedInspection(
        inspection_id="insp_seed_defect_001",
        verdict=Verdict.FAIL,
        product_id="batch-B04-part-0007",
        defect_type="edge_burr",
        confidence=0.94,
        minutes_ago=3,
    ),
)


def _default_db_path() -> Path:
    return Path(
        os.environ.get("INSPECTION_DB_PATH")
        or os.environ.get("SQLITE_DB_PATH")
        or os.environ.get("DB_PATH")
        or "data/operator-console.db"
    )


def _default_evidence_dir() -> Path:
    return Path(os.environ.get("EVIDENCE_DIR", "data/evidence"))


def _write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(PLACEHOLDER_JPEG)


def seed_operator_console(
    db_path: str | Path,
    evidence_dir: str | Path,
    station_id: str,
    workspace_id: str = "",
    operator_id: str = "seed_operator",
    model_version: str = "operator-console-seed-v1",
) -> list[InspectionEvent]:
    """Create or replace deterministic local inspection rows for UI validation."""
    db_path = Path(db_path)
    evidence_dir = Path(evidence_dir)
    store = InspectionStore(db_path)
    now = datetime.now().replace(microsecond=0)
    events: list[InspectionEvent] = []

    for index, row in enumerate(SEED_ROWS, start=1):
        original_rel = Path("operator-console-seed") / f"{row.inspection_id}_original.jpg"
        annotated_rel = Path("operator-console-seed") / f"{row.inspection_id}_annotated.jpg"
        _write_placeholder(evidence_dir / original_rel)
        _write_placeholder(evidence_dir / annotated_rel)

        detections = []
        if row.verdict != Verdict.PASS:
            detections.append(
                Detection(
                    defect_type=row.defect_type,
                    confidence=row.confidence,
                    severity=0.72 if row.verdict == Verdict.REVIEW else 0.88,
                    threshold_used=0.75,
                    model_version=model_version,
                    bbox=BoundingBox(
                        x=32.0 + index,
                        y=44.0 + index,
                        width=128.0,
                        height=72.0,
                    ),
                )
            )

        event = InspectionEvent(
            inspection_id=row.inspection_id,
            timestamp=now - timedelta(minutes=row.minutes_ago),
            station_id=station_id,
            workspace_id=workspace_id,
            product_id=row.product_id,
            operator_id=operator_id,
            decision=row.verdict,
            confidence=row.confidence,
            detections=detections,
            num_detections=len(detections),
            image_original_path=str(original_rel),
            image_annotated_path=str(annotated_rel),
            model_version=model_version,
            model_name="seeded-local-tensorrt-placeholder",
            capture_ms=42.0 + index,
            inference_ms=18.0 + index,
            total_ms=66.0 + index,
        )
        store.save(event)
        events.append(event)

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed local PASS, REVIEW, and DEFECT inspections for /inspect UI validation."
    )
    parser.add_argument("--db-path", type=Path, default=_default_db_path())
    parser.add_argument("--evidence-dir", type=Path, default=_default_evidence_dir())
    parser.add_argument("--station-id", default=os.environ.get("STATION_ID", "station_dev"))
    parser.add_argument("--workspace-id", default=os.environ.get("WORKSPACE_ID", ""))
    parser.add_argument("--operator-id", default="seed_operator")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = seed_operator_console(
        db_path=args.db_path,
        evidence_dir=args.evidence_dir,
        station_id=args.station_id,
        workspace_id=args.workspace_id,
        operator_id=args.operator_id,
    )
    print(f"Seeded {len(events)} inspections into {args.db_path}")
    print(f"Evidence placeholders written under {args.evidence_dir / 'operator-console-seed'}")
    for event in events:
        print(f"- {event.inspection_id}: {event.decision.value} {event.product_id} {event.confidence:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
