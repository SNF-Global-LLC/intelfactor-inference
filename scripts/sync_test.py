"""
IntelFactor.ai — Manual Inspection Sync Test
Pushes one (or all) pending inspection(s) to cloud synchronously with verbose output.

Usage:
    python3 scripts/sync_test.py                          # sync one pending inspection
    python3 scripts/sync_test.py --all                    # sync all pending
    python3 scripts/sync_test.py --id insp_20260326_...  # sync specific inspection

Required env vars:
    CLOUD_API_URL     e.g. https://api.intelbase.ai
    CLOUD_API_KEY     bearer token

Optional env vars:
    SQLITE_DB_PATH    default: /data/local.db
    EVIDENCE_DIR      default: /data/evidence
    WORKSPACE_ID      injects workspace_id if the inspection record has none
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("sync_test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual inspection sync test")
    parser.add_argument("--all", action="store_true", help="Sync all pending inspections")
    parser.add_argument("--id", dest="inspection_id", help="Sync a specific inspection_id")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without sending")
    args = parser.parse_args()

    api_url = os.environ.get("CLOUD_API_URL", "").rstrip("/")
    api_key = os.environ.get("CLOUD_API_KEY", "")
    db_path = os.environ.get("SQLITE_DB_PATH", "/data/local.db")
    evidence_dir = os.environ.get("EVIDENCE_DIR", "/data/evidence")
    workspace_id_override = os.environ.get("WORKSPACE_ID", "")

    if not api_url and not args.dry_run:
        logger.error("CLOUD_API_URL not set — cannot sync. Use --dry-run to preview payload.")
        return 1
    if not api_key and not args.dry_run:
        logger.error("CLOUD_API_KEY not set — refusing unauthenticated sync. Use --dry-run to preview payload.")
        return 1

    logger.info("DB:       %s", db_path)
    logger.info("Evidence: %s", evidence_dir)
    logger.info("API:      %s", api_url or "(dry-run)")

    from packages.inference.storage.inspection_store import InspectionStore
    from packages.inference.sync_inspections import InspectionSyncWorker

    store = InspectionStore(db_path=db_path)

    if args.inspection_id:
        events = [e for e in store.get_pending_sync(limit=500) if e.inspection_id == args.inspection_id]
        if not events:
            logger.error("Inspection %s not found in pending queue", args.inspection_id)
            return 1
    else:
        limit = 500 if args.all else 1
        events = store.get_pending_sync(limit=limit)

    if not events:
        logger.info("No pending inspections found")
        return 0

    logger.info("Found %d inspection(s) to sync", len(events))

    # Inject workspace_id override if provided
    if workspace_id_override:
        for e in events:
            if not e.workspace_id:
                logger.info("Injecting WORKSPACE_ID=%s into %s", workspace_id_override, e.inspection_id)
                e.workspace_id = workspace_id_override

    if args.dry_run:
        import json
        worker = InspectionSyncWorker(store, evidence_dir, api_url, api_key)
        for event in events:
            payload = worker._build_payload(event, "", "")
            print(f"\n--- Payload for {event.inspection_id} ---")
            print(json.dumps(payload, indent=2, default=str))
        return 0

    worker = InspectionSyncWorker(store, evidence_dir, api_url, api_key)

    success = 0
    failure = 0
    for event in events:
        try:
            worker._sync_one(event)
            success += 1
        except Exception as exc:
            logger.error("FAILED %s: %s", event.inspection_id, exc)
            failure += 1

    logger.info("Done: %d synced, %d failed", success, failure)
    return 0 if failure == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
