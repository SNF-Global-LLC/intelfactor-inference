"""
IntelFactor.ai — Storage Factory
Returns appropriate storage backend based on STORAGE_MODE env var.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import EventStore, EvidenceStore, TripleStore

# Singleton instances
_event_store: "EventStore | None" = None
_evidence_store: "EvidenceStore | None" = None
_triple_store: "TripleStore | None" = None


def get_storage_mode() -> str:
    """Get the current storage mode (local or cloud)."""
    return os.environ.get("STORAGE_MODE", "local")


def get_event_store() -> "EventStore":
    """Get the event store instance."""
    global _event_store
    if _event_store is None:
        mode = get_storage_mode()
        if mode == "local":
            from .sqlite_events import SQLiteEventStore
            db_path = os.environ.get("SQLITE_DB_PATH", "/opt/intelfactor/data/local.db")
            _event_store = SQLiteEventStore(db_path)
        else:
            # Cloud mode - would use DynamoDB or Postgres
            raise NotImplementedError("Cloud storage not yet implemented")
    return _event_store


def get_evidence_store() -> "EvidenceStore":
    """Get the evidence store instance."""
    global _evidence_store
    if _evidence_store is None:
        mode = get_storage_mode()
        if mode == "local":
            from .sqlite_evidence import SQLiteEvidenceStore
            db_path = os.environ.get("SQLITE_DB_PATH", "/opt/intelfactor/data/local.db")
            evidence_dir = os.environ.get("EVIDENCE_DIR", "/opt/intelfactor/data/evidence")
            _evidence_store = SQLiteEvidenceStore(db_path, evidence_dir)
        else:
            raise NotImplementedError("Cloud storage not yet implemented")
    return _evidence_store


def get_triple_store() -> "TripleStore":
    """Get the triple store instance."""
    global _triple_store
    if _triple_store is None:
        mode = get_storage_mode()
        if mode == "local":
            from .sqlite_triples import SQLiteTripleStore
            db_path = os.environ.get("SQLITE_DB_PATH", "/opt/intelfactor/data/local.db")
            _triple_store = SQLiteTripleStore(db_path)
        else:
            raise NotImplementedError("Cloud storage not yet implemented")
    return _triple_store
