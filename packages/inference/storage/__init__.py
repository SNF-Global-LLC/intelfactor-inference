"""
IntelFactor.ai — Storage Abstraction Layer
Supports STORAGE_MODE=local (SQLite) or cloud (DynamoDB/Postgres).
"""

from .factory import (
    get_event_store,
    get_evidence_store,
    get_triple_store,
    get_storage_mode,
)

__all__ = [
    "get_event_store",
    "get_evidence_store",
    "get_triple_store",
    "get_storage_mode",
]
