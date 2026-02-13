"""
IntelFactor.ai — SQLite Triple Store
Local storage for causal triples.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from .base import TripleStore
from .sqlite_base import get_connection

logger = logging.getLogger(__name__)


class SQLiteTripleStore(TripleStore):
    """SQLite-backed causal triple storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = get_connection(db_path)

    def insert(self, triple: dict[str, Any]) -> str:
        """Insert a causal triple."""
        triple_id = triple.get("triple_id", "")
        if not triple_id:
            raise ValueError("triple_id is required")

        # Serialize outcome_measured dict to JSON
        outcome = triple.get("outcome_measured", {})
        outcome_json = json.dumps(outcome, ensure_ascii=False) if outcome else None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO causal_triples
                (triple_id, timestamp, station_id, defect_event_id, defect_type,
                 defect_severity, cause_parameter, cause_value, cause_target,
                 cause_drift_pct, cause_confidence, cause_explanation_zh,
                 cause_explanation_en, recommendation_id, operator_action,
                 operator_id, outcome_measured, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                triple_id,
                triple.get("timestamp", datetime.now().isoformat()),
                triple.get("station_id", ""),
                triple.get("defect_event_id", ""),
                triple.get("defect_type", ""),
                triple.get("defect_severity", 0),
                triple.get("cause_parameter", ""),
                triple.get("cause_value"),
                triple.get("cause_target"),
                triple.get("cause_drift_pct"),
                triple.get("cause_confidence"),
                triple.get("cause_explanation_zh", ""),
                triple.get("cause_explanation_en", ""),
                triple.get("recommendation_id", ""),
                triple.get("operator_action", "pending"),
                triple.get("operator_id", ""),
                outcome_json,
                triple.get("status", "pending"),
            ),
        )
        self._conn.commit()
        return triple_id

    def get(self, triple_id: str) -> dict[str, Any] | None:
        """Get a single triple by ID."""
        row = self._conn.execute(
            "SELECT * FROM causal_triples WHERE triple_id = ?", (triple_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_dict(row)

    def list(
        self,
        limit: int = 50,
        status: str | None = None,
        station_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List triples with optional filters."""
        query = "SELECT * FROM causal_triples WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update(self, triple_id: str, updates: dict[str, Any]) -> bool:
        """Update a triple. Returns True if found and updated."""
        # Build dynamic update
        allowed_fields = {
            "operator_action", "operator_id", "outcome_measured", "status"
        }
        set_clauses = []
        params: list[Any] = []

        for key, value in updates.items():
            if key in allowed_fields:
                if key == "outcome_measured":
                    value = json.dumps(value, ensure_ascii=False) if value else None
                set_clauses.append(f"{key} = ?")
                params.append(value)

        if not set_clauses:
            return False

        query = f"UPDATE causal_triples SET {', '.join(set_clauses)} WHERE triple_id = ?"
        params.append(triple_id)

        cursor = self._conn.execute(query, params)
        self._conn.commit()
        return cursor.rowcount > 0

    def _row_to_dict(self, row) -> dict[str, Any]:
        """Convert a sqlite Row to dict with parsed JSON fields."""
        d = dict(row)

        # Parse outcome_measured JSON
        if d.get("outcome_measured"):
            try:
                d["outcome_measured"] = json.loads(d["outcome_measured"])
            except json.JSONDecodeError:
                d["outcome_measured"] = {}
        else:
            d["outcome_measured"] = {}

        return d
