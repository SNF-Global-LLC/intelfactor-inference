"""
IntelFactor.ai — Action Recommender (RCA Layer 4)
SOP-linked corrective actions with NOTIFY-ONLY safety rails.
Also responsible for generating and storing causal triples.

NON-NEGOTIABLE: Agent suggests. Agent NEVER executes.
No automated parameter changes. No automated line stops.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from packages.inference.schemas import (
    ActionRecommendation,
    AnomalyAlert,
    CausalTriple,
    OperatorAction,
    ProcessCorrelation,
    RCAExplanation,
    TripleStatus,
)

logger = logging.getLogger(__name__)

TRIPLE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS causal_triples (
    triple_id           TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    station_id          TEXT NOT NULL,

    -- Defect
    defect_event_id     TEXT NOT NULL,
    defect_type         TEXT NOT NULL,
    defect_severity     REAL DEFAULT 0.0,

    -- Cause
    cause_parameter     TEXT DEFAULT '',
    cause_value         REAL DEFAULT 0.0,
    cause_target        REAL DEFAULT 0.0,
    cause_drift_pct     REAL DEFAULT 0.0,
    cause_confidence    REAL DEFAULT 0.0,
    cause_explanation_zh TEXT DEFAULT '',
    cause_explanation_en TEXT DEFAULT '',

    -- Outcome
    recommendation_id   TEXT DEFAULT '',
    operator_action     TEXT DEFAULT 'pending',
    operator_id         TEXT DEFAULT '',
    outcome_measured    TEXT DEFAULT '{}',

    -- Status
    status              TEXT DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_triples_station
    ON causal_triples(station_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_triples_status
    ON causal_triples(status);

CREATE INDEX IF NOT EXISTS idx_triples_defect_type
    ON causal_triples(defect_type);
"""


class ActionRecommender:
    """
    Generates SOP-linked corrective actions from RCA analysis.
    Stores causal triples for the data flywheel.

    SAFETY RAILS:
    - Output is RECOMMENDATION ONLY.
    - All recommendations include evidence links.
    - Operator must explicitly accept/reject.
    - Rejection reason stored as training signal.
    """

    def __init__(
        self,
        sop_map: dict[str, Any] | None = None,
        db_path: str | Path = "/opt/intelfactor/data/triples.db",
    ):
        self.sop_map = sop_map or {}
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def start(self) -> None:
        """Initialize triple storage."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(TRIPLE_SCHEMA_SQL)
        self._conn.commit()
        logger.info("ActionRecommender started: %s", self.db_path)

    def recommend(
        self,
        alert: AnomalyAlert,
        correlations: list[ProcessCorrelation],
        explanation: RCAExplanation,
    ) -> ActionRecommendation:
        """
        Generate SOP-linked action recommendation.

        NEVER EXECUTES. Only produces a recommendation for operator display.
        """
        if correlations:
            top = correlations[0]
            sop_section = self._find_sop_section(top.parameter_name, alert.defect_type)

            action_en = (
                f"Station {alert.station_id}: {top.parameter_name} at {top.current_value} "
                f"(target {top.target_value} ±{top.tolerance}). "
                f"Recalibrate per {sop_section}."
            )
            action_zh = (
                f"{alert.station_id}站：{top.parameter_name}当前值{top.current_value}"
                f"（目标{top.target_value}±{top.tolerance}）。"
                f"请按{sop_section}重新校准。"
            )
            urgency = "high" if abs(top.drift_pct) > 10 else "normal"
            param_target = f"{top.parameter_name}: {top.target_value} ±{top.tolerance}"
        else:
            sop_section = "SOP (general inspection)"
            action_en = (
                f"Station {alert.station_id}: {alert.defect_type} rate anomaly detected "
                f"(z={alert.z_score}). Inspect station and review recent process changes."
            )
            action_zh = (
                f"{alert.station_id}站：{alert.defect_type}缺陷率异常"
                f"（z={alert.z_score}）。请检查工位并复查近期工艺变更。"
            )
            urgency = "normal"
            param_target = ""

        rec = ActionRecommendation(
            sop_section=sop_section,
            action_en=action_en,
            action_zh=action_zh,
            parameter_target=param_target,
            urgency=urgency,
            evidence_ids=alert.event_ids[:10],  # cap evidence links
        )

        logger.info(
            "Recommendation generated: station=%s sop=%s urgency=%s",
            alert.station_id, sop_section, urgency,
        )
        return rec

    def create_triple(
        self,
        alert: AnomalyAlert,
        correlations: list[ProcessCorrelation],
        explanation: RCAExplanation,
        recommendation: ActionRecommendation,
    ) -> CausalTriple:
        """
        Create and store a causal triple. The fundamental unit of IntelFactor's data moat.
        Triple is initially PENDING until operator provides feedback.
        """
        top_corr = correlations[0] if correlations else None

        triple = CausalTriple(
            station_id=alert.station_id,
            defect_event_id=alert.event_ids[0] if alert.event_ids else alert.alert_id,
            defect_type=alert.defect_type,
            defect_severity=alert.current_rate / max(alert.baseline_rate, 0.01),
            cause_parameter=top_corr.parameter_name if top_corr else "",
            cause_value=top_corr.current_value if top_corr else 0.0,
            cause_target=top_corr.target_value if top_corr else 0.0,
            cause_drift_pct=top_corr.drift_pct if top_corr else 0.0,
            cause_confidence=explanation.confidence,
            cause_explanation_zh=explanation.explanation_zh,
            cause_explanation_en=explanation.explanation_en,
            recommendation_id=recommendation.recommendation_id,
            status=TripleStatus.PENDING,
        )

        self._store_triple(triple)
        return triple

    def record_operator_feedback(
        self,
        triple_id: str,
        action: OperatorAction,
        operator_id: str = "",
        outcome: dict[str, Any] | None = None,
        rejection_reason: str = "",
    ) -> CausalTriple | None:
        """
        Record operator feedback on a recommendation.
        This closes the loop and turns PENDING into VERIFIED or DISPUTED.
        """
        if self._conn is None:
            raise RuntimeError("Recommender not started.")

        status = TripleStatus.VERIFIED if action == OperatorAction.ACCEPTED else TripleStatus.DISPUTED
        outcome_json = json.dumps(outcome or {}, default=str)

        if rejection_reason:
            # Store rejection reason in outcome for training signal
            outcome_data = outcome or {}
            outcome_data["rejection_reason"] = rejection_reason
            outcome_json = json.dumps(outcome_data, default=str)

        self._conn.execute(
            """UPDATE causal_triples
               SET operator_action = ?, operator_id = ?,
                   outcome_measured = ?, status = ?
               WHERE triple_id = ?""",
            (action.value, operator_id, outcome_json, status.value, triple_id),
        )
        self._conn.commit()

        logger.info(
            "Operator feedback recorded: triple=%s action=%s status=%s",
            triple_id, action.value, status.value,
        )

        # Return updated triple
        return self._load_triple(triple_id)

    def get_pending_triples(self, station_id: str | None = None) -> list[dict[str, Any]]:
        """Get triples awaiting operator feedback."""
        if self._conn is None:
            raise RuntimeError("Recommender not started.")

        query = "SELECT * FROM causal_triples WHERE status = 'pending'"
        params: list[str] = []
        if station_id:
            query += " AND station_id = ?"
            params.append(station_id)
        query += " ORDER BY timestamp DESC LIMIT 50"

        rows = self._conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in self._conn.execute(query, params).description] if rows else []
        return [dict(zip(columns, row)) for row in rows]

    def get_triple_stats(self) -> dict[str, Any]:
        """Get triple collection statistics for monitoring."""
        if self._conn is None:
            return {"status": "not_started"}

        total = self._conn.execute("SELECT COUNT(*) FROM causal_triples").fetchone()[0]
        verified = self._conn.execute("SELECT COUNT(*) FROM causal_triples WHERE status = 'verified'").fetchone()[0]
        pending = self._conn.execute("SELECT COUNT(*) FROM causal_triples WHERE status = 'pending'").fetchone()[0]
        disputed = self._conn.execute("SELECT COUNT(*) FROM causal_triples WHERE status = 'disputed'").fetchone()[0]

        acceptance_rate = (verified / total * 100) if total > 0 else 0

        return {
            "total_triples": total,
            "verified": verified,
            "pending": pending,
            "disputed": disputed,
            "acceptance_rate_pct": round(acceptance_rate, 1),
        }

    def _find_sop_section(self, parameter_name: str, defect_type: str) -> str:
        """Look up the relevant SOP section for a parameter + defect combination."""
        # Check parameter-specific SOP mappings
        param_sops = self.sop_map.get("parameter_sops", {})
        if parameter_name in param_sops:
            return param_sops[parameter_name]

        # Check defect-specific SOP mappings
        defect_sops = self.sop_map.get("defect_sops", {})
        if defect_type in defect_sops:
            return defect_sops[defect_type]

        # Default
        return self.sop_map.get("default_section", "SOP (refer to quality manual)")

    def _store_triple(self, triple: CausalTriple) -> None:
        """Persist triple to SQLite."""
        if self._conn is None:
            raise RuntimeError("Recommender not started.")

        self._conn.execute(
            """INSERT OR IGNORE INTO causal_triples
               (triple_id, timestamp, station_id,
                defect_event_id, defect_type, defect_severity,
                cause_parameter, cause_value, cause_target, cause_drift_pct,
                cause_confidence, cause_explanation_zh, cause_explanation_en,
                recommendation_id, operator_action, operator_id, outcome_measured,
                status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                triple.triple_id,
                triple.timestamp.isoformat(),
                triple.station_id,
                triple.defect_event_id,
                triple.defect_type,
                triple.defect_severity,
                triple.cause_parameter,
                triple.cause_value,
                triple.cause_target,
                triple.cause_drift_pct,
                triple.cause_confidence,
                triple.cause_explanation_zh,
                triple.cause_explanation_en,
                triple.recommendation_id,
                triple.operator_action.value,
                triple.operator_id,
                json.dumps(triple.outcome_measured, default=str),
                triple.status.value,
            ),
        )
        self._conn.commit()

    def _load_triple(self, triple_id: str) -> CausalTriple | None:
        """Load a single triple from storage."""
        if self._conn is None:
            return None

        row = self._conn.execute(
            "SELECT * FROM causal_triples WHERE triple_id = ?", (triple_id,)
        ).fetchone()

        if row is None:
            return None

        columns = [desc[0] for desc in self._conn.execute(
            "SELECT * FROM causal_triples LIMIT 0"
        ).description]
        data = dict(zip(columns, row))

        return CausalTriple(
            triple_id=data["triple_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            station_id=data["station_id"],
            defect_event_id=data["defect_event_id"],
            defect_type=data["defect_type"],
            defect_severity=data["defect_severity"],
            cause_parameter=data["cause_parameter"],
            cause_value=data["cause_value"],
            cause_target=data["cause_target"],
            cause_drift_pct=data["cause_drift_pct"],
            cause_confidence=data["cause_confidence"],
            cause_explanation_zh=data["cause_explanation_zh"],
            cause_explanation_en=data["cause_explanation_en"],
            recommendation_id=data["recommendation_id"],
            operator_action=OperatorAction(data["operator_action"]),
            operator_id=data["operator_id"],
            outcome_measured=json.loads(data["outcome_measured"]),
            status=TripleStatus(data["status"]),
        )

    def stop(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
