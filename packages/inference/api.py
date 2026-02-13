"""
IntelFactor.ai — Station REST API
Minimal endpoints for the local operator dashboard.

Served from the edge node itself. No external dependency.
Chinese-primary interface for Guangdong factory operators.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def create_app(runtime: Any = None) -> Any:
    """
    Create Flask app for station API.
    Pass in a running StationRuntime for live data.
    """
    try:
        from flask import Flask, jsonify, request
    except ImportError:
        raise RuntimeError("Flask required for station API (pip install flask)")

    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False  # Allow Chinese in JSON responses

    # Store runtime reference
    app.runtime = runtime

    # ── Dashboard ───────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def dashboard():
        """Serve the operator dashboard."""
        from flask import send_from_directory
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, "index.html")

    # ── Health ──────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health():
        """Health check for monitoring."""
        stats = {}
        if app.runtime:
            stats = app.runtime.get_stats()
        return jsonify({"status": "ok", "station": stats})

    # ── Station Status ─────────────────────────────────────────────

    @app.route("/api/status", methods=["GET"])
    def station_status():
        """Full station status including pipeline stats."""
        if not app.runtime:
            return jsonify({"error": "Runtime not initialized"}), 503

        stats = app.runtime.get_stats()

        # Add camera stats if available
        if hasattr(app.runtime, "_ingest") and app.runtime._ingest:
            stats["camera"] = app.runtime._ingest.get_stats()

        return jsonify(stats)

    # ── Recent Events ──────────────────────────────────────────────

    @app.route("/api/events", methods=["GET"])
    def recent_events():
        """Get recent detection events."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"events": [], "error": "Pipeline not ready"}), 503

        limit = request.args.get("limit", 50, type=int)
        verdict = request.args.get("verdict")  # PASS, FAIL, REVIEW

        acc = app.runtime.pipeline.accumulator
        if acc._conn is None:
            return jsonify({"events": []}), 200

        query = "SELECT * FROM defect_events ORDER BY timestamp DESC LIMIT ?"
        params: list[Any] = [limit]

        if verdict:
            # Note: we only store FAIL/REVIEW events
            pass

        rows = acc._conn.execute(query, params).fetchall()
        columns = [desc[0] for desc in acc._conn.execute(
            "SELECT * FROM defect_events LIMIT 0"
        ).description] if rows else []

        events = [dict(zip(columns, row)) for row in rows]
        return jsonify({"events": events, "count": len(events)})

    # ── Anomaly Alerts ─────────────────────────────────────────────

    @app.route("/api/alerts", methods=["GET"])
    def anomaly_alerts():
        """Get current anomaly alerts."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"alerts": []}), 503

        acc = app.runtime.pipeline.accumulator
        if acc._conn is None:
            return jsonify({"alerts": []}), 200

        rows = acc._conn.execute(
            """SELECT * FROM anomaly_alerts
               WHERE acknowledged = 0
               ORDER BY timestamp DESC LIMIT 20"""
        ).fetchall()

        columns = [desc[0] for desc in acc._conn.execute(
            "SELECT * FROM anomaly_alerts LIMIT 0"
        ).description] if rows else []

        alerts = [dict(zip(columns, row)) for row in rows]
        return jsonify({"alerts": alerts})

    # ── RCA Recommendations (Pending) ──────────────────────────────

    @app.route("/api/recommendations", methods=["GET"])
    def pending_recommendations():
        """Get pending recommendations awaiting operator response."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"recommendations": []}), 503

        station_id = request.args.get("station_id")
        triples = app.runtime.pipeline.recommender.get_pending_triples(station_id)
        return jsonify({"recommendations": triples})

    # ── Operator Feedback ──────────────────────────────────────────

    @app.route("/api/feedback", methods=["POST"])
    def record_feedback():
        """
        Record operator feedback on a recommendation.
        Body: {triple_id, action: accepted|rejected|modified, operator_id, outcome?, rejection_reason?}
        """
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"error": "Pipeline not ready"}), 503

        data = request.get_json()
        if not data or "triple_id" not in data or "action" not in data:
            return jsonify({"error": "Missing triple_id or action"}), 400

        from packages.inference.schemas import OperatorAction

        try:
            action = OperatorAction(data["action"])
        except ValueError:
            return jsonify({"error": f"Invalid action: {data['action']}"}), 400

        updated = app.runtime.pipeline.recommender.record_operator_feedback(
            triple_id=data["triple_id"],
            action=action,
            operator_id=data.get("operator_id", ""),
            outcome=data.get("outcome"),
            rejection_reason=data.get("rejection_reason", ""),
        )

        if updated is None:
            return jsonify({"error": "Triple not found"}), 404

        return jsonify({
            "status": "recorded",
            "triple_id": updated.triple_id,
            "new_status": updated.status.value,
        })

    # ── Triple Stats ───────────────────────────────────────────────

    @app.route("/api/triples/stats", methods=["GET"])
    def triple_stats():
        """Get causal triple collection statistics."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({}), 503

        stats = app.runtime.pipeline.recommender.get_triple_stats()
        return jsonify(stats)

    # ── Process Parameter Drift ────────────────────────────────────

    @app.route("/api/drift", methods=["GET"])
    def parameter_drift():
        """Get current process parameter drift status."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"drifts": []}), 503

        station_id = request.args.get("station_id")
        drifts = app.runtime.pipeline.correlator.check_drift(station_id)
        return jsonify({"drifts": drifts})

    # ── Manual Parameter Reading ───────────────────────────────────

    @app.route("/api/reading", methods=["POST"])
    def record_reading():
        """
        Record a manual process parameter reading from operator.
        Body: {parameter_name, value, station_id?}
        Used when sensors aren't available (most Yangjiang factories).
        """
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"error": "Pipeline not ready"}), 503

        data = request.get_json()
        if not data or "parameter_name" not in data or "value" not in data:
            return jsonify({"error": "Missing parameter_name or value"}), 400

        from packages.inference.rca.correlator import ParameterReading

        reading = ParameterReading(
            parameter_name=data["parameter_name"],
            value=float(data["value"]),
            station_id=data.get("station_id", app.runtime.config.station_id),
        )
        app.runtime.pipeline.correlator.record_reading(reading)

        return jsonify({"status": "recorded", "parameter": data["parameter_name"]})

    # ── Pipeline Stats ─────────────────────────────────────────────

    @app.route("/api/pipeline/stats", methods=["GET"])
    def pipeline_stats():
        """Get full pipeline statistics."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({}), 503
        return jsonify(app.runtime.pipeline.get_stats())

    # ── Evidence Frames ────────────────────────────────────────────

    @app.route("/api/evidence/<event_id>", methods=["GET"])
    def get_evidence(event_id):
        """Serve evidence JPEG frame for a given event."""
        from flask import send_file, abort

        if not app.runtime or not hasattr(app.runtime, "evidence_writer") or not app.runtime.evidence_writer:
            abort(404)

        frame_path = app.runtime.evidence_writer.get_frame_path(event_id)
        if frame_path is None or not frame_path.exists():
            abort(404)

        return send_file(str(frame_path), mimetype="image/jpeg")

    # ── Evidence Stats ─────────────────────────────────────────────

    @app.route("/api/evidence/stats", methods=["GET"])
    def evidence_stats():
        """Get evidence disk usage stats."""
        if not app.runtime or not hasattr(app.runtime, "evidence_writer") or not app.runtime.evidence_writer:
            return jsonify({}), 503
        return jsonify(app.runtime.evidence_writer.get_stats())

    return app


def run_api(runtime: Any = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the station API server."""
    app = create_app(runtime)
    logger.info("Station API starting on %s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)
