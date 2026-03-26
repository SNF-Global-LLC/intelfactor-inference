"""
IntelFactor.ai — Station REST API v2
Enhanced API with storage abstraction and evidence serving.

Supports both STORAGE_MODE=local (SQLite) and cloud modes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_app(
    runtime: Any = None,
    sensor_service: Any = None,
    maintenance_iq: Any = None,
    machine_health_config: dict[str, Any] | None = None,
) -> Any:
    """
    Create Flask app for station API.
    Pass in a running StationRuntime for live data.
    """
    try:
        from flask import Flask, jsonify, request, send_file, abort
    except ImportError:
        raise RuntimeError("Flask required for station API (pip install flask)")

    from packages.inference.storage import (
        get_event_store,
        get_evidence_store,
        get_triple_store,
        get_storage_mode,
    )

    # Production visibility metrics
    from packages.visibility.metrics_api import metrics_bp, init_metrics

    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False  # Allow Chinese in JSON responses

    # Store runtime reference
    app.runtime = runtime

    # Get evidence directory from env
    evidence_dir = Path(os.environ.get("EVIDENCE_DIR", "/opt/intelfactor/data/evidence"))

    # Initialize production metrics
    db_path = os.environ.get("DB_PATH", "/opt/intelfactor/data/local.db")
    station_id = os.environ.get("STATION_ID", "SNF-Vision-1")
    metrics = init_metrics(app, db_path=db_path, station_id=station_id)
    app.register_blueprint(metrics_bp)

    # ── Dashboard ───────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def dashboard():
        """Serve the operator dashboard."""
        from flask import send_from_directory
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, "index.html")

    @app.route("/inspect", methods=["GET"])
    def inspect_page():
        """Serve the manual QC inspection page."""
        from flask import send_from_directory
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, "inspect.html")

    # ── Health ──────────────────────────────────────────────────────

    @app.route("/health", methods=["GET"])
    def health():
        """Health check for monitoring."""
        stats = {"storage_mode": get_storage_mode()}
        if app.runtime:
            stats.update(app.runtime.get_stats())
        return jsonify({"status": "ok", "station": stats})

    # ── Station Status ─────────────────────────────────────────────

    @app.route("/api/status", methods=["GET"])
    def station_status():
        """Full station status including pipeline stats."""
        if not app.runtime:
            return jsonify({
                "storage_mode": get_storage_mode(),
                "runtime": "not initialized"
            })

        stats = app.runtime.get_stats()
        stats["storage_mode"] = get_storage_mode()

        # Add camera stats if available
        if hasattr(app.runtime, "_ingest") and app.runtime._ingest:
            stats["camera"] = app.runtime._ingest.get_stats()

        return jsonify(stats)

    # ── Events (using storage abstraction) ──────────────────────────

    @app.route("/api/events", methods=["GET"])
    def list_events():
        """Get recent detection events."""
        limit = request.args.get("limit", 50, type=int)
        verdict = request.args.get("verdict")  # PASS, FAIL, REVIEW
        station_id = request.args.get("station_id")

        event_store = get_event_store()
        events = event_store.list(limit=limit, verdict=verdict, station_id=station_id)
        return jsonify({"events": events, "count": len(events)})

    @app.route("/api/events/<event_id>", methods=["GET"])
    def get_event(event_id):
        """Get a single event by ID."""
        event_store = get_event_store()
        event = event_store.get(event_id)
        if event is None:
            abort(404)
        return jsonify(event)

    @app.route("/api/events", methods=["POST"])
    def create_event():
        """Create a new event."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        event_store = get_event_store()
        try:
            event_id = event_store.insert(data)
            # Feed production metrics
            metrics.on_event(data)
            return jsonify({"status": "created", "event_id": event_id}), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # ── Evidence Endpoints ──────────────────────────────────────────

    @app.route("/api/v1/evidence/<event_id>", methods=["GET"])
    def get_evidence_metadata(event_id):
        """Get evidence metadata for an event."""
        evidence_store = get_evidence_store()
        metadata = evidence_store.get_metadata(event_id)
        if metadata is None:
            abort(404)
        return jsonify(metadata)

    @app.route("/api/v1/evidence/<event_id>/image.jpg", methods=["GET"])
    def get_evidence_image(event_id):
        """Serve evidence JPEG image."""
        evidence_store = get_evidence_store()
        path = evidence_store.get_image_path(event_id)
        if path is None or not path.exists():
            abort(404)
        return send_file(str(path), mimetype="image/jpeg")

    @app.route("/api/v1/evidence/<event_id>/thumb.jpg", methods=["GET"])
    def get_evidence_thumb(event_id):
        """Serve evidence thumbnail."""
        evidence_store = get_evidence_store()
        path = evidence_store.get_thumb_path(event_id)
        if path is None or not path.exists():
            # Fall back to main image
            path = evidence_store.get_image_path(event_id)
            if path is None or not path.exists():
                abort(404)
        return send_file(str(path), mimetype="image/jpeg")

    @app.route("/api/v1/evidence/manifest", methods=["GET"])
    def get_evidence_manifest():
        """Get evidence manifest for a date."""
        date = request.args.get("date")
        if not date:
            # Default to today
            date = datetime.now().strftime("%Y-%m-%d")

        evidence_store = get_evidence_store()
        entries = evidence_store.list_by_date(date)
        return jsonify({"date": date, "entries": entries, "count": len(entries)})

    # Legacy evidence endpoint (compatibility)
    @app.route("/api/evidence/<event_id>", methods=["GET"])
    def get_evidence_legacy(event_id):
        """Legacy: Serve evidence JPEG frame for a given event."""
        evidence_store = get_evidence_store()
        path = evidence_store.get_image_path(event_id)
        if path is None or not path.exists():
            abort(404)
        return send_file(str(path), mimetype="image/jpeg")

    @app.route("/api/evidence/stats", methods=["GET"])
    def evidence_stats():
        """Get evidence disk usage stats."""
        if not evidence_dir.exists():
            return jsonify({"error": "Evidence directory not found"}), 404

        total_bytes = 0
        date_dirs = []
        for d in evidence_dir.iterdir():
            if d.is_dir() and len(d.name) == 10:  # YYYY-MM-DD
                date_dirs.append(d.name)
                for f in d.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size

        return jsonify({
            "evidence_dir": str(evidence_dir),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / (1024 * 1024), 1),
            "date_dirs": len(date_dirs),
            "dates": sorted(date_dirs, reverse=True)[:10],  # Last 10 dates
        })

    # ── Triples (using storage abstraction) ─────────────────────────

    @app.route("/api/triples", methods=["GET"])
    def list_triples():
        """Get causal triples."""
        limit = request.args.get("limit", 50, type=int)
        status = request.args.get("status")
        station_id = request.args.get("station_id")

        triple_store = get_triple_store()
        triples = triple_store.list(limit=limit, status=status, station_id=station_id)
        return jsonify({"triples": triples, "count": len(triples)})

    @app.route("/api/triples/<triple_id>", methods=["GET"])
    def get_triple(triple_id):
        """Get a single triple by ID."""
        triple_store = get_triple_store()
        triple = triple_store.get(triple_id)
        if triple is None:
            abort(404)
        return jsonify(triple)

    @app.route("/api/triples/<triple_id>", methods=["PATCH"])
    def update_triple(triple_id):
        """Update a triple (operator feedback)."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        triple_store = get_triple_store()
        updated = triple_store.update(triple_id, data)
        if not updated:
            abort(404)
        return jsonify({"status": "updated", "triple_id": triple_id})

    # ── Anomaly Alerts ─────────────────────────────────────────────

    @app.route("/api/alerts", methods=["GET"])
    def anomaly_alerts():
        """Get current anomaly alerts."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"alerts": []}), 200

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
            return jsonify({"recommendations": []}), 200

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
        data = request.get_json()
        if not data or "triple_id" not in data or "action" not in data:
            return jsonify({"error": "Missing triple_id or action"}), 400

        triple_store = get_triple_store()
        updated = triple_store.update(data["triple_id"], {
            "operator_action": data["action"],
            "operator_id": data.get("operator_id", ""),
            "outcome_measured": data.get("outcome", {}),
        })

        if not updated:
            return jsonify({"error": "Triple not found"}), 404

        return jsonify({
            "status": "recorded",
            "triple_id": data["triple_id"],
            "action": data["action"],
        })

    # ── Process Parameter Drift ────────────────────────────────────

    @app.route("/api/drift", methods=["GET"])
    def parameter_drift():
        """Get current process parameter drift status."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"drifts": []}), 200

        station_id = request.args.get("station_id")
        drifts = app.runtime.pipeline.correlator.check_drift(station_id)
        return jsonify({"drifts": drifts})

    # ── Manual Parameter Reading ───────────────────────────────────

    @app.route("/api/reading", methods=["POST"])
    def record_reading():
        """
        Record a manual process parameter reading from operator.
        Body: {parameter_name, value, station_id?}
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
            return jsonify({}), 200
        return jsonify(app.runtime.pipeline.get_stats())

    # ── Triple Stats ───────────────────────────────────────────────

    @app.route("/api/triples/stats", methods=["GET"])
    def triple_stats():
        """Get causal triple collection statistics."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({}), 200

        stats = app.runtime.pipeline.recommender.get_triple_stats()
        return jsonify(stats)

    # ── Inspection (Manual QC Station) ────────────────────────────

    @app.route("/api/inspect", methods=["POST"])
    def inspect():
        """
        Trigger a single inspection transaction.
        Thin route: parse input, call service, return JSON.

        Body (optional): {"product_id": "...", "operator_id": "...", "workspace_id": "..."}
        """
        from packages.inference.inspection import run_inspection

        if not app.runtime:
            return jsonify({"error": "Runtime not initialized"}), 503

        metadata = request.get_json(silent=True) or {}
        result = run_inspection(app.runtime, metadata)

        if result.get("verdict") == "ERROR":
            return jsonify(result), 500

        return jsonify(result)

    @app.route("/api/inspect/<inspection_id>/feedback", methods=["POST"])
    def inspect_feedback(inspection_id):
        """
        Record operator feedback on an inspection result.
        Body: {"action": "accepted"|"rejected", "operator_id": "...", "reason": "...", "notes": "..."}
        """
        data = request.get_json()
        if not data or "action" not in data:
            return jsonify({"error": "Missing action field"}), 400

        action = data["action"]
        accepted = action == "accepted"
        operator_id = data.get("operator_id", "")
        reason = data.get("reason", "")
        notes = data.get("notes", "")

        # Persist to inspection store
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({"error": "Inspection store not available"}), 503

        updated = store.update_feedback(
            inspection_id, accepted=accepted,
            operator_id=operator_id, reason=reason, notes=notes,
        )
        if not updated:
            return jsonify({"error": "Inspection not found"}), 404

        return jsonify({
            "status": "recorded",
            "inspection_id": inspection_id,
            "action": action,
        })

    @app.route("/api/inspections", methods=["GET"])
    def list_inspections():
        """
        List inspection events with optional filters.
        Query params: station_id, decision, sync_status, limit, offset
        """
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({"inspections": [], "total": 0}), 200

        events = store.list_inspections(
            station_id=request.args.get("station_id"),
            decision=request.args.get("decision"),
            sync_status=request.args.get("sync_status"),
            limit=int(request.args.get("limit", 50)),
            offset=int(request.args.get("offset", 0)),
        )

        return jsonify({
            "inspections": [_inspection_to_dict(e) for e in events],
            "count": len(events),
        })

    @app.route("/api/inspections/<inspection_id>", methods=["GET"])
    def get_inspection(inspection_id):
        """Get a single inspection event by ID."""
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({"error": "Inspection store not available"}), 503

        event = store.get(inspection_id)
        if not event:
            return jsonify({"error": "Not found"}), 404

        return jsonify(_inspection_to_dict(event))

    @app.route("/api/inspections/stats", methods=["GET"])
    def inspection_stats():
        """Get inspection store statistics."""
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({}), 200
        return jsonify(store.get_stats())

    @app.route("/api/inspections/sync", methods=["GET"])
    def inspection_sync_stats():
        """Get sync worker statistics."""
        worker = getattr(app.runtime, "_sync_worker", None) if app.runtime else None
        if not worker:
            return jsonify({"running": False, "message": "Sync worker not configured"}), 200
        return jsonify(worker.get_stats())

    return app


def _inspection_to_dict(event: Any) -> dict[str, Any]:
    """Convert an InspectionEvent to a JSON-serializable dict."""
    detections = []
    for d in (event.detections or []):
        detections.append({
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

    return {
        "inspection_id": event.inspection_id,
        "timestamp": event.timestamp.isoformat() if event.timestamp else "",
        "station_id": event.station_id,
        "workspace_id": event.workspace_id,
        "product_id": event.product_id,
        "operator_id": event.operator_id,
        "decision": event.decision.value if hasattr(event.decision, "value") else event.decision,
        "confidence": round(event.confidence, 4),
        "detections": detections,
        "num_detections": event.num_detections,
        "image_original_path": event.image_original_path,
        "image_annotated_path": event.image_annotated_path,
        "image_original_url": event.image_original_url,
        "image_annotated_url": event.image_annotated_url,
        "model_version": event.model_version,
        "model_name": event.model_name,
        "timing": {
            "capture_ms": round(event.capture_ms, 1),
            "inference_ms": round(event.inference_ms, 1),
            "total_ms": round(event.total_ms, 1),
        },
        "accepted": event.accepted,
        "rejection_reason": event.rejection_reason,
        "notes": event.notes,
        "sync_status": event.sync_status.value if hasattr(event.sync_status, "value") else event.sync_status,
        "synced_at": event.synced_at.isoformat() if event.synced_at else None,
    }


def run_api(runtime: Any = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the station API server."""
    app = create_app(runtime)
    logger.info("Station API v2 starting on %s:%d (mode=%s)", host, port,
                os.environ.get("STORAGE_MODE", "local"))
    app.run(host=host, port=port, debug=False, threaded=True)
