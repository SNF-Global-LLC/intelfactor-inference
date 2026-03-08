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

    Pass in a running StationRuntime for live vision/RCA data.
    Pass in a running SensorService and MaintenanceIQ for machine health routes.
    machine_health_config is the machine_health block from station.yaml.
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

    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False  # Allow Chinese in JSON responses

    # Store component references in app context
    app.runtime = runtime
    app.sensor_service = sensor_service
    app.maintenance_iq = maintenance_iq
    app.machine_health_config = machine_health_config or {}

    # Get evidence directory from env
    evidence_dir = Path(os.environ.get("EVIDENCE_DIR", "/opt/intelfactor/data/evidence"))

    # ── Serialisation helpers ────────────────────────────────────────

    def _sensor_event_to_dict(event: Any) -> dict[str, Any]:
        """Serialise a SensorEvent dataclass to a JSON-safe dict."""
        return {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "station_id": event.station_id,
            "machine_id": event.machine_id,
            "sensor_type": event.sensor_type.value,
            "raw_values": event.raw_values,
            "anomaly_score": event.anomaly_score,
            "confidence": event.confidence,
            "edge_verdict": event.edge_verdict.value,
        }

    def _verdict_to_dict(v: Any) -> dict[str, Any]:
        """Serialise a MaintenanceVerdict to a JSON-safe dict."""
        return {
            "verdict_id": v.verdict_id,
            "timestamp": v.timestamp.isoformat(),
            "machine_id": v.machine_id,
            "station_id": v.station_id,
            "verdict": v.verdict.value,
            "z_score": v.z_score,
            "confidence": v.confidence,
            "contributing_factors": v.contributing_factors,
            "warning_threshold": v.warning_threshold,
            "critical_threshold": v.critical_threshold,
        }

    def _action_to_dict(a: Any) -> dict[str, Any]:
        """Serialise a MaintenanceAction to a JSON-safe dict."""
        return {
            "action_id": a.action_id,
            "timestamp": a.timestamp.isoformat(),
            "machine_id": a.machine_id,
            "station_id": a.station_id,
            "verdict": _verdict_to_dict(a.verdict),
            "action_type": a.action_type.value,
            "action_en": a.action_en,
            "action_zh": a.action_zh,
            "sop_section": a.sop_section,
            "urgency": a.urgency,
            "operator_action": a.operator_action,
            "operator_id": a.operator_id,
            "rejection_reason": a.rejection_reason,
        }

    def _health_score(z_score: float) -> float:
        """
        Map a z-score to a 0–100 health score.

        z=0 → 100 (nominal), z=2 → 70 (warning zone), z=3.5 → 47.5 (critical).
        """
        return max(0.0, min(100.0, round(100.0 - z_score * 15.0, 1)))

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
        stats: dict[str, Any] = {"storage_mode": get_storage_mode()}
        if app.runtime:
            stats.update(app.runtime.get_stats())
        if app.sensor_service:
            stats["sensor_service"] = app.sensor_service.get_stats()
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

        if hasattr(app.runtime, "_ingest") and app.runtime._ingest:
            stats["camera"] = app.runtime._ingest.get_stats()

        return jsonify(stats)

    # ── Events (using storage abstraction) ──────────────────────────

    @app.route("/api/events", methods=["GET"])
    def list_events():
        """Get recent detection events."""
        limit = request.args.get("limit", 50, type=int)
        verdict = request.args.get("verdict")
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
            path = evidence_store.get_image_path(event_id)
            if path is None or not path.exists():
                abort(404)
        return send_file(str(path), mimetype="image/jpeg")

    @app.route("/api/v1/evidence/manifest", methods=["GET"])
    def get_evidence_manifest():
        """Get evidence manifest for a date."""
        date = request.args.get("date")
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        evidence_store = get_evidence_store()
        entries = evidence_store.list_by_date(date)
        return jsonify({"date": date, "entries": entries, "count": len(entries)})

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
            "dates": sorted(date_dirs, reverse=True)[:10],
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

    # ── Anomaly Alerts (storage abstraction) ───────────────────────

    @app.route("/api/alerts", methods=["GET"])
    def anomaly_alerts():
        """Get current unacknowledged anomaly alerts."""
        event_store = get_event_store()
        alerts = event_store.list_alerts(limit=20)
        return jsonify({"alerts": alerts})

    # ── RCA Recommendations ────────────────────────────────────────

    @app.route("/api/recommendations", methods=["GET"])
    def pending_recommendations():
        """Get pending defect recommendations awaiting operator response."""
        if not app.runtime or not app.runtime.pipeline:
            return jsonify({"recommendations": []}), 200

        station_id = request.args.get("station_id")
        triples = app.runtime.pipeline.recommender.get_pending_triples(station_id)
        return jsonify({"recommendations": triples})

    # ── Operator Feedback ──────────────────────────────────────────

    @app.route("/api/feedback", methods=["POST"])
    def record_feedback():
        """
        Record operator feedback on a causal triple.
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

    # ── Machine Health: Fleet Overview ─────────────────────────────

    @app.route("/api/maintenance/health", methods=["GET"])
    def maintenance_health():
        """
        Return the current health verdict for every machine in station.yaml.

        For each asset in machine_health.assets, fetches the most recent
        sensor events, evaluates via MaintenanceIQ, and returns a health entry.
        Returns HEALTHY with confidence=0.0 when no sensor data exists yet.
        """
        assets = app.machine_health_config.get("assets", [])
        station_id = (
            app.runtime.config.station_id if app.runtime else "station_1"
        )

        results = []
        for asset in assets:
            machine_id = asset["machine_id"]
            entry: dict[str, Any] = {
                "machine_id": machine_id,
                "display_name": asset.get("display_name", machine_id),
                "asset_type": asset.get("asset_type", ""),
                "verdict": "HEALTHY",
                "health_score": 100.0,
                "z_score": 0.0,
                "confidence": 0.0,
                "contributing_factors": [],
                "last_reading_at": None,
            }

            if app.sensor_service and app.maintenance_iq:
                events = app.sensor_service.get_recent_events_for_machine(
                    machine_id, limit=20
                )
                verdict_obj = app.maintenance_iq.evaluate(
                    machine_id=machine_id,
                    station_id=station_id,
                    events=events,
                )
                entry["verdict"] = verdict_obj.verdict.value
                entry["health_score"] = _health_score(verdict_obj.z_score)
                entry["z_score"] = verdict_obj.z_score
                entry["confidence"] = verdict_obj.confidence
                entry["contributing_factors"] = verdict_obj.contributing_factors

                if events:
                    entry["last_reading_at"] = events[0].timestamp.isoformat()

            results.append(entry)

        return jsonify({"machines": results, "count": len(results)})

    # ── Machine Health: Sensor Events ─────────────────────────────

    @app.route("/api/maintenance/events", methods=["GET"])
    def maintenance_list_events():
        """
        Query sensor events with optional filters.

        Params: machine_id, sensor_type, verdict (HEALTHY|WARNING|CRITICAL), limit (default 100).
        """
        if not app.sensor_service:
            return jsonify({"events": [], "count": 0})

        machine_id = request.args.get("machine_id")
        sensor_type_raw = request.args.get("sensor_type")
        verdict = request.args.get("verdict")
        limit = request.args.get("limit", 100, type=int)

        from packages.ingestion.schemas import SensorType

        sensor_type = None
        if sensor_type_raw:
            try:
                sensor_type = SensorType(sensor_type_raw)
            except ValueError:
                return jsonify({"error": f"Unknown sensor_type: {sensor_type_raw!r}"}), 400

        events = app.sensor_service.list_events(
            machine_id=machine_id,
            sensor_type=sensor_type,
            verdict=verdict,
            limit=limit,
        )
        return jsonify({"events": events, "count": len(events)})

    @app.route("/api/maintenance/events/<event_id>", methods=["GET"])
    def maintenance_get_event(event_id):
        """Get a single sensor event by ID."""
        if not app.sensor_service:
            abort(404)

        event = app.sensor_service.get_event(event_id)
        if event is None:
            abort(404)
        return jsonify(event)

    # ── Machine Health: Baselines ──────────────────────────────────

    @app.route("/api/maintenance/baselines", methods=["GET"])
    def maintenance_baselines():
        """Return all stored baseline profiles."""
        if not app.sensor_service:
            return jsonify({"baselines": [], "count": 0})

        baselines = app.sensor_service.list_baselines()
        return jsonify({"baselines": baselines, "count": len(baselines)})

    # ── Machine Health: Incidents ──────────────────────────────────

    @app.route("/api/maintenance/incidents", methods=["GET"])
    def maintenance_incidents():
        """
        Return WARNING and CRITICAL events grouped by machine.

        Each entry: machine_id, severity, event_count, first_seen, last_seen,
        contributing_factors (list of sensor type strings).
        Optional filter: machine_id query param.
        """
        if not app.sensor_service:
            return jsonify({"incidents": [], "count": 0})

        machine_id = request.args.get("machine_id")
        incidents = app.sensor_service.get_incidents(machine_id=machine_id)
        return jsonify({"incidents": incidents, "count": len(incidents)})

    # ── Machine Health: Ingest Single Reading ──────────────────────

    @app.route("/api/maintenance/sensor-events", methods=["POST"])
    def maintenance_ingest_event():
        """
        Ingest a single SensorReading, score it, and persist it.

        Body: {station_id?, machine_id, sensor_type, raw_values: {...}}
        Returns 202 with the resulting scored SensorEvent.
        """
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not initialised"}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        required = ("machine_id", "sensor_type", "raw_values")
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        from packages.ingestion.schemas import SensorReading, SensorType

        try:
            sensor_type = SensorType(data["sensor_type"])
        except ValueError:
            return jsonify({"error": f"Unknown sensor_type: {data['sensor_type']!r}"}), 400

        raw_values = data.get("raw_values", {})
        if not isinstance(raw_values, dict):
            return jsonify({"error": "raw_values must be a JSON object"}), 400

        reading = SensorReading(
            station_id=data.get("station_id", ""),
            machine_id=data["machine_id"],
            sensor_type=sensor_type,
            raw_values={k: float(v) for k, v in raw_values.items()},
            mqtt_topic=data.get("mqtt_topic", ""),
        )

        try:
            event = app.sensor_service.ingest_reading(reading)
        except Exception as exc:
            logger.error("ingest_reading failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

        return jsonify({"status": "accepted", "event": _sensor_event_to_dict(event)}), 202

    # ── Machine Health: Ingest Batch ──────────────────────────────

    @app.route("/api/maintenance/sensor-events/batch", methods=["POST"])
    def maintenance_ingest_batch():
        """
        Ingest a batch of SensorReadings (max 500).

        Body: {"readings": [...]}
        Returns 202 with summary: events_processed, warnings, criticals.
        """
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not initialised"}), 503

        data = request.get_json()
        if not data or "readings" not in data:
            return jsonify({"error": "JSON body with 'readings' list required"}), 400

        readings_raw = data["readings"]
        if not isinstance(readings_raw, list):
            return jsonify({"error": "'readings' must be a list"}), 400
        if len(readings_raw) > 500:
            return jsonify({"error": "Batch size exceeds 500"}), 400

        from packages.ingestion.schemas import SensorReading, SensorType, HealthVerdict

        processed = 0
        warnings = 0
        criticals = 0
        errors: list[str] = []

        for i, item in enumerate(readings_raw):
            try:
                sensor_type = SensorType(item["sensor_type"])
                reading = SensorReading(
                    station_id=item.get("station_id", ""),
                    machine_id=item["machine_id"],
                    sensor_type=sensor_type,
                    raw_values={k: float(v) for k, v in item.get("raw_values", {}).items()},
                    mqtt_topic=item.get("mqtt_topic", ""),
                )
                event = app.sensor_service.ingest_reading(reading)
                processed += 1
                if event.edge_verdict == HealthVerdict.WARNING:
                    warnings += 1
                elif event.edge_verdict == HealthVerdict.CRITICAL:
                    criticals += 1
            except Exception as exc:
                errors.append(f"[{i}] {exc}")

        result: dict[str, Any] = {
            "status": "accepted",
            "events_processed": processed,
            "warnings": warnings,
            "criticals": criticals,
        }
        if errors:
            result["errors"] = errors[:20]  # Cap error list at 20 entries

        return jsonify(result), 202

    # ── Machine Health: Recommendations ──────────────────────────

    @app.route("/api/maintenance/recommendations", methods=["GET"])
    def maintenance_recommendations():
        """
        Get the latest maintenance recommendation for each machine in warning/critical state.

        Fetches the current verdict per machine from sensor history and calls
        MaintenanceIQ.recommend() for any machine not in HEALTHY state.
        """
        if not app.sensor_service or not app.maintenance_iq:
            return jsonify({"recommendations": [], "count": 0})

        assets = app.machine_health_config.get("assets", [])
        station_id = (
            app.runtime.config.station_id if app.runtime else "station_1"
        )

        recommendations = []
        for asset in assets:
            machine_id = asset["machine_id"]
            events = app.sensor_service.get_recent_events_for_machine(machine_id, limit=20)
            verdict_obj = app.maintenance_iq.evaluate(
                machine_id=machine_id,
                station_id=station_id,
                events=events,
            )

            from packages.ingestion.schemas import HealthVerdict
            if verdict_obj.verdict != HealthVerdict.HEALTHY:
                action = app.maintenance_iq.recommend(verdict_obj)
                recommendations.append(_action_to_dict(action))

        return jsonify({"recommendations": recommendations, "count": len(recommendations)})

    # ── Machine Health: Operator Feedback ────────────────────────

    @app.route("/api/maintenance/feedback", methods=["POST"])
    def maintenance_feedback():
        """
        Record operator feedback on a sensor event.

        Body: {event_id, operator_action: confirm|reject|uncertain,
               operator_id?, rejection_reason?, comment?}
        Mirrors the pattern of POST /api/feedback for causal triples.
        """
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not initialised"}), 503

        data = request.get_json()
        if not data or "event_id" not in data or "operator_action" not in data:
            return jsonify({"error": "Missing event_id or operator_action"}), 400

        allowed = {"confirm", "reject", "uncertain"}
        if data["operator_action"] not in allowed:
            return jsonify({"error": f"operator_action must be one of {sorted(allowed)}"}), 400

        try:
            updated = app.sensor_service.update_event_feedback(
                event_id=data["event_id"],
                operator_action=data["operator_action"],
                operator_id=data.get("operator_id", ""),
                rejection_reason=data.get("rejection_reason", ""),
                comment=data.get("comment", ""),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not updated:
            return jsonify({"error": "Event not found"}), 404

        return jsonify({
            "status": "recorded",
            "event_id": data["event_id"],
            "operator_action": data["operator_action"],
        })

    # ── Machine Health: Stats ────────────────────────────────────

    @app.route("/api/maintenance/stats", methods=["GET"])
    def maintenance_stats():
        """
        Return aggregated sensor statistics.

        Fields: total_events, events_by_verdict, events_by_sensor_type,
        events_by_machine, baseline_count, last_reading_at.
        """
        if not app.sensor_service:
            return jsonify({"status": "sensor_service_not_initialised"})

        raw = app.sensor_service.get_stats()
        if raw.get("status") == "not_started":
            return jsonify(raw)

        conn = app.sensor_service._conn
        if conn is None:
            return jsonify(raw)

        def _agg(group_col: str, where: str = "") -> dict[str, int]:
            clause = f"WHERE {where}" if where else ""
            rows = conn.execute(
                f"SELECT {group_col}, COUNT(*) FROM sensor_events {clause} GROUP BY {group_col}"
            ).fetchall()
            return {r[0]: r[1] for r in rows}

        last_row = conn.execute(
            "SELECT MAX(timestamp) FROM sensor_events"
        ).fetchone()
        last_reading_at = last_row[0] if last_row else None

        return jsonify({
            "total_events": raw.get("total_events", 0),
            "events_by_verdict": _agg("edge_verdict"),
            "events_by_sensor_type": _agg("sensor_type"),
            "events_by_machine": _agg("machine_id"),
            "baseline_count": raw.get("baseline_profiles", 0),
            "last_reading_at": last_reading_at,
        })

    return app


def run_api(runtime: Any = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the station API server."""
    app = create_app(runtime)
    logger.info("Station API v2 starting on %s:%d (mode=%s)", host, port,
                os.environ.get("STORAGE_MODE", "local"))
    app.run(host=host, port=port, debug=False, threaded=True)
