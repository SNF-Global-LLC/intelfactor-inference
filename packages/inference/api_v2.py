"""
IntelFactor.ai — Station REST API v2
Enhanced API with storage abstraction and evidence serving.

Supports both STORAGE_MODE=local (SQLite) and cloud modes.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from packages.ingestion.schemas import (
    HealthVerdict,
    MaintenanceAction,
    SensorEvent,
    SensorReading,
    SensorType,
)

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
        from flask import Flask, abort, jsonify, request, send_file
    except ImportError:
        raise RuntimeError("Flask required for station API (pip install flask)")

    from packages.inference.storage import (
        get_event_store,
        get_evidence_store,
        get_storage_mode,
        get_triple_store,
    )

    # Production visibility metrics
    from packages.visibility.metrics_api import init_metrics, metrics_bp

    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        static_url_path="/static",
    )
    app.config["JSON_AS_ASCII"] = False  # Allow Chinese in JSON responses

    # ── API Key Auth ────────────────────────────────────────────────
    _api_key = os.environ.get("EDGE_API_KEY") or os.environ.get("STATION_API_KEY", "")
    _workspace_id = os.environ.get("WORKSPACE_ID", "").strip()
    if not _api_key:
        logger.warning(
            "EDGE_API_KEY/STATION_API_KEY not set — API endpoints will fail closed. "
            "Set EDGE_API_KEY or STATION_API_KEY before deployment."
        )

    def _require_api_key() -> tuple[Any, int] | None:
        """
        Check API key for station API endpoints.
        Returns a (response, status) tuple to return immediately if auth fails,
        or None if auth passes.
        """
        if not _api_key:
            logger.error(
                "auth_failed reason=api_key_not_configured path=%s", request.path
            )
            return jsonify({"error": "Station API key is not configured"}), 503

        # Accept edge-specific, legacy station, or Authorization: Bearer <token>.
        provided = (
            request.headers.get("X-Edge-Api-Key")
            or request.headers.get("X-API-Key")
            or ""
        )
        if not provided:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                provided = auth_header[7:]

        if provided != _api_key:
            logger.warning("auth_failed reason=bad_api_key path=%s", request.path)
            return jsonify({"error": "Unauthorized"}), 401

        return None

    @app.before_request
    def _protect_api_routes():
        """Require edge auth for every API route. Health, UI shells, and maintenance stay open."""
        if request.path.startswith("/api/maintenance/"):
            return None
        if request.path.startswith("/api/"):
            return _require_api_key()
        return None

    # Store runtime and service references
    app.runtime = runtime
    app.sensor_service = sensor_service
    app.maintenance_iq = maintenance_iq
    app.machine_health_config = machine_health_config

    # Get evidence directory from env
    evidence_dir = Path(
        os.environ.get("EVIDENCE_DIR", "/opt/intelfactor/data/evidence")
    )

    # ── Security helpers ──────────────────────────────────────────────
    _MAX_LIMIT = 1000

    def _safe_limit() -> int:
        """Parse and cap the ?limit= query parameter."""
        raw = request.args.get("limit", 50, type=int)
        return max(1, min(raw, _MAX_LIMIT))

    def _safe_offset() -> int:
        """Parse and floor the ?offset= query parameter."""
        raw = request.args.get("offset", 0, type=int)
        return max(0, raw)

    def _resolve_evidence_path(relative_path: str) -> Path | None:
        """Resolve a relative evidence path, rejecting traversal attempts."""
        full = (evidence_dir / relative_path).resolve()
        try:
            full.relative_to(evidence_dir.resolve())
        except ValueError:
            return None  # path escapes evidence_dir
        return full if full.exists() else None

    def _workspace_mismatch(candidate: str | None) -> bool:
        return bool(_workspace_id and candidate and candidate != _workspace_id)

    def _workspace_for_new_record(
        data: dict[str, Any],
    ) -> tuple[str, tuple[Any, int] | None]:
        requested = (
            data.get("workspace_id") or request.headers.get("X-Workspace-Id") or ""
        ).strip()
        if _workspace_mismatch(requested):
            logger.warning(
                "workspace_mismatch authenticated_workspace=%s requested_workspace=%s path=%s",
                _workspace_id,
                requested,
                request.path,
            )
            return "", (jsonify({"error": "Workspace mismatch"}), 403)
        return requested or _workspace_id, None

    def _require_event_workspace(event: Any) -> tuple[Any, int] | None:
        event_workspace = (
            getattr(event, "workspace_id", "")
            if not isinstance(event, dict)
            else event.get("workspace_id", "")
        )
        if _workspace_mismatch(event_workspace):
            logger.warning(
                "workspace_mismatch authenticated_workspace=%s event_workspace=%s path=%s",
                _workspace_id,
                event_workspace,
                request.path,
            )
            return jsonify({"error": "Workspace mismatch"}), 403
        return None

    def _metadata_workspace(metadata: dict[str, Any] | None) -> str:
        if not metadata:
            return ""
        return str(
            metadata.get("workspace_id")
            or metadata.get("workspace")
            or metadata.get("metadata", {}).get("workspace_id", "")
        )

    # Initialize production metrics
    db_path = os.environ.get("DB_PATH", "/opt/intelfactor/data/local.db")
    station_id = os.environ.get("STATION_ID", "SNF-Vision-1")
    metrics = init_metrics(app, db_path=db_path, station_id=station_id)
    app.register_blueprint(metrics_bp)

    # ── Pages ────────────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def root():
        """Redirect to the inspection page — primary operator surface."""
        from flask import redirect

        return redirect("/inspect")

    @app.route("/inspect", methods=["GET"])
    def inspect_page():
        """Serve the manual QC inspection page."""
        from flask import send_from_directory

        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, "inspect.html")

    @app.route("/status", methods=["GET"])
    def status_page():
        """Serve the station status / diagnostics page."""
        from flask import send_from_directory

        static_dir = os.path.join(os.path.dirname(__file__), "static")
        return send_from_directory(static_dir, "index.html")

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
            return jsonify(
                {"storage_mode": get_storage_mode(), "runtime": "not initialized"}
            )

        stats = app.runtime.get_stats()
        stats["storage_mode"] = get_storage_mode()

        # Camera backend info (set by cli.py after capture init)
        stats["camera_backend"] = getattr(app.runtime, "_camera_backend", "unknown")
        stats["camera_connected"] = getattr(app.runtime, "_camera_connected", False)
        stats["vision_model_key"] = getattr(app.runtime, "_vision_model_key", "unknown")
        stats["language_model_key"] = getattr(
            app.runtime, "_language_model_key", "unknown"
        )

        # Add continuous ingest stats if available (RTSP/USB streaming mode)
        if hasattr(app.runtime, "_ingest") and app.runtime._ingest:
            stats["camera"] = app.runtime._ingest.get_stats()

        return jsonify(stats)

    # ── Events (using storage abstraction) ──────────────────────────

    @app.route("/api/events", methods=["GET"])
    def list_events():
        """Get recent detection events."""
        limit = _safe_limit()
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
        if (err := _require_api_key()) is not None:
            return err
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        workspace_id, workspace_err = _workspace_for_new_record(data)
        if workspace_err is not None:
            return workspace_err
        if workspace_id:
            data["workspace_id"] = workspace_id

        event_store = get_event_store()
        try:
            event_id = event_store.insert(data)
            # Feed production metrics
            metrics.on_event(data)
            return jsonify({"status": "created", "event_id": event_id}), 201
        except ValueError:
            logger.exception("Event insertion failed")
            return jsonify({"error": "Invalid event data"}), 400

    # ── Evidence Endpoints ──────────────────────────────────────────

    @app.route("/api/v1/evidence/<event_id>", methods=["GET"])
    def get_evidence_metadata(event_id):
        """Get evidence metadata for an event."""
        evidence_store = get_evidence_store()
        metadata = evidence_store.get_metadata(event_id)
        if metadata is None:
            abort(404)
        if _workspace_mismatch(_metadata_workspace(metadata)):
            logger.warning(
                "workspace_mismatch authenticated_workspace=%s path=%s",
                _workspace_id,
                request.path,
            )
            return jsonify({"error": "Workspace mismatch"}), 403
        return jsonify(metadata)

    @app.route("/api/v1/evidence/<event_id>/image.jpg", methods=["GET"])
    def get_evidence_image(event_id):
        """Serve evidence JPEG image."""
        evidence_store = get_evidence_store()
        metadata = evidence_store.get_metadata(event_id)
        if _workspace_mismatch(_metadata_workspace(metadata)):
            logger.warning(
                "workspace_mismatch authenticated_workspace=%s path=%s",
                _workspace_id,
                request.path,
            )
            return jsonify({"error": "Workspace mismatch"}), 403
        path = evidence_store.get_image_path(event_id)
        if path is None or not path.exists():
            abort(404)
        return send_file(str(path), mimetype="image/jpeg")

    @app.route("/api/v1/evidence/<event_id>/thumb.jpg", methods=["GET"])
    def get_evidence_thumb(event_id):
        """Serve evidence thumbnail."""
        evidence_store = get_evidence_store()
        metadata = evidence_store.get_metadata(event_id)
        if _workspace_mismatch(_metadata_workspace(metadata)):
            logger.warning(
                "workspace_mismatch authenticated_workspace=%s path=%s",
                _workspace_id,
                request.path,
            )
            return jsonify({"error": "Workspace mismatch"}), 403
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
        if _workspace_id:
            entries = [
                e
                for e in entries
                if not _metadata_workspace(e) or _metadata_workspace(e) == _workspace_id
            ]
        return jsonify({"date": date, "entries": entries, "count": len(entries)})

    # Legacy evidence endpoint (compatibility)
    @app.route("/api/evidence/<event_id>", methods=["GET"])
    def get_evidence_legacy(event_id):
        """Legacy: Serve evidence JPEG frame for a given event."""
        evidence_store = get_evidence_store()
        metadata = evidence_store.get_metadata(event_id)
        if _workspace_mismatch(_metadata_workspace(metadata)):
            logger.warning(
                "workspace_mismatch authenticated_workspace=%s path=%s",
                _workspace_id,
                request.path,
            )
            return jsonify({"error": "Workspace mismatch"}), 403
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
            if d.is_symlink():
                continue
            if d.is_dir() and len(d.name) == 10:  # YYYY-MM-DD
                date_dirs.append(d.name)
                for f in d.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size

        return jsonify(
            {
                "total_bytes": total_bytes,
                "total_mb": round(total_bytes / (1024 * 1024), 1),
                "date_dirs": len(date_dirs),
                "dates": sorted(date_dirs, reverse=True)[:10],  # Last 10 dates
            }
        )

    # ── Triples (using storage abstraction) ─────────────────────────

    @app.route("/api/triples", methods=["GET"])
    def list_triples():
        """Get causal triples."""
        limit = _safe_limit()
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
        if (err := _require_api_key()) is not None:
            return err
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

        columns = (
            [
                desc[0]
                for desc in acc._conn.execute(
                    "SELECT * FROM anomaly_alerts LIMIT 0"
                ).description
            ]
            if rows
            else []
        )

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
        if (err := _require_api_key()) is not None:
            return err
        data = request.get_json()
        if not data or "triple_id" not in data or "action" not in data:
            return jsonify({"error": "Missing triple_id or action"}), 400

        triple_store = get_triple_store()
        updated = triple_store.update(
            data["triple_id"],
            {
                "operator_action": data["action"],
                "operator_id": data.get("operator_id", ""),
                "outcome_measured": data.get("outcome", {}),
            },
        )

        if not updated:
            return jsonify({"error": "Triple not found"}), 404

        return jsonify(
            {
                "status": "recorded",
                "triple_id": data["triple_id"],
                "action": data["action"],
            }
        )

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
        if (err := _require_api_key()) is not None:
            return err
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
        if (err := _require_api_key()) is not None:
            return err
        from packages.inference.inspection import run_inspection

        if not app.runtime:
            return jsonify({"error": "Runtime not initialized"}), 503

        metadata = request.get_json(silent=True) or {}
        workspace_id, workspace_err = _workspace_for_new_record(metadata)
        if workspace_err is not None:
            return workspace_err
        if workspace_id:
            metadata["workspace_id"] = workspace_id
        result = run_inspection(app.runtime, metadata)

        if result.get("verdict") == "ERROR":
            return jsonify(result), 500

        # Add direct image URLs so the operator UI doesn't need to know about
        # the evidence store internals. Routes are defined below.
        iid = result["inspection_id"]
        result["image_original_url"] = f"/api/inspections/{iid}/original.jpg"
        result["image_annotated_url"] = f"/api/inspections/{iid}/annotated.jpg"

        return jsonify(result)

    # ── Inspection Image Serving ────────────────────────────────────

    @app.route("/api/inspections/<inspection_id>/original.jpg", methods=["GET"])
    def get_inspection_original_img(inspection_id):
        """Serve the original (unannotated) JPEG for an inspection."""
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            abort(404)
        event = store.get(inspection_id)
        if not event or not event.image_original_path:
            abort(404)
        if (err := _require_event_workspace(event)) is not None:
            return err
        full_path = _resolve_evidence_path(event.image_original_path)
        if full_path is None:
            abort(404)
        return send_file(str(full_path), mimetype="image/jpeg")

    @app.route("/api/inspections/<inspection_id>/annotated.jpg", methods=["GET"])
    def get_inspection_annotated_img(inspection_id):
        """Serve the annotated JPEG (bounding boxes drawn) for an inspection."""
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            abort(404)
        event = store.get(inspection_id)
        if not event or not event.image_annotated_path:
            abort(404)
        if (err := _require_event_workspace(event)) is not None:
            return err
        full_path = _resolve_evidence_path(event.image_annotated_path)
        if full_path is None:
            abort(404)
        return send_file(str(full_path), mimetype="image/jpeg")

    @app.route("/api/inspect/<inspection_id>/feedback", methods=["POST"])
    def inspect_feedback(inspection_id):
        """
        Record operator feedback on an inspection result.
        Body: {"action": "confirm_defect"|"override_to_pass", "operator_id": "...", "reason": "...", "notes": "..."}
        Legacy "accepted"/"rejected" actions are still accepted for older edge clients.
        """
        if (err := _require_api_key()) is not None:
            return err
        data = request.get_json()
        if not data or "action" not in data:
            return jsonify({"error": "Missing action field"}), 400

        action = data["action"]
        if action in {"confirm_defect", "accepted"}:
            accepted = True
            reason = data.get("reason", "")
        elif action in {"override_to_pass", "rejected"}:
            accepted = False
            reason = data.get("reason") or action
        else:
            return jsonify({"error": "Invalid action"}), 400
        operator_id = data.get("operator_id", "")
        notes = data.get("notes", "")

        # Persist to inspection store
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({"error": "Inspection store not available"}), 503
        event = store.get(inspection_id)
        if event and (err := _require_event_workspace(event)) is not None:
            return err

        updated = store.update_feedback(
            inspection_id,
            accepted=accepted,
            operator_id=operator_id,
            action=action,
            reason=reason,
            notes=notes,
        )
        if not updated:
            return jsonify({"error": "Inspection not found"}), 404

        return jsonify(
            {
                "status": "recorded",
                "inspection_id": inspection_id,
                "action": action,
            }
        )

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
            workspace_id=_workspace_id or None,
            decision=request.args.get("decision"),
            sync_status=request.args.get("sync_status"),
            limit=_safe_limit(),
            offset=_safe_offset(),
        )

        return jsonify(
            {
                "inspections": [_inspection_to_dict(e) for e in events],
                "count": len(events),
            }
        )

    @app.route("/api/inspections/<inspection_id>", methods=["GET"])
    def get_inspection(inspection_id):
        """Get a single inspection event by ID."""
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({"error": "Inspection store not available"}), 503

        event = store.get(inspection_id)
        if not event:
            return jsonify({"error": "Not found"}), 404
        if (err := _require_event_workspace(event)) is not None:
            return err

        return jsonify(
            _inspection_to_dict(
                event,
                operator_actions=store.list_operator_actions(inspection_id),
            )
        )

    @app.route("/api/inspections/stats", methods=["GET"])
    def inspection_stats():
        """Get inspection store statistics."""
        store = getattr(app.runtime, "_inspection_store", None) if app.runtime else None
        if not store:
            return jsonify({}), 200
        return jsonify(store.get_stats(workspace_id=_workspace_id or None))

    @app.route("/api/inspections/sync", methods=["GET"])
    def inspection_sync_stats():
        """Get sync worker statistics."""
        worker = getattr(app.runtime, "_sync_worker", None) if app.runtime else None
        if not worker:
            return jsonify(
                {"running": False, "message": "Sync worker not configured"}
            ), 200
        return jsonify(worker.get_stats())

    # ── Sync Heartbeat ───────────────────────────────────────────────

    @app.route("/api/sync/heartbeat", methods=["GET"])
    def sync_heartbeat():
        """Get sync agent heartbeat status."""
        worker = getattr(app.runtime, "_sync_worker", None) if app.runtime else None
        if not worker:
            return jsonify(
                {
                    "running": False,
                    "message": "Sync worker not configured",
                    "timestamp": datetime.now().isoformat(),
                }
            ), 200
        stats = worker.get_stats()
        stats["timestamp"] = datetime.now().isoformat()
        return jsonify(stats)

    # ── Maintenance API ──────────────────────────────────────────────

    @app.route("/api/maintenance/health", methods=["GET"])
    def maintenance_health():
        """Get machine health status for all configured assets."""
        if not app.sensor_service or not app.maintenance_iq or not app.machine_health_config:
            return jsonify({"machines": [], "count": 0}), 200

        assets = app.machine_health_config.get("assets", [])
        machines = []
        for asset in assets:
            machine_id = asset["machine_id"]
            events = app.sensor_service.get_recent_events_for_machine(machine_id, limit=20)
            verdict = app.maintenance_iq.evaluate(
                machine_id=machine_id,
                station_id=app.sensor_service.station_id,
                events=events,
            )
            last_reading = events[0].timestamp if events else None
            machines.append(
                {
                    "machine_id": machine_id,
                    "asset_type": asset.get("asset_type", ""),
                    "display_name": asset.get("display_name", machine_id),
                    "verdict": verdict.verdict.value
                    if hasattr(verdict.verdict, "value")
                    else verdict.verdict,
                    "confidence": round(verdict.confidence, 4),
                    "health_score": 100.0
                    if verdict.verdict == HealthVerdict.HEALTHY
                    else (50.0 if verdict.verdict == HealthVerdict.WARNING else 0.0),
                    "last_reading_at": last_reading.isoformat() if last_reading else None,
                    "z_score": round(verdict.z_score, 3),
                    "warning_threshold": verdict.warning_threshold,
                    "critical_threshold": verdict.critical_threshold,
                }
            )

        return jsonify({"machines": machines, "count": len(machines)})

    @app.route("/api/maintenance/sensor-events", methods=["POST"])
    def post_sensor_event():
        """Ingest a single sensor reading."""
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not available"}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        machine_id = data.get("machine_id")
        sensor_type_str = data.get("sensor_type")
        raw_values = data.get("raw_values")

        if not machine_id or not sensor_type_str or raw_values is None:
            return (
                jsonify(
                    {"error": "Missing required fields: machine_id, sensor_type, raw_values"}
                ),
                400,
            )

        try:
            sensor_type = SensorType(sensor_type_str)
        except ValueError:
            return jsonify({"error": f"Unknown sensor type: {sensor_type_str}"}), 400

        reading = SensorReading(
            station_id=data.get("station_id", app.sensor_service.station_id),
            machine_id=machine_id,
            sensor_type=sensor_type,
            raw_values=raw_values,
        )
        event = app.sensor_service.ingest_reading(reading)

        return jsonify({"status": "accepted", "event": _sensor_event_to_dict(event)}), 202

    @app.route("/api/maintenance/sensor-events/batch", methods=["POST"])
    def post_sensor_events_batch():
        """Ingest a batch of sensor readings."""
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not available"}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        readings_data = data.get("readings", [])
        if not isinstance(readings_data, list):
            return jsonify({"error": "readings must be a list"}), 400

        if len(readings_data) > 500:
            return jsonify({"error": "Batch size exceeds 500"}), 400

        warnings_count = 0
        criticals_count = 0

        for rd in readings_data:
            machine_id = rd.get("machine_id")
            sensor_type_str = rd.get("sensor_type")
            raw_values = rd.get("raw_values")

            if not machine_id or not sensor_type_str or raw_values is None:
                continue

            try:
                sensor_type = SensorType(sensor_type_str)
            except ValueError:
                continue

            reading = SensorReading(
                station_id=rd.get("station_id", app.sensor_service.station_id),
                machine_id=machine_id,
                sensor_type=sensor_type,
                raw_values=raw_values,
            )
            event = app.sensor_service.ingest_reading(reading)
            if event.edge_verdict == HealthVerdict.WARNING:
                warnings_count += 1
            elif event.edge_verdict == HealthVerdict.CRITICAL:
                criticals_count += 1

        return jsonify(
            {
                "status": "accepted",
                "events_processed": len(readings_data),
                "warnings": warnings_count,
                "criticals": criticals_count,
            }
        ), 202

    @app.route("/api/maintenance/events", methods=["GET"])
    def list_maintenance_events():
        """List sensor events with optional filters."""
        if not app.sensor_service:
            return jsonify({"events": [], "count": 0}), 200

        machine_id = request.args.get("machine_id")
        sensor_type_str = request.args.get("sensor_type")
        verdict = request.args.get("verdict")
        limit = _safe_limit()

        sensor_type = None
        if sensor_type_str:
            try:
                sensor_type = SensorType(sensor_type_str)
            except ValueError:
                return jsonify({"error": f"Unknown sensor type: {sensor_type_str}"}), 400

        events = app.sensor_service.list_events(
            machine_id=machine_id,
            sensor_type=sensor_type,
            verdict=verdict,
            limit=limit,
        )

        return jsonify({"events": events, "count": len(events)})

    @app.route("/api/maintenance/events/<event_id>", methods=["GET"])
    def get_maintenance_event(event_id):
        """Get a single sensor event by ID."""
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not available"}), 503

        event = app.sensor_service.get_event(event_id)
        if event is None:
            return jsonify({"error": "Not found"}), 404

        return jsonify(event)

    @app.route("/api/maintenance/baselines", methods=["GET"])
    def list_baselines():
        """List computed baseline profiles."""
        if not app.sensor_service:
            return jsonify({"baselines": [], "count": 0}), 200

        baselines = app.sensor_service.list_baselines()
        return jsonify({"baselines": baselines, "count": len(baselines)})

    @app.route("/api/maintenance/incidents", methods=["GET"])
    def list_incidents():
        """List incidents (grouped WARNING/CRITICAL events)."""
        if not app.sensor_service:
            return jsonify({"incidents": [], "count": 0}), 200

        machine_id = request.args.get("machine_id")
        incidents = app.sensor_service.get_incidents(machine_id=machine_id)
        return jsonify({"incidents": incidents, "count": len(incidents)})

    @app.route("/api/maintenance/recommendations", methods=["GET"])
    def maintenance_recommendations():
        """Get maintenance recommendations for machines with non-HEALTHY status."""
        if not app.sensor_service or not app.maintenance_iq:
            return jsonify({"recommendations": [], "count": 0}), 200

        station_id = request.args.get("station_id", app.sensor_service.station_id)
        recommendations = []

        if app.machine_health_config:
            machine_ids = [
                a["machine_id"] for a in app.machine_health_config.get("assets", [])
            ]
        else:
            events = app.sensor_service.list_events(limit=1000)
            machine_ids = list({e["machine_id"] for e in events})

        for machine_id in machine_ids:
            events = app.sensor_service.get_recent_events_for_machine(
                machine_id, limit=20
            )
            verdict = app.maintenance_iq.evaluate(machine_id, station_id, events)
            if verdict.verdict != HealthVerdict.HEALTHY:
                action = app.maintenance_iq.recommend(verdict)
                recommendations.append(_maintenance_action_to_dict(action))

        return jsonify({"recommendations": recommendations, "count": len(recommendations)})

    @app.route("/api/maintenance/feedback", methods=["POST"])
    def maintenance_feedback():
        """Record operator feedback on a sensor event."""
        if not app.sensor_service:
            return jsonify({"error": "Sensor service not available"}), 503

        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        event_id = data.get("event_id")
        operator_action = data.get("operator_action")

        if not event_id or not operator_action:
            return jsonify({"error": "Missing event_id or operator_action"}), 400

        allowed = {"confirm", "reject", "uncertain"}
        if operator_action not in allowed:
            return (
                jsonify(
                    {"error": f"Invalid operator_action. Must be one of: {allowed}"}
                ),
                400,
            )

        try:
            updated = app.sensor_service.update_event_feedback(
                event_id=event_id,
                operator_action=operator_action,
                operator_id=data.get("operator_id", ""),
                rejection_reason=data.get("rejection_reason", ""),
                comment=data.get("comment", ""),
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if not updated:
            return jsonify({"error": "Event not found"}), 404

        return jsonify({"status": "recorded", "event_id": event_id})

    @app.route("/api/maintenance/stats", methods=["GET"])
    def maintenance_stats():
        """Get sensor service statistics."""
        if not app.sensor_service:
            return jsonify(
                {
                    "total_events": 0,
                    "baseline_count": 0,
                    "last_reading_at": None,
                    "events_by_machine": {},
                    "events_by_sensor_type": {},
                }
            ), 200

        conn = app.sensor_service._conn
        if conn is None:
            return jsonify(
                {
                    "total_events": 0,
                    "baseline_count": 0,
                    "last_reading_at": None,
                    "events_by_machine": {},
                    "events_by_sensor_type": {},
                }
            ), 200

        total_events = conn.execute(
            "SELECT COUNT(*) FROM sensor_events"
        ).fetchone()[0]
        baseline_count = conn.execute(
            "SELECT COUNT(*) FROM baseline_profiles"
        ).fetchone()[0]

        last_reading_row = conn.execute(
            "SELECT MAX(timestamp) FROM sensor_events"
        ).fetchone()[0]
        last_reading_at = last_reading_row if last_reading_row else None

        events_by_machine = {}
        for row in conn.execute(
            "SELECT machine_id, COUNT(*) FROM sensor_events GROUP BY machine_id"
        ).fetchall():
            events_by_machine[row[0]] = row[1]

        events_by_sensor_type = {}
        for row in conn.execute(
            "SELECT sensor_type, COUNT(*) FROM sensor_events GROUP BY sensor_type"
        ).fetchall():
            events_by_sensor_type[row[0]] = row[1]

        return jsonify(
            {
                "total_events": total_events,
                "baseline_count": baseline_count,
                "last_reading_at": last_reading_at,
                "events_by_machine": events_by_machine,
                "events_by_sensor_type": events_by_sensor_type,
            }
        )

    return app


def _sensor_event_to_dict(event: SensorEvent) -> dict[str, Any]:
    """Convert a SensorEvent to a JSON-serializable dict."""
    return {
        "event_id": event.event_id,
        "timestamp": event.timestamp.isoformat() if event.timestamp else "",
        "station_id": event.station_id,
        "machine_id": event.machine_id,
        "sensor_type": event.sensor_type.value
        if hasattr(event.sensor_type, "value")
        else event.sensor_type,
        "raw_values": event.raw_values,
        "anomaly_score": event.anomaly_score,
        "confidence": event.confidence,
        "edge_verdict": event.edge_verdict.value
        if hasattr(event.edge_verdict, "value")
        else event.edge_verdict,
    }


def _maintenance_action_to_dict(action: MaintenanceAction) -> dict[str, Any]:
    """Convert a MaintenanceAction to a JSON-serializable dict."""
    verdict = action.verdict
    return {
        "action_id": action.action_id,
        "timestamp": action.timestamp.isoformat() if action.timestamp else "",
        "machine_id": action.machine_id,
        "station_id": action.station_id,
        "action_type": action.action_type.value
        if hasattr(action.action_type, "value")
        else action.action_type,
        "action_en": action.action_en,
        "action_zh": action.action_zh,
        "sop_section": action.sop_section,
        "urgency": action.urgency,
        "operator_action": action.operator_action,
        "operator_id": action.operator_id,
        "rejection_reason": action.rejection_reason,
        "verdict": {
            "verdict_id": verdict.verdict_id,
            "timestamp": verdict.timestamp.isoformat() if verdict.timestamp else "",
            "machine_id": verdict.machine_id,
            "station_id": verdict.station_id,
            "verdict": verdict.verdict.value
            if hasattr(verdict.verdict, "value")
            else verdict.verdict,
            "z_score": verdict.z_score,
            "confidence": verdict.confidence,
            "warning_threshold": verdict.warning_threshold,
            "critical_threshold": verdict.critical_threshold,
            "contributing_factors": verdict.contributing_factors,
        },
    }


def _inspection_to_dict(
    event: Any, operator_actions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Convert an InspectionEvent to a JSON-serializable dict."""
    detections = []
    for d in event.detections or []:
        detections.append(
            {
                "defect_type": d.defect_type,
                "confidence": round(d.confidence, 4),
                "severity": round(d.severity, 4),
                "bbox": {
                    "x": round(d.bbox.x, 1),
                    "y": round(d.bbox.y, 1),
                    "width": round(d.bbox.width, 1),
                    "height": round(d.bbox.height, 1),
                },
            }
        )

    return {
        "inspection_id": event.inspection_id,
        "timestamp": event.timestamp.isoformat() if event.timestamp else "",
        "station_id": event.station_id,
        "workspace_id": event.workspace_id,
        "product_id": event.product_id,
        "operator_id": event.operator_id,
        "decision": event.decision.value
        if hasattr(event.decision, "value")
        else event.decision,
        "confidence": round(event.confidence, 4),
        "detections": detections,
        "num_detections": event.num_detections,
        "image_original_path": event.image_original_path,
        "image_annotated_path": event.image_annotated_path,
        "image_original_url": _safe_inspection_image_url(
            event.inspection_id, "original", event.image_original_path
        ),
        "image_annotated_url": _safe_inspection_image_url(
            event.inspection_id, "annotated", event.image_annotated_path
        ),
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
        "operator_actions": operator_actions or [],
        "sync_status": event.sync_status.value
        if hasattr(event.sync_status, "value")
        else event.sync_status,
        "synced_at": event.synced_at.isoformat() if event.synced_at else None,
    }


def _safe_inspection_image_url(
    inspection_id: str, kind: str, path_value: str
) -> str | None:
    """Expose auth-protected local image routes instead of raw public object URLs."""
    if not path_value:
        return None
    return f"/api/inspections/{inspection_id}/{kind}.jpg"


def run_api(runtime: Any = None, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the station API server."""
    app = create_app(runtime)
    logger.info(
        "Station API v2 starting on %s:%d (mode=%s)",
        host,
        port,
        os.environ.get("STORAGE_MODE", "local"),
    )
    app.run(host=host, port=port, debug=False, threaded=True)
