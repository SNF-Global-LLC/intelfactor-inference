"""
IntelFactor Metrics API Blueprint
==================================
Adds /api/metrics/* endpoints to the existing Flask station API.

Integration: Add to your existing station app.py:
    from visibility.metrics_api import metrics_bp, init_metrics
    metrics = init_metrics(app)
    app.register_blueprint(metrics_bp)

Deploy to: /opt/intelfactor/packages/visibility/metrics_api.py
"""

from flask import Blueprint, jsonify, request, current_app
from .production_metrics import ProductionMetrics
import threading
import time
import logging

logger = logging.getLogger("intelfactor.visibility.api")

metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")

# Module-level reference to the metrics engine
_metrics: ProductionMetrics = None
_idle_checker_thread: threading.Thread = None


def init_metrics(app, db_path=None, station_id=None) -> ProductionMetrics:
    """
    Initialize the metrics engine. Call once at station startup.
    Returns the ProductionMetrics instance so the main event handler
    can call metrics.on_event(event).
    """
    global _metrics, _idle_checker_thread

    db_path = db_path or app.config.get("DB_PATH", "/opt/intelfactor/data/local.db")
    station_id = station_id or app.config.get("STATION_ID", "SNF-Vision-1")

    _metrics = ProductionMetrics(db_path=db_path, station_id=station_id)

    # Start background idle checker (every 30 seconds)
    def idle_loop():
        while True:
            try:
                if _metrics:
                    _metrics.check_idle()
            except Exception as e:
                logger.error(f"Idle check error: {e}")
            time.sleep(30)

    _idle_checker_thread = threading.Thread(target=idle_loop, daemon=True, name="idle-checker")
    _idle_checker_thread.start()
    logger.info(f"Production metrics initialized for station {station_id}")

    return _metrics


def get_metrics() -> ProductionMetrics:
    """Get the global metrics instance."""
    if _metrics is None:
        raise RuntimeError("Metrics not initialized. Call init_metrics(app) first.")
    return _metrics


# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------

@metrics_bp.route("/live", methods=["GET"])
def live_snapshot():
    """
    Real-time snapshot: current throughput, cycle time, utilization.
    Poll this every 5-10 seconds from the dashboard.

    GET /api/metrics/live
    """
    m = get_metrics()
    return jsonify(m.get_live_snapshot())


@metrics_bp.route("/throughput", methods=["GET"])
def throughput():
    """
    Production counts by hour.

    GET /api/metrics/throughput?hours=8
    """
    hours = request.args.get("hours", 8, type=int)
    m = get_metrics()
    return jsonify(m.get_throughput(hours=hours))


@metrics_bp.route("/cycle-time", methods=["GET"])
def cycle_time():
    """
    Cycle time statistics.

    GET /api/metrics/cycle-time?hours=1
    """
    hours = request.args.get("hours", 1, type=int)
    m = get_metrics()
    return jsonify(m.get_cycle_times(hours=hours))


@metrics_bp.route("/utilization", methods=["GET"])
def utilization():
    """
    Station utilization (active vs idle).

    GET /api/metrics/utilization?hours=8
    """
    hours = request.args.get("hours", 8, type=int)
    m = get_metrics()
    return jsonify(m.get_utilization(hours=hours))


@metrics_bp.route("/shift-summary", methods=["GET"])
def shift_summary():
    """
    Generate or retrieve shift summary.

    GET /api/metrics/shift-summary
    GET /api/metrics/shift-summary?shift_id=shift_1&date=2026-02-16
    """
    shift_id = request.args.get("shift_id", None)
    date = request.args.get("date", None)
    m = get_metrics()
    summary = m.generate_shift_summary(shift_id=shift_id, shift_date=date)
    return jsonify(summary)


@metrics_bp.route("/health", methods=["GET"])
def metrics_health():
    """Health check for the metrics subsystem."""
    m = get_metrics()
    return jsonify({
        "status": "ok",
        "station_id": m.station_id,
        "current_state": m._current_state,
        "last_detection": m._last_detection_time,
        "idle_threshold_seconds": m.idle_threshold,
    })
