"""
IntelFactor.ai — CLI Entry Points
Run station or hub from command line.

Usage:
    intelfactor-station --config /opt/intelfactor/config/station.yaml
    intelfactor-hub --config /opt/intelfactor/config/hub.yaml
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("intelfactor")


def _load_yaml(path: str) -> dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        import json
        with open(path) as f:
            return json.load(f)


def run_station():
    """CLI: run station runtime + API server."""
    parser = argparse.ArgumentParser(description="IntelFactor Station Node")
    parser.add_argument("--config", default="/opt/intelfactor/config/station.yaml", help="Station config file")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server bind address")
    parser.add_argument("--no-camera", action="store_true", help="Skip camera ingest (API + RCA only)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--doctor", action="store_true", help="Run pre-flight diagnostics and exit")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Doctor mode: run diagnostics and exit
    if args.doctor:
        from packages.inference.doctor import run_doctor
        report = run_doctor(
            config_path=args.config,
            skip_camera=args.no_camera,
        )
        report.print_report()
        sys.exit(0 if report.all_passed else 1)

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        raw = _load_yaml(str(config_path))
        logger.info("Config loaded: %s", config_path)
    else:
        raw = {}
        logger.warning("Config not found at %s, using defaults", config_path)

    # Build StationConfig from YAML
    from packages.inference.modes.runtime import StationConfig, StationRuntime
    from packages.inference.schemas import DeploymentMode

    mode_str = raw.get("mode", "station_only")
    mode = DeploymentMode(mode_str)

    station_config = StationConfig(
        station_id=raw.get("station_id", "station_1"),
        data_dir=raw.get("data_dir", "/opt/intelfactor/data"),
        model_dir=raw.get("model_dir", "/opt/intelfactor/models"),
        edge_yaml_path=raw.get("edge_yaml_path", "/opt/intelfactor/config/edge.yaml"),
        vision_model_override=raw.get("vision_model"),
        language_model_override=raw.get("language_model"),
        confidence_threshold=raw.get("confidence_threshold", 0.5),
        anomaly_check_interval_sec=raw.get("rca", {}).get("anomaly_check_interval_sec", 300),
        z_score_threshold=raw.get("rca", {}).get("z_score_threshold", 2.5),
        defect_taxonomy=raw.get("defect_taxonomy", {}),
        sop_map=raw.get("sop_map", {}),
        sop_context=raw.get("sop_context", {}),
    )

    # Start runtime
    runtime = StationRuntime(station_config, mode=mode)
    runtime.start()

    # Optionally start camera ingest
    if not args.no_camera and raw.get("camera", {}).get("source"):
        from packages.inference.ingest import CameraConfig, CameraIngest, CameraProtocol

        cam_cfg = raw.get("camera", {})
        camera_config = CameraConfig(
            source=cam_cfg.get("source", ""),
            protocol=CameraProtocol(cam_cfg.get("protocol", "rtsp")),
            station_id=station_config.station_id,
            fps_target=cam_cfg.get("fps_target", 30),
        )

        ingest = CameraIngest(camera_config, on_frame=lambda frame, meta: runtime.process_frame(frame))
        ingest.start()
        runtime._ingest = ingest  # attach for stats
        logger.info("Camera ingest started: %s", camera_config.source)

    # Start API server in background thread
    from packages.inference.api_v2 import create_app

    app = create_app(runtime)

    api_thread = threading.Thread(
        target=lambda: app.run(host=args.host, port=args.port, debug=False, threaded=True),
        daemon=True,
        name="api-server",
    )
    api_thread.start()
    logger.info("Station API running on http://%s:%d", args.host, args.port)

    # Handle shutdown signals
    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Station ready: %s (mode=%s)", station_config.station_id, mode.value)
    logger.info("  Health: http://%s:%d/health", args.host, args.port)
    logger.info("  API:    http://%s:%d/api/status", args.host, args.port)

    # Block until shutdown
    shutdown_event.wait()

    # Graceful shutdown
    runtime.stop()
    logger.info("Station shutdown complete")


def run_hub():
    """CLI: run site hub runtime."""
    parser = argparse.ArgumentParser(description="IntelFactor Site Hub")
    parser.add_argument("--config", default="/opt/intelfactor/config/hub.yaml", help="Hub config file")
    parser.add_argument("--port", type=int, default=8090, help="Hub API port")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    from packages.inference.modes.runtime import SiteHubRuntime, HubConfig

    config = HubConfig()
    if Path(args.config).exists():
        raw = _load_yaml(args.config)
        config.hub_url = raw.get("hub_url", config.hub_url)
        config.hub_model_override = raw.get("language_model")

    hub = SiteHubRuntime(config)
    hub.start()

    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Site Hub ready on port %d", args.port)
    shutdown_event.wait()

    hub.stop()
    logger.info("Hub shutdown complete")


def run_doctor():
    """CLI: run pre-flight diagnostics."""
    parser = argparse.ArgumentParser(
        description="IntelFactor System Doctor - Pre-flight diagnostics for edge deployment"
    )
    parser.add_argument("--full", action="store_true", help="Run full checks including camera")
    parser.add_argument("--camera", type=str, help="Camera URI to test (overrides CAMERA_URI env)")
    parser.add_argument("--config", default="/opt/intelfactor/config/station.yaml", help="Station config file")
    parser.add_argument("--skip-camera", action="store_true", help="Skip camera check even with --full")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Import the doctor script
    import os
    import json as json_module

    # Set environment from config if available
    config_path = Path(args.config)
    if config_path.exists():
        raw = _load_yaml(str(config_path))
        if "camera" in raw and "source" in raw["camera"]:
            os.environ.setdefault("CAMERA_URI", raw["camera"]["source"])
        if "data_dir" in raw:
            os.environ.setdefault("EVIDENCE_DIR", f"{raw['data_dir']}/evidence")
            os.environ.setdefault("SQLITE_DB_PATH", f"{raw['data_dir']}/local.db")
        if "model_dir" in raw:
            os.environ.setdefault("VISION_MODEL_PATH", f"{raw['model_dir']}/vision")
            os.environ.setdefault("LLM_MODEL_PATH", f"{raw['model_dir']}/language")

    # Run doctor checks
    from scripts.doctor import (
        check_python,
        check_gpu,
        check_storage,
        check_evidence_dir,
        check_disk_space,
        check_api_health,
        check_models,
        check_camera,
    )

    results = {}

    if not args.json:
        print("=" * 60)
        print("IntelFactor.ai System Doctor")
        print("=" * 60)
        print()

    results["python"] = check_python()
    results["gpu"] = check_gpu()
    results["storage"] = check_storage()
    results["evidence"] = check_evidence_dir()
    results["disk"] = check_disk_space()
    results["api"] = check_api_health()
    results["models"] = check_models()

    # Camera check
    if (args.full or args.camera) and not args.skip_camera:
        camera_uri = args.camera or os.environ.get("CAMERA_URI", "/dev/video0")
        results["camera"] = check_camera(camera_uri)

    # Summary
    passed = sum(1 for ok in results.values() if ok)
    total = len(results)

    if args.json:
        output = {
            "checks": results,
            "passed": passed,
            "total": total,
            "ready": passed == total,
        }
        print(json_module.dumps(output, indent=2))
    else:
        print()
        print("=" * 60)
        print(f"Results: {passed}/{total} checks passed")

        if passed == total:
            print("System is ready for deployment.")
        else:
            print("Some checks failed. Review warnings above.")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "hub":
        sys.argv.pop(1)
        run_hub()
    elif len(sys.argv) > 1 and sys.argv[1] == "doctor":
        sys.argv.pop(1)
        run_doctor()
    else:
        run_station()
