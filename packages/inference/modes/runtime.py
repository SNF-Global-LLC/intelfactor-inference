"""
IntelFactor.ai — Deployment Modes
StationOnly: everything local on the station node.
StationPlusHub: station does realtime; hub does cross-line RCA + VLM review.

Both modes use identical RCA pipeline. The difference is WHERE the heavier
models run and whether cross-line intelligence is available.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packages.inference.providers.base import LanguageProvider, VisionProvider
from packages.inference.providers.resolver import CapabilityResolver
from packages.inference.rca.accumulator import DefectAccumulator
from packages.inference.rca.correlator import ProcessCorrelator
from packages.inference.rca.explainer import RCAExplainer
from packages.inference.rca.pipeline import RCAPipeline
from packages.inference.rca.recommender import ActionRecommender
from packages.inference.schemas import (
    DeploymentMode,
    DetectionResult,
    Verdict,
)

logger = logging.getLogger(__name__)


@dataclass
class StationConfig:
    """Configuration for a station node."""
    station_id: str = "station_1"
    data_dir: str = "/opt/intelfactor/data"
    model_dir: str = "/opt/intelfactor/models"
    edge_yaml_path: str = "/opt/intelfactor/config/edge.yaml"

    # Inference
    vision_model_override: str | None = None
    language_model_override: str | None = None
    confidence_threshold: float = 0.5
    fail_threshold: float = 0.5
    review_threshold: float = 0.3
    # Defect class names (0-based, must match YOLO engine output order).
    # Loaded from station.yaml defect_classes.
    # Falls back to _CANONICAL_DEFECT_CLASSES in resolver.py if empty.
    defect_classes: list[str] = field(default_factory=list)
    provider_config: dict[str, Any] = field(default_factory=dict)

    # RCA
    anomaly_check_interval_sec: int = 300  # 5 minutes
    z_score_threshold: float = 2.5

    # Defect taxonomy and SOP (loaded from edge.yaml)
    defect_taxonomy: dict[str, Any] = field(default_factory=dict)
    sop_map: dict[str, Any] = field(default_factory=dict)
    sop_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class HubConfig:
    """Configuration for a site hub (StationPlusHub mode only)."""
    hub_url: str = "http://hub.local:8080"
    hub_model_override: str | None = None
    review_queue_enabled: bool = True
    cross_line_rca_enabled: bool = True


class StationRuntime:
    """
    The runtime that runs on every station node.
    Handles: vision inference → event recording → local RCA → operator UI serving.

    In StationOnly mode: this IS the entire system.
    In StationPlusHub mode: this handles realtime, hub handles heavy analysis.
    """

    def __init__(self, config: StationConfig, mode: DeploymentMode = DeploymentMode.STATION_ONLY):
        self.config = config
        self.mode = mode
        self._running = False

        # Will be initialized in start()
        self.vision: VisionProvider | None = None
        self.language: LanguageProvider | None = None
        self.pipeline: RCAPipeline | None = None
        self.evidence_writer = None
        self._rca_thread: threading.Thread | None = None
        self._review_queue: queue.Queue[DetectionResult] = queue.Queue(maxsize=1000)

        # Machine health copilot — initialized in start() when machine_health.enabled
        self.sensor_service: Any = None
        self.maintenance_iq: Any = None
        self.machine_health_config: dict[str, Any] = {}

    def start(self) -> None:
        """Initialize all components and start the RCA background loop."""
        logger.info("Starting StationRuntime: station=%s mode=%s", self.config.station_id, self.mode.value)

        # Resolve hardware capabilities and select optimal providers
        resolver = CapabilityResolver(config={
            **self.config.provider_config,
            "model_dir": self.config.model_dir,
            "station_id": self.config.station_id,
            "confidence_threshold": self.config.confidence_threshold,
            "fail_threshold": self.config.fail_threshold,
            "review_threshold": self.config.review_threshold,
        })

        caps = resolver.detect_capabilities()
        logger.info("Device: %s (%s, %dMB VRAM)", caps.device_class.value, caps.gpu_name, caps.vram_mb)

        # Initialize providers
        # Pass defect_classes so TensorRTVisionProvider maps class IDs → names correctly.
        # station.yaml defect_classes takes precedence; resolver falls back to canonical defaults.
        vision_provider_config: dict[str, Any] = {"station_id": self.config.station_id}
        if self.config.defect_classes:
            vision_provider_config["defect_classes"] = self.config.defect_classes

        self.vision = resolver.resolve_vision_provider(
            override_model=self.config.vision_model_override,
            provider_config=vision_provider_config,
        )
        self.vision.load()

        self.language = resolver.resolve_language_provider(
            override_model=self.config.language_model_override,
        )
        self.language.load()

        # Load edge.yaml for process parameters
        edge_yaml = self._load_edge_yaml()

        # ── Machine Health Copilot ────────────────────────────────────
        mh_cfg = edge_yaml.get("machine_health", {})
        if mh_cfg.get("enabled", False):
            try:
                from packages.ingestion.sensor_service import SensorService
                from packages.policy.maintenance_iq import MaintenanceIQ

                sensors_db = Path(self.config.data_dir) / "sensors.db"
                thresholds = mh_cfg.get("thresholds", {})
                warning_t = float(thresholds.get("warning", 2.0))
                critical_t = float(thresholds.get("critical", 3.5))

                self.sensor_service = SensorService(
                    station_id=self.config.station_id,
                    db_path=sensors_db,
                    warning_threshold=warning_t,
                    critical_threshold=critical_t,
                )
                self.sensor_service.start()

                self.maintenance_iq = MaintenanceIQ.from_edge_yaml(edge_yaml)
                self.machine_health_config = mh_cfg

                n_topics = len(self.sensor_service.topics)
                logger.info(
                    "Sensor ingestion started for %d sensor topics (warning=%.1f, critical=%.1f)",
                    n_topics, warning_t, critical_t,
                )
            except Exception as exc:
                logger.warning(
                    "Machine health copilot init failed — sensor monitoring disabled: %s", exc
                )

        # Initialize RCA pipeline
        data_dir = Path(self.config.data_dir)

        accumulator = DefectAccumulator(db_path=data_dir / "accumulator.db")
        accumulator.start()

        correlator = ProcessCorrelator.from_edge_yaml(edge_yaml)

        explainer = RCAExplainer(
            language_provider=self.language,
            sop_context=self.config.sop_context,
            defect_taxonomy=self.config.defect_taxonomy,
        )

        recommender = ActionRecommender(
            sop_map=self.config.sop_map,
            db_path=data_dir / "triples.db",
        )
        recommender.start()

        self.pipeline = RCAPipeline(accumulator, correlator, explainer, recommender)

        # Initialize evidence writer
        try:
            from packages.inference.evidence import EvidenceWriter
            evidence_dir = data_dir / "evidence"
            max_disk_gb = getattr(self.config, "max_evidence_disk_gb", 50.0)
            self.evidence_writer = EvidenceWriter(
                evidence_dir=str(evidence_dir),
                max_disk_gb=max_disk_gb,
            )
        except Exception as e:
            logger.warning("Evidence writer init failed (frames will not be saved): %s", e)
            self.evidence_writer = None

        # Start background RCA check loop
        self._running = True
        self._rca_thread = threading.Thread(target=self._rca_loop, daemon=True, name="rca-loop")
        self._rca_thread.start()

        logger.info("StationRuntime started successfully")

    def process_frame(self, frame: Any) -> DetectionResult:
        """
        Process a single frame through the full pipeline.
        Called by the camera ingest service for every frame.

        1. Run vision inference.
        2. Record in accumulator (if FAIL/REVIEW).
        3. If REVIEW and hub mode, enqueue for hub VLM triage.
        4. Return result for operator display.
        """
        if self.vision is None or self.pipeline is None:
            raise RuntimeError("StationRuntime not started.")

        # Vision inference
        result = self.vision.detect(frame)

        # Write evidence for FAIL and REVIEW events
        if self.evidence_writer and result.verdict in (Verdict.FAIL, Verdict.REVIEW):
            try:
                import numpy as np
                if isinstance(frame, np.ndarray):
                    bbox_list = [
                        {"x": d.bbox.x, "y": d.bbox.y, "width": d.bbox.width, "height": d.bbox.height,
                         "label": d.defect_type, "confidence": d.confidence, "severity": d.severity}
                        for d in result.detections
                    ] if result.detections else None
                    frame_ref = self.evidence_writer.write(
                        frame=frame,
                        event_id=result.event_id,
                        metadata={
                            "station_id": result.station_id,
                            "verdict": result.verdict.value,
                            "confidence": result.confidence,
                            "detections": len(result.detections),
                            "model_version": result.model_version,
                        },
                        bbox_list=bbox_list,
                    )
                    result.frame_ref = frame_ref
            except Exception as e:
                logger.warning("Evidence write failed for %s: %s", result.event_id, e)

        # Record in accumulator
        self.pipeline.ingest(result)

        # Route REVIEW cases to hub in StationPlusHub mode
        if (
            result.verdict == Verdict.REVIEW
            and self.mode == DeploymentMode.STATION_PLUS_HUB
            and not self._review_queue.full()
        ):
            self._review_queue.put_nowait(result)

        return result

    def get_review_queue(self) -> list[DetectionResult]:
        """Get queued REVIEW cases (for hub to pull)."""
        items = []
        while not self._review_queue.empty():
            try:
                items.append(self._review_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def _rca_loop(self) -> None:
        """Background loop that checks for anomalies on a schedule."""
        logger.info("RCA background loop started (interval=%ds)", self.config.anomaly_check_interval_sec)
        while self._running:
            try:
                if self.pipeline:
                    results = self.pipeline.run_rca(
                        station_id=self.config.station_id,
                        z_threshold=self.config.z_score_threshold,
                    )
                    if results:
                        logger.info("RCA triggered: %d anomalies processed", len(results))
                        for r in results:
                            logger.info(
                                "  → triple=%s defect=%s confidence=%.2f",
                                r.triple.triple_id, r.alert.defect_type, r.explanation.confidence,
                            )
            except Exception as e:
                logger.error("RCA loop error: %s", e)

            time.sleep(self.config.anomaly_check_interval_sec)

    def _load_edge_yaml(self) -> dict[str, Any]:
        """Load edge.yaml config bundle."""
        yaml_path = Path(self.config.edge_yaml_path)
        if not yaml_path.exists():
            logger.warning("edge.yaml not found at %s, using empty config", yaml_path)
            return {}

        try:
            import yaml
            with open(yaml_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            # Fallback: try JSON
            try:
                with open(yaml_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, Exception):
                return {}

    def get_stats(self) -> dict[str, Any]:
        """Get station runtime statistics."""
        stats: dict[str, Any] = {
            "station_id": self.config.station_id,
            "mode": self.mode.value,
            "running": self._running,
            "review_queue_depth": self._review_queue.qsize(),
            "vision_backend": type(self.vision).__name__ if self.vision else "none",
            "language_backend": type(self.language).__name__ if self.language else "none",
            "vision_is_stub": type(self.vision).__name__ == "StubVisionProvider" if self.vision else None,
            "language_is_stub": type(self.language).__name__ == "StubLanguageProvider" if self.language else None,
        }
        if self.vision is not None:
            model_name = getattr(self.vision.model_spec, "model_name", "")
            if model_name:
                stats["vision_model_name"] = model_name

            if type(self.vision).__name__ == "RoboflowHostedVisionProvider":
                workspace = getattr(self.vision, "workspace", "")
                workflow_id = getattr(self.vision, "workflow_id", "")
                project = getattr(self.vision, "project", "")
                version = getattr(self.vision, "version", "")
                experimental = bool(getattr(self.vision, "experimental_mode", False))
                verdict_policy = str(getattr(self.vision, "verdict_policy", "default"))
                if workflow_id:
                    parts = [p for p in (workspace, workflow_id) if p]
                else:
                    parts = [p for p in (workspace, project) if p]
                detail = "/".join(parts)
                if version and not workflow_id:
                    detail = f"{detail} v{version}" if detail else f"v{version}"
                stats["vision_provider_label"] = "Roboflow Hosted (Experimental)" if experimental else "Roboflow Hosted"
                if detail:
                    stats["vision_provider_detail"] = detail
                stats["vision_provider_experimental"] = experimental
                stats["vision_provider_policy"] = verdict_policy
            elif model_name:
                stats["vision_provider_label"] = model_name
        if self.pipeline:
            stats["pipeline"] = self.pipeline.get_stats()
        if self.sensor_service is not None:
            stats["sensor_service"] = self.sensor_service.get_stats()
        return stats

    def stop(self) -> None:
        """Graceful shutdown."""
        self._running = False
        if self._rca_thread and self._rca_thread.is_alive():
            self._rca_thread.join(timeout=10)
        if self.vision:
            self.vision.unload()
        if self.language:
            self.language.unload()
        if self.pipeline:
            self.pipeline.accumulator.stop()
            self.pipeline.recommender.stop()
        if self.sensor_service is not None:
            self.sensor_service.stop()
        logger.info("StationRuntime stopped")


class SiteHubRuntime:
    """
    Site hub runtime for StationPlusHub mode.
    Runs on a more powerful device (AGX Orin / Thor / GPU server).

    Responsibilities:
    - Cross-line RCA (aggregate anomalies from multiple stations).
    - VLM triage for REVIEW cases (image-aware analysis).
    - Larger SLM for deeper explanations.
    - Fleet-level dashboard data.
    """

    def __init__(self, config: HubConfig, station_ids: list[str] | None = None):
        self.config = config
        self.station_ids = station_ids or []
        self.language: LanguageProvider | None = None
        self._running = False

    def start(self) -> None:
        """Initialize hub with larger models."""
        logger.info("Starting SiteHubRuntime: %s", self.config.hub_url)

        resolver = CapabilityResolver(config={"model_dir": "/opt/intelfactor/models"})
        caps = resolver.detect_capabilities()

        # Hub always gets the biggest language model the hardware supports
        self.language = resolver.resolve_language_provider(
            override_model=self.config.hub_model_override,
        )
        self.language.load()

        self._running = True
        logger.info("SiteHubRuntime started: device=%s", caps.device_class.value)

    def triage_review(self, result: DetectionResult, frame_bytes: bytes | None = None) -> dict[str, Any]:
        """
        Triage a REVIEW case with the hub's larger model.
        If VLM is available and frame_bytes provided, do image-aware analysis.
        Otherwise, do text-only analysis with the bigger SLM.
        """
        if self.language is None:
            raise RuntimeError("Hub not started.")

        prompt = (
            f"A REVIEW case requires triage.\n"
            f"Station: {result.station_id}\n"
            f"Detections: {len(result.detections)} potential defects\n"
            f"Max confidence: {result.confidence:.2f}\n"
            f"Model verdict: {result.verdict.value}\n\n"
            f"Should this be escalated to FAIL, or confirmed as PASS?\n"
            f"Provide your judgment as JSON with: verdict, confidence, reasoning_zh, reasoning_en"
        )

        explanation = self.language.generate(prompt)
        return {
            "original_event_id": result.event_id,
            "hub_explanation": explanation,
            "hub_model": self.language.model_spec.model_name,
        }

    def stop(self) -> None:
        """Shutdown hub."""
        self._running = False
        if self.language:
            self.language.unload()
        logger.info("SiteHubRuntime stopped")
