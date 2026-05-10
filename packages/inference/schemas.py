"""
IntelFactor.ai — Core Data Schemas
Shared types for the entire inference + RCA pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enums ──────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"


class TripleStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    DISPUTED = "disputed"


class OperatorAction(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"
    PENDING = "pending"


class SyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"


class DeviceClass(str, Enum):
    """Hardware capability tiers."""
    ORIN_NANO = "orin_nano"       # 8GB, 67 TOPS
    ORIN_NX = "orin_nx"           # 16GB, 157 TOPS
    AGX_ORIN = "agx_orin"         # 64GB, 275 TOPS
    THOR_T4000 = "thor_t4000"     # 64GB, 1200 FP4 TFLOPS
    THOR_T5000 = "thor_t5000"     # 128GB, 2070 FP4 TFLOPS
    GPU_SERVER = "gpu_server"     # RTX/L4/A10/A100 etc.


class DeploymentMode(str, Enum):
    STATION_ONLY = "station_only"
    STATION_PLUS_HUB = "station_plus_hub"


class InferenceBackend(str, Enum):
    TENSORRT = "tensorrt"
    TRITON = "triton"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    TENSORRT_LLM = "tensorrt_llm"
    TENSORRT_EDGE_LLM = "tensorrt_edge_llm"


# ── Detection Results ──────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


@dataclass
class Detection:
    defect_type: str
    confidence: float
    bbox: BoundingBox
    severity: float = 0.0
    threshold_used: float = 0.0  # Per-class threshold applied
    model_version: str = ""  # Model version that made this detection


@dataclass
class DetectionResult:
    """Output from VisionProvider.detect()"""
    event_id: str = field(default_factory=lambda: f"evt_{datetime.now(tz=timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None))
    station_id: str = ""
    sku: str = ""
    shift: str = ""
    detections: list[Detection] = field(default_factory=list)
    verdict: Verdict = Verdict.PASS
    confidence: float = 0.0
    inference_ms: float = 0.0
    model_version: str = ""  # Version from model bundle metadata
    model_name: str = ""  # Model name from bundle
    frame_ref: str = ""  # path to evidence frame
    provider_metadata: dict[str, Any] = field(default_factory=dict)


# ── Inspection Event (edge-to-cloud handoff) ──────────────────────────

@dataclass
class InspectionEvent:
    """
    One discrete inspection transaction.
    This is the canonical record produced by the edge station and synced to cloud.
    """
    inspection_id: str = field(default_factory=lambda: f"insp_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None))
    station_id: str = ""
    workspace_id: str = ""

    # Product
    product_id: str = ""
    operator_id: str = ""

    # Verdict
    decision: Verdict = Verdict.PASS
    confidence: float = 0.0
    detections: list[Detection] = field(default_factory=list)
    num_detections: int = 0

    # Evidence paths (local on Jetson)
    image_original_path: str = ""
    image_annotated_path: str = ""
    report_path: str = ""

    # Evidence URLs (populated after cloud sync)
    image_original_url: str = ""
    image_annotated_url: str = ""

    # Model
    model_version: str = ""
    model_name: str = ""

    # Timing
    capture_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0

    # Operator feedback (filled later)
    accepted: bool | None = None
    rejection_reason: str = ""
    notes: str = ""

    # Sync
    sync_status: SyncStatus = SyncStatus.PENDING
    sync_error: str = ""
    last_attempt_at: datetime | None = None
    synced_at: datetime | None = None


# ── RCA Types ──────────────────────────────────────────────────────────

@dataclass
class AnomalyAlert:
    """Output from the Defect Pattern Accumulator."""
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None))
    station_id: str = ""
    defect_type: str = ""
    current_rate: float = 0.0
    baseline_rate: float = 0.0
    z_score: float = 0.0
    window_hours: float = 4.0
    event_ids: list[str] = field(default_factory=list)


@dataclass
class ProcessCorrelation:
    """Output from the Process Parameter Correlator."""
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    anomaly_alert_id: str = ""
    parameter_name: str = ""          # e.g. "grinding_rpm"
    current_value: float = 0.0
    target_value: float = 0.0
    tolerance: float = 0.0
    drift_pct: float = 0.0
    pearson_r: float = 0.0
    confidence: float = 0.0
    time_window_minutes: int = 30


@dataclass
class RCAExplanation:
    """Output from the SLM/VLM Explainer."""
    explanation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    explanation_zh: str = ""          # Chinese explanation (primary)
    explanation_en: str = ""          # English explanation
    confidence: float = 0.0
    model_used: str = ""
    generation_ms: float = 0.0


@dataclass
class ActionRecommendation:
    """Output from the Action Recommender."""
    recommendation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sop_section: str = ""             # e.g. "SOP 4.2.3"
    action_zh: str = ""
    action_en: str = ""
    parameter_target: str = ""        # e.g. "grinding_rpm: 3000 ±50"
    urgency: str = "normal"           # normal | high | critical
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class CausalTriple:
    """The fundamental unit of IntelFactor's data moat."""
    triple_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc).replace(tzinfo=None))
    station_id: str = ""

    # Defect
    defect_event_id: str = ""
    defect_type: str = ""
    defect_severity: float = 0.0

    # Cause
    cause_parameter: str = ""
    cause_value: float = 0.0
    cause_target: float = 0.0
    cause_drift_pct: float = 0.0
    cause_confidence: float = 0.0
    cause_explanation_zh: str = ""
    cause_explanation_en: str = ""

    # Outcome
    recommendation_id: str = ""
    operator_action: OperatorAction = OperatorAction.PENDING
    operator_id: str = ""
    outcome_measured: dict[str, Any] = field(default_factory=dict)

    # Status
    status: TripleStatus = TripleStatus.PENDING


# ── Configuration Types ────────────────────────────────────────────────

@dataclass
class DeviceCapabilities:
    """Detected hardware capabilities."""
    device_class: DeviceClass = DeviceClass.ORIN_NANO
    gpu_name: str = ""
    vram_mb: int = 0
    compute_capability: str = ""
    cuda_cores: int = 0
    tensor_cores: int = 0
    max_power_w: int = 0
    jetson: bool = False
    jetpack_version: str = ""


@dataclass
class ModelSpec:
    """Resolved model specification for a given device."""
    model_name: str = ""
    model_path: str = ""
    quantization: str = ""            # FP16, INT8, INT4
    backend: InferenceBackend = InferenceBackend.TENSORRT
    max_tokens: int = 512
    context_window: int = 4096
    expected_latency_ms: float = 0.0
