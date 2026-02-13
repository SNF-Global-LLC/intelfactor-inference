# IntelFactor Inference Engine

Edge-native manufacturing quality inspection + closed-loop root cause analysis.

Runs entirely on NVIDIA Jetson — no cloud required. Detects defects, explains why they happened, recommends what to do about it, and learns from operator feedback.

## Architecture

```
Camera ──► Vision (YOLOv8/TensorRT) ──► DefectIQ Rules ──► RCA Pipeline
                                                              │
                ┌─────────────────────────────────────────────┘
                ▼
    ┌──── Accumulator (SQLite, 30-day rolling, z-score anomaly)
    ├──── Correlator (process parameters, Pearson drift detection)
    ├──── Explainer (Qwen 3B via llama.cpp, bilingual CN/EN)
    └──── Recommender (SOP-linked actions, causal triples)
                │
                ▼
    Operator Dashboard (Chinese-primary, accept/reject feedback)
                │
                ▼
    Causal Triple Store (defect → cause → outcome, verified by operators)
```

## Quick Start

```bash
# Install (Jetson)
pip install -e ".[jetson]"

# Pre-flight check
intelfactor-station doctor --config configs/station.yaml

# Run station
intelfactor-station run --config configs/station.yaml

# Open dashboard at http://localhost:8080
```

## Project Structure

```
packages/inference/
├── providers/              # GPU-agnostic vision + language providers
│   ├── base.py             #   Provider interfaces
│   ├── resolver.py         #   Auto-detect hardware, select optimal models
│   ├── vision_trt.py       #   TensorRT inference (YOLOv8)
│   ├── language_llama.py   #   llama.cpp inference (Qwen 3B)
│   └── language_vllm.py    #   vLLM inference (GPU server hub)
├── rca/                    # 4-Layer Root Cause Analysis
│   ├── accumulator.py      #   Defect pattern accumulation + anomaly detection
│   ├── correlator.py       #   Process parameter correlation + drift
│   ├── explainer.py        #   Bilingual explanation via SLM
│   ├── recommender.py      #   SOP-linked recommendations + causal triples
│   └── pipeline.py         #   Full RCA pipeline orchestrator
├── modes/runtime.py        # Station + Hub deployment modes
├── ingest.py               # Camera capture (RTSP/GigE/USB) with watchdog
├── api.py                  # Station REST API (Flask, 10 endpoints)
├── evidence.py             # JPEG evidence writer with disk quota
├── doctor.py               # Pre-flight diagnostics
├── cli.py                  # CLI entry points
├── schemas.py              # Core data structures
├── sync.py                 # SQLite → PostgreSQL sync service
└── static/index.html       # Operator dashboard (Chinese-primary)

configs/
├── station.yaml            # Station configuration template
└── wiko_taxonomy.yaml      # 13 defect types (Wiko cutlery)

deploy/
├── station/                # Jetson station deployment
├── hub/                    # Docker Compose hub stack
└── systemd/                # Production service unit

scripts/                    # Build + test automation
tests/                      # 59 tests, all passing
docs/DEPLOYMENT_GUIDE.md    # Bilingual CN/EN deployment guide
```

## Hardware Support

| Device | VRAM | Role | Vision Model | Language Model |
|---|---|---|---|---|
| Orin Nano Super | 8GB | Station (pilot) | YOLOv8n FP16 | Qwen-2.5-3B Q4 |
| Orin NX 16GB | 16GB | Station (production) | YOLOv8m FP16 | Qwen-2.5-7B Q4 |
| AGX Orin 64GB | 64GB | Multi-line hub | YOLOv8l FP16 | Qwen-2.5-14B Q8 |
| GPU Server | 24GB+ | Site hub | YOLOv8x FP16 | Qwen-2.5-14B (vLLM) |

The system auto-detects hardware and selects optimal models via `CapabilityResolver`.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
# 59 passed
```

## Deployment

See [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) for full bilingual (CN/EN) deployment instructions.

## License

Proprietary — IntelFactor.ai
