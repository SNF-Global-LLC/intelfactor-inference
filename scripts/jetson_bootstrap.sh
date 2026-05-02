#!/usr/bin/env bash
# IntelFactor.ai — Jetson Bootstrap Script
#
# One-command setup for NVIDIA Jetson Orin Nano / Orin Nano Super.
# Checks prerequisites, creates directory structure, validates environment.
#
# Usage:
#   chmod +x scripts/jetson_bootstrap.sh
#   ./scripts/jetson_bootstrap.sh
#   ./scripts/jetson_bootstrap.sh --models-dir /mnt/ssd/models
#
# After this script completes, follow the printed next steps.

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

MODELS_DIR="/opt/intelfactor/models"
DATA_DIR="/opt/intelfactor/data"
EVIDENCE_DIR="/opt/intelfactor/data/evidence"
CONFIG_DIR="/opt/intelfactor/config"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; EVIDENCE_DIR="$DATA_DIR/evidence"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--models-dir DIR] [--data-dir DIR]"
            exit 0
            ;;
        *) die "Unknown option: $1" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  IntelFactor Jetson Bootstrap${NC}"
echo -e "${BOLD}  Target: NVIDIA Jetson Orin Nano / Orin Nano Super${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""

ERRORS=0

# ── 1. Check JetPack / L4T ──────────────────────────────────────────────────

info "Checking JetPack / L4T version..."

if [ -f /etc/nv_tegra_release ]; then
    L4T_VERSION=$(head -1 /etc/nv_tegra_release | sed 's/.*R\([0-9]*\).*/\1/')
    L4T_FULL=$(head -1 /etc/nv_tegra_release)
    ok "L4T detected: $L4T_FULL"

    if [[ "$L4T_VERSION" -ge 36 ]]; then
        ok "JetPack 6.x (L4T R36+) — supported"
    elif [[ "$L4T_VERSION" -ge 35 ]]; then
        warn "JetPack 5.x (L4T R35) — supported but JetPack 6.x recommended"
    else
        warn "JetPack version may be too old (L4T R${L4T_VERSION})"
    fi
else
    fail "Not running on Jetson (/etc/nv_tegra_release not found)"
    warn "This script is intended for NVIDIA Jetson devices."
    warn "On GPU servers, use Docker directly: cd deploy/edge-only && docker compose up -d"
    ERRORS=$((ERRORS + 1))
fi

# Detect Jetson model
if [ -f /proc/device-tree/model ]; then
    JETSON_MODEL=$(tr -d '\0' < /proc/device-tree/model)
    ok "Device: $JETSON_MODEL"
else
    warn "Could not detect Jetson model"
fi

# ── 2. Check Docker ─────────────────────────────────────────────────────────

echo ""
info "Checking Docker..."

if command -v docker &>/dev/null; then
    DOCKER_VERSION=$(docker --version | head -1)
    ok "Docker: $DOCKER_VERSION"

    if docker info &>/dev/null 2>&1; then
        ok "Docker daemon running"
    else
        fail "Docker daemon not running or permission denied"
        warn "Try: sudo systemctl start docker"
        warn "  Or: sudo usermod -aG docker \$USER && newgrp docker"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "Docker not installed"
    warn "Install: sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin"
    ERRORS=$((ERRORS + 1))
fi

# Check docker compose (v2 plugin)
if docker compose version &>/dev/null 2>&1; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
    ok "Docker Compose: $COMPOSE_VERSION"
else
    fail "docker compose plugin not found"
    warn "Install: sudo apt-get install -y docker-compose-plugin"
    ERRORS=$((ERRORS + 1))
fi

# ── 3. Check NVIDIA Container Toolkit ───────────────────────────────────────

echo ""
info "Checking NVIDIA Container Toolkit..."

if dpkg -l nvidia-container-toolkit &>/dev/null 2>&1 \
   || dpkg -l nvidia-docker2 &>/dev/null 2>&1 \
   || [ -f /etc/nvidia-container-runtime/config.toml ]; then
    ok "NVIDIA Container Toolkit installed"
else
    fail "NVIDIA Container Toolkit not found"
    warn "Install:"
    warn "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    warn "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    warn "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\"
    warn "    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    warn "    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    warn "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
    warn "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    ERRORS=$((ERRORS + 1))
fi

# Quick GPU test via Docker
if docker info 2>/dev/null | grep -qi "nvidia"; then
    ok "Docker NVIDIA runtime registered"
elif [ -f /etc/docker/daemon.json ] && grep -q "nvidia" /etc/docker/daemon.json 2>/dev/null; then
    ok "NVIDIA runtime configured in daemon.json"
else
    warn "NVIDIA runtime may not be the default Docker runtime"
    warn "If 'docker compose up' fails with GPU errors, run:"
    warn "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
fi

# ── 4. Create directories ───────────────────────────────────────────────────

echo ""
info "Creating directory structure..."

for DIR in \
    "$MODELS_DIR/vision" \
    "$MODELS_DIR/llm" \
    "$DATA_DIR" \
    "$EVIDENCE_DIR" \
    "$CONFIG_DIR"; do
    if [ -d "$DIR" ]; then
        ok "Exists: $DIR"
    else
        sudo mkdir -p "$DIR"
        sudo chown "$(id -u):$(id -g)" "$DIR" 2>/dev/null || true
        ok "Created: $DIR"
    fi
done

# ── 5. Copy default config ──────────────────────────────────────────────────

echo ""
info "Checking configuration..."

if [ -f "$CONFIG_DIR/station.yaml" ]; then
    ok "station.yaml exists at $CONFIG_DIR/station.yaml"
else
    if [ -f "$REPO_DIR/configs/station.yaml" ]; then
        sudo cp "$REPO_DIR/configs/station.yaml" "$CONFIG_DIR/station.yaml"
        sudo chown "$(id -u):$(id -g)" "$CONFIG_DIR/station.yaml" 2>/dev/null || true
        ok "Copied default station.yaml to $CONFIG_DIR/"
        warn "Edit $CONFIG_DIR/station.yaml for your deployment"
    else
        warn "No station.yaml found in repo configs/"
    fi
fi

if [ -f "$REPO_DIR/configs/wiko_taxonomy.yaml" ]; then
    sudo cp "$REPO_DIR/configs/wiko_taxonomy.yaml" "$CONFIG_DIR/wiko_taxonomy.yaml" 2>/dev/null || true
fi

# ── 6. Check/copy .env ──────────────────────────────────────────────────────

echo ""
info "Checking .env for Docker deployment..."

ENV_FILE="$REPO_DIR/deploy/edge-only/.env"
ENV_EXAMPLE="$REPO_DIR/deploy/edge-only/.env.example"

if [ -f "$ENV_FILE" ]; then
    ok ".env exists at $ENV_FILE"
    warn "Review and update if needed: $ENV_FILE"
else
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        ok "Copied .env.example → .env"
        warn "IMPORTANT: Edit $ENV_FILE before running docker compose"
    else
        warn ".env.example not found — create $ENV_FILE manually"
    fi
fi

# ── 7. Check available disk space ───────────────────────────────────────────

echo ""
info "Checking disk space..."

DATA_DISK=$(df -BG "$DATA_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ -n "$DATA_DISK" ] && [ "$DATA_DISK" -ge 20 ]; then
    ok "Data partition: ${DATA_DISK}GB free"
elif [ -n "$DATA_DISK" ]; then
    warn "Data partition: ${DATA_DISK}GB free (recommend ≥20GB for evidence storage)"
fi

MODEL_DISK=$(df -BG "$MODELS_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ -n "$MODEL_DISK" ] && [ "$MODEL_DISK" -ge 5 ]; then
    ok "Models partition: ${MODEL_DISK}GB free"
elif [ -n "$MODEL_DISK" ]; then
    warn "Models partition: ${MODEL_DISK}GB free (need ~3GB for TRT engine + LLM)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
if [ "$ERRORS" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  Bootstrap complete — no blockers found${NC}"
else
    echo -e "${YELLOW}${BOLD}  Bootstrap complete — $ERRORS issue(s) need attention${NC}"
fi
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Directory layout:${NC}"
echo "  Models:   $MODELS_DIR/vision/  (TRT engines)"
echo "            $MODELS_DIR/llm/     (Qwen GGUF)"
echo "  Data:     $DATA_DIR/           (SQLite DB)"
echo "  Evidence: $EVIDENCE_DIR/       (JPEG ring buffer)"
echo "  Config:   $CONFIG_DIR/         (station.yaml)"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo ""
echo "  1. Copy your ONNX model to the Jetson and build TRT engine:"
echo "     ${CYAN}sudo cp yolov8n.onnx $MODELS_DIR/vision/${NC}"
echo "     ${CYAN}./scripts/build_trt_engine.sh $MODELS_DIR/vision/yolov8n.onnx fp16${NC}"
echo ""
echo "  2. (Optional) Download Qwen LLM for RCA explanations:"
echo "     ${CYAN}./scripts/setup_models.sh --quick${NC}"
echo "     Or set ENABLE_LOCAL_LLM=false in .env to skip."
echo ""
echo "  3. Edit .env for your station:"
echo "     ${CYAN}nano $REPO_DIR/deploy/edge-only/.env${NC}"
echo ""
echo "  4. Start the station:"
echo "     ${CYAN}cd $REPO_DIR/deploy/edge-only${NC}"
echo "     ${CYAN}docker compose up -d${NC}"
echo ""
echo "  5. Verify:"
echo "     ${CYAN}curl http://localhost:8080/health${NC}"
echo "     ${CYAN}./scripts/validate_station.sh${NC}"
echo ""
