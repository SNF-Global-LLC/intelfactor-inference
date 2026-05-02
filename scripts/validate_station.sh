#!/usr/bin/env bash
# IntelFactor.ai — Station Validation Script
#
# Pre-flight and post-deploy checks for edge station.
# Run after docker compose up or systemd start.
#
# Usage:
#   ./scripts/validate_station.sh
#   ./scripts/validate_station.sh --host 192.168.1.50 --port 8080
#   ./scripts/validate_station.sh --skip-api   # check files only, no API

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[PASS]${NC} $*"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $*"; FAIL=$((FAIL + 1)); }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $*"; WARN=$((WARN + 1)); }
skip() { echo -e "  ${CYAN}[SKIP]${NC} $*"; }

API_HOST="localhost"
API_PORT="8080"
SKIP_API=false
MODELS_DIR="/opt/intelfactor/models"
DATA_DIR="/opt/intelfactor/data"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)       API_HOST="$2"; shift 2 ;;
        --port)       API_PORT="$2"; shift 2 ;;
        --skip-api)   SKIP_API=true; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--host HOST] [--port PORT] [--skip-api] [--models-dir DIR] [--data-dir DIR]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
WARN=0

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  IntelFactor Station Validation${NC}"
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""

# ── Docker ───────────────────────────────────────────────────────────────────

echo -e "${BOLD}Docker:${NC}"

if command -v docker &>/dev/null; then
    ok "Docker installed"
else
    fail "Docker not installed"
fi

if docker info &>/dev/null 2>&1; then
    ok "Docker daemon running"
else
    fail "Docker daemon not running or no permission"
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "intelfactor-station"; then
    ok "Container 'intelfactor-station' running"
else
    warn "Container 'intelfactor-station' not running"
fi

# ── NVIDIA Runtime ───────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}NVIDIA Runtime:${NC}"

if docker info 2>/dev/null | grep -qi "nvidia" \
   || [ -f /etc/docker/daemon.json ] && grep -q "nvidia" /etc/docker/daemon.json 2>/dev/null; then
    ok "NVIDIA runtime available to Docker"
else
    fail "NVIDIA runtime not configured for Docker"
fi

if [ -f /etc/nv_tegra_release ]; then
    ok "Jetson platform detected"
elif command -v nvidia-smi &>/dev/null; then
    ok "NVIDIA GPU detected (non-Jetson)"
else
    fail "No NVIDIA GPU detected"
fi

# ── Model Files ──────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}Model Files:${NC}"

ENGINE_COUNT=$(find "$MODELS_DIR/vision" -name "*.engine" 2>/dev/null | wc -l)
if [ "$ENGINE_COUNT" -gt 0 ]; then
    ENGINE_FILE=$(find "$MODELS_DIR/vision" -name "*.engine" 2>/dev/null | head -1)
    ENGINE_SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
    ok "TRT engine found: $(basename "$ENGINE_FILE") ($ENGINE_SIZE)"
else
    fail "No TRT engine in $MODELS_DIR/vision/ — run build_trt_engine.sh first"
fi

MANIFEST_COUNT=$(find "$MODELS_DIR/vision" -name "*_manifest.json" 2>/dev/null | wc -l)
if [ "$MANIFEST_COUNT" -gt 0 ]; then
    ok "Engine manifest found"
else
    warn "No engine manifest — engine may be from manual copy"
fi

GGUF_COUNT=$(find "$MODELS_DIR/llm" -name "*.gguf" 2>/dev/null | wc -l)
if [ "$GGUF_COUNT" -gt 0 ]; then
    GGUF_FILE=$(find "$MODELS_DIR/llm" -name "*.gguf" 2>/dev/null | head -1)
    GGUF_SIZE=$(du -h "$GGUF_FILE" | cut -f1)
    ok "Language model found: $(basename "$GGUF_FILE") ($GGUF_SIZE)"
else
    warn "No GGUF in $MODELS_DIR/llm/ — RCA explanations will use statistical fallback"
fi

# ── Data Paths ───────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}Data Paths:${NC}"

if [ -d "$DATA_DIR" ]; then
    ok "Data directory exists: $DATA_DIR"
    if [ -w "$DATA_DIR" ] || [ "$(id -u)" -eq 0 ]; then
        ok "Data directory writable"
    else
        fail "Data directory not writable: $DATA_DIR"
    fi
else
    fail "Data directory missing: $DATA_DIR"
fi

EVIDENCE_DIR="${DATA_DIR}/evidence"
if [ -d "$EVIDENCE_DIR" ]; then
    ok "Evidence directory exists: $EVIDENCE_DIR"
else
    warn "Evidence directory missing: $EVIDENCE_DIR (will be created on first write)"
fi

# Check disk space
DATA_FREE_GB=$(df -BG "$DATA_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | tr -d 'G')
if [ -n "$DATA_FREE_GB" ] && [ "$DATA_FREE_GB" -ge 10 ]; then
    ok "Disk space: ${DATA_FREE_GB}GB free"
elif [ -n "$DATA_FREE_GB" ]; then
    warn "Disk space: ${DATA_FREE_GB}GB free (recommend ≥10GB)"
fi

# ── Camera Config ────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}Camera Config:${NC}"

CONFIG_FILE="/opt/intelfactor/config/station.yaml"
if [ -f "$CONFIG_FILE" ]; then
    ok "station.yaml present"
    if command -v python3 &>/dev/null; then
        CAM_PROTOCOL=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
cam = cfg.get('camera', {})
print(cam.get('protocol', 'not set'))
" 2>/dev/null || echo "parse error")
        ok "Camera protocol: $CAM_PROTOCOL"
    fi
else
    warn "station.yaml not found at $CONFIG_FILE"
fi

# ── API Health ───────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}API Health:${NC}"

if [ "$SKIP_API" = true ]; then
    skip "API checks (--skip-api)"
else
    HEALTH_URL="http://${API_HOST}:${API_PORT}/health"

    if command -v curl &>/dev/null; then
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$HEALTH_URL" 2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            HEALTH_BODY=$(curl -s --max-time 5 "$HEALTH_URL" 2>/dev/null)
            ok "Health endpoint: $HEALTH_URL → 200"
            ok "Response: $HEALTH_BODY"
        elif [ "$HTTP_CODE" = "000" ]; then
            fail "Cannot reach $HEALTH_URL — is the station running?"
        else
            fail "Health endpoint returned HTTP $HTTP_CODE"
        fi

        STATUS_URL="http://${API_HOST}:${API_PORT}/api/status"
        STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$STATUS_URL" 2>/dev/null || echo "000")
        if [ "$STATUS_CODE" = "200" ]; then
            ok "Status endpoint: $STATUS_URL → 200"
        else
            warn "Status endpoint: HTTP $STATUS_CODE"
        fi
    else
        skip "curl not installed — cannot check API"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────

TOTAL=$((PASS + FAIL + WARN))

echo ""
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}PASS: $PASS${NC}  ${RED}FAIL: $FAIL${NC}  ${YELLOW}WARN: $WARN${NC}  Total: $TOTAL"

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  Station validation passed${NC}"
else
    echo -e "${RED}${BOLD}  Station has $FAIL failing check(s) — review above${NC}"
fi
echo -e "${BOLD}════════════════════════════════════════════════════${NC}"
echo ""

exit "$FAIL"
