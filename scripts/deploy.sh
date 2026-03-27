#!/bin/bash
# IntelFactor Deployment Script
# Usage: ./scripts/deploy.sh [edge-only|hybrid|hub]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${DATA_DIR:-/opt/intelfactor/data}"
MODELS_DIR="${MODELS_DIR:-/opt/intelfactor/models}"

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${BLUE}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║           IntelFactor Inference Engine                    ║"
    echo "║                  Deployment Script                         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi
    log_info "✓ Docker installed"

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose v2."
        exit 1
    fi
    log_info "✓ Docker Compose installed"

    # Check NVIDIA runtime (optional)
    if docker info 2>/dev/null | grep -q "nvidia"; then
        log_info "✓ NVIDIA runtime available"
        HAS_GPU=true
    else
        log_warn "NVIDIA runtime not detected. GPU features disabled."
        HAS_GPU=false
    fi
}

setup_directories() {
    log_info "Setting up directories..."

    sudo mkdir -p "$DATA_DIR/evidence"
    sudo mkdir -p "$MODELS_DIR"
    sudo chown -R "$(id -u):$(id -g)" "$DATA_DIR" 2>/dev/null || true

    log_info "✓ Data directory: $DATA_DIR"
    log_info "✓ Models directory: $MODELS_DIR"
}

configure_environment() {
    local mode=$1
    local env_file="$PROJECT_ROOT/deploy/$mode/.env"
    local example_file="$PROJECT_ROOT/deploy/$mode/.env.example"

    if [[ ! -f "$env_file" ]]; then
        if [[ -f "$example_file" ]]; then
            cp "$example_file" "$env_file"
            log_info "Created $env_file from example"
            log_warn "Please edit $env_file with your configuration"

            if [[ "$mode" == "hybrid" ]]; then
                echo ""
                log_warn "For hybrid mode, you need to configure:"
                echo "  - CLOUD_API_URL (https://api.intelfactor.ai)"
                echo "  - CLOUD_API_KEY (from app.intelfactor.ai)"
                echo "  - S3_BUCKET (optional, for evidence upload)"
                echo ""
                read -p "Do you want to configure now? [y/N] " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    configure_hybrid "$env_file"
                fi
            fi
        else
            log_error "Environment example file not found: $example_file"
            exit 1
        fi
    else
        log_info "Using existing configuration: $env_file"
    fi
}

configure_hybrid() {
    local env_file=$1

    read -p "Station ID [station_01]: " station_id
    station_id=${station_id:-station_01}

    read -p "Cloud API URL [https://api.intelfactor.ai]: " api_url
    api_url=${api_url:-https://api.intelfactor.ai}

    read -p "Cloud API Key: " api_key
    if [[ -z "$api_key" ]]; then
        log_warn "No API key provided. Cloud sync will be disabled."
    fi

    read -p "Camera URI [/dev/video0]: " camera_uri
    camera_uri=${camera_uri:-/dev/video0}

    # Update .env file
    sed -i.bak "s|^STATION_ID=.*|STATION_ID=$station_id|" "$env_file"
    sed -i.bak "s|^CLOUD_API_URL=.*|CLOUD_API_URL=$api_url|" "$env_file"
    sed -i.bak "s|^CLOUD_API_KEY=.*|CLOUD_API_KEY=$api_key|" "$env_file"
    sed -i.bak "s|^CAMERA_URI=.*|CAMERA_URI=$camera_uri|" "$env_file"
    rm -f "$env_file.bak"

    log_info "Configuration saved to $env_file"
}

deploy_edge_only() {
    log_info "Deploying edge-only mode..."

    cd "$PROJECT_ROOT/deploy/edge-only"

    # Build if needed
    if ! docker images | grep -q "intelfactor-inference"; then
        log_info "Building Docker image..."
        docker compose build
    fi

    # Deploy
    docker compose up -d

    log_info "✓ Edge-only deployment complete"
    echo ""
    echo -e "${GREEN}Dashboard:${NC} http://localhost:8080"
    echo -e "${GREEN}API:${NC}       http://localhost:8080/api/"
    echo -e "${GREEN}Health:${NC}    http://localhost:8080/health"
}

deploy_hybrid() {
    log_info "Deploying hybrid mode..."

    cd "$PROJECT_ROOT/deploy/hybrid"

    # Verify cloud config
    if grep -q "CLOUD_API_KEY=$" .env 2>/dev/null || grep -q "CLOUD_API_KEY=your_api_key" .env 2>/dev/null; then
        log_warn "Cloud API key not configured. Sync will fail."
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi

    # Build if needed
    if ! docker images | grep -q "intelfactor-inference"; then
        log_info "Building Docker images..."
        docker compose build
    fi

    # Deploy
    docker compose up -d

    log_info "✓ Hybrid deployment complete"
    echo ""
    echo -e "${GREEN}Dashboard:${NC} http://localhost:8080"
    echo -e "${GREEN}API:${NC}       http://localhost:8080/api/"
    echo -e "${GREEN}Cloud:${NC}     Syncing to $(grep CLOUD_API_URL .env | cut -d= -f2)"
    echo ""
    echo "View sync logs: docker logs -f intelfactor-sync-agent"
}

deploy_hub() {
    log_info "Deploying site hub..."

    cd "$PROJECT_ROOT/deploy/hub"

    # Deploy
    docker compose up -d

    log_info "✓ Hub deployment complete"
    echo ""
    echo -e "${GREEN}Grafana:${NC}    http://localhost:3000 (admin/<GRAFANA_PASSWORD from .env>)"
    echo -e "${GREEN}Prometheus:${NC} http://localhost:9090"
    echo -e "${GREEN}MinIO:${NC}      http://localhost:9001"
    echo -e "${GREEN}Postgres:${NC}   localhost:5432"
}

show_status() {
    echo ""
    log_info "Current deployments:"
    echo ""

    for mode in edge-only hybrid hub; do
        if [[ -f "$PROJECT_ROOT/deploy/$mode/docker-compose.yml" ]]; then
            cd "$PROJECT_ROOT/deploy/$mode"
            if docker compose ps --status running 2>/dev/null | grep -q "running"; then
                echo -e "  ${GREEN}●${NC} $mode: running"
                docker compose ps --format "table {{.Name}}\t{{.Status}}" 2>/dev/null | tail -n +2 | sed 's/^/    /'
            else
                echo -e "  ${RED}○${NC} $mode: stopped"
            fi
        fi
    done
}

run_doctor() {
    log_info "Running system health checks..."
    python3 "$PROJECT_ROOT/scripts/doctor.py" --full
}

show_help() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  edge-only   Deploy edge-only mode (no cloud)"
    echo "  hybrid      Deploy hybrid mode (with cloud sync)"
    echo "  hub         Deploy site hub (Postgres + Grafana)"
    echo "  status      Show deployment status"
    echo "  doctor      Run system health checks"
    echo "  stop        Stop all deployments"
    echo "  logs        Show station logs"
    echo ""
    echo "Examples:"
    echo "  $0 edge-only          # Deploy edge-only mode"
    echo "  $0 hybrid             # Deploy with cloud sync"
    echo "  $0 status             # Check what's running"
}

# Main
print_banner

case "${1:-help}" in
    edge-only)
        check_prerequisites
        setup_directories
        configure_environment "edge-only"
        deploy_edge_only
        ;;
    hybrid)
        check_prerequisites
        setup_directories
        configure_environment "hybrid"
        deploy_hybrid
        ;;
    hub)
        check_prerequisites
        deploy_hub
        ;;
    status)
        show_status
        ;;
    doctor)
        run_doctor
        ;;
    stop)
        log_info "Stopping all deployments..."
        cd "$PROJECT_ROOT/deploy/edge-only" && docker compose down 2>/dev/null || true
        cd "$PROJECT_ROOT/deploy/hybrid" && docker compose down 2>/dev/null || true
        cd "$PROJECT_ROOT/deploy/hub" && docker compose down 2>/dev/null || true
        log_info "All deployments stopped"
        ;;
    logs)
        docker logs -f intelfactor-station
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
