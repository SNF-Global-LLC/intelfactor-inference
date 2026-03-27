#!/bin/bash
# IntelFactor.ai — Station Setup Script
# Idempotent installer for Jetson station nodes.
#
# What it does:
#   1. Creates directory structure
#   2. Creates intelfactor system user
#   3. Copies config files
#   4. Installs Python package
#   5. Installs systemd service
#   6. Enables and starts the service
#
# Usage:
#   sudo ./deploy/station/setup.sh
#   sudo ./deploy/station/setup.sh --config /path/to/station.yaml
#   sudo ./deploy/station/setup.sh --no-start  # install only, don't start

set -euo pipefail

INSTALL_DIR="/opt/intelfactor"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG_SRC=""
NO_START=false
STATION_USER="intelfactor"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_SRC="$2"; shift 2 ;;
        --no-start) NO_START=true; shift ;;
        --install-dir) INSTALL_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: sudo $0 [--config station.yaml] [--no-start]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Check root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: Run with sudo"
    exit 1
fi

echo "========================================"
echo " IntelFactor Station Setup"
echo "========================================"
echo " Install dir: $INSTALL_DIR"
echo " Repo dir:    $REPO_DIR"
echo "========================================"

# ── 1. Create directories ──
echo ""
echo "[1/6] Creating directories..."
mkdir -p "$INSTALL_DIR"/{config,data,data/evidence,models,logs}
echo "  ✓ $INSTALL_DIR/{config,data,data/evidence,models,logs}"

# ── 2. Create system user ──
echo ""
echo "[2/6] Creating system user..."
if id "$STATION_USER" &>/dev/null; then
    echo "  ✓ User $STATION_USER already exists"
else
    useradd --system --shell /usr/sbin/nologin --home "$INSTALL_DIR" "$STATION_USER"
    echo "  ✓ Created user $STATION_USER"
fi

# Add to video group for camera access
usermod -aG video "$STATION_USER" 2>/dev/null || true
echo "  ✓ Added to video group"

# ── 3. Copy config files ──
echo ""
echo "[3/6] Copying configuration..."

# Station config
if [ -n "$CONFIG_SRC" ] && [ -f "$CONFIG_SRC" ]; then
    cp "$CONFIG_SRC" "$INSTALL_DIR/config/station.yaml"
    echo "  ✓ station.yaml from $CONFIG_SRC"
elif [ ! -f "$INSTALL_DIR/config/station.yaml" ]; then
    cp "$REPO_DIR/configs/station.yaml" "$INSTALL_DIR/config/station.yaml"
    echo "  ✓ station.yaml from defaults"
else
    echo "  ✓ station.yaml already exists (not overwriting)"
fi

# Taxonomy
if [ -f "$REPO_DIR/configs/wiko_taxonomy.yaml" ]; then
    cp "$REPO_DIR/configs/wiko_taxonomy.yaml" "$INSTALL_DIR/config/wiko_taxonomy.yaml"
    echo "  ✓ wiko_taxonomy.yaml"
fi

# ── 4. Install Python package ──
echo ""
echo "[4/6] Installing Python package..."
cd "$REPO_DIR"

# Detect if on Jetson
if [ -f /etc/nv_tegra_release ] || dpkg -l nvidia-jetpack &>/dev/null 2>&1; then
    EXTRAS="jetson"
    echo "  Detected: Jetson (using [jetson] extras)"
else
    EXTRAS="server"
    echo "  Detected: Server (using [server] extras)"
fi

pip3 install -e ".[$EXTRAS]" --break-system-packages 2>&1 | tail -5
echo "  ✓ Package installed"

# ── 5. Install systemd service ──
echo ""
echo "[5/6] Installing systemd service..."
SERVICE_SRC="$REPO_DIR/deploy/systemd/intelfactor-station.service"
SERVICE_DST="/etc/systemd/system/intelfactor-station.service"

if [ -f "$SERVICE_SRC" ]; then
    # Update paths in service file
    sed \
        -e "s|/opt/intelfactor|$INSTALL_DIR|g" \
        -e "s|User=intelfactor|User=$STATION_USER|g" \
        "$SERVICE_SRC" > "$SERVICE_DST"

    systemctl daemon-reload
    systemctl enable intelfactor-station.service
    echo "  ✓ Service installed and enabled"
else
    echo "  ⚠ Service file not found: $SERVICE_SRC"
fi

# ── 6. Set permissions and start ──
echo ""
echo "[6/6] Setting permissions..."
chown -R "$STATION_USER:$STATION_USER" "$INSTALL_DIR"
chmod -R 750 "$INSTALL_DIR"
chmod -R 700 "$INSTALL_DIR/data"
echo "  ✓ Ownership: $STATION_USER"

if [ "$NO_START" = false ]; then
    echo ""
    echo "Starting station..."
    systemctl start intelfactor-station.service
    sleep 2

    if systemctl is-active --quiet intelfactor-station.service; then
        echo "  ✓ Station is running"
        echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8080/"
        echo "  Health:    http://$(hostname -I | awk '{print $1}'):8080/health"
        echo "  Logs:      journalctl -u intelfactor-station -f"
    else
        echo "  ✗ Station failed to start. Check logs:"
        echo "    journalctl -u intelfactor-station --no-pager -n 20"
    fi
fi

echo ""
echo "========================================"
echo " Setup complete"
echo "========================================"
echo " Config:    $INSTALL_DIR/config/station.yaml"
echo " Data:      $INSTALL_DIR/data/"
echo " Models:    $INSTALL_DIR/models/"
echo " Service:   systemctl status intelfactor-station"
echo " Doctor:    intelfactor-station --doctor --config $INSTALL_DIR/config/station.yaml"
echo "========================================"
