# IntelFactor Station Deployment Guide / 部署指南

## Hardware BOM / 硬件清单

| Item / 设备 | Spec / 规格 | Qty / 数量 |
|---|---|---|
| NVIDIA Jetson Orin Nano Super | 8GB, JetPack 6.x | 1 per station / 每工位1台 |
| Industrial Camera / 工业相机 | USB3 or GigE, 1080p, global shutter | 1 per station |
| Mounting Arm / 安装支架 | Adjustable, vibration-dampened | 1 per station |
| LED Ring Light / 环形灯 | 60W diffused, 5500K daylight | 1 per station |
| Ethernet Switch / 以太网交换机 | Gigabit, 8-port minimum | 1 per site |
| UPS / 不间断电源 | 600W, 15min runtime | 1 per site |
| 10" Tablet (optional) / 平板（可选） | Android or browser-based, for operator UI | 1 per station |

## Network Diagram / 网络拓扑

```
┌────────────────────────────────────────────────────────┐
│ Production Zone (OT) / 生产区                          │
│                                                        │
│  [Camera] ──USB3──► [Jetson Station 1] ──┐             │
│  [Camera] ──USB3──► [Jetson Station 2] ──┤             │
│                                          │             │
│                              ┌───────────┘             │
│                              ▼                         │
│                    [Gigabit Switch]                     │
│                              │                         │
└──────────────────────────────┼─────────────────────────┘
                               │ (optional — station works
                               │  without this link)
┌──────────────────────────────┼─────────────────────────┐
│ Site Data Zone / 数据区      │                         │
│                              ▼                         │
│                    [Site Hub Server]                    │
│                    Docker Compose:                      │
│                    - Postgres                           │
│                    - MinIO                              │
│                    - Prometheus                         │
│                    - Grafana (:3000)                    │
│                                                        │
│              Optional VPN ──► [Cloud Sync]              │
└────────────────────────────────────────────────────────┘
```

**Key Rule / 关键原则:** Station works fully offline. Hub is optional.
工位完全离线运行。集线器可选。

---

## Station Setup (Jetson Orin Nano) / 工位设置

### 1. Flash JetPack / 刷写系统

```bash
# Use NVIDIA SDK Manager on a host PC
# Download JetPack 6.x for Orin Nano Super
# Flash via USB-C recovery mode
```

### 2. Install IntelFactor / 安装 IntelFactor

```bash
# Clone repo
git clone https://github.com/intelfactor/intelfactor-inference.git
cd intelfactor-inference

# Install with Jetson extras
pip install -e ".[jetson]"

# Verify installation
intelfactor-station --help
```

### 3. Build TensorRT Engine / 构建推理引擎

```bash
# Download YOLOv8n base model
./scripts/build_trt_engine.sh

# This exports yolov8n.pt → yolov8n_fp16.engine
# Expected: ~5 minutes on Orin Nano
```

### 4. Download Language Model / 下载语言模型

```bash
# Download Qwen-2.5-3B GGUF for bilingual RCA
./scripts/download_qwen.sh

# Expected: ~2GB download, stored in models/
```

### 5. Configure Station / 配置工位

Edit `configs/station.yaml`:

```yaml
station:
  id: "wiko_line_1"
  name: "阳江工厂一号线"

camera:
  source: "/dev/video0"     # or rtsp://192.168.1.100/stream
  protocol: "usb"           # usb, rtsp, gige, file
  width: 1920
  height: 1080

models:
  vision_engine: "models/yolov8n_fp16.engine"
  language_model: "models/qwen2.5-3b-instruct.Q4_K_M.gguf"

data:
  db_path: "data/station.db"
  evidence_dir: "data/evidence"
  max_evidence_gb: 50
```

### 6. Run Pre-Flight Check / 运行预检

```bash
intelfactor-station doctor --config configs/station.yaml
```

Expected output:
```
[✓] Disk space: 112GB free (>10GB required)
[✓] Data directory writable
[✓] Config file valid
[✓] TensorRT engine found: models/yolov8n_fp16.engine
[✓] Language model found: models/qwen2.5-3b-instruct.Q4_K_M.gguf
[✓] Camera accessible: /dev/video0
[✓] Taxonomy valid: 13 defect types loaded

READY — All 7 checks passed
```

### 7. Start Station / 启动工位

```bash
# Foreground (for testing)
intelfactor-station run --config configs/station.yaml

# Or install as system service (production)
sudo cp deploy/systemd/intelfactor-station.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now intelfactor-station
```

### 8. Access Operator Dashboard / 访问操作面板

Open browser on tablet or phone:
```
http://<station-ip>:8080/
```

Dashboard shows: 检测结果 (detection results), 异常警报 (anomaly alerts), RCA推荐 (recommendations), 操作反馈 (operator feedback).

---

## Hub Setup (Docker Compose) / 集线器设置

The hub aggregates data from multiple stations. **Stations do not require the hub to operate.**

### 1. Requirements / 要求

- Linux server with Docker + Docker Compose
- 4GB+ RAM, 100GB+ disk
- Network access to station IPs

### 2. Deploy / 部署

```bash
cd deploy/hub
docker compose up -d

# Verify services
docker compose ps
# Expected: postgres, minio, prometheus, grafana all running
```

### 3. Configure Station Sync / 配置同步

On each station:
```bash
# Add to crontab (syncs every 5 minutes)
crontab -e
# Add: */5 * * * * /path/to/deploy/station/sync_to_hub.sh
```

### 4. Access Grafana / 访问监控面板

```
http://<hub-ip>:3000
Default login: admin / admin
```

Four dashboards pre-configured:
- **Defect Rate by Station** / 各工位缺陷率
- **Cross-Line Drift** / 跨线漂移
- **Triple Acceptance** / 因果三元组接受率
- **System Health** / 系统健康

---

## Troubleshooting / 故障排除

| Symptom / 症状 | Cause / 原因 | Fix / 解决方案 |
|---|---|---|
| Camera not detected / 相机未检测到 | USB not connected or wrong device | Check `ls /dev/video*`, reconnect USB |
| TRT engine build fails / 引擎构建失败 | JetPack version mismatch | Ensure JetPack 6.x, re-run `build_trt_engine.sh` |
| High inference latency (>50ms) / 推理延迟高 | GPU thermal throttling | Check ventilation, `tegrastats` for temp |
| Dashboard shows "Pipeline not ready" / 面板显示未就绪 | Station still initializing | Wait 30s after start, check `systemctl status` |
| Sync to hub fails / 同步失败 | Hub unreachable | Check network, station continues offline |
| SQLite locked errors / 数据库锁定 | Multiple processes accessing DB | Only one station process per DB file |

---

## Emergency Recovery / 紧急恢复

If the station crashes during a production shift:

```bash
# Restart station service
sudo systemctl restart intelfactor-station

# If that fails, check logs
journalctl -u intelfactor-station -n 50

# Nuclear option: reset data and restart
# WARNING: This deletes local inspection history
sudo systemctl stop intelfactor-station
rm -rf data/station.db
sudo systemctl start intelfactor-station
```

**The station will resume inspecting within 30 seconds of restart.**
**工位将在重启后30秒内恢复检测。**
