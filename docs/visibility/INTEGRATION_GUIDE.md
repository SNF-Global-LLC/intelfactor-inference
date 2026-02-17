"""
IntelFactor Station API — Visibility Integration Patch
========================================================
This file shows the EXACT changes to make in your existing station app.

Your current station runs at /opt/intelfactor/packages/inference/
with a Flask app on port 9000. The bridge POSTs detections to /api/events.

These are the minimal changes to add production metrics.
"""

# =====================================================================
# STEP 1: Add to your existing station app.py (near the top imports)
# =====================================================================
"""
# Add these imports to your station app.py:

from visibility.metrics_api import metrics_bp, init_metrics

# After app = Flask(__name__), add:
metrics = init_metrics(app, db_path="/opt/intelfactor/data/local.db",
                       station_id="SNF-Vision-1")
app.register_blueprint(metrics_bp)
"""

# =====================================================================
# STEP 2: Add metrics.on_event() to your existing /api/events handler
# =====================================================================
"""
Your existing event handler probably looks something like this:

@app.route('/api/events', methods=['POST'])
def ingest_event():
    data = request.get_json()
    inspection_id = store_event(data)  # your existing SQLite write
    trigger_rca(data)                  # your existing RCA trigger
    return jsonify({"inspection_id": inspection_id}), 201

Add ONE line after storing the event:

@app.route('/api/events', methods=['POST'])
def ingest_event():
    data = request.get_json()
    inspection_id = store_event(data)
    trigger_rca(data)

    # >>> ADD THIS LINE <<<
    metrics.on_event(data)

    return jsonify({"inspection_id": inspection_id}), 201

That's it. Every detection now feeds the production counter,
cycle time tracker, and utilization tracker automatically.
"""

# =====================================================================
# STEP 3: Copy the visibility package to the Jetson
# =====================================================================
"""
scp -r packages/visibility/ jetson:/opt/intelfactor/packages/visibility/

Or via Tailscale:
scp -r packages/visibility/ 100.90.67.57:/opt/intelfactor/packages/visibility/

Directory structure after:
/opt/intelfactor/
├── packages/
│   ├── inference/          # existing - camera, ingest, station API
│   └── visibility/         # NEW - production metrics
│       ├── __init__.py
│       ├── production_metrics.py
│       └── metrics_api.py
├── data/
│   └── local.db            # existing - gets new tables auto-created
├── config/
│   └── station.yaml        # existing - no changes needed
└── models/                 # existing - no changes needed
"""

# =====================================================================
# STEP 4: Restart the station service
# =====================================================================
"""
sudo systemctl restart intelfactor-station.service

Verify:
curl http://localhost:9000/api/metrics/health
curl http://localhost:9000/api/metrics/live
curl http://localhost:9000/api/metrics/throughput?hours=1
"""

# =====================================================================
# STEP 5: Verify with a manual test
# =====================================================================
"""
# Simulate a detection event (same format the bridge sends):
curl -X POST http://localhost:9000/api/events \\
  -H "Content-Type: application/json" \\
  -d '{
    "inspection_id": "test-001",
    "timestamp": "2026-02-16T14:23:45.123",
    "class": "knife",
    "confidence": 0.92,
    "verdict": "PASS",
    "station_id": "SNF-Vision-1"
  }'

# Wait 2 seconds, send another:
curl -X POST http://localhost:9000/api/events \\
  -H "Content-Type: application/json" \\
  -d '{
    "inspection_id": "test-002",
    "timestamp": "2026-02-16T14:23:47.456",
    "class": "knife",
    "confidence": 0.89,
    "verdict": "PASS",
    "station_id": "SNF-Vision-1"
  }'

# Check metrics:
curl http://localhost:9000/api/metrics/live
# Should show: current_hour_units: 2, avg_cycle_seconds: ~2.0

curl http://localhost:9000/api/metrics/throughput?hours=1
# Should show: hourly_breakdown with 2 units
"""
