# IntelFactor Production Visibility — Build Guide

## What This Is

The production visibility layer that sits on top of your existing edge inspection pipeline. No changes to DeepStream, the bridge, or the camera. Every COCO detection you're already making becomes a production count, a cycle time measurement, and a utilization data point.

**This is the product for Feb 20 and the first real value you deliver.**

---

## Architecture (What Goes Where)

```
┌─────────────────────────────────────────────────────────┐
│  JETSON (SNF-Vision-1)                                  │
│                                                         │
│  DeepStream ──► Bridge ──► Station API (port 9000)      │
│                              │                          │
│                              ├─► events table (existing)│
│                              │                          │
│                              └─► metrics.on_event()     │
│                                   │                     │
│                                   ├─► production_counts │
│                                   ├─► cycle_times       │
│                                   └─► station_util      │
│                                                         │
│  /api/metrics/live         ◄── Dashboard polls this     │
│  /api/metrics/throughput                                │
│  /api/metrics/cycle-time                                │
│  /api/metrics/utilization                               │
│  /api/metrics/shift-summary                             │
│                                                         │
│  batch_sync.py + metrics_sync.py (60s timer)            │
│       │                                                 │
└───────┼─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  AWS CLOUD                                              │
│                                                         │
│  DynamoDB: intelfactor-production-metrics                │
│  DynamoDB: intelfactor-shift-summaries                   │
│                                                         │
│  Lambda API: /api/v1/metrics/*                          │
│  IntelAgent: production_narrative tool (#8)              │
│                                                         │
│  Dashboard: app.intelfactor.ai/edge/production          │
│       └─► ProductionDashboard.tsx                       │
│           ├─► LiveTicker (polls /metrics/live)          │
│           ├─► ThroughputChart (hourly bars)             │
│           ├─► CycleTimeChart (trend + variance)         │
│           ├─► UtilizationGauge (donut)                  │
│           └─► ShiftComparison (table)                   │
└─────────────────────────────────────────────────────────┘
```

---

## Build Order (5 Days)

### Day 1: Edge Metrics Engine
**Goal:** Every detection produces a count, cycle time, and utilization update.

1. Copy `edge/packages/visibility/` to Jetson:
   ```bash
   scp -r edge/packages/visibility/ 100.90.67.57:/opt/intelfactor/packages/visibility/
   scp edge/migrations/001_production_metrics.sql 100.90.67.57:/opt/intelfactor/migrations/
   ```

2. Integrate with station API (see `INTEGRATION.py` for exact changes):
   - Add 2 import lines
   - Add 1 line to event handler: `metrics.on_event(data)`
   - Register blueprint: `app.register_blueprint(metrics_bp)`

3. Restart and verify:
   ```bash
   ssh 100.90.67.57
   sudo systemctl restart intelfactor-station.service
   
   # Tables auto-create on first init
   curl http://localhost:9000/api/metrics/health
   
   # Run DeepStream, let it detect a few objects
   curl http://localhost:9000/api/metrics/live
   curl http://localhost:9000/api/metrics/throughput?hours=1
   ```

**Done when:** You see unit counts incrementing in /api/metrics/live while the camera runs.

### Day 2: Cycle Time + Utilization
**Goal:** Cycle time statistics and active/idle tracking working.

1. Put 5-10 objects in front of the camera at varying intervals
2. Verify cycle times:
   ```bash
   curl http://localhost:9000/api/metrics/cycle-time?hours=1
   # Should show avg, min, max, stddev
   ```
3. Wait 2+ minutes with no objects → check idle detection:
   ```bash
   curl http://localhost:9000/api/metrics/utilization?hours=1
   # Should show utilization_pct < 100, idle_seconds > 0
   ```
4. Generate a shift summary:
   ```bash
   curl http://localhost:9000/api/metrics/shift-summary
   ```

**Done when:** Cycle times have non-zero stddev and utilization shows both active and idle periods.

### Day 3: Cloud Sync + API
**Goal:** Metrics visible from the cloud dashboard.

1. Create DynamoDB tables:
   ```bash
   # On your dev machine
   cd edge/scripts
   python -c "from metrics_sync import create_tables; create_tables()"
   ```

2. Deploy metrics sync to Jetson:
   ```bash
   scp edge/scripts/metrics_sync.py 100.90.67.57:/opt/intelfactor/scripts/
   ```

3. Add to existing batch_sync.py on Jetson:
   ```python
   from metrics_sync import sync_metrics
   # After existing event sync:
   sync_metrics()
   ```

4. Add cloud API router to Lambda:
   ```python
   # In your handler.py:
   from routers.metrics import router as metrics_router
   app.include_router(metrics_router)
   ```

5. Deploy Lambda (your existing process):
   ```bash
   # Build + deploy
   cd backend
   ./deploy.sh  # or manual zip deploy
   ```

6. Verify:
   ```bash
   curl https://api.intelfactor.ai/api/v1/metrics/throughput?station_id=SNF-Vision-1&hours=1
   ```

**Done when:** Cloud API returns real data from the Jetson.

### Day 4: Dashboard
**Goal:** Production visibility page live at app.intelfactor.ai/edge/production.

1. Add `ProductionDashboard.tsx` to your dashboard src:
   ```bash
   cp dashboard/src/pages/ProductionDashboard.tsx \
      ~/intelfactor-dashboard/src/pages/
   ```

2. Add route in your router:
   ```tsx
   import ProductionDashboard from './pages/ProductionDashboard';
   // In your Routes:
   <Route path="/edge/production" element={<ProductionDashboard />} />
   ```

3. Add nav link (in your existing sidebar/nav):
   ```tsx
   <NavLink to="/edge/production">Production</NavLink>
   ```

4. Build and deploy:
   ```bash
   npm run build
   aws s3 sync dist/ s3://your-dashboard-bucket/
   aws cloudfront create-invalidation --distribution-id XXXXX --paths "/*"
   ```

**Done when:** app.intelfactor.ai/edge/production shows live data with updating charts.

### Day 5: Agent Integration + Demo Polish
**Goal:** IntelAgent can answer "how is production running?" with real data.

1. Add production_narrative tool to your agent:
   - Copy `cloud/api/tools/production_narrative.py`
   - Add `PRODUCTION_NARRATIVE_TOOL` to your tool list
   - Register handler in your agent router

2. Test in /edge/ask:
   ```
   "How many units did we produce this hour?"
   "What was throughput on the last shift?"
   "Is cycle time stable today?"
   ```

3. Demo polish:
   - Set up a repeating demo: place knives on conveyor/desk at regular intervals
   - Dashboard should show counts climbing, cycle time stabilizing
   - Agent should narrate production state in natural language

---

## Feb 20 Confluent Share-Out Demo Script

1. **Open dashboard** at app.intelfactor.ai/edge/production
2. **Show live camera** feed (existing /edge/live page)
3. **Demonstrate detection:** Place knife in view → counter increments
4. **Show throughput chart** building in real time
5. **Show cycle time:** Place objects at regular intervals → chart stabilizes
6. **Show idle detection:** Remove all objects → status changes to Idle
7. **Open agent chat** at /edge/ask → "How is production running?"
8. **Show architecture slide:** Edge camera → Jetson YOLO → SQLite → DynamoDB → Dashboard

**The story:** "We deploy cameras on production lines. Within one hour, plant managers have real-time throughput, cycle time, and utilization data they've never had before. No MES integration. No IT project. Just cameras and edge compute. The same architecture scales to AI-powered defect detection as we train custom models on the image data we're already collecting."

---

## Files in This Package

```
intelfactor-visibility/
├── edge/
│   ├── migrations/
│   │   └── 001_production_metrics.sql    # SQLite schema
│   ├── packages/
│   │   └── visibility/
│   │       ├── __init__.py
│   │       ├── production_metrics.py      # Core metrics engine
│   │       └── metrics_api.py             # Flask blueprint
│   ├── scripts/
│   │   └── metrics_sync.py                # DynamoDB sync
│   └── INTEGRATION.py                     # Wiring guide
├── cloud/
│   └── api/
│       ├── routers/
│       │   └── metrics.py                 # FastAPI router
│       └── tools/
│           └── production_narrative.py     # Agent tool
├── dashboard/
│   └── src/
│       └── pages/
│           └── ProductionDashboard.tsx     # React page
└── BUILD_GUIDE.md                         # This file
```

---

## What This Does NOT Change

- DeepStream pipeline (run_prod.py) — untouched
- Bridge (bridge_to_station.py) — untouched
- Camera config — untouched
- Existing events table — untouched (read-only from metrics)
- Existing RCA pipeline — untouched
- Existing dashboard pages — untouched (new page added)
- Existing batch_sync.py — 2 lines added

## What Comes After (Not This Week)

- [ ] Custom YOLO26 defect model (needs Wiko labeled images)
- [ ] Kafka streaming (Confluent requirement, can demo without)
- [ ] Operator review UI (needs defect model first)
- [ ] Multi-station fleet view (needs second Jetson)
- [ ] Shift boundary auto-detection (currently uses fixed schedule)
- [ ] Bottleneck analysis across stations (needs multi-station)
