-- IntelFactor.ai — Hub PostgreSQL Schema
-- Mirrors station SQLite tables for cross-line analytics.

CREATE TABLE IF NOT EXISTS defect_events (
    event_id        TEXT PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    station_id      TEXT NOT NULL,
    defect_type     TEXT NOT NULL,
    severity        REAL NOT NULL,
    confidence      REAL NOT NULL,
    shift           TEXT DEFAULT '',
    sku             TEXT DEFAULT '',
    sop_criterion   TEXT DEFAULT '',
    model_version   TEXT DEFAULT '',
    frame_ref       TEXT DEFAULT '',
    synced_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hub_events_station_ts ON defect_events(station_id, timestamp);
CREATE INDEX idx_hub_events_defect_type ON defect_events(defect_type, timestamp);

CREATE TABLE IF NOT EXISTS causal_triples (
    triple_id           TEXT PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL,
    station_id          TEXT NOT NULL,
    defect_event_id     TEXT NOT NULL,
    defect_type         TEXT NOT NULL,
    defect_severity     REAL DEFAULT 0.0,
    cause_parameter     TEXT DEFAULT '',
    cause_value         REAL DEFAULT 0.0,
    cause_target        REAL DEFAULT 0.0,
    cause_drift_pct     REAL DEFAULT 0.0,
    cause_confidence    REAL DEFAULT 0.0,
    cause_explanation_zh TEXT DEFAULT '',
    cause_explanation_en TEXT DEFAULT '',
    recommendation_id   TEXT DEFAULT '',
    operator_action     TEXT DEFAULT 'pending',
    operator_id         TEXT DEFAULT '',
    outcome_measured    JSONB DEFAULT '{}',
    status              TEXT DEFAULT 'pending',
    synced_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hub_triples_station ON causal_triples(station_id, timestamp);
CREATE INDEX idx_hub_triples_status ON causal_triples(status);
CREATE INDEX idx_hub_triples_defect ON causal_triples(defect_type);

CREATE TABLE IF NOT EXISTS anomaly_alerts (
    alert_id        TEXT PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    station_id      TEXT NOT NULL,
    defect_type     TEXT NOT NULL,
    current_rate    REAL NOT NULL,
    baseline_rate   REAL NOT NULL,
    z_score         REAL NOT NULL,
    window_hours    REAL NOT NULL,
    acknowledged    INTEGER DEFAULT 0,
    synced_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Cross-line analytics views

CREATE OR REPLACE VIEW v_defect_rate_by_station AS
SELECT
    station_id,
    defect_type,
    DATE_TRUNC('hour', timestamp) AS hour,
    COUNT(*) AS event_count
FROM defect_events
GROUP BY station_id, defect_type, DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

CREATE OR REPLACE VIEW v_triple_acceptance AS
SELECT
    station_id,
    defect_type,
    status,
    COUNT(*) AS count,
    AVG(cause_confidence) AS avg_confidence
FROM causal_triples
GROUP BY station_id, defect_type, status;

CREATE OR REPLACE VIEW v_cross_line_drift AS
SELECT
    station_id,
    cause_parameter,
    AVG(cause_drift_pct) AS avg_drift_pct,
    MAX(cause_drift_pct) AS max_drift_pct,
    COUNT(*) AS alert_count
FROM causal_triples
WHERE cause_parameter != '' AND timestamp > NOW() - INTERVAL '7 days'
GROUP BY station_id, cause_parameter
ORDER BY avg_drift_pct DESC;
