# Edge-to-Cloud Inspection Event Contract

**Version:** 1.0  
**Date:** 2026-03-25  
**Status:** FROZEN — Backend implementation in progress

This document defines the contract between the Jetson edge station (producer) and the IntelFactor.ai cloud backend (consumer) for inspection event synchronization.

---

## Overview

The edge station operates **local-first**: inspections are captured, inferred, and verdicted entirely on-device without network dependency. Events are persisted to SQLite with `sync_status=pending`. A background sync worker periodically uploads pending events to the cloud.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Jetson Edge   │────▶│  IntelFactor.ai │────▶│   Review UI     │
│  (Producer)     │     │   (Consumer)    │     │  (Dashboard)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Sync Flow

1. **Edge:** Run inspection → Persist to SQLite (`sync_status=pending`)
2. **Edge:** Sync worker queries pending events
3. **Edge → Cloud:** `POST /api/v1/edge/inspections/upload-urls` (get presigned S3 URLs)
4. **Edge → S3:** `PUT` to presigned URLs (original.jpg, annotated.jpg)
5. **Edge → Cloud:** `POST /api/v1/edge/inspections` (metadata + S3 URLs)
6. **Cloud:** Store in DynamoDB, return success
7. **Edge:** Update `sync_status=synced`

---

## Required Cloud Endpoints

### 1. `POST /api/v1/edge/inspections/upload-urls`

Request presigned S3 PUT URLs for evidence files.

**Request:**
```json
{
  "inspection_id": "station_01-20260325-143022-a1b2c3",
  "station_id": "station_01",
  "workspace_id": "ws_abc123",
  "has_original": true,
  "has_annotated": true
}
```

**Response (200):**
```json
{
  "original_url": "https://s3.amazonaws.com/bucket/evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/original.jpg?X-Amz-Algorithm=...",
  "annotated_url": "https://s3.amazonaws.com/bucket/evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/annotated.jpg?X-Amz-Algorithm=...",
  "original_key": "evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/original.jpg",
  "annotated_key": "evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/annotated.jpg"
}
```

**S3 Key Pattern:**
```
evidence/{workspace_id}/manual-qc/{yyyy}/{mm}/{dd}/{inspection_id}/{asset_name}
```

---

### 2. `POST /api/v1/edge/inspections`

Ingest inspection metadata after assets are uploaded.

**Request:**
```json
{
  "inspection_id": "station_01-20260325-143022-a1b2c3",
  "timestamp": "2026-03-25T14:30:22.123456+00:00",
  "station_id": "station_01",
  "workspace_id": "ws_abc123",
  "product_id": "knife-8inch-damascus",
  "operator_id": "op_john_doe",
  "decision": "FAIL",
  "confidence": 0.92,
  "detections": [
    {
      "class": "scratch",
      "confidence": 0.94,
      "severity": 0.75,
      "threshold_used": 0.5,
      "bbox": {"x": 120.5, "y": 200.0, "width": 45.0, "height": 30.0}
    }
  ],
  "num_detections": 1,
  "image_original_url": "https://s3.amazonaws.com/bucket/evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/original.jpg",
  "image_annotated_url": "https://s3.amazonaws.com/bucket/evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/annotated.jpg",
  "model_version": "yolov8n-v1.2.3",
  "model_name": "yolov8n_cutlery_fp16",
  "timing": {
    "capture_ms": 45.2,
    "inference_ms": 23.1,
    "total_ms": 68.3
  },
  "accepted": null,
  "rejection_reason": "",
  "notes": ""
}
```

**Response (201):**
```json
{
  "status": "created",
  "inspection_id": "station_01-20260325-143022-a1b2c3",
  "synced_at": "2026-03-25T14:30:25.456789+00:00"
}
```

---

### 3. `POST /api/v1/edge/inspections/{id}/feedback`

Record operator feedback on an inspection (accept/reject/modify verdict).

**Request:**
```json
{
  "action": "accepted",
  "operator_id": "op_supervisor_1",
  "reason": "",
  "notes": "False positive — reflection, not scratch"
}
```

**Response (200):**
```json
{
  "status": "recorded",
  "inspection_id": "station_01-20260325-143022-a1b2c3",
  "action": "accepted"
}
```

---

### 4. `GET /api/v1/edge/inspections`

List inspection events with filtering.

**Query Parameters:**
- `workspace_id` (required) — Filter by workspace
- `station_id` — Filter by station
- `decision` — Filter by decision: `PASS`, `FAIL`, `REVIEW`
- `start_date` — ISO date (inclusive)
- `end_date` — ISO date (inclusive)
- `operator_id` — Filter by operator
- `product_id` — Filter by product
- `limit` — Max results (default: 50, max: 100)
- `offset` — Pagination offset

**Response (200):**
```json
{
  "inspections": [...],
  "count": 50,
  "total": 1247,
  "offset": 0,
  "limit": 50
}
```

---

### 5. `GET /api/v1/edge/inspections/{id}`

Get a single inspection event by ID.

**Response (200):**
```json
{
  "inspection_id": "station_01-20260325-143022-a1b2c3",
  "timestamp": "2026-03-25T14:30:22.123456+00:00",
  "station_id": "station_01",
  "workspace_id": "ws_abc123",
  "product_id": "knife-8inch-damascus",
  "operator_id": "op_john_doe",
  "decision": "FAIL",
  "confidence": 0.92,
  "detections": [...],
  "num_detections": 1,
  "image_original_url": "...",
  "image_annotated_url": "...",
  "model_version": "yolov8n-v1.2.3",
  "model_name": "yolov8n_cutlery_fp16",
  "timing": {...},
  "accepted": true,
  "rejection_reason": "",
  "notes": "False positive — reflection, not scratch",
  "synced_at": "2026-03-25T14:30:25.456789+00:00",
  "feedback_at": "2026-03-25T14:35:10.123456+00:00",
  "feedback_operator_id": "op_supervisor_1"
}
```

---

## Data Model

### DynamoDB Schema

**Table:** `inspection_events`

| Field | Type | Description |
|-------|------|-------------|
| `PK` | String | `WORKSPACE#{workspace_id}` |
| `SK` | String | `INSPECTION#{timestamp}#{inspection_id}` |
| `inspection_id` | String | Unique ID from edge |
| `station_id` | String | Station that produced the inspection |
| `workspace_id` | String | Workspace/tenant ID |
| `product_id` | String | Product being inspected |
| `operator_id` | String | Operator who triggered inspection |
| `decision` | String | `PASS`, `FAIL`, `REVIEW` |
| `confidence` | Number | 0.0–1.0 aggregate confidence |
| `detections` | List | Nested detection objects |
| `severity_counts` | Map | `{critical: 0, major: 1, minor: 0}` |
| `image_original_url` | String | S3 URL to original image |
| `image_annotated_url` | String | S3 URL to annotated image |
| `report_url` | String | S3 URL to JSON report (if separate) |
| `model_version` | String | Model bundle version |
| `model_name` | String | Model bundle name |
| `timing` | Map | `{capture_ms, inference_ms, total_ms}` |
| `accepted` | Boolean | `true`/`false`/`null` (pending) |
| `rejection_reason` | String | Why operator rejected |
| `notes` | String | Operator notes |
| `created_at` | String | ISO timestamp from edge |
| `synced_at` | String | ISO timestamp when received |
| `feedback_at` | String | ISO timestamp of feedback |
| `feedback_operator_id` | String | Operator who gave feedback |
| `ttl` | Number | Optional: TTL for data retention |

**GSI1:** `station_id` (for station-level queries)  
**GSI2:** `product_id` (for product-level aggregation)  
**GSI3:** `operator_id` (for operator performance metrics)

---

## Enums

### Verdict (decision)
- `PASS` — No defects detected or all below thresholds
- `FAIL` — Defect(s) exceeded fail threshold
- `REVIEW` — Ambiguous confidence, requires human review

### OperatorAction (feedback action)
- `accepted` — Operator agrees with verdict
- `rejected` — Operator disagrees with verdict
- `modified` — Operator changed verdict

---

## Error Handling

**Edge behavior on cloud errors:**
- Network timeout: Retry with exponential backoff (max 5 retries)
- 4xx errors: Mark as FAILED, log error, alert operator if persistent
- 5xx errors: Retry with backoff, mark FAILED after max retries
- S3 upload failure: Mark FAILED, will retry on next sync cycle

**Cloud error responses:**
- `400` — Invalid payload (missing required fields)
- `401` — Unauthorized (invalid API key)
- `403` — Forbidden (workspace access denied)
- `404` — Inspection not found (for feedback endpoint)
- `409` — Duplicate inspection_id
- `429` — Rate limited
- `500` — Internal server error

---

## Security

- All endpoints require `Authorization: Bearer {api_key}` header
- API keys are workspace-scoped
- Presigned S3 URLs expire after 15 minutes
- S3 bucket should have CORS configured for PUT from edge IPs
- CloudFront or signed URLs recommended for image serving

---

## Implementation Checklist

### Phase A: Upload URLs
- [ ] `POST /api/v1/edge/inspections/upload-urls`
- [ ] S3 presigned URL generation
- [ ] Proper key naming with workspace/date prefix

### Phase B: Metadata Ingest
- [ ] `POST /api/v1/edge/inspections`
- [ ] DynamoDB persistence
- [ ] Payload validation
- [ ] Duplicate detection handling

### Phase C: Feedback
- [ ] `POST /api/v1/edge/inspections/{id}/feedback`
- [ ] Update existing record (not new table)
- [ ] Audit trail fields (feedback_at, feedback_operator_id)

### Phase D: Read Endpoints
- [ ] `GET /api/v1/edge/inspections`
- [ ] `GET /api/v1/edge/inspections/{id}`
- [ ] Filtering and pagination
- [ ] GSI queries for efficient listing

### Phase E: Frontend
- [ ] `/quality/manual-inspections` page (table view)
- [ ] `/quality/manual-inspections/[id]` page (detail view)
- [ ] Image viewer with original/annotated toggle

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-25 | Initial contract freeze |
