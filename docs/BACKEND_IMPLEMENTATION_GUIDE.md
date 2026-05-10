# IntelFactor.ai Backend Implementation Guide

**For:** Cloud-side ingestion of manual QC inspection events  
**From:** Jetson Edge Station (intelfactor-inference)  
**Contract Version:** 1.0

---

## Context

The Jetson edge station now produces inspection events locally and syncs them asynchronously to the cloud. The edge implementation is complete and tested. This guide provides the copy-pasteable specification for implementing the cloud-side counterpart.

---

## Edge-Side Summary

The edge station:
1. Captures images, runs TRT inference, produces verdicts (PASS/FAIL/REVIEW)
2. Saves evidence locally (original.jpg, annotated.jpg, report.json)
3. Persists `InspectionEvent` to SQLite with `sync_status=pending`
4. Background `InspectionSyncWorker` polls and uploads to cloud

The sync worker expects these cloud endpoints:
- `POST /api/v1/edge/inspections/upload-urls` — Get presigned S3 URLs
- `POST /api/v1/edge/inspections` — Ingest metadata after upload
- `POST /api/v1/edge/inspections/{id}/feedback` — Record operator feedback

Plus read endpoints for the dashboard:
- `GET /api/v1/edge/inspections` — List with filters
- `GET /api/v1/edge/inspections/{id}` — Get single inspection

---

## Implementation Order

### Phase A: Upload URLs Endpoint

**Endpoint:** `POST /api/v1/edge/inspections/upload-urls`

**Purpose:** Return presigned S3 PUT URLs so the edge can upload images directly to S3 without streaming through your API.

**Request Schema:**
```python
class UploadUrlsRequest(BaseModel):
    inspection_id: str          # From edge: "{station_id}-{timestamp}-{uuid}"
    station_id: str             # Station identifier
    workspace_id: str           # Workspace/tenant ID
    has_original: bool          # Edge has original.jpg
    has_annotated: bool         # Edge has annotated.jpg
```

**Response Schema:**
```python
class UploadUrlsResponse(BaseModel):
    original_url: str | None    # Presigned PUT URL for original.jpg
    annotated_url: str | None   # Presigned PUT URL for annotated.jpg
    original_key: str | None    # S3 key for original (for DB storage)
    annotated_key: str | None   # S3 key for annotated (for DB storage)
```

**S3 Key Pattern:**
```python
def make_s3_key(workspace_id: str, inspection_id: str, asset: str) -> str:
    from datetime import datetime
    # inspection_id format: "{station_id}-{YYYYMMDD}-{HHMMSS}-{uuid}"
    parts = inspection_id.split("-")
    if len(parts) >= 2:
        date_str = parts[1]  # YYYYMMDD
        year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    else:
        now = datetime.utcnow()
        year, month, day = now.strftime("%Y/%m/%d").split("/")
    
    return f"evidence/{workspace_id}/manual-qc/{year}/{month}/{day}/{inspection_id}/{asset}"

# Examples:
# evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/original.jpg
# evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/annotated.jpg
```

**Implementation Notes:**
- Use `boto3.generate_presigned_url('put_object', ...)` with 15-minute expiry
- Set `ContentType='image/jpeg'` in the presigned URL params
- Return `original_key` and `annotated_key` so the edge can store private object keys instead of public URLs

---

### Phase B: Metadata Ingest Endpoint

**Endpoint:** `POST /api/v1/edge/inspections`

**Purpose:** Receive inspection metadata after images are uploaded to S3.

**Request Schema:**
```python
class DetectionPayload(BaseModel):
    class_: str = Field(alias="class")  # defect_type: "scratch", "dent", etc.
    confidence: float                   # 0.0–1.0
    severity: float                     # 0.0–1.0 (normalized)
    threshold_used: float               # Per-class threshold applied
    bbox: BoundingBox

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class TimingPayload(BaseModel):
    capture_ms: float
    inference_ms: float
    total_ms: float

class InspectionIngestRequest(BaseModel):
    inspection_id: str
    timestamp: str                      # ISO 8601 with timezone
    station_id: str
    workspace_id: str
    product_id: str = ""
    operator_id: str = ""
    decision: str                       # "PASS", "FAIL", "REVIEW"
    confidence: float
    detections: list[DetectionPayload]
    num_detections: int
    image_original_url: str  # Private object key, despite legacy field name
    image_annotated_url: str # Private object key, despite legacy field name
    model_version: str
    model_name: str
    timing: TimingPayload
    accepted: bool | None = None
    rejection_reason: str = ""
    notes: str = ""
```

**DynamoDB Item Structure:**
```python
{
    # Primary Keys
    "PK": "WORKSPACE#ws_abc123",
    "SK": "INSPECTION#2026-03-25T14:30:22.123456#station_01-20260325-143022-a1b2c3",
    
    # Attributes
    "inspection_id": "station_01-20260325-143022-a1b2c3",
    "timestamp": "2026-03-25T14:30:22.123456+00:00",
    "station_id": "station_01",
    "workspace_id": "ws_abc123",
    "product_id": "knife-8inch-damascus",
    "operator_id": "op_john_doe",
    "decision": "FAIL",
    "confidence": 0.92,
    "detections": [...],  # Store as-is (List<Map>)
    "num_detections": 1,
    "image_original_url": "evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/original.jpg",
    "image_annotated_url": "evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/annotated.jpg",
    "model_version": "yolov8n-v1.2.3",
    "model_name": "yolov8n_cutlery_fp16",
    "timing": {"capture_ms": 45.2, "inference_ms": 23.1, "total_ms": 68.3},
    "accepted": None,
    "rejection_reason": "",
    "notes": "",
    "created_at": "2026-03-25T14:30:22.123456+00:00",
    "synced_at": "2026-03-25T14:30:25.456789+00:00",
    
    # GSI Keys
    "GSI1PK": "STATION#station_01",
    "GSI1SK": "2026-03-25T14:30:22.123456",
}
```

**Response (201 Created):**
```json
{
  "status": "created",
  "inspection_id": "station_01-20260325-143022-a1b2c3",
  "synced_at": "2026-03-25T14:30:25.456789+00:00"
}
```

**Validation Rules:**
- `decision` must be one of: `PASS`, `FAIL`, `REVIEW`
- `timestamp` must be valid ISO 8601
- `confidence` must be 0.0–1.0
- `detections` length must match `num_detections`
- Treat duplicate `inspection_id` as idempotent when the workspace and payload are compatible; reject only conflicting duplicates

---

### Phase C: Feedback Endpoint

**Endpoint:** `POST /api/v1/edge/inspections/{inspection_id}/feedback`

**Purpose:** Record operator feedback (accept/reject) on an inspection.

**Request Schema:**
```python
class FeedbackRequest(BaseModel):
    action: str              # "accepted", "rejected", "modified"
    operator_id: str
    reason: str = ""         # Required if action == "rejected"
    notes: str = ""
```

**DynamoDB Update:**
```python
# Update existing item, don't create new table
table.update_item(
    Key={
        "PK": f"WORKSPACE#{workspace_id}",
        "SK": f"INSPECTION#{timestamp}#{inspection_id}"
    },
    UpdateExpression="SET accepted = :a, rejection_reason = :r, notes = :n, feedback_at = :f, feedback_operator_id = :o",
    ExpressionAttributeValues={
        ":a": action == "accepted",
        ":r": reason,
        ":n": notes,
        ":f": datetime.utcnow().isoformat(),
        ":o": operator_id,
    }
)
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

### Phase D: Read Endpoints

#### `GET /api/v1/edge/inspections`

**Query Parameters:**
```
workspace_id (required)
station_id (optional)
decision (optional): PASS | FAIL | REVIEW
start_date (optional): YYYY-MM-DD
end_date (optional): YYYY-MM-DD
operator_id (optional)
product_id (optional)
limit (default: 50, max: 100)
offset (default: 0)
```

**Implementation:**
- If `station_id` provided: Query GSI1 with `GSI1PK = STATION#{station_id}`
- Otherwise: Query main table with `PK = WORKSPACE#{workspace_id}`
- Apply filters client-side or use FilterExpression

**Response:**
```json
{
  "inspections": [...],
  "count": 50,
  "total": 1247,
  "offset": 0,
  "limit": 50
}
```

#### `GET /api/v1/edge/inspections/{inspection_id}`

**Implementation:**
- Use `PK = WORKSPACE#{workspace_id}` and `begins_with(SK, INSPECTION#)` with FilterExpression on `inspection_id`
- Or maintain a reverse GSI if you need frequent lookups by inspection_id alone

**Response:** Full inspection object (same as ingest payload + synced_at, feedback fields)

---

## FastAPI Router Template

```python
# app/routers/edge_inspections.py
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal
import boto3
from datetime import datetime

router = APIRouter(prefix="/api/v1/edge", tags=["edge-inspections"])

# --- Dependencies ---
async def get_current_workspace(request: Request) -> str:
    """Extract workspace_id from API key."""
    # Your existing auth logic here
    return "ws_abc123"

# --- Phase A: Upload URLs ---
@router.post("/inspections/upload-urls")
async def get_upload_urls(
    req: UploadUrlsRequest,
    workspace_id: str = Depends(get_current_workspace),
):
    """Generate presigned S3 URLs for edge evidence upload."""
    s3 = boto3.client("s3")
    bucket = "your-evidence-bucket"
    
    urls = {}
    keys = {}
    
    if req.has_original:
        key = make_s3_key(workspace_id, req.inspection_id, "original.jpg")
        urls["original_url"] = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": "image/jpeg",
            },
            ExpiresIn=900,  # 15 minutes
        )
        keys["original_key"] = key
    
    if req.has_annotated:
        key = make_s3_key(workspace_id, req.inspection_id, "annotated.jpg")
        urls["annotated_url"] = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": bucket,
                "Key": key,
                "ContentType": "image/jpeg",
            },
            ExpiresIn=900,
        )
        keys["annotated_key"] = key
    
    return {**urls, **keys}

# --- Phase B: Ingest ---
@router.post("/inspections", status_code=201)
async def ingest_inspection(
    req: InspectionIngestRequest,
    workspace_id: str = Depends(get_current_workspace),
):
    """Ingest inspection metadata from edge station."""
    # Validate workspace matches
    if req.workspace_id != workspace_id:
        raise HTTPException(403, "Workspace mismatch")
    
    # Check for duplicate
    existing = get_inspection(req.inspection_id)
    if existing:
        raise HTTPException(409, "Inspection already exists")
    
    # Build DynamoDB item
    item = build_dynamodb_item(req)
    
    # Store
    table.put_item(Item=item)
    
    return {
        "status": "created",
        "inspection_id": req.inspection_id,
        "synced_at": datetime.utcnow().isoformat(),
    }

# --- Phase C: Feedback ---
@router.post("/inspections/{inspection_id}/feedback")
async def record_feedback(
    inspection_id: str,
    req: FeedbackRequest,
    workspace_id: str = Depends(get_current_workspace),
):
    """Record operator feedback on an inspection."""
    # Find the inspection
    inspection = get_inspection(inspection_id)
    if not inspection:
        raise HTTPException(404, "Inspection not found")
    
    if inspection["workspace_id"] != workspace_id:
        raise HTTPException(403, "Access denied")
    
    # Update
    update_inspection_feedback(inspection_id, req)
    
    return {
        "status": "recorded",
        "inspection_id": inspection_id,
        "action": req.action,
    }

# --- Phase D: Read ---
@router.get("/inspections")
async def list_inspections(
    station_id: str | None = None,
    decision: Literal["PASS", "FAIL", "REVIEW"] | None = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = 0,
    workspace_id: str = Depends(get_current_workspace),
):
    """List inspection events."""
    # Query DynamoDB with filters
    inspections = query_inspections(
        workspace_id=workspace_id,
        station_id=station_id,
        decision=decision,
        limit=limit,
        offset=offset,
    )
    return {
        "inspections": inspections,
        "count": len(inspections),
        "offset": offset,
        "limit": limit,
    }

@router.get("/inspections/{inspection_id}")
async def get_inspection_detail(
    inspection_id: str,
    workspace_id: str = Depends(get_current_workspace),
):
    """Get a single inspection by ID."""
    inspection = get_inspection(inspection_id)
    if not inspection:
        raise HTTPException(404, "Inspection not found")
    if inspection["workspace_id"] != workspace_id:
        raise HTTPException(403, "Access denied")
    return inspection
```

---

## Testing Locally

```bash
# 1. Test upload URLs
curl -X POST http://localhost:8000/api/v1/edge/inspections/upload-urls \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inspection_id": "station_01-20260325-143022-a1b2c3",
    "station_id": "station_01",
    "workspace_id": "ws_abc123",
    "has_original": true,
    "has_annotated": true
  }'

# 2. Test ingest (after uploading to S3)
curl -X POST http://localhost:8000/api/v1/edge/inspections \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inspection_id": "station_01-20260325-143022-a1b2c3",
    "timestamp": "2026-03-25T14:30:22.123456+00:00",
    "station_id": "station_01",
    "workspace_id": "ws_abc123",
    "decision": "FAIL",
    "confidence": 0.92,
    "detections": [],
    "num_detections": 0,
    "image_original_url": "evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/original.jpg",
    "image_annotated_url": "evidence/ws_abc123/manual-qc/2026/03/25/station_01-20260325-143022-a1b2c3/annotated.jpg",
    "model_version": "yolov8n-v1.2.3",
    "model_name": "yolov8n_cutlery_fp16",
    "timing": {"capture_ms": 45.2, "inference_ms": 23.1, "total_ms": 68.3}
  }'

# 3. Test feedback
curl -X POST http://localhost:8000/api/v1/edge/inspections/station_01-20260325-143022-a1b2c3/feedback \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"action": "accepted", "operator_id": "op_1", "notes": "Looks correct"}'

# 4. Test list
curl "http://localhost:8000/api/v1/edge/inspections?limit=10" \
  -H "Authorization: Bearer $API_KEY"
```

---

## Files to Modify in IntelFactor.ai

1. **Create:** `app/routers/edge_inspections.py` — Main router
2. **Create:** `app/schemas/edge_inspections.py` — Pydantic schemas
3. **Create:** `app/services/inspection_service.py` — Business logic (optional)
4. **Modify:** `app/main.py` — Include new router
5. **Modify:** `app/core/config.py` — Add S3 bucket setting
6. **Create:** `tests/test_edge_inspections.py` — Unit tests

---

## Constraints (Read Carefully)

- **No new databases** — Use existing DynamoDB
- **No background jobs** — Cloud app is request/response only
- **No frontend changes** — Focus on backend API only
- **Reuse existing patterns** — Auth, logging, error handling
- **Minimal abstractions** — One inspection = one DynamoDB item
- **Production-safe** — Input validation, rate limiting, proper error codes

---

## Questions?

See `EDGE_CLOUD_CONTRACT.md` for the full protocol specification.
