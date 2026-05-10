# Jetson Operator Console Validation

This checklist validates the local `/inspect` operator console on a Jetson station with FLIR camera input. It is not a cloud-sync test. The edge station remains authoritative, SQLite remains the local source of truth, and the console must continue to work without internet access.

## Environment Setup

- Jetson is powered from its factory power supply and has stable cooling.
- FLIR camera is physically mounted, focused, and connected over the expected interface.
- TensorRT engine and station config are already installed on the Jetson.
- Local data directories exist and are writable:
  - `/opt/intelfactor/data`
  - `/opt/intelfactor/data/evidence`
- `EDGE_API_KEY` or `STATION_API_KEY` is set for the local station API.
- `STORAGE_MODE=local`.
- `DB_PATH` and `SQLITE_DB_PATH` point to the same SQLite file.
- No cloud credentials are required for this validation.

## Launch Command

For API-only validation without the full supervisor, run:

```bash
mkdir -p /opt/intelfactor/data/evidence
EDGE_API_KEY=<local-edge-key> \
STATION_API_KEY=<local-edge-key> \
DB_PATH=/opt/intelfactor/data/operator-console.db \
SQLITE_DB_PATH=/opt/intelfactor/data/operator-console.db \
STORAGE_MODE=local \
EVIDENCE_DIR=/opt/intelfactor/data/evidence \
STATION_ID=<station-id> \
python3 -c "from packages.inference.api_v2 import create_app; app=create_app(); app.run(host='0.0.0.0', port=8080, debug=False)"
```

Open:

```text
http://<jetson-ip>:8080/inspect
```

Enter the local edge API key in the console before loading queue, history, or status data.

## Optional Local Seed

Use seeded records before camera validation when you need to verify UI behavior only:

```bash
python3 scripts/seed_operator_console.py \
  --db-path /opt/intelfactor/data/operator-console.db \
  --evidence-dir /opt/intelfactor/data/evidence \
  --station-id <station-id>
```

Refresh `/inspect`. The console should show PASS, REVIEW, and DEFECT local inspection rows, with the REVIEW row visible in Review Queue.

## Camera And Feed Validation

- Open Live Inspection.
- Confirm the latest captured frame or evidence image appears after an inspection.
- Confirm original and annotated image tabs switch without broken images.
- Disconnect and reconnect the FLIR camera once.
- Confirm the status panel changes camera state or the station logs a camera error.

Pass criteria:
- The operator can see the part or latest evidence image.
- Broken camera state is visible and does not crash the console.

Fail criteria:
- Live inspection cannot produce a local evidence frame.
- The console displays stale camera status as connected after the camera is disconnected.

## TensorRT And Model Status Validation

- Open Local System Status.
- Confirm Model loaded shows the expected TensorRT/model bundle version.
- Run one inspection and confirm verdict, defect class, confidence, and timing values update.

Pass criteria:
- Model status is not `unknown`.
- Inference timing is visible after inspection.
- No cloud or remote API is required to produce a verdict.

Fail criteria:
- The model status is `unknown` when the real station runtime is active.
- Inspection requires network access.

## SQLite Persistence Validation

- Run or seed at least three inspections.
- Restart the API process.
- Reopen `/inspect`, enter the edge API key, and open Inspection History.

Pass criteria:
- Previous inspections remain visible after restart.
- Review Queue still shows unresolved REVIEW rows.

Fail criteria:
- History is empty after restart despite using the same DB path.
- REVIEW rows disappear without operator action.

## Evidence Image Path Validation

- For a real inspection, open Inspection History and select the row.
- Confirm original and annotated local evidence paths are shown.
- Confirm image URLs are local authenticated `/api/inspections/...jpg` routes.
- Confirm no S3, CloudFront, or external URL appears in the operator console.

Pass criteria:
- Evidence files exist under `EVIDENCE_DIR`.
- Image previews load from local API routes.

Fail criteria:
- Evidence points to a remote object URL.
- Evidence is missing from disk immediately after a successful inspection.

## Review Queue Validation

- Create or seed at least one REVIEW inspection.
- Open Review Queue.
- Confirm each row shows thumbnail area, timestamp, station, batch/product, defect class, confidence, and both actions.
- Press `Confirm defect` on one REVIEW row.
- Seed or create another REVIEW row and press `Override to pass`.

Pass criteria:
- Both actions record successfully.
- The row leaves the queue after the action is recorded or no longer requires a decision.
- The action trail is visible in Inspection History.

Fail criteria:
- The console uses approve/reject wording.
- The action cannot be recorded while the local API is healthy.

## Touch-Screen Ergonomics Checks

- Test on the actual operator display at factory brightness.
- Confirm the verdict can be read from one meter away.
- Confirm `Confirm defect`, `Override to pass`, `Run inspection`, and nav buttons are usable with a gloved finger.
- Confirm filters do not overlap on the display resolution.
- Confirm the operator can understand PASS / REVIEW / DEFECT state in under two seconds.

Pass criteria:
- No overlapping controls or clipped text.
- Primary actions can be tapped reliably.
- Operator has only the choices needed for the inspection state.

Fail criteria:
- Action buttons are too small for touch use.
- Verdict, confidence, or defect class is hard to scan.

## Shutdown And Restart Recovery Checks

- Run one inspection and confirm it appears in history.
- Stop the API process.
- Start the API process with the same `DB_PATH`, `SQLITE_DB_PATH`, and `EVIDENCE_DIR`.
- Reopen `/inspect`.
- Confirm history, evidence paths, review queue count, and status return.

Pass criteria:
- Local inspection history survives restart.
- Evidence files remain on disk.
- The console recovers after the API process restarts.

Fail criteria:
- Restart loses local SQLite rows.
- Restart deletes or moves evidence unexpectedly.

## Final Pass Criteria

- `/inspect` works on the Jetson without internet.
- FLIR capture produces evidence locally.
- TensorRT/model status is visible and correct.
- SQLite rows persist across restart.
- Evidence remains on local disk and loads in the console.
- REVIEW queue supports `confirm_defect` and `override_to_pass`.
- Touch targets are usable on the factory display.
- No cloud sync, Kafka, training, billing, or remote dashboard dependency is introduced.

## Final Fail Criteria

- The console cannot run with cloud/network disabled.
- Operators cannot identify inspection state in under two seconds.
- Evidence is remote-only or missing locally.
- REVIEW actions fail on a healthy local API.
- Station restart loses local inspection state.
