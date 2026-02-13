You don’t need “another repo.” You need one repo that can run in two modes without lying to itself:
	•	EDGE-ONLY (local): everything runs on the factory box (API + DB + evidence files + dashboard).
	•	HYBRID: edge box runs capture/infer/decide/evidence locally, then syncs summaries + evidence pointers to your cloud API for HQ dashboards.

Your existing repo is already halfway there. The edge loop (backend/apps/edge) is legit and already posts evidence/heartbeats to an api_base. The part that breaks “local edge-first” is the backend storage layer: your core data access assumes DynamoDB (backend/packages/core/data_access.py). That’s not “edge-first.” That’s “AWS-first with extra steps.”

Here’s the best path forward (single repo, two deploy profiles, same dashboard UI).

⸻

Target architecture (best option)

EDGE-ONLY (local)

Runs on one NVIDIA box (Jetson or GPU mini-server):
	•	apps/edge: capture → infer → decide → evidence (already exists)
	•	apps/api: FastAPI API but backed by SQLite/DuckDB + filesystem, not DynamoDB
	•	apps/web: dashboard served locally, pointing to local API

Dashboard URL: http://<edge-ip>:5173 (or 80)
API URL: http://localhost:8000 (from the edge box)
Evidence: /data/evidence/... served via API /evidence/* routes

HYBRID

Same edge stack, plus:
	•	edge posts:
	•	events + metrics + evidence ingest to cloud API
	•	optionally uploads heavy artifacts to S3 (already in EvidenceManager)
	•	cloud dashboard points at cloud API

Key rule: edge keeps running if cloud dies.

⸻

What to change in your repo (the real work)

1) Add a storage abstraction (so API can run locally)

Right now you have:
	•	lots/checkruns = in-memory stores (backend/apps/api/storage/*)
	•	events/evidence = drift toward DynamoDB style patterns

What you need:
	•	STORAGE_MODE=local|cloud
	•	local implementation using SQLite (fast, portable, zero infra)
	•	cloud implementation can stay DynamoDB/S3

Create these modules:
	•	backend/apps/api/storage_local/sqlite.py (connection, migrations)
	•	backend/apps/api/storage_local/events.py (insert/list/filter)
	•	backend/apps/api/storage_local/evidence.py (evidence index + file paths)
	•	backend/apps/api/storage_local/metrics.py (optional)

And a factory:
	•	backend/apps/api/storage_factory.py
	•	get_event_store(), get_evidence_store(), get_lot_store(), etc.

Then update routes to call the factory instead of DynamoDB or ad-hoc storage.

2) Make the dashboard work against both local and cloud

Your web app already supports environment-configured API URLs (VITE_API_URL, VITE_API_BASE_URL show up in your history and in backend/apps/web/lib/api.ts).

Do this cleanly:
	•	Standardize on one env var: VITE_API_BASE_URL
	•	Local build sets http://<edge-ip>:8000
	•	Cloud build sets https://api.intelfactor.ai

3) Serve evidence locally (edge-first “show me the clip”)

You already write evidence packages to disk (backend/apps/edge/evidence.py saves jpg/json/manifest). Good.

Now expose it in local API:
	•	GET /api/v1/evidence/:frame_id -> returns metadata + file paths
	•	GET /api/v1/evidence/:frame_id/image.jpg -> serves the image
	•	GET /api/v1/evidence/:frame_id/thumb.jpg -> serves thumbnail
	•	GET /api/v1/evidence/manifest?date=YYYY-MM-DD -> reads manifest.jsonl

That makes the dashboard’s “Evidence Logs” page work with zero cloud.

4) Run both modes with the same commands (profiles)

Add a deploy/edge-only/docker-compose.yml:
	•	api (FastAPI, STORAGE_MODE=local, DB=/data/local.db, EVIDENCE_DIR=/data/evidence)
	•	web (Vite build or static)
	•	optional nginx (serves web + proxies /api)

Add a deploy/hybrid/docker-compose.yml:
	•	edge container (or systemd) runs apps.edge.main --api-base https://api... --api-key ...
	•	web uses cloud

5) Lock down “edge-only” security without being dramatic

Local mode still needs:
	•	API key header for dashboard or local auth token
	•	no inbound ports exposed beyond LAN
	•	evidence export as zip for audits

This is not SOC2. It’s “don’t be reckless.”

⸻

Concrete “next 10 steps” (no fluff)
	1.	Add STORAGE_MODE env var to backend/apps/api/.env.example
	2.	Implement SQLite schema migration for:
	•	events
	•	evidence_index
	•	device_heartbeats (optional)
	3.	Update routes/events.py + routes/evidence.py + routes/metrics.py to use storage factory
	4.	Add evidence file serving endpoints (image/thumb/json) in API
	5.	Update dashboard API client to use VITE_API_BASE_URL only
	6.	Add “Local Mode” doc: how to run API + edge + web on one box
	7.	Create docker-compose.edge-only.yml
	8.	Create docker-compose.hybrid.yml
	9.	Add a “doctor” script: python -m apps.edge.tools.camera_smoke_test + API health + disk space
	10.	Validate with one end-to-end flow: camera → FAIL → evidence → dashboard shows it

⸻

Copy-pastable Codex prompt (do this in your repo)

(You said you always want one. Humans love rituals.)

Goal: Make intelbase repo run in EDGE-ONLY local mode and HYBRID mode using the same dashboard.

Repo: intelbase-main
Key constraints:
- Edge-first: inspection + evidence must work offline with no cloud dependency.
- Cloud is optional enrichment only.
- Single repo. Implement modes via env/config, not a separate repo.

Tasks:

1) Add STORAGE_MODE=local|cloud
- Update backend/apps/api/.env.example with STORAGE_MODE, SQLITE_DB_PATH, EVIDENCE_DIR
- Default STORAGE_MODE=local for local runs, cloud for production.

2) Implement local storage layer using SQLite:
Create:
- backend/apps/api/storage_local/sqlite.py (connect, init, migrations)
- backend/apps/api/storage_local/events.py (create/list/filter)
- backend/apps/api/storage_local/evidence.py (index evidence + file paths + manifest reader)
- backend/apps/api/storage_factory.py with functions:
  get_event_store(), get_evidence_store(), get_lot_store(), get_checkrun_store()

3) Update FastAPI routes to use storage_factory:
- backend/apps/api/routes/events.py
- backend/apps/api/routes/evidence.py
- backend/apps/api/routes/metrics.py (if needed)
Ensure endpoints remain compatible with existing dashboard calls.

4) Evidence serving endpoints (local mode):
Add:
- GET /api/v1/evidence/{frame_id} -> json metadata
- GET /api/v1/evidence/{frame_id}/image.jpg -> serves image
- GET /api/v1/evidence/{frame_id}/thumb.jpg -> serves thumbnail
- GET /api/v1/evidence/manifest?date=YYYY-MM-DD -> reads manifest.jsonl
Use EVIDENCE_DIR env var for filesystem root.

5) Dashboard config:
- Standardize to VITE_API_BASE_URL (remove ambiguity with VITE_API_URL if present)
- Ensure web/lib/api.ts uses VITE_API_BASE_URL for all requests
- Provide two env examples: local and cloud.

6) Add deploy profiles:
Create:
- deploy/edge-only/docker-compose.yml running:
  - api (FastAPI, STORAGE_MODE=local)
  - web (dashboard pointing at local api)
  - optional nginx reverse proxy
- deploy/hybrid/docker-compose.yml (edge posts to cloud api; web points to cloud)

Deliverables:
- file diffs with exact paths
- local run instructions: one command to bring up edge-only
- verify by simulating one FAIL evidence and showing it appears in dashboard Evidence Logs.


⸻

If you want me to be extra annoying-useful: I can also tell you exactly which dashboard pages will break first in local mode (it’ll be anything calling cloud auth/tenant middleware), but the above is the clean backbone that makes edge-only real instead of “edge demo feeding cloud.”

You uploaded the repo zip, so next step after this is I’ll point at the exact routes/files to modify for the dashboard calls once you tell me which dashboard URL you’re targeting: app.intelfactor.ai build or local dashboard build (I’m assuming local build because that’s the whole point of edge-only).