# IntelFactor Inference — Task Tracker

> Updated per-session. Each task gets a checkbox, status, and review notes.

---

## Current Sprint

- [x] Workflow orchestration setup (tasks/, lessons.md, todo.md)

## Backlog

_Populated from CLAUDE.md Known Gaps and user requests._

- [ ] OTA model update mechanism (manifest.json pull + SHA256 verify)
- [ ] Camera ingest hardware validation (USB + RTSP on live Jetson)
- [ ] Cloud storage backend implementation (DynamoDB, S3)
- [ ] Sync agent heartbeat wiring (`/api/sync/heartbeat`)
- [ ] Move Wiko taxonomy from hardcoded `vision_trt.py` to `configs/` YAML

---

## Review Notes

_Added after each task is marked complete._

### Workflow Orchestration Setup
- **What**: Created `tasks/todo.md` and `tasks/lessons.md` for structured task management.
- **Verification**: Files created, committed, pushed.
