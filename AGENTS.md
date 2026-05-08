# AGENTS.md

## Cursor Cloud specific instructions

### Services Overview

This is an edge-first AI platform for manufacturing quality inspection. The main development service is the **Flask REST API** (`packages/inference/api_v2.py`).

### Running the Dev Server (API-only, no hardware)

The `make dev` target expects TensorRT models and hardware that won't be present in cloud VMs. Instead, start the API standalone:

```bash
mkdir -p /workspace/data/evidence
DB_PATH=/workspace/data/dev.db \
STORAGE_MODE=local \
SQLITE_DB_PATH=/workspace/data/dev.db \
EVIDENCE_DIR=/workspace/data/evidence \
STATION_ID=station_dev \
python3 -c "
import os
os.makedirs('/workspace/data/evidence', exist_ok=True)
from packages.inference.api_v2 import create_app
app = create_app()
app.run(host='0.0.0.0', port=8080, debug=False)
"
```

This starts the Flask API on port 8080 with local SQLite storage and no camera/vision/language model dependencies.

### Key Commands

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev]" flask` |
| Lint | `python3 -m ruff check packages/ tests/` |
| All tests | `python3 -m pytest tests/ -v` |
| API tests only | `python3 -m pytest tests/test_api_v2.py -v` |
| Sensor tests | `python3 -m pytest tests/test_sensor_service.py tests/test_maintenance_iq.py -v` |

### Gotchas

- **PATH**: After `pip install`, executables land in `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` is set.
- **DB_PATH vs SQLITE_DB_PATH**: The API reads `DB_PATH` for production metrics and `SQLITE_DB_PATH` for the event store. Both must be set to the same path for local dev.
- **`make dev` won't work in cloud**: It attempts full station startup including TensorRT model loading. Use the API-only snippet above.
- **Pre-existing test failures**: ~25 test failures and 7 errors in `test_maintenance_api.py` and `test_taxonomy.py` are pre-existing codebase issues, not environment problems. 198+ tests pass.
- **No external services needed**: All storage is SQLite (stdlib). No MQTT broker, Docker, or GPU required for dev/test.
