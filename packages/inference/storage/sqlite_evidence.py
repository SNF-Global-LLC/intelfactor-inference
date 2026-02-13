"""
IntelFactor.ai — SQLite Evidence Store
Local storage for evidence index and file paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import EvidenceStore
from .sqlite_base import get_connection

logger = logging.getLogger(__name__)


class SQLiteEvidenceStore(EvidenceStore):
    """SQLite-backed evidence storage with filesystem file serving."""

    def __init__(self, db_path: str, evidence_dir: str):
        self.db_path = db_path
        self.evidence_dir = Path(evidence_dir)
        self._conn = get_connection(db_path)

    def index(self, event_id: str, metadata: dict[str, Any]) -> None:
        """Index evidence metadata for an event."""
        date_dir = metadata.get("date_dir", "")
        image_path = metadata.get("image_path", "")
        thumb_path = metadata.get("thumb_path", "")
        json_path = metadata.get("json_path", "")
        file_size = metadata.get("file_size_bytes", 0)

        # Store full metadata as JSON
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO evidence_index
                (event_id, date_dir, image_path, thumb_path, json_path,
                 file_size_bytes, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, date_dir, image_path, thumb_path, json_path, file_size, metadata_json),
        )
        self._conn.commit()

    def get_metadata(self, event_id: str) -> dict[str, Any] | None:
        """Get evidence metadata for an event."""
        row = self._conn.execute(
            "SELECT * FROM evidence_index WHERE event_id = ?", (event_id,)
        ).fetchone()

        if row is None:
            # Fall back to reading from filesystem
            return self._read_metadata_from_fs(event_id)

        d = dict(row)
        if d.get("metadata_json"):
            try:
                return json.loads(d["metadata_json"])
            except json.JSONDecodeError:
                pass
        return d

    def get_image_path(self, event_id: str) -> Path | None:
        """Get the image file path for an event."""
        row = self._conn.execute(
            "SELECT image_path, date_dir FROM evidence_index WHERE event_id = ?",
            (event_id,),
        ).fetchone()

        if row and row["image_path"]:
            path = self.evidence_dir / row["image_path"]
            if path.exists():
                return path

        # Fall back to filesystem search
        return self._find_file_on_fs(event_id, ".jpg")

    def get_thumb_path(self, event_id: str) -> Path | None:
        """Get the thumbnail file path for an event."""
        row = self._conn.execute(
            "SELECT thumb_path FROM evidence_index WHERE event_id = ?",
            (event_id,),
        ).fetchone()

        if row and row["thumb_path"]:
            path = self.evidence_dir / row["thumb_path"]
            if path.exists():
                return path

        # Fall back to main image if no thumb
        return self.get_image_path(event_id)

    def list_by_date(self, date: str) -> list[dict[str, Any]]:
        """List all evidence entries for a given date (YYYY-MM-DD)."""
        # First check database
        rows = self._conn.execute(
            "SELECT * FROM evidence_index WHERE date_dir = ? ORDER BY event_id",
            (date,),
        ).fetchall()

        if rows:
            results = []
            for row in rows:
                d = dict(row)
                if d.get("metadata_json"):
                    try:
                        d["metadata"] = json.loads(d["metadata_json"])
                    except json.JSONDecodeError:
                        d["metadata"] = {}
                results.append(d)
            return results

        # Fall back to reading manifest from filesystem
        return self._read_manifest_from_fs(date)

    def _find_file_on_fs(self, event_id: str, extension: str) -> Path | None:
        """Search filesystem for evidence file."""
        if not self.evidence_dir.exists():
            return None

        # Search date directories (newest first)
        for day_dir in sorted(self.evidence_dir.iterdir(), reverse=True):
            if day_dir.is_dir() and len(day_dir.name) == 10:  # YYYY-MM-DD
                path = day_dir / f"{event_id}{extension}"
                if path.exists():
                    return path
        return None

    def _read_metadata_from_fs(self, event_id: str) -> dict[str, Any] | None:
        """Read metadata JSON from filesystem."""
        json_path = self._find_file_on_fs(event_id, ".json")
        if json_path and json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        return None

    def _read_manifest_from_fs(self, date: str) -> list[dict[str, Any]]:
        """Read manifest.jsonl from filesystem for a date."""
        date_dir = self.evidence_dir / date
        manifest_path = date_dir / "manifest.jsonl"

        if not manifest_path.exists():
            # Fall back to listing all JSON files
            results = []
            if date_dir.exists():
                for json_file in date_dir.glob("*.json"):
                    if json_file.name != "manifest.jsonl":
                        try:
                            with open(json_file) as f:
                                results.append(json.load(f))
                        except (json.JSONDecodeError, OSError):
                            pass
            return results

        results = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return results
