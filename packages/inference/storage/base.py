"""
IntelFactor.ai — Storage Base Classes
Abstract interfaces for storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any


class EventStore(ABC):
    """Abstract interface for defect event storage."""

    @abstractmethod
    def insert(self, event: dict[str, Any]) -> str:
        """Insert a defect event. Returns event_id."""
        ...

    @abstractmethod
    def get(self, event_id: str) -> dict[str, Any] | None:
        """Get a single event by ID."""
        ...

    @abstractmethod
    def list(
        self,
        limit: int = 50,
        verdict: str | None = None,
        station_id: str | None = None,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """List events with optional filters."""
        ...

    @abstractmethod
    def count(
        self,
        verdict: str | None = None,
        station_id: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Count events matching filters."""
        ...

    @abstractmethod
    def list_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        """List unacknowledged anomaly alerts, newest first."""
        ...


class EvidenceStore(ABC):
    """Abstract interface for evidence storage."""

    @abstractmethod
    def index(self, event_id: str, metadata: dict[str, Any]) -> None:
        """Index evidence metadata for an event."""
        ...

    @abstractmethod
    def get_metadata(self, event_id: str) -> dict[str, Any] | None:
        """Get evidence metadata for an event."""
        ...

    @abstractmethod
    def get_image_path(self, event_id: str) -> Path | None:
        """Get the image file path for an event."""
        ...

    @abstractmethod
    def get_thumb_path(self, event_id: str) -> Path | None:
        """Get the thumbnail file path for an event."""
        ...

    @abstractmethod
    def list_by_date(self, date: str) -> list[dict[str, Any]]:
        """List all evidence entries for a given date (YYYY-MM-DD)."""
        ...


class TripleStore(ABC):
    """Abstract interface for causal triple storage."""

    @abstractmethod
    def insert(self, triple: dict[str, Any]) -> str:
        """Insert a causal triple. Returns triple_id."""
        ...

    @abstractmethod
    def get(self, triple_id: str) -> dict[str, Any] | None:
        """Get a single triple by ID."""
        ...

    @abstractmethod
    def list(
        self,
        limit: int = 50,
        status: str | None = None,
        station_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List triples with optional filters."""
        ...

    @abstractmethod
    def update(self, triple_id: str, updates: dict[str, Any]) -> bool:
        """Update a triple. Returns True if found and updated."""
        ...
