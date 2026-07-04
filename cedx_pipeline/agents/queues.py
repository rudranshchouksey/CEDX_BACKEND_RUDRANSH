"""Result queues for the agent fleet.

Two thread-safe queues collect the terminal outcomes of record processing:

    * :class:`ApprovedQueue` — records that passed both worker and verifier.
    * :class:`ExceptionQueue` — records that were budget-exceeded, rejected
      after max retries, or escalated to ``needs_human``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from cedx_pipeline.agents.contracts import FleetRecordResult, RecordState


class ApprovedQueue:
    """Thread-safe queue for records that passed the agent pipeline.

    Every entry is a :class:`FleetRecordResult` with
    ``state == RecordState.APPROVED``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[FleetRecordResult] = []

    def append(self, result: FleetRecordResult) -> None:
        """Add an approved result to the queue."""
        with self._lock:
            self._items.append(result)

    def snapshot(self) -> list[FleetRecordResult]:
        """Return a shallow copy of all approved results."""
        with self._lock:
            return list(self._items)

    def count(self) -> int:
        """Return the number of approved records."""
        with self._lock:
            return len(self._items)


class ExceptionQueue:
    """Thread-safe queue for records that failed or need human review.

    Entries may have states: ``EXCEPTION``, ``REJECTED``, or
    ``NEEDS_HUMAN``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[FleetRecordResult] = []

    def append(self, result: FleetRecordResult) -> None:
        """Add an exception result to the queue."""
        with self._lock:
            self._items.append(result)

    def snapshot(self) -> list[FleetRecordResult]:
        """Return a shallow copy of all exception results."""
        with self._lock:
            return list(self._items)

    def count(self) -> int:
        """Return the number of exception records."""
        with self._lock:
            return len(self._items)
