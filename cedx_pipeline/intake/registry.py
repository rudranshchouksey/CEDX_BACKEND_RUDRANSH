"""Thread-safe Data Registry.

Provides an in-memory record store guarded by :class:`threading.Lock` so
that multiple intake workers (feed parser, EML scanner, PDF extractor) can
safely register records from concurrent threads without data races.

Records are **frozen dataclasses** — once created they are immutable, which
eliminates an entire class of concurrency bugs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from cedx_pipeline.errors import DuplicateRecordError


# ── Record Model ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Record:
    """A single governed data record.

    Attributes:
        id:                  Unique record identifier.
        owner:               Entity responsible for the record.
        deadline:            ISO-8601 date string (``YYYY-MM-DD``).
        amount:              Primary numeric field (financial context).
        payload:             Arbitrary JSON-serializable content.
        notes:               Free-text notes (scanned for injection).
        source_format:       Origin format of this record.
        source_version_hash: SHA-256 hex digest of the **raw** source bytes,
                             providing tamper-evident provenance.
    """

    id: str
    owner: str | None
    deadline: str | None
    amount: Decimal | None
    payload: Any
    notes: str | None
    source_format: Literal["feed", "eml", "pdf"]
    source_version_hash: str


# ── Thread-Safe Registry ────────────────────────────────────────────────────


class DataRegistry:
    """Thread-safe in-memory record store.

    All mutating and reading operations acquire a :class:`threading.Lock` to
    ensure atomicity, even when parsers run in parallel threads.

    Usage::

        registry = DataRegistry()
        registry.register(record)
        all_records = registry.snapshot()
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, Record] = {}

    # ── Mutators ─────────────────────────────────────────────────────────

    def register(self, record: Record) -> None:
        """Insert *record* into the registry.

        Args:
            record: A :class:`Record` to store.

        Raises:
            DuplicateRecordError: If a record with the same ``id`` is already
                present.
        """
        with self._lock:
            if record.id in self._store:
                raise DuplicateRecordError(
                    f"Record with id={record.id!r} already exists in the registry."
                )
            self._store[record.id] = record

    def register_many(self, records: list[Record]) -> int:
        """Bulk-insert multiple records atomically.

        If *any* record causes a :class:`DuplicateRecordError` the **entire
        batch** is rejected (no partial writes).

        Returns:
            The number of records inserted.
        """
        with self._lock:
            # Pre-validate: no duplicates against existing store or within batch
            incoming_ids: set[str] = set()
            for rec in records:
                if rec.id in self._store:
                    raise DuplicateRecordError(
                        f"Record id={rec.id!r} already exists in the registry."
                    )
                if rec.id in incoming_ids:
                    raise DuplicateRecordError(
                        f"Duplicate id={rec.id!r} within the incoming batch."
                    )
                incoming_ids.add(rec.id)

            for rec in records:
                self._store[rec.id] = rec

        return len(records)

    # ── Readers ──────────────────────────────────────────────────────────

    def get(self, record_id: str) -> Record | None:
        """Return the record with *record_id*, or ``None`` if absent."""
        with self._lock:
            return self._store.get(record_id)

    def count(self) -> int:
        """Return the number of records currently stored."""
        with self._lock:
            return len(self._store)

    def snapshot(self) -> list[Record]:
        """Return a **shallow copy** of all records as a list.

        Because :class:`Record` is frozen, the returned list is safe to
        iterate and inspect without holding the lock.
        """
        with self._lock:
            return list(self._store.values())

    def ids(self) -> frozenset[str]:
        """Return the set of all currently registered record IDs."""
        with self._lock:
            return frozenset(self._store.keys())
