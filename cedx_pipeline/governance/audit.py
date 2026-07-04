"""Compliant Audit & Files Packaging Generator.

Maintains a strictly sequential, append-only event log and atomically
generates the final audited output files in the /out directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cedx_pipeline.agents.contracts import FleetRecordResult
from cedx_pipeline.agents.trace import SpanCollector
from cedx_pipeline.amendment import Amendment

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """An append-only event in the audit stream."""

    seq: int
    event_type: str
    timestamp: str
    details: dict[str, Any]


class AuditEngine:
    """Atomic file writer and append-only event stream manager."""

    def __init__(self, amendment: Amendment):
        self._amendment = amendment
        self._lock = threading.Lock()
        self._events: list[AuditEvent] = []
        self._seq = 0

    def append_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Atomically append a new event to the stream."""
        with self._lock:
            event = AuditEvent(
                seq=self._seq,
                event_type=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                details=details,
            )
            self._events.append(event)
            self._seq += 1
            logger.debug("Audit event appended: seq=%d type=%s", event.seq, event.event_type)

    def export(
        self,
        out_dir: str | Path,
        exception_records: list[FleetRecordResult],
        approved_records: list[FleetRecordResult],
        collector: SpanCollector,
        package_payload: dict[str, Any] | list[dict[str, Any]],
        branded_package_name: str = "branded_package.json",
    ) -> None:
        """Atomically write the final audit and package files to out_dir."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. branded_package
        package_file = out_path / branded_package_name
        package_bytes = json.dumps(package_payload, indent=2).encode("utf-8")
        package_hash = hashlib.sha256(package_bytes).hexdigest()
        
        # Write to temporary file then move for atomicity
        temp_package = package_file.with_suffix(".tmp")
        temp_package.write_bytes(package_bytes)
        temp_package.replace(package_file)

        # 2. exception_queue.json
        exceptions_file = out_path / "exception_queue.json"
        exceptions_data = [
            {
                "record_id": r.record_id,
                "state": r.state.value if hasattr(r.state, "value") else r.state,
                "reason_codes": r.reason_codes,
            }
            for r in exception_records
        ]
        temp_exceptions = exceptions_file.with_suffix(".tmp")
        with open(temp_exceptions, "w", encoding="utf-8") as f:
            json.dump(exceptions_data, f, indent=2)
        temp_exceptions.replace(exceptions_file)

        # 3. audit.json
        audit_file = out_path / "audit.json"
        with self._lock:
            events_dump = [asdict(e) for e in self._events]

        agent_trace: dict[str, list[dict[str, Any]]] = {}
        for r in exception_records + approved_records:
            spans = collector.spans_for_record(r.record_id)
            span_dicts = []
            for span in spans:
                d = span.to_dict()
                d.pop("record_id", None)
                span_dicts.append(d)
            agent_trace[r.record_id] = span_dicts

        audit_data = {
            "agents": ["orchestrator", "worker", "verifier", "router"],
            "cost": collector.total_cost(),
            "amendment": {
                "case_id": self._amendment.case_id,
                "role": self._amendment.role,
                "threshold": self._amendment.threshold,
                "digest": self._amendment.digest,
            },
            "output_package_hash": package_hash,
            "events": events_dump,
            "agent_trace": agent_trace,
        }

        temp_audit = audit_file.with_suffix(".tmp")
        with open(temp_audit, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)
        temp_audit.replace(audit_file)

        logger.info("Exported audit package to %s (package_hash=%s)", out_dir, package_hash)
