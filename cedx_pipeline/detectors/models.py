"""Anomaly data models shared across all detectors.

Every detector returns instances of :class:`Anomaly` so that the detection
engine and downstream consumers operate on a uniform contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class AnomalyType(str, Enum):
    """Enumeration of anomaly categories the pipeline can detect."""

    STALE = "STALE"
    MISSING_INPUT = "MISSING_INPUT"
    OUTLIER = "OUTLIER"
    INJECTION_BLOCKED = "INJECTION_BLOCKED"


@dataclass(frozen=True, slots=True)
class Anomaly:
    """An immutable anomaly finding attached to a specific record.

    Attributes:
        record_id:    The ``id`` of the record that triggered this anomaly.
        anomaly_type: Category of the anomaly.
        detail:       Human-readable explanation of *why* this was flagged.
        severity:     Impact severity for triage and escalation.
    """

    record_id: str
    anomaly_type: AnomalyType
    detail: str
    severity: Literal["low", "medium", "high", "critical"]
