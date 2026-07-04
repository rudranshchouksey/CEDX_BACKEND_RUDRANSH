"""Schema validator — enforces non-null constraints on mandatory fields.

This detector catches records where required governance fields are ``None``,
which indicates either a parsing gap or corrupt source data.  Rather than
silently dropping or defaulting such records, the pipeline surfaces them
as ``MISSING_INPUT`` anomalies for human triage.
"""

from __future__ import annotations

import logging
from typing import Any

from cedx_pipeline.config import MANDATORY_FIELDS
from cedx_pipeline.detectors.models import Anomaly, AnomalyType
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)


def _get_field(record: Record, field_name: str) -> Any:
    """Safely retrieve *field_name* from *record* via :func:`getattr`."""
    return getattr(record, field_name, None)


def detect_missing_input(records: list[Record]) -> list[Anomaly]:
    """Return an :class:`Anomaly` for every mandatory field that is ``None``
    on any record.

    The set of mandatory fields is defined in
    :data:`~cedx_pipeline.config.MANDATORY_FIELDS`.

    Args:
        records: Snapshot of records to validate.

    Returns:
        List of ``MISSING_INPUT`` anomalies — at most one per
        (record, field) pair.
    """
    anomalies: list[Anomaly] = []

    for rec in records:
        for field in MANDATORY_FIELDS:
            value = _get_field(rec, field)
            if value is None:
                anomalies.append(
                    Anomaly(
                        record_id=rec.id,
                        anomaly_type=AnomalyType.MISSING_INPUT,
                        detail=f"Mandatory field '{field}' is null.",
                        severity="high",
                    )
                )

    logger.info("Schema validator: flagged %d missing input(s).", len(anomalies))
    return anomalies
