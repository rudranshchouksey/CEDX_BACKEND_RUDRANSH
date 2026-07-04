"""STALE detector — flags records whose deadline has already passed.

Comparison reference date is ``PIPELINE_NOW``, read from the environment
variable of the same name (default: ``2026-06-26``).
"""

from __future__ import annotations

import logging
from datetime import date

from cedx_pipeline.detectors.models import Anomaly, AnomalyType
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)


def detect_stale(records: list[Record], pipeline_now: date) -> list[Anomaly]:
    """Return an :class:`Anomaly` for every record whose deadline is in the
    past relative to *pipeline_now*.

    Records with a ``None`` or unparseable ``deadline`` are **not** flagged
    here — that is the responsibility of the schema (``MISSING_INPUT``)
    detector.

    Args:
        records:      Snapshot of records to scan.
        pipeline_now: Reference date for staleness comparison.

    Returns:
        List of ``STALE`` anomalies (may be empty).
    """
    anomalies: list[Anomaly] = []

    for rec in records:
        if rec.deadline is None:
            continue  # handled by schema detector

        try:
            deadline_date = date.fromisoformat(rec.deadline)
        except (ValueError, TypeError):
            logger.debug(
                "Record %s has unparseable deadline %r — skipping stale check.",
                rec.id,
                rec.deadline,
            )
            continue

        if deadline_date < pipeline_now:
            anomalies.append(
                Anomaly(
                    record_id=rec.id,
                    anomaly_type=AnomalyType.STALE,
                    detail=(
                        f"Deadline {rec.deadline} is before pipeline reference "
                        f"date {pipeline_now.isoformat()}."
                    ),
                    severity="medium",
                )
            )

    logger.info("Stale detector: flagged %d record(s).", len(anomalies))
    return anomalies
