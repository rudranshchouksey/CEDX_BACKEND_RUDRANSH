"""Detection engine — orchestrates all anomaly detectors.

Accepts a :class:`DataRegistry` and the pipeline reference date, runs every
detector over a consistent snapshot of the registry, and returns a unified
list of :class:`Anomaly` findings.
"""

from __future__ import annotations

import logging
from datetime import date

from cedx_pipeline.detectors.injection import detect_injection
from cedx_pipeline.detectors.models import Anomaly
from cedx_pipeline.detectors.outlier import detect_outliers
from cedx_pipeline.detectors.schema import detect_missing_input
from cedx_pipeline.detectors.stale import detect_stale
from cedx_pipeline.intake.registry import DataRegistry, Record

logger = logging.getLogger(__name__)


def run_detectors(
    registry: DataRegistry,
    pipeline_now: date,
) -> list[Anomaly]:
    """Execute all anomaly detectors against a consistent registry snapshot.

    The snapshot is taken **once** under the registry lock, then each detector
    operates on the same immutable list — guaranteeing deterministic,
    repeatable results even if another thread mutates the registry concurrently.

    Args:
        registry:     The populated :class:`DataRegistry`.
        pipeline_now: Reference date for the staleness detector.

    Returns:
        Merged list of all :class:`Anomaly` findings, ordered by detector
        execution sequence (stale → schema → outlier → injection).
    """
    snapshot: list[Record] = registry.snapshot()
    logger.info("Detection engine: analysing %d record(s).", len(snapshot))

    anomalies: list[Anomaly] = []

    # ── 1. Staleness ─────────────────────────────────────────────────────
    anomalies.extend(detect_stale(snapshot, pipeline_now))

    # ── 2. Schema (missing mandatory fields) ─────────────────────────────
    anomalies.extend(detect_missing_input(snapshot))

    # ── 3. Statistical outliers ──────────────────────────────────────────
    anomalies.extend(detect_outliers(snapshot))

    # ── 4. Prompt injection ──────────────────────────────────────────────
    anomalies.extend(detect_injection(snapshot))

    logger.info(
        "Detection engine: total anomalies = %d  "
        "(STALE=%d, MISSING_INPUT=%d, OUTLIER=%d, INJECTION_BLOCKED=%d)",
        len(anomalies),
        sum(1 for a in anomalies if a.anomaly_type.value == "STALE"),
        sum(1 for a in anomalies if a.anomaly_type.value == "MISSING_INPUT"),
        sum(1 for a in anomalies if a.anomaly_type.value == "OUTLIER"),
        sum(1 for a in anomalies if a.anomaly_type.value == "INJECTION_BLOCKED"),
    )

    return anomalies
