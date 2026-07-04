"""Outlier detector — Median Absolute Deviation (MAD).

Uses the **modified Z-score** method to detect outliers in the ``amount``
field without relying on hardcoded dollar thresholds.  This is robust to
skewed distributions and is resistant to masking by multiple outliers
(unlike mean/stddev).

Algorithm
---------
::

    median  = median(amounts)
    MAD     = median(|x_i − median|) × 1.4826   (consistency constant)
    score_i = 0.6745 × (x_i − median) / MAD
    outlier if |score_i| > 3.5

Graceful degradation:
    * Batch size < 3  → skip (not enough data).
    * MAD = 0 (all values identical) → no outliers by definition.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from statistics import median as py_median

from cedx_pipeline.config import (
    MAD_CONSISTENCY_CONSTANT,
    MAD_ZSCORE_THRESHOLD,
    MIN_OUTLIER_BATCH_SIZE,
)
from cedx_pipeline.detectors.models import Anomaly, AnomalyType
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)

#: Factor for the modified Z-score numerator (0.6745 ≈ Φ⁻¹(0.75)).
_MODIFIED_ZSCORE_FACTOR = Decimal("0.6745")


def _compute_mad(values: list[Decimal]) -> tuple[Decimal, Decimal]:
    """Return ``(median, MAD)`` for *values*.

    The MAD (Median Absolute Deviation) is scaled by the consistency constant
    so that it estimates the standard deviation under a normal distribution.
    """
    med = py_median(values)
    deviations = [abs(v - med) for v in values]
    raw_mad = py_median(deviations)
    scaled_mad = raw_mad * Decimal(str(MAD_CONSISTENCY_CONSTANT))
    return med, scaled_mad


def detect_outliers(records: list[Record]) -> list[Anomaly]:
    """Scan the ``amount`` field across all records for statistical outliers.

    Records with ``amount is None`` are silently excluded from the
    calculation — they are handled by the schema validator.

    Args:
        records: Snapshot of records to analyse.

    Returns:
        List of ``OUTLIER`` anomalies (may be empty).
    """
    # Collect (record_id, amount) pairs for records that have a valid amount
    valid_pairs: list[tuple[str, Decimal]] = [
        (rec.id, rec.amount)
        for rec in records
        if rec.amount is not None
    ]

    if len(valid_pairs) < MIN_OUTLIER_BATCH_SIZE:
        logger.info(
            "Outlier detector: only %d valid amount(s) — minimum is %d, skipping.",
            len(valid_pairs),
            MIN_OUTLIER_BATCH_SIZE,
        )
        return []

    amounts: list[Decimal] = [pair[1] for pair in valid_pairs]
    med, mad = _compute_mad(amounts)

    if mad == 0:
        logger.info(
            "Outlier detector: MAD=0 (all amounts identical at %s) — no outliers.",
            med,
        )
        return []

    anomalies: list[Anomaly] = []

    for record_id, amount in valid_pairs:
        modified_z = _MODIFIED_ZSCORE_FACTOR * (amount - med) / mad
        if abs(modified_z) > Decimal(str(MAD_ZSCORE_THRESHOLD)):
            anomalies.append(
                Anomaly(
                    record_id=record_id,
                    anomaly_type=AnomalyType.OUTLIER,
                    detail=(
                        f"Amount {amount} has modified Z-score "
                        f"{float(modified_z):.2f} (threshold ±{MAD_ZSCORE_THRESHOLD}). "
                        f"Batch median={float(med):.2f}, MAD={float(mad):.2f}."
                    ),
                    severity="high",
                )
            )

    logger.info("Outlier detector: flagged %d record(s).", len(anomalies))
    return anomalies
