"""Prompt-injection detector — regex signature matrix.

Scans the ``notes`` field of each record against a curated matrix of
compiled regular expressions.  Each pattern targets a known class of
social-engineering or governance-bypass strings that an adversary might
embed in free-text fields to manipulate downstream LLM agents or
human reviewers.

Design decisions:
    * Patterns are **case-insensitive** (``re.IGNORECASE``).
    * All matched signatures are reported — not just the first hit.
    * Every match is severity ``critical`` because injection attempts
      represent a deliberate policy circumvention.
"""

from __future__ import annotations

import logging
import re

from cedx_pipeline.detectors.models import Anomaly, AnomalyType
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)


# ── Signature Matrix ────────────────────────────────────────────────────────
#
# Each entry is (signature_name, compiled_regex).  The name appears in the
# anomaly detail for traceability during incident review.

INJECTION_SIGNATURES: list[tuple[str, re.Pattern[str]]] = [
    # Direct governance bypass
    (
        "governance_bypass",
        re.compile(r"approve\s+immediately", re.IGNORECASE),
    ),
    (
        "review_skip",
        re.compile(r"skip\s+review", re.IGNORECASE),
    ),
    (
        "rule_override",
        re.compile(r"ignore\s+rules?", re.IGNORECASE),
    ),
    (
        "compliance_override",
        re.compile(r"override\s+(compliance|policy|controls?)", re.IGNORECASE),
    ),
    # Urgency / pressure tactics
    (
        "urgency_pressure",
        re.compile(
            r"(urgent|asap|no\s+time)\s*(approval|review|sign[\s-]?off)",
            re.IGNORECASE,
        ),
    ),
    # Authority impersonation
    (
        "role_impersonation",
        re.compile(
            r"(acting\s+as|on\s+behalf\s+of|authorized\s+by)\s+"
            r"(ceo|cfo|cto|director|board|president)",
            re.IGNORECASE,
        ),
    ),
    # Threshold manipulation
    (
        "threshold_manipulation",
        re.compile(
            r"(raise|increase|remove|disable)\s+(the\s+)?(threshold|limit)",
            re.IGNORECASE,
        ),
    ),
    # Audit suppression
    (
        "audit_suppression",
        re.compile(
            r"(skip|bypass|disable|suppress)\s+(audit|logging|tracking)",
            re.IGNORECASE,
        ),
    ),
    # Direct LLM prompt injection patterns
    (
        "llm_injection",
        re.compile(
            r"(ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?))"
            r"|(disregard\s+(previous|all|above))"
            r"|(you\s+are\s+now\s+(an?\s+)?)",
            re.IGNORECASE,
        ),
    ),
    # Blanket exception demands
    (
        "exception_demand",
        re.compile(
            r"(grant|give|allow)\s+(an?\s+)?(exception|exemption|waiver)",
            re.IGNORECASE,
        ),
    ),
]


def detect_injection(records: list[Record]) -> list[Anomaly]:
    """Scan the ``notes`` field of every record against the signature matrix.

    Records with ``None`` or empty ``notes`` are silently skipped.

    Args:
        records: Snapshot of records to scan.

    Returns:
        List of ``INJECTION_BLOCKED`` anomalies — one per matched signature
        per record (a single record can trigger multiple anomalies if it
        matches multiple patterns).
    """
    anomalies: list[Anomaly] = []

    for rec in records:
        if not rec.notes:
            continue

        for sig_name, pattern in INJECTION_SIGNATURES:
            match = pattern.search(rec.notes)
            if match:
                anomalies.append(
                    Anomaly(
                        record_id=rec.id,
                        anomaly_type=AnomalyType.INJECTION_BLOCKED,
                        detail=(
                            f"Injection signature '{sig_name}' matched in notes: "
                            f"'{match.group()}'"
                        ),
                        severity="critical",
                    )
                )

    logger.info("Injection detector: flagged %d hit(s).", len(anomalies))
    return anomalies
