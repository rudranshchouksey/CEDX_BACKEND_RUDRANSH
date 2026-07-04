"""Cryptographic Amendment Core.

Derives a deterministic regulatory role and financial threshold from the
``CASE_ID`` environment variable using SHA-256.  The derivation is pure,
stateless, and reproducible across any platform with a conforming hash
implementation.

Algorithm
---------
::

    H  = sha256(CASE_ID.lower().encode("utf-8")).hexdigest()
    R  = REGULATORY_ROLES[ int(H[0], 16) % 4 ]
    T  = 10_000 + (int(H[1:3], 16) % 50) * 1_000
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from cedx_pipeline.config import REGULATORY_ROLES, get_case_id

logger = logging.getLogger(__name__)


# ── Data Model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Amendment:
    """Immutable result of the cryptographic amendment derivation.

    Attributes:
        case_id:   The raw ``CASE_ID`` that seeded the computation.
        role:      Regulatory role selected from :data:`REGULATORY_ROLES`.
        threshold: Financial threshold in whole currency units.
        digest:    Full SHA-256 hex digest for auditability.
    """

    case_id: str
    role: str
    threshold: int
    digest: str


# ── Core Logic ───────────────────────────────────────────────────────────────


def compute_amendment(case_id: str) -> Amendment:
    """Derive an :class:`Amendment` deterministically from *case_id*.

    This function is **pure** — it has no side-effects, reads no environment
    variables, and performs no I/O.  It is safe to call from any thread.

    Args:
        case_id: Identifier string (e.g. ``"CEDX-7F3A"``).

    Returns:
        A frozen :class:`Amendment` dataclass.
    """
    digest: str = hashlib.sha256(
        case_id.lower().encode("utf-8"),
    ).hexdigest()

    role_index: int = int(digest[0], 16) % 4
    role: str = REGULATORY_ROLES[role_index]

    threshold: int = 10_000 + (int(digest[1:3], 16) % 50) * 1_000

    return Amendment(
        case_id=case_id,
        role=role,
        threshold=threshold,
        digest=digest,
    )


# ── Initialisation Helper ───────────────────────────────────────────────────


def init_amendment() -> Amendment:
    """Read ``CASE_ID`` from the environment, compute the amendment, and log
    the result to stdout.

    This is the **only** function in this module that performs side-effects
    (env-var read + logging).

    Returns:
        The computed :class:`Amendment`.

    Raises:
        ConfigError: Propagated from :func:`~cedx_pipeline.config.get_case_id`
            when ``CASE_ID`` is missing or empty.
    """
    case_id: str = get_case_id()
    amendment: Amendment = compute_amendment(case_id)

    logger.info("AMENDMENT: role=%s threshold=%d", amendment.role, amendment.threshold)

    return amendment
