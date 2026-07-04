"""Centralised configuration readers for the CEDX pipeline.

Every environment-variable access passes through this module so that the
rest of the codebase never calls :func:`os.getenv` directly.  This makes
configuration auditable, testable, and easy to stub.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from cedx_pipeline.errors import ConfigError

# ── Backend Roots ────────────────────────────────────────────────────────────
BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Public Constants ─────────────────────────────────────────────────────────

#: Regulatory roles that the amendment core may assign.
REGULATORY_ROLES: tuple[str, ...] = (
    "risk_officer",
    "legal_counsel",
    "compliance",
    "finance_controller",
)

#: Minimum batch size required for the MAD outlier detector to produce
#: statistically meaningful results.
MIN_OUTLIER_BATCH_SIZE: int = 3

#: Modified Z-score threshold for MAD outlier detection.
MAD_ZSCORE_THRESHOLD: float = 3.5

#: Consistency constant for the MAD estimator under normality assumption.
MAD_CONSISTENCY_CONSTANT: float = 1.4826

#: Default ``PIPELINE_NOW`` value when the env var is unset (ISO-8601).
DEFAULT_PIPELINE_NOW: str = "2026-06-26"

#: Default seed directory path when ``SEED_DIR`` is unset.
DEFAULT_SEED_DIR: str = str(BACKEND_ROOT / "seed")

#: Mandatory record fields that must be non-null.
MANDATORY_FIELDS: tuple[str, ...] = ("id", "owner", "deadline", "amount")

# ── Agent System Constants ───────────────────────────────────────────────────

#: Maximum execution steps (worker + verifier calls) allowed per record.
MAX_STEPS_PER_RECORD: int = 6

#: Maximum cumulative inference cost (USD) allowed per record.
MAX_COST_USD_PER_RECORD: float = 0.50

#: Maximum number of worker retries after verifier rejection before
#: escalating to ``needs_human``.
MAX_WORKER_RETRIES: int = 2

#: Cost per 1,000 tokens for each authorized model (simulated gateway).
MODEL_COST_PER_1K_TOKENS: dict[str, float] = {
    "gemini-1.5-flash": 0.001,
    "gpt-4o-mini": 0.002,
    "gemini-1.5-pro": 0.008,
    "gpt-4o": 0.010,
    "claude-sonnet-4": 0.012,
}

# ── Phase 3 Constants ────────────────────────────────────────────────────────

#: Directory for storing and retrieving LLM transcripts.
TRANSCRIPTS_DIR: Path = Path(os.environ.get("TRANSCRIPTS_DIR", BACKEND_ROOT / "transcripts")).resolve()

#: If true, intercept LLM interactions and replay from transcripts.
REPLAY_LLM: bool = os.environ.get("REPLAY_LLM", "false").strip().lower() == "true"



# ── Environment Readers ──────────────────────────────────────────────────────


def get_case_id() -> str:
    """Return the ``CASE_ID`` environment variable or raise.

    Raises:
        ConfigError: If ``CASE_ID`` is not set or is empty/whitespace.
    """
    value = os.environ.get("CASE_ID", "").strip()
    if not value:
        raise ConfigError(
            "Environment variable CASE_ID is required but not set or empty."
        )
    return value


def get_seed_dir() -> Path:
    """Return the intake seed directory as a resolved :class:`~pathlib.Path`.

    Falls back to :data:`DEFAULT_SEED_DIR` when ``SEED_DIR`` is unset.
    """
    raw = os.environ.get("SEED_DIR", DEFAULT_SEED_DIR).strip()
    return Path(raw).resolve()


def get_pipeline_now() -> date:
    """Return the pipeline reference date.

    Falls back to :data:`DEFAULT_PIPELINE_NOW` when ``PIPELINE_NOW`` is unset.

    Raises:
        ConfigError: If the value is present but not a valid ISO-8601 date.
    """
    raw = os.environ.get("PIPELINE_NOW", DEFAULT_PIPELINE_NOW).strip()
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(
            f"PIPELINE_NOW value {raw!r} is not a valid ISO-8601 date."
        ) from exc
