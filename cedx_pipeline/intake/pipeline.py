"""Intake pipeline orchestrator.

Reads the ``SEED_DIR`` environment variable (defaulting to ``/app/seed``),
discovers ``feed.json`` and the ``inbox/`` sub-directory, dispatches the
appropriate parsers, and populates a :class:`DataRegistry`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from cedx_pipeline.config import get_seed_dir
from cedx_pipeline.intake.parsers import (
    parse_eml_directory,
    parse_feed,
    parse_pdf_directory,
)
from cedx_pipeline.intake.registry import DataRegistry

logger = logging.getLogger(__name__)


def run_intake(*, seed_dir: Path | None = None) -> DataRegistry:
    """Execute the full intake pipeline and return a populated registry.

    Steps:
        1. Resolve the seed directory (parameter > ``SEED_DIR`` env > default).
        2. Parse ``feed.json`` from the seed root.
        3. Scan ``inbox/`` for ``.eml`` and ``.pdf`` files.
        4. Register all records into a new :class:`DataRegistry`.

    Args:
        seed_dir: Explicit seed directory override.  When ``None`` the value
            is read from :func:`~cedx_pipeline.config.get_seed_dir`.

    Returns:
        A :class:`DataRegistry` containing every successfully parsed record.
    """
    resolved_dir: Path = seed_dir if seed_dir is not None else get_seed_dir()
    logger.info("Intake: seed directory resolved to %s", resolved_dir)

    if not resolved_dir.is_dir():
        logger.warning(
            "Seed directory %s does not exist — returning empty registry.",
            resolved_dir,
        )
        return DataRegistry()

    # ── Discover sources ─────────────────────────────────────────────────

    feed_path: Path = resolved_dir / "feed.json"
    inbox_path: Path = resolved_dir / "inbox"

    # ── Parse all formats ────────────────────────────────────────────────

    feed_records = parse_feed(feed_path)
    eml_records = parse_eml_directory(inbox_path)
    pdf_records = parse_pdf_directory(inbox_path)

    # ── Populate registry ────────────────────────────────────────────────

    registry = DataRegistry()

    for batch_name, batch in [
        ("feed", feed_records),
        ("eml", eml_records),
        ("pdf", pdf_records),
    ]:
        if batch:
            registered = registry.register_many(batch)
            logger.info("Registry: loaded %d %s record(s).", registered, batch_name)

    logger.info("Intake complete: %d total record(s) in registry.", registry.count())
    return registry
