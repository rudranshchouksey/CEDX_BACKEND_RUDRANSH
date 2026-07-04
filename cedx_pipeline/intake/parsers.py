"""Multi-format intake parsers.

Each parser reads a specific source format, computes a provenance hash of the
raw bytes, and returns a list of :class:`~cedx_pipeline.intake.registry.Record`
objects ready for registry insertion.

Supported formats:
    * ``feed``  — structured JSON array (``feed.json``).
    * ``eml``   — raw RFC-5322 email messages (``inbox/*.eml``).
    * ``pdf``   — text-extractable PDF documents (``inbox/*.pdf``).
"""

from __future__ import annotations

import email
import email.parser
import email.policy
import hashlib
import json
import logging
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from cedx_pipeline.errors import EmlParseError, FeedParseError, PdfParseError
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _safe_decimal(value: Any) -> Decimal | None:
    """Coerce *value* to :class:`Decimal`, returning ``None`` on failure.

    This avoids float-precision issues in financial pipelines.
    """
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _generate_id(prefix: str, *args: Any) -> str:
    """Generate a deterministic record ID with a format prefix based on the arguments."""
    m = hashlib.sha256()
    for arg in args:
        m.update(str(arg).encode("utf-8"))
    return f"{prefix}-{m.hexdigest()[:12]}"


# ── Feed Parser ──────────────────────────────────────────────────────────────


def parse_feed(feed_path: Path) -> list[Record]:
    """Parse ``feed.json`` into records.

    The file must contain a top-level JSON array of objects.  Each object
    **should** have keys ``id``, ``owner``, ``deadline``, ``amount``,
    ``payload``, and ``notes``.  Missing keys are stored as ``None`` so that
    downstream schema validation can flag them as ``MISSING_INPUT``.

    Args:
        feed_path: Absolute path to the ``feed.json`` file.

    Returns:
        List of parsed records.

    Raises:
        FeedParseError: If the file is missing, unreadable, or does not
            contain a JSON array.
    """
    if not feed_path.is_file():
        logger.warning("Feed file not found at %s — skipping.", feed_path)
        return []

    raw_bytes: bytes = b""
    try:
        raw_bytes = feed_path.read_bytes()
        data = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeedParseError(f"Cannot parse feed at {feed_path}: {exc}") from exc

    if not isinstance(data, list):
        raise FeedParseError(
            f"feed.json must contain a JSON array, got {type(data).__name__}."
        )

    version_hash: str = _sha256_hex(raw_bytes)
    records: list[Record] = []

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.warning("Feed entry #%d is not an object — skipping.", idx)
            continue

        records.append(
            Record(
                id=str(entry.get("id") or _generate_id("feed", version_hash, idx)),
                owner=entry.get("owner"),
                deadline=entry.get("deadline"),
                amount=_safe_decimal(entry.get("amount")),
                payload=entry.get("payload"),
                notes=entry.get("notes"),
                source_format="feed",
                source_version_hash=version_hash,
            )
        )

    logger.info("Feed parser: ingested %d record(s) from %s", len(records), feed_path)
    return records


# ── EML Parser ───────────────────────────────────────────────────────────────


def _extract_eml_body(msg: email.message.EmailMessage) -> str:
    """Extract the best plain-text body from a parsed email message."""
    body = msg.get_body(preferencelist=("plain", "html"))
    if body is not None:
        content = body.get_content()
        return content if isinstance(content, str) else str(content)
    return ""


def parse_eml_directory(inbox_path: Path) -> list[Record]:
    """Scan *inbox_path* for ``.eml`` files and parse each into a record.

    Each email becomes a single record.  The ``From`` header maps to
    ``owner``, the body maps to ``payload``, and the ``Subject`` maps to
    ``notes``.  The ``amount`` and ``deadline`` fields are set to ``None``
    (schema validation will flag these as ``MISSING_INPUT`` if they remain
    unresolved by a later enrichment stage).

    Args:
        inbox_path: Directory containing ``.eml`` files.

    Returns:
        List of parsed records.
    """
    if not inbox_path.is_dir():
        logger.warning("Inbox directory not found at %s — skipping.", inbox_path)
        return []

    eml_files = sorted(inbox_path.glob("*.eml"))
    if not eml_files:
        logger.info("No .eml files found in %s.", inbox_path)
        return []

    records: list[Record] = []
    parser = email.parser.BytesParser(policy=email.policy.default)

    for eml_path in eml_files:
        try:
            raw_bytes = eml_path.read_bytes()
            msg = parser.parsebytes(raw_bytes)

            records.append(
                Record(
                    id=msg.get("Message-ID", _generate_id("eml", _sha256_hex(raw_bytes))).strip("<>"),
                    owner=msg.get("From"),
                    deadline=None,
                    amount=None,
                    payload=_extract_eml_body(msg),
                    notes=msg.get("Subject"),
                    source_format="eml",
                    source_version_hash=_sha256_hex(raw_bytes),
                )
            )
        except Exception as exc:
            raise EmlParseError(
                f"Failed to parse EML file {eml_path.name}: {exc}"
            ) from exc

    logger.info(
        "EML parser: ingested %d record(s) from %s", len(records), inbox_path
    )
    return records


# ── PDF Parser ───────────────────────────────────────────────────────────────


def parse_pdf_directory(inbox_path: Path) -> list[Record]:
    """Scan *inbox_path* for ``.pdf`` files and extract text into records.

    Uses ``pdfplumber`` for text extraction.  Each PDF becomes a single
    record with the extracted text stored in ``payload``.  Like EML records,
    ``amount`` and ``deadline`` start as ``None``.

    Args:
        inbox_path: Directory containing ``.pdf`` files.

    Returns:
        List of parsed records.
    """
    if not inbox_path.is_dir():
        logger.warning("Inbox directory not found at %s — skipping.", inbox_path)
        return []

    pdf_files = sorted(inbox_path.glob("*.pdf"))
    if not pdf_files:
        logger.info("No .pdf files found in %s.", inbox_path)
        return []

    # Lazy import: pdfplumber is only needed if PDFs exist
    try:
        import pdfplumber  # type: ignore[import-untyped]
    except ImportError as exc:
        raise PdfParseError(
            "pdfplumber is required for PDF intake but is not installed. "
            "Install with: pip install pdfplumber"
        ) from exc

    records: list[Record] = []

    for pdf_path in pdf_files:
        try:
            raw_bytes = pdf_path.read_bytes()
            version_hash = _sha256_hex(raw_bytes)

            pages_text: list[str] = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)

            records.append(
                Record(
                    id=_generate_id("pdf", version_hash),
                    owner=None,
                    deadline=None,
                    amount=None,
                    payload="\n\n".join(pages_text) if pages_text else "",
                    notes=pdf_path.stem,  # filename as fallback notes
                    source_format="pdf",
                    source_version_hash=version_hash,
                )
            )
        except PdfParseError:
            raise
        except Exception as exc:
            raise PdfParseError(
                f"Failed to extract text from {pdf_path.name}: {exc}"
            ) from exc

    logger.info(
        "PDF parser: ingested %d record(s) from %s", len(records), inbox_path
    )
    return records
