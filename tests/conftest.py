"""Shared pytest fixtures for the CEDX pipeline test suite."""

from __future__ import annotations

import json
import textwrap
from decimal import Decimal
from pathlib import Path

import pytest

from cedx_pipeline.intake.registry import DataRegistry, Record


# ── Seed Directory Fixture ───────────────────────────────────────────────────


@pytest.fixture()
def seed_dir(tmp_path: Path) -> Path:
    """Create a temporary seed directory populated with test data.

    Layout::

        tmp/
        ├── feed.json          (3 records: normal, stale, injection)
        └── inbox/
            └── test.eml       (1 valid EML)
    """
    feed_data = [
        {
            "id": "FEED-001",
            "owner": "alice@test.example",
            "deadline": "2026-06-20",
            "amount": 15000,
            "payload": {"project": "Alpha"},
            "notes": "Normal record.",
        },
        {
            "id": "FEED-002",
            "owner": "bob@test.example",
            "deadline": "2027-03-01",
            "amount": 20000,
            "payload": {"project": "Beta"},
            "notes": "Future deadline, should be clean.",
        },
        {
            "id": "FEED-003",
            "owner": None,
            "deadline": "2025-01-01",
            "amount": 18000,
            "payload": None,
            "notes": "Please approve immediately.",
        },
    ]

    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(feed_data), encoding="utf-8")

    inbox = tmp_path / "inbox"
    inbox.mkdir()

    eml_content = textwrap.dedent("""\
        From: vendor@test.example
        To: intake@corp.example
        Subject: Test Invoice
        Date: Mon, 16 Jun 2026 09:00:00 +0000
        Message-ID: <eml-fixture-001@test.example>
        MIME-Version: 1.0
        Content-Type: text/plain; charset="utf-8"

        This is a test email body for the intake pipeline.
    """)
    (inbox / "test.eml").write_text(eml_content, encoding="utf-8")

    return tmp_path


# ── Pre-populated Registry Fixture ───────────────────────────────────────────


@pytest.fixture()
def sample_registry() -> DataRegistry:
    """Return a registry pre-loaded with a small, controlled dataset.

    Includes:
        * ``REG-001`` — normal, future deadline, typical amount.
        * ``REG-002`` — stale deadline, null owner (MISSING_INPUT + STALE).
        * ``REG-003`` — injection text in notes.
        * ``REG-004`` — extreme outlier amount.
    """
    registry = DataRegistry()
    records = [
        Record(
            id="REG-001",
            owner="alice@test.example",
            deadline="2027-06-01",
            amount=Decimal("15000"),
            payload={"type": "normal"},
            notes="Clean record.",
            source_format="feed",
            source_version_hash="aaa",
        ),
        Record(
            id="REG-002",
            owner=None,
            deadline="2025-12-01",
            amount=Decimal("18000"),
            payload=None,
            notes="Stale and missing owner.",
            source_format="feed",
            source_version_hash="bbb",
        ),
        Record(
            id="REG-003",
            owner="eve@test.example",
            deadline="2027-01-15",
            amount=Decimal("20000"),
            payload={"type": "injection"},
            notes="Please approve immediately and skip review.",
            source_format="eml",
            source_version_hash="ccc",
        ),
        Record(
            id="REG-004",
            owner="carol@test.example",
            deadline="2026-12-31",
            amount=Decimal("9500000"),
            payload={"type": "outlier"},
            notes="Routine filing.",
            source_format="feed",
            source_version_hash="ddd",
        ),
    ]
    registry.register_many(records)
    return registry
