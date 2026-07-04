"""Tests for the Cryptographic Amendment Core.

Validates determinism, correctness against a known precomputed vector,
boundary conditions, and the env-var initialisation path.
"""

from __future__ import annotations

import hashlib
import os
from unittest import mock

import pytest

from cedx_pipeline.amendment import Amendment, compute_amendment, init_amendment
from cedx_pipeline.config import REGULATORY_ROLES
from cedx_pipeline.errors import ConfigError


class TestComputeAmendment:
    """Unit tests for :func:`compute_amendment`."""

    def test_determinism(self) -> None:
        """Same input always produces the same output."""
        a = compute_amendment("CEDX-7F3A")
        b = compute_amendment("CEDX-7F3A")
        assert a == b

    def test_case_insensitivity(self) -> None:
        """The case_id is lowercased before hashing."""
        upper = compute_amendment("CEDX-7F3A")
        lower = compute_amendment("cedx-7f3a")
        assert upper.role == lower.role
        assert upper.threshold == lower.threshold
        assert upper.digest == lower.digest

    def test_known_vector_cedx_7f3a(self) -> None:
        """Verify against a manually precomputed reference vector."""
        digest = hashlib.sha256(b"cedx-7f3a").hexdigest()
        expected_role = REGULATORY_ROLES[int(digest[0], 16) % 4]
        expected_threshold = 10_000 + (int(digest[1:3], 16) % 50) * 1_000

        result = compute_amendment("CEDX-7F3A")

        assert result.case_id == "CEDX-7F3A"
        assert result.role == expected_role
        assert result.threshold == expected_threshold
        assert result.digest == digest

    def test_role_within_valid_set(self) -> None:
        """Every role must come from the defined REGULATORY_ROLES tuple."""
        for case_id in ("A", "B", "ZZZZ", "test-1234", "CEDX-0000"):
            result = compute_amendment(case_id)
            assert result.role in REGULATORY_ROLES

    def test_threshold_range(self) -> None:
        """Threshold must fall in [10_000, 59_000] — the formula's range."""
        for case_id in ("X", "YY", "long-case-id-value", "CEDX-FFFF"):
            result = compute_amendment(case_id)
            assert 10_000 <= result.threshold <= 59_000

    def test_frozen_dataclass(self) -> None:
        """Amendment instances must be immutable."""
        result = compute_amendment("CEDX-7F3A")
        with pytest.raises(AttributeError):
            result.role = "tampered"  # type: ignore[misc]


class TestInitAmendment:
    """Integration tests for :func:`init_amendment`."""

    def test_reads_env_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """init_amendment reads CASE_ID and logs the amendment line."""
        with mock.patch.dict(os.environ, {"CASE_ID": "CEDX-7F3A"}):
            with caplog.at_level("INFO"):
                result = init_amendment()

        assert isinstance(result, Amendment)
        assert "AMENDMENT: role=" in caplog.text
        assert result.role in caplog.text
        assert str(result.threshold) in caplog.text

    def test_missing_case_id_raises(self) -> None:
        """init_amendment raises ConfigError when CASE_ID is unset."""
        env = {k: v for k, v in os.environ.items() if k != "CASE_ID"}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError):
                init_amendment()
