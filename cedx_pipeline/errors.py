"""Custom exception hierarchy for the CEDX pipeline.

All pipeline-specific exceptions inherit from :class:`CedxPipelineError`
to allow coarse-grained ``except`` clauses at the orchestration layer while
preserving fine-grained error identity for individual subsystems.
"""

from __future__ import annotations


class CedxPipelineError(Exception):
    """Root exception for every error raised within the CEDX pipeline."""


# ── Configuration & Environment ──────────────────────────────────────────────


class ConfigError(CedxPipelineError):
    """A required environment variable or configuration value is missing or
    malformed."""


# ── Intake / Parsing ────────────────────────────────────────────────────────


class IntakeError(CedxPipelineError):
    """An unrecoverable problem occurred during data intake (file I/O,
    deserialization, or format conversion)."""


class FeedParseError(IntakeError):
    """``feed.json`` could not be read or contains structurally invalid data."""


class EmlParseError(IntakeError):
    """An ``.eml`` file in the inbox could not be parsed."""


class PdfParseError(IntakeError):
    """A ``.pdf`` file in the inbox could not be read or text-extracted."""


# ── Registry ────────────────────────────────────────────────────────────────


class RegistryError(CedxPipelineError):
    """An invariant of the :class:`DataRegistry` was violated (e.g. duplicate
    record ID insertion)."""


class DuplicateRecordError(RegistryError):
    """A record with the same ``id`` already exists in the registry."""


# ── Agent System ────────────────────────────────────────────────────────────


class AgentError(CedxPipelineError):
    """Base error for the multi-agent subsystem."""


class AgentContractViolation(AgentError):
    """An agent attempted to call another agent not in its ``can_call``
    whitelist."""


class BudgetExceededError(AgentError):
    """A record exceeded its per-record cost or step budget."""


class VerifierRejectionError(AgentError):
    """The verifier rejected the worker's draft after all retries were
    exhausted."""


class MissingTranscriptError(AgentError):
    """A required transcript was not found during offline replay mode."""


