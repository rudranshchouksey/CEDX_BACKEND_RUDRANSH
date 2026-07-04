"""Agent contract interface and shared type vocabulary.

Every agent in the fleet **must** subclass :class:`AgentContract` and
declare four structural properties:

    * ``name``              — unique string identifier.
    * ``role``              — one of the :class:`AgentRole` enum members.
    * ``authorized_models`` — frozenset of :class:`ModelId` values the agent
                              is permitted to invoke.
    * ``can_call``          — frozenset of agent *names* this agent may
                              interact with (whitelist).

The :meth:`validate_call` method enforces the ``can_call`` whitelist at
runtime, raising :class:`AgentContractViolation` on any unauthorised
inter-agent communication.

All enums are derived from the ``audit.schema.json`` governance schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from cedx_pipeline.errors import AgentContractViolation


# ── Enums (mirroring audit.schema.json) ──────────────────────────────────────


class AgentRole(str, Enum):
    """Agent roles within the fleet.  Values match ``audit.schema.json``."""

    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    VERIFIER = "verifier"
    ROUTER = "router"


class ModelId(str, Enum):
    """Authorized model identifiers.  Values match ``audit.schema.json``."""

    GEMINI_FLASH = "gemini-1.5-flash"
    GPT4O_MINI = "gpt-4o-mini"
    GEMINI_PRO = "gemini-1.5-pro"
    GPT4O = "gpt-4o"
    CLAUDE_SONNET = "claude-sonnet-4"


class VerifierVerdict(str, Enum):
    """Outcome of the verifier's evaluation.  Values match
    ``audit.schema.json``."""

    PASS = "pass"
    FAIL = "fail"
    NEEDS_HUMAN = "needs_human"


class RecordState(str, Enum):
    """Lifecycle state of a record within the agent pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXCEPTION = "exception"
    NEEDS_HUMAN = "needs_human"


# ── Data Transfer Objects ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Immutable context bundle passed into an agent's ``execute`` method.

    Attributes:
        record_id:      ID of the record being processed.
        record_fields:  Flat dict of the record's raw fields for inspection.
        anomalies:      List of anomaly type strings from Phase 1 detection.
        past_failures:  How many times the worker draft was rejected so far.
        amendment_role: The regulatory role from the amendment core.
        amendment_threshold: The financial threshold from the amendment core.
    """

    record_id: str
    record_fields: dict[str, Any]
    anomalies: list[str]
    past_failures: int = 0
    amendment_role: str = ""
    amendment_threshold: int = 0


@dataclass(frozen=True, slots=True)
class WorkerDraft:
    """Immutable output produced by the :class:`WorkerAgent`.

    Attributes:
        record_id:     ID of the source record.
        assembly:      The structured assembly payload draft.
        model_used:    Which model produced this draft.
        tokens_in:     Input token count.
        tokens_out:    Output token count.
        cost_usd:      Inference cost for this draft.
        latency_ms:    Inference latency.
        prompt_version: Semantic version of the prompt template.
    """

    record_id: str
    assembly: dict[str, Any]
    model_used: ModelId
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    prompt_version: str = "1.0.0"


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Immutable output produced by the :class:`VerifierAgent`.

    Attributes:
        record_id:      ID of the source record.
        verdict:        The verifier's determination.
        reason:         Human-readable explanation.
        issues:         Specific issues found (empty on pass).
        model_used:     Which model the verifier ran.
        tokens_in:      Input token count.
        tokens_out:     Output token count.
        cost_usd:       Inference cost.
        latency_ms:     Inference latency.
        prompt_version: Semantic version of the prompt template.
    """

    record_id: str
    verdict: VerifierVerdict
    reason: str
    issues: list[str] = field(default_factory=list)
    model_used: ModelId = ModelId.CLAUDE_SONNET
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    prompt_version: str = "1.0.0"


@dataclass(frozen=True, slots=True)
class FleetRecordResult:
    """Final outcome for a single record after the agent fleet processes it.

    Attributes:
        record_id:    ID of the record.
        state:        Terminal lifecycle state.
        reason_codes: Accumulated reason codes (e.g. BUDGET_EXCEEDED,
                      AGENT_HALLUCINATION).
        draft:        The last worker draft (may be ``None`` if budget was
                      exceeded before the worker ran).
        retries:      Number of worker retries performed.
        total_cost:   Cumulative cost across all agent calls for this record.
        total_steps:  Cumulative steps consumed.
    """

    record_id: str
    state: RecordState
    reason_codes: list[str] = field(default_factory=list)
    draft: WorkerDraft | None = None
    retries: int = 0
    total_cost: float = 0.0
    total_steps: int = 0


# ── Agent Contract ABC ──────────────────────────────────────────────────────


class AgentContract(ABC):
    """Abstract base class defining the typed contract for every fleet agent.

    Subclasses **must** implement the four structural properties and the
    ``execute`` method.  The ``validate_call`` enforcement is provided for
    free.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique agent identifier (e.g. ``"worker"``, ``"verifier"``)."""
        ...

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """The agent's role within the fleet."""
        ...

    @property
    @abstractmethod
    def authorized_models(self) -> frozenset[ModelId]:
        """Set of model IDs this agent is permitted to invoke."""
        ...

    @property
    @abstractmethod
    def can_call(self) -> frozenset[str]:
        """Whitelist of agent *names* this agent may interact with.

        An empty frozenset means the agent is a terminal node (e.g. the
        verifier).
        """
        ...

    def validate_call(self, target_name: str) -> None:
        """Raise :class:`AgentContractViolation` if *target_name* is not in
        this agent's ``can_call`` whitelist.

        Args:
            target_name: The ``name`` property of the agent being called.

        Raises:
            AgentContractViolation: If the call is not authorised.
        """
        if target_name not in self.can_call:
            raise AgentContractViolation(
                f"Agent '{self.name}' (role={self.role.value}) is not "
                f"authorised to call agent '{target_name}'.  "
                f"Allowed targets: {sorted(self.can_call) or '(none)'}."
            )
