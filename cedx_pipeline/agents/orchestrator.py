"""Orchestrator Agent — distributed planner and execution governor.

The orchestrator owns the execution graph across the ingestion registry.
For each record it:

    1. Checks budget (steps and cost) **before** dispatching to the worker.
    2. Dispatches to the :class:`WorkerAgent` for draft generation.
    3. Dispatches to the :class:`VerifierAgent` for independent evaluation.
    4. Handles retries on verifier rejection (up to ``MAX_WORKER_RETRIES``).
    5. Routes terminal outcomes to the :class:`ApprovedQueue` or
       :class:`ExceptionQueue`.

Budget enforcement:
    * ``MAX_STEPS_PER_RECORD`` — hard cap on worker+verifier call count.
    * ``MAX_COST_USD_PER_RECORD`` — hard cap on cumulative inference cost.
    * On breach: immediately halt, append ``BUDGET_EXCEEDED``, route to
      exception queue.

The orchestrator does **not** call any model directly — its
``authorized_models`` is empty.  It coordinates; it does not compute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cedx_pipeline.agents.contracts import (
    AgentContext,
    AgentContract,
    AgentRole,
    FleetRecordResult,
    ModelId,
    RecordState,
    VerifierVerdict,
)
from cedx_pipeline.agents.model_router import ModelGateway, ModelRouter
from cedx_pipeline.agents.queues import ApprovedQueue, ExceptionQueue
from cedx_pipeline.agents.trace import SpanCollector
from cedx_pipeline.agents.verifier import VerifierAgent
from cedx_pipeline.agents.worker import WorkerAgent
from cedx_pipeline.amendment import Amendment
from cedx_pipeline.config import (
    MAX_COST_USD_PER_RECORD,
    MAX_STEPS_PER_RECORD,
    MAX_WORKER_RETRIES,
)
from cedx_pipeline.detectors.models import Anomaly
from cedx_pipeline.intake.registry import DataRegistry, Record

logger = logging.getLogger(__name__)


# ── Fleet Result Aggregate ──────────────────────────────────────────────────


@dataclass
class FleetResult:
    """Aggregate result of the entire fleet run across all records.

    Attributes:
        approved:       Queue of approved records.
        exceptions:     Queue of exception/needs_human records.
        collector:      The span collector with all trace data.
        record_results: Per-record outcome details.
    """

    approved: ApprovedQueue
    exceptions: ExceptionQueue
    collector: SpanCollector
    record_results: list[FleetRecordResult] = field(default_factory=list)


# ── Orchestrator Agent ──────────────────────────────────────────────────────


class OrchestratorAgent(AgentContract):
    """Concrete orchestrator — execution governor for the agent fleet.

    Attributes (contract):
        name:              ``"orchestrator"``
        role:              :attr:`AgentRole.ORCHESTRATOR`
        authorized_models: empty (orchestrator does not call models)
        can_call:          ``{"worker"}`` only
    """

    # ── Contract Properties ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "orchestrator"

    @property
    def role(self) -> AgentRole:
        return AgentRole.ORCHESTRATOR

    @property
    def authorized_models(self) -> frozenset[ModelId]:
        return frozenset()  # does not call models

    @property
    def can_call(self) -> frozenset[str]:
        return frozenset({"worker"})

    # ── Constructor ──────────────────────────────────────────────────────

    def __init__(
        self,
        registry: DataRegistry,
        anomalies: list[Anomaly],
        amendment: Amendment,
    ) -> None:
        self._registry = registry
        self._anomalies_by_record = self._group_anomalies(anomalies)
        self._amendment = amendment

        # ── Instantiate fleet components ─────────────────────────────────
        self._collector = SpanCollector()
        self._gateway = ModelGateway()
        self._router = ModelRouter()
        self._worker = WorkerAgent(self._gateway, self._router, self._collector)
        self._verifier = VerifierAgent(self._collector, self._gateway)
        self._approved = ApprovedQueue()
        self._exceptions = ExceptionQueue()

    # ── Public API ───────────────────────────────────────────────────────

    def run(self) -> FleetResult:
        """Execute the agent fleet across all records in the registry.

        Returns:
            A :class:`FleetResult` with approved/exception queues, trace
            data, and per-record outcomes.
        """
        # Validate inter-agent call contract
        self.validate_call(self._worker.name)
        self._worker.validate_call(self._verifier.name)

        records = self._registry.snapshot()
        logger.info(
            "Orchestrator: starting fleet run across %d record(s).",
            len(records),
        )

        result = FleetResult(
            approved=self._approved,
            exceptions=self._exceptions,
            collector=self._collector,
        )

        for record in records:
            record_result = self._process_record(record)
            result.record_results.append(record_result)

        logger.info(
            "Orchestrator: fleet run complete. "
            "approved=%d exceptions=%d total_cost=$%.6f",
            self._approved.count(),
            self._exceptions.count(),
            self._collector.total_cost(),
        )

        return result

    # ── Per-Record Processing ────────────────────────────────────────────

    def _process_record(self, record: Record) -> FleetRecordResult:
        """Process a single *record* through the worker→verifier loop.

        Enforces step and cost budgets.  On budget breach, halts immediately
        and routes to exception queue with ``BUDGET_EXCEEDED``.
        """
        anomaly_types = self._anomalies_by_record.get(record.id, [])
        steps = 0
        cost = 0.0
        reason_codes: list[str] = []
        last_draft = None

        for attempt in range(MAX_WORKER_RETRIES + 1):
            # ── Budget pre-check ─────────────────────────────────────────
            if steps >= MAX_STEPS_PER_RECORD:
                reason_codes.append("BUDGET_EXCEEDED")
                logger.warning(
                    "Orchestrator: record=%s exceeded step limit (%d/%d).",
                    record.id,
                    steps,
                    MAX_STEPS_PER_RECORD,
                )
                return self._route_exception(
                    record, reason_codes, last_draft, attempt, cost, steps
                )

            if cost >= MAX_COST_USD_PER_RECORD:
                reason_codes.append("BUDGET_EXCEEDED")
                logger.warning(
                    "Orchestrator: record=%s exceeded cost cap ($%.6f/$%.2f).",
                    record.id,
                    cost,
                    MAX_COST_USD_PER_RECORD,
                )
                return self._route_exception(
                    record, reason_codes, last_draft, attempt, cost, steps
                )

            # ── Worker step ──────────────────────────────────────────────
            context = AgentContext(
                record_id=record.id,
                record_fields={
                    "id": record.id,
                    "owner": record.owner,
                    "deadline": record.deadline,
                    "amount": str(record.amount) if record.amount else None,
                    "notes": record.notes,
                },
                anomalies=anomaly_types,
                past_failures=attempt,
                amendment_role=self._amendment.role,
                amendment_threshold=self._amendment.threshold,
            )

            draft = self._worker.execute(record, context)
            last_draft = draft
            steps += 1
            cost += draft.cost_usd

            # ── Budget re-check after worker ─────────────────────────────
            if cost >= MAX_COST_USD_PER_RECORD:
                reason_codes.append("BUDGET_EXCEEDED")
                logger.warning(
                    "Orchestrator: record=%s budget exceeded after worker "
                    "($%.6f/$%.2f).",
                    record.id,
                    cost,
                    MAX_COST_USD_PER_RECORD,
                )
                return self._route_exception(
                    record, reason_codes, last_draft, attempt, cost, steps
                )

            # ── Verifier step ────────────────────────────────────────────
            verdict_result = self._verifier.evaluate(draft, record, attempt)
            steps += 1
            cost += verdict_result.cost_usd

            if verdict_result.verdict == VerifierVerdict.PASS:
                # ── Approved ─────────────────────────────────────────────
                approved_result = FleetRecordResult(
                    record_id=record.id,
                    state=RecordState.APPROVED,
                    reason_codes=reason_codes,
                    draft=draft,
                    retries=attempt,
                    total_cost=round(cost, 6),
                    total_steps=steps,
                )
                self._approved.append(approved_result)
                logger.info(
                    "Orchestrator: record=%s APPROVED after %d attempt(s), "
                    "cost=$%.6f",
                    record.id,
                    attempt + 1,
                    cost,
                )
                return approved_result

            # ── Verifier rejected ────────────────────────────────────────
            for issue in verdict_result.issues:
                if issue.startswith("AGENT_HALLUCINATION"):
                    reason_codes.append("AGENT_HALLUCINATION")
                elif issue.startswith("AGENT_MALFORMED"):
                    reason_codes.append("AGENT_MALFORMED")

            logger.warning(
                "Orchestrator: record=%s verifier REJECTED (attempt %d/%d). "
                "Issues: %s",
                record.id,
                attempt + 1,
                MAX_WORKER_RETRIES + 1,
                "; ".join(verdict_result.issues),
            )

        # ── All retries exhausted → needs_human ─────────────────────────
        reason_codes.append("RETRIES_EXHAUSTED")
        logger.warning(
            "Orchestrator: record=%s all retries exhausted → needs_human.",
            record.id,
        )
        return self._route_exception(
            record,
            reason_codes,
            last_draft,
            MAX_WORKER_RETRIES + 1,
            cost,
            steps,
            state=RecordState.NEEDS_HUMAN,
        )

    # ── Routing Helpers ──────────────────────────────────────────────────

    def _route_exception(
        self,
        record: Record,
        reason_codes: list[str],
        draft: Any,
        retries: int,
        cost: float,
        steps: int,
        state: RecordState = RecordState.EXCEPTION,
    ) -> FleetRecordResult:
        """Route a record to the exception queue."""
        result = FleetRecordResult(
            record_id=record.id,
            state=state,
            reason_codes=reason_codes,
            draft=draft,
            retries=retries,
            total_cost=round(cost, 6),
            total_steps=steps,
        )
        self._exceptions.append(result)
        return result

    # ── Internal Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _group_anomalies(
        anomalies: list[Anomaly],
    ) -> dict[str, list[str]]:
        """Group anomaly type strings by record ID for fast lookup."""
        grouped: dict[str, list[str]] = {}
        for anomaly in anomalies:
            if anomaly.record_id not in grouped:
                grouped[anomaly.record_id] = []
            grouped[anomaly.record_id].append(anomaly.anomaly_type.value)
        return grouped
