"""Worker Agent — produces assembly payload drafts.

The worker is the primary inference agent in the fleet.  It:

    1. Receives a record + anomaly context from the orchestrator.
    2. Asks the :class:`ModelRouter` to select the right model.
    3. Calls the :class:`ModelGateway` for inference.
    4. Records a :class:`TraceSpan` with full token/cost metrics.
    5. Returns a :class:`WorkerDraft` for the verifier.

The worker may **only** interact with the verifier (enforced by the
``can_call`` whitelist).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cedx_pipeline.agents.contracts import (
    AgentContract,
    AgentContext,
    AgentRole,
    ModelId,
    WorkerDraft,
)
from cedx_pipeline.agents.model_router import ModelGateway, ModelRouter
from cedx_pipeline.agents.trace import SpanCollector, TraceSpan
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)

#: Prompt template version for audit trail.
_PROMPT_VERSION = "1.0.0"


class WorkerAgent(AgentContract):
    """Concrete worker agent — drafts assembly payloads via model inference.

    Attributes (contract):
        name:              ``"worker"``
        role:              :attr:`AgentRole.WORKER`
        authorized_models: flash, mini, pro, gpt-4o (no claude — that's
                           reserved for the verifier)
        can_call:          ``{"verifier"}`` only
    """

    # ── Contract Properties ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "worker"

    @property
    def role(self) -> AgentRole:
        return AgentRole.WORKER

    @property
    def authorized_models(self) -> frozenset[ModelId]:
        return frozenset({
            ModelId.GEMINI_FLASH,
            ModelId.GPT4O_MINI,
            ModelId.GEMINI_PRO,
            ModelId.GPT4O,
        })

    @property
    def can_call(self) -> frozenset[str]:
        return frozenset({"verifier"})

    # ── Constructor ──────────────────────────────────────────────────────

    def __init__(
        self,
        gateway: ModelGateway,
        router: ModelRouter,
        collector: SpanCollector,
    ) -> None:
        self._gateway = gateway
        self._router = router
        self._collector = collector

    # ── Core Execution ───────────────────────────────────────────────────

    def execute(
        self,
        record: Record,
        context: AgentContext,
    ) -> WorkerDraft:
        """Produce an assembly draft for *record*.

        Args:
            record:  The source record to process.
            context: Contextual metadata (anomalies, past failures, etc.).

        Returns:
            A :class:`WorkerDraft` containing the model output and metrics.
        """
        # ── 1. Model selection ───────────────────────────────────────────
        model = self._router.select_model(
            record,
            anomaly_types=context.anomalies,
            past_failures=context.past_failures,
        )

        if model not in self.authorized_models:
            # Fallback safety — should never happen with a well-configured
            # router, but defence-in-depth.
            logger.warning(
                "Worker: router selected unauthorised model %s, "
                "falling back to %s.",
                model.value,
                ModelId.GEMINI_FLASH.value,
            )
            model = ModelId.GEMINI_FLASH

        # ── 2. Build prompt ──────────────────────────────────────────────
        prompt = self._build_prompt(record, context)

        # ── 3. Inference ─────────────────────────────────────────────────
        result = self._gateway.infer(self.name, model, prompt, record)

        # ── 4. Parse assembly from output ────────────────────────────────
        try:
            assembly: dict[str, Any] = json.loads(result.output)
        except (json.JSONDecodeError, TypeError):
            assembly = {"raw_output": result.output}

        # ── 5. Record trace span ─────────────────────────────────────────
        span = TraceSpan(
            record_id=record.id,
            agent=self.name,
            model=model.value,
            prompt_version=_PROMPT_VERSION,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            retries=context.past_failures,
            status="success",
            verdict=None,
        )
        self._collector.record(span)

        draft = WorkerDraft(
            record_id=record.id,
            assembly=assembly,
            model_used=model,
            tokens_in=result.tokens_in,
            tokens_out=result.tokens_out,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            prompt_version=_PROMPT_VERSION,
        )

        logger.info(
            "Worker: record=%s model=%s cost=$%.6f tokens=%d/%d",
            record.id,
            model.value,
            result.cost_usd,
            result.tokens_in,
            result.tokens_out,
        )

        return draft

    # ── Prompt Construction ──────────────────────────────────────────────

    @staticmethod
    def _build_prompt(record: Record, context: AgentContext) -> str:
        """Build the inference prompt from *record* and *context*.

        This is a structured prompt template — production systems would
        version this externally.
        """
        anomaly_section = (
            f"Anomalies detected: {', '.join(context.anomalies)}"
            if context.anomalies
            else "No anomalies detected."
        )

        return (
            f"=== CEDX Assembly Draft Prompt v{_PROMPT_VERSION} ===\n"
            f"Record ID: {record.id}\n"
            f"Owner: {record.owner}\n"
            f"Deadline: {record.deadline}\n"
            f"Amount: {record.amount}\n"
            f"Source: {record.source_format}\n"
            f"Notes: {record.notes}\n"
            f"{anomaly_section}\n"
            f"Past failures: {context.past_failures}\n"
            f"Regulatory role: {context.amendment_role}\n"
            f"Financial threshold: {context.amendment_threshold}\n\n"
            f"Produce a structured assembly payload for governance review."
        )
