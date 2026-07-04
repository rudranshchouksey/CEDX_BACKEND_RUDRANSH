"""Verifier Agent — independent critic that evaluates worker drafts.

The verifier is **architecturally independent** from the worker:

    * It uses a **different model** (``claude-sonnet-4``).
    * Its ``can_call`` whitelist is **empty** — it cannot delegate to any
      other agent (terminal node in the execution graph).
    * It has full programmatic authority to **OVERRULE** or **REJECT** the
      worker's draft.

Validation checks:
    1. Draft is well-formed (non-empty, valid structure).
    2. ``record_id`` in draft matches the source record.
    3. ``amount`` in draft matches source (no hallucinated values).
    4. No injection patterns leaked into the draft output.
    5. All mandatory source fields are reflected in the draft.

When a check fails, the verifier emits ``AGENT_HALLUCINATION`` or
``AGENT_MALFORMED`` reason codes and returns a ``"fail"`` verdict.
"""

from __future__ import annotations

import json
import logging
import re
import time

from cedx_pipeline.agents.contracts import (
    AgentContract,
    AgentRole,
    ModelId,
    VerifierResult,
    VerifierVerdict,
    WorkerDraft,
)
from cedx_pipeline.agents.model_router import ModelGateway
from cedx_pipeline.agents.trace import SpanCollector, TraceSpan
from cedx_pipeline.config import MODEL_COST_PER_1K_TOKENS
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)

#: Prompt template version for audit trail.
_PROMPT_VERSION = "1.0.0"

#: Lightweight injection patterns to scan draft output.
_DRAFT_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("approve_immediately", re.compile(r"approve\s+immediately", re.IGNORECASE)),
    ("skip_review", re.compile(r"skip\s+review", re.IGNORECASE)),
    ("ignore_rules", re.compile(r"ignore\s+rules?", re.IGNORECASE)),
    ("override_compliance", re.compile(r"override\s+(compliance|policy)", re.IGNORECASE)),
    ("ignore_instructions", re.compile(r"ignore\s+(previous|all)\s+instructions?", re.IGNORECASE)),
]


class VerifierAgent(AgentContract):
    """Concrete verifier agent — independent critic of worker output.

    Attributes (contract):
        name:              ``"verifier"``
        role:              :attr:`AgentRole.VERIFIER`
        authorized_models: ``{claude-sonnet-4}`` only (independent model)
        can_call:          empty frozenset (terminal node)
    """

    # ── Contract Properties ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "verifier"

    @property
    def role(self) -> AgentRole:
        return AgentRole.VERIFIER

    @property
    def authorized_models(self) -> frozenset[ModelId]:
        return frozenset({ModelId.CLAUDE_SONNET})

    @property
    def can_call(self) -> frozenset[str]:
        return frozenset()  # terminal node — cannot call anyone

    # ── Constructor ──────────────────────────────────────────────────────

    def __init__(self, collector: SpanCollector, gateway: ModelGateway) -> None:
        self._collector = collector
        self._gateway = gateway

    # ── Core Execution ───────────────────────────────────────────────────

    def evaluate(
        self,
        draft: WorkerDraft,
        record: Record,
        retry_count: int = 0,
    ) -> VerifierResult:
        """Evaluate *draft* against the raw *record* fields.

        Performs structural validation checks and returns a
        :class:`VerifierResult` with the verdict.

        Args:
            draft:       The worker's assembly payload draft.
            record:      The original source record.
            retry_count: Current retry iteration (for span metadata).

        Returns:
            A :class:`VerifierResult` — verdict is ``"pass"``, ``"fail"``,
            or ``"needs_human"``.
        """
        start = time.monotonic()
        issues: list[str] = []

        # ── Check 1: Draft is non-empty ──────────────────────────────────
        if not draft.assembly:
            issues.append("AGENT_MALFORMED: Draft assembly is empty.")

        # ── Check 2: Record ID consistency ───────────────────────────────
        draft_id = draft.assembly.get("record_id")
        if draft_id is not None and draft_id != record.id:
            issues.append(
                f"AGENT_HALLUCINATION: Draft record_id '{draft_id}' "
                f"does not match source '{record.id}'."
            )

        # ── Check 3: Amount consistency ──────────────────────────────────
        draft_amount = draft.assembly.get("amount")
        if record.amount is not None and draft_amount is not None:
            source_str = str(record.amount)
            if str(draft_amount) != source_str:
                issues.append(
                    f"AGENT_HALLUCINATION: Draft amount '{draft_amount}' "
                    f"does not match source '{source_str}'."
                )

        # ── Check 4: Injection leakage ───────────────────────────────────
        draft_text = str(draft.assembly)
        for pattern_name, pattern in _DRAFT_INJECTION_PATTERNS:
            if pattern.search(draft_text):
                issues.append(
                    f"AGENT_MALFORMED: Injection pattern '{pattern_name}' "
                    f"detected in draft output."
                )

        # ── Check 5: Mandatory field coverage ────────────────────────────
        for field_name in ("record_id", "owner", "deadline", "amount"):
            if field_name not in draft.assembly:
                issues.append(
                    f"AGENT_MALFORMED: Mandatory field '{field_name}' "
                    f"missing from draft assembly."
                )

        # ── Determine verdict ────────────────────────────────────────────
        if issues:
            verdict = VerifierVerdict.FAIL
            reason = f"Verifier found {len(issues)} issue(s)."
        else:
            verdict = VerifierVerdict.PASS
            reason = "Draft passed all validation checks."

        # ── Intercept via Gateway (for Replay/Record) ────────────────────
        prompt = f"Verify draft for record {record.id} against source:\n{record}\nDraft:\n{draft.assembly}"
        simulated_response = json.dumps({"verdict": verdict.value, "reason": reason})

        gateway_result = self._gateway.infer(
            agent=self.name,
            model=ModelId.CLAUDE_SONNET,
            prompt=prompt,
            record=record,
            simulated_response=simulated_response,
        )

        # ── Record trace span (with verdict) ─────────────────────────────
        span = TraceSpan(
            record_id=record.id,
            agent=self.name,
            model=ModelId.CLAUDE_SONNET.value,
            prompt_version=_PROMPT_VERSION,
            tokens_in=gateway_result.tokens_in,
            tokens_out=gateway_result.tokens_out,
            cost_usd=gateway_result.cost_usd,
            latency_ms=gateway_result.latency_ms,
            retries=retry_count,
            status="success",
            verdict=verdict,
        )
        self._collector.record(span)

        # ── Log independently on both sides ──────────────────────────────
        if verdict == VerifierVerdict.FAIL:
            logger.warning(
                "Verifier REJECTED record=%s — %d issue(s): %s",
                record.id,
                len(issues),
                "; ".join(issues),
            )
        else:
            logger.info("Verifier APPROVED record=%s.", record.id)

        return VerifierResult(
            record_id=record.id,
            verdict=verdict,
            reason=reason,
            issues=issues,
            model_used=ModelId.CLAUDE_SONNET,
            tokens_in=gateway_result.tokens_in,
            tokens_out=gateway_result.tokens_out,
            cost_usd=gateway_result.cost_usd,
            latency_ms=gateway_result.latency_ms,
            prompt_version=_PROMPT_VERSION,
        )
