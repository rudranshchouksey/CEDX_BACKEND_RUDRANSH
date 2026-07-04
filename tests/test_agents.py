"""Tests for the multi-agent system — worker, verifier, and orchestrator.

Covers model routing, draft generation, verifier judgement, budget
enforcement, retry logic, and the full orchestrator loop.
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest

from cedx_pipeline.agents.contracts import (
    AgentContext,
    ModelId,
    RecordState,
    VerifierVerdict,
    WorkerDraft,
)
from cedx_pipeline.agents.model_router import ModelGateway, ModelRouter
from cedx_pipeline.agents.orchestrator import OrchestratorAgent
from cedx_pipeline.agents.trace import SpanCollector
from cedx_pipeline.agents.verifier import VerifierAgent
from cedx_pipeline.agents.worker import WorkerAgent
from cedx_pipeline.amendment import compute_amendment
from cedx_pipeline.detectors.models import Anomaly, AnomalyType
from cedx_pipeline.intake.registry import DataRegistry, Record


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_record(**overrides) -> Record:
    defaults = dict(
        id="AGT-001",
        owner="tester@example.com",
        deadline="2027-06-01",
        amount=Decimal("15000"),
        payload={"project": "Test"},
        notes="Standard record.",
        source_format="feed",
        source_version_hash="abc123def456",
    )
    defaults.update(overrides)
    return Record(**defaults)


def _make_anomaly(record_id: str, atype: AnomalyType) -> Anomaly:
    return Anomaly(
        record_id=record_id,
        anomaly_type=atype,
        detail=f"Test {atype.value}",
        severity="high",
    )


# ── Model Router Tests ───────────────────────────────────────────────────────


class TestModelRouter:
    """Tests for complexity-based model selection."""

    def test_clean_record_selects_flash(self) -> None:
        router = ModelRouter()
        record = _make_record()
        model = router.select_model(record, [], past_failures=0)
        assert model == ModelId.GEMINI_FLASH

    def test_moderate_anomalies_select_mini(self) -> None:
        router = ModelRouter()
        record = _make_record()
        model = router.select_model(record, ["STALE"], past_failures=0)
        assert model == ModelId.GPT4O_MINI

    def test_many_anomalies_select_pro(self) -> None:
        router = ModelRouter()
        record = _make_record()
        model = router.select_model(
            record,
            ["STALE", "MISSING_INPUT", "OUTLIER"],
            past_failures=0,
        )
        assert model == ModelId.GEMINI_PRO

    def test_injection_selects_pro(self) -> None:
        router = ModelRouter()
        record = _make_record()
        model = router.select_model(
            record, ["INJECTION_BLOCKED"], past_failures=0
        )
        assert model == ModelId.GEMINI_PRO

    def test_retry_escalates_to_gpt4o(self) -> None:
        router = ModelRouter()
        record = _make_record()
        model = router.select_model(record, [], past_failures=1)
        assert model == ModelId.GPT4O


# ── Worker Agent Tests ───────────────────────────────────────────────────────


class TestWorkerAgent:
    """Tests for the WorkerAgent draft generation."""

    def test_produces_valid_draft(self) -> None:
        collector = SpanCollector()
        worker = WorkerAgent(ModelGateway(), ModelRouter(), collector)
        record = _make_record()
        context = AgentContext(
            record_id=record.id,
            record_fields={"id": record.id},
            anomalies=[],
            past_failures=0,
        )

        draft = worker.execute(record, context)

        assert draft.record_id == record.id
        assert isinstance(draft.assembly, dict)
        assert draft.tokens_in > 0
        assert draft.tokens_out > 0
        assert draft.cost_usd > 0
        assert collector.count() == 1

    def test_records_trace_span(self) -> None:
        collector = SpanCollector()
        worker = WorkerAgent(ModelGateway(), ModelRouter(), collector)
        record = _make_record(id="SPAN-TEST")
        context = AgentContext(
            record_id=record.id,
            record_fields={},
            anomalies=[],
        )

        worker.execute(record, context)

        spans = collector.spans_for_record("SPAN-TEST")
        assert len(spans) == 1
        assert spans[0].agent == "worker"
        assert spans[0].verdict is None  # only verifier sets verdict


# ── Verifier Agent Tests ─────────────────────────────────────────────────────


class TestVerifierAgent:
    """Tests for the VerifierAgent evaluation."""

    def test_approves_clean_draft(self) -> None:
        collector = SpanCollector()
        verifier = VerifierAgent(collector, ModelGateway())
        record = _make_record()
        draft = WorkerDraft(
            record_id=record.id,
            assembly={
                "record_id": record.id,
                "owner": record.owner,
                "deadline": record.deadline,
                "amount": str(record.amount),
                "summary": "Test.",
            },
            model_used=ModelId.GEMINI_FLASH,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=5.0,
        )

        result = verifier.evaluate(draft, record)

        assert result.verdict == VerifierVerdict.PASS
        assert len(result.issues) == 0

    def test_rejects_hallucinated_amount(self) -> None:
        collector = SpanCollector()
        verifier = VerifierAgent(collector, ModelGateway())
        record = _make_record(amount=Decimal("15000"))
        draft = WorkerDraft(
            record_id=record.id,
            assembly={
                "record_id": record.id,
                "owner": record.owner,
                "deadline": record.deadline,
                "amount": "99999",  # hallucinated
                "summary": "Test.",
            },
            model_used=ModelId.GEMINI_FLASH,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=5.0,
        )

        result = verifier.evaluate(draft, record)

        assert result.verdict == VerifierVerdict.FAIL
        assert any("AGENT_HALLUCINATION" in i for i in result.issues)

    def test_rejects_mismatched_record_id(self) -> None:
        collector = SpanCollector()
        verifier = VerifierAgent(collector, ModelGateway())
        record = _make_record(id="REAL-ID")
        draft = WorkerDraft(
            record_id="REAL-ID",
            assembly={
                "record_id": "WRONG-ID",  # hallucinated
                "owner": record.owner,
                "deadline": record.deadline,
                "amount": str(record.amount),
            },
            model_used=ModelId.GEMINI_FLASH,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=5.0,
        )

        result = verifier.evaluate(draft, record)

        assert result.verdict == VerifierVerdict.FAIL
        assert any("AGENT_HALLUCINATION" in i for i in result.issues)

    def test_rejects_empty_draft(self) -> None:
        collector = SpanCollector()
        verifier = VerifierAgent(collector, ModelGateway())
        record = _make_record()
        draft = WorkerDraft(
            record_id=record.id,
            assembly={},  # empty
            model_used=ModelId.GEMINI_FLASH,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=5.0,
        )

        result = verifier.evaluate(draft, record)

        assert result.verdict == VerifierVerdict.FAIL
        assert any("AGENT_MALFORMED" in i for i in result.issues)

    def test_span_includes_verdict(self) -> None:
        collector = SpanCollector()
        verifier = VerifierAgent(collector, ModelGateway())
        record = _make_record(id="VERDICT-TEST")
        draft = WorkerDraft(
            record_id=record.id,
            assembly={
                "record_id": record.id,
                "owner": record.owner,
                "deadline": record.deadline,
                "amount": str(record.amount),
            },
            model_used=ModelId.GEMINI_FLASH,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.001,
            latency_ms=5.0,
        )

        verifier.evaluate(draft, record)

        spans = collector.spans_for_record("VERDICT-TEST")
        assert len(spans) == 1
        assert spans[0].agent == "verifier"
        assert spans[0].verdict == VerifierVerdict.PASS


# ── Orchestrator Tests ───────────────────────────────────────────────────────


class TestOrchestrator:
    """Integration tests for the OrchestratorAgent."""

    def test_clean_record_approved(self) -> None:
        """A clean record should pass worker+verifier and land in approved."""
        registry = DataRegistry()
        registry.register(_make_record(id="CLEAN-001"))

        orch = OrchestratorAgent(
            registry, [], compute_amendment("TEST-ORCH")
        )
        result = orch.run()

        assert result.approved.count() == 1
        assert result.exceptions.count() == 0
        assert result.record_results[0].state == RecordState.APPROVED

    def test_budget_exceeded_routes_to_exception(self) -> None:
        """Setting a tiny cost cap should trigger BUDGET_EXCEEDED."""
        registry = DataRegistry()
        registry.register(_make_record(id="BUDGET-001"))

        orch = OrchestratorAgent(
            registry, [], compute_amendment("TEST-BUDGET")
        )

        with mock.patch(
            "cedx_pipeline.agents.orchestrator.MAX_COST_USD_PER_RECORD", 0.0
        ):
            result = orch.run()

        assert result.exceptions.count() == 1
        exc_result = result.exceptions.snapshot()[0]
        assert "BUDGET_EXCEEDED" in exc_result.reason_codes

    def test_multiple_records_processed(self) -> None:
        """All records in the registry should be processed."""
        registry = DataRegistry()
        for i in range(4):
            registry.register(_make_record(id=f"MULTI-{i:03d}"))

        orch = OrchestratorAgent(
            registry, [], compute_amendment("TEST-MULTI")
        )
        result = orch.run()

        total = result.approved.count() + result.exceptions.count()
        assert total == 4
        assert len(result.record_results) == 4

    def test_anomalous_record_uses_higher_tier_model(self) -> None:
        """Records with many anomalies should route to premium models."""
        registry = DataRegistry()
        record = _make_record(id="ANOM-001")
        registry.register(record)

        anomalies = [
            _make_anomaly("ANOM-001", AnomalyType.STALE),
            _make_anomaly("ANOM-001", AnomalyType.MISSING_INPUT),
            _make_anomaly("ANOM-001", AnomalyType.OUTLIER),
        ]

        orch = OrchestratorAgent(
            registry, anomalies, compute_amendment("TEST-ANOM")
        )
        result = orch.run()

        # The worker should have used gemini-1.5-pro for 3+ anomalies
        rr = result.record_results[0]
        assert rr.draft is not None
        assert rr.draft.model_used == ModelId.GEMINI_PRO

    def test_trace_spans_collected(self) -> None:
        """Every agent interaction should produce a trace span."""
        registry = DataRegistry()
        registry.register(_make_record(id="TRACE-001"))

        orch = OrchestratorAgent(
            registry, [], compute_amendment("TEST-TRACE")
        )
        result = orch.run()

        spans = result.collector.all_spans()
        # At minimum: 1 worker span + 1 verifier span
        assert len(spans) >= 2

        agents_seen = {s.agent for s in spans}
        assert "worker" in agents_seen
        assert "verifier" in agents_seen

        # Verifier span must include verdict
        verifier_spans = [s for s in spans if s.agent == "verifier"]
        assert all(s.verdict is not None for s in verifier_spans)
