"""Tests for the Agent Contract Interface.

Validates that every concrete agent correctly declares its contract
properties, and that ``validate_call`` enforces the ``can_call`` whitelist.
"""

from __future__ import annotations

import pytest

from cedx_pipeline.agents.contracts import (
    AgentRole,
    ModelId,
)
from cedx_pipeline.agents.model_router import ModelGateway, ModelRouter
from cedx_pipeline.agents.orchestrator import OrchestratorAgent
from cedx_pipeline.agents.trace import SpanCollector
from cedx_pipeline.agents.verifier import VerifierAgent
from cedx_pipeline.agents.worker import WorkerAgent
from cedx_pipeline.amendment import compute_amendment
from cedx_pipeline.errors import AgentContractViolation
from cedx_pipeline.intake.registry import DataRegistry


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def collector() -> SpanCollector:
    return SpanCollector()


@pytest.fixture()
def worker(collector: SpanCollector) -> WorkerAgent:
    return WorkerAgent(ModelGateway(), ModelRouter(), collector)


@pytest.fixture()
def verifier(collector: SpanCollector) -> VerifierAgent:
    return VerifierAgent(collector, ModelGateway())


@pytest.fixture()
def orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent(
        DataRegistry(),
        [],
        compute_amendment("TEST-CONTRACT"),
    )


# ── Contract Declaration Tests ───────────────────────────────────────────────


class TestWorkerContract:
    """Verify the WorkerAgent's contract declaration."""

    def test_name(self, worker: WorkerAgent) -> None:
        assert worker.name == "worker"

    def test_role(self, worker: WorkerAgent) -> None:
        assert worker.role == AgentRole.WORKER

    def test_authorized_models(self, worker: WorkerAgent) -> None:
        expected = frozenset({
            ModelId.GEMINI_FLASH,
            ModelId.GPT4O_MINI,
            ModelId.GEMINI_PRO,
            ModelId.GPT4O,
        })
        assert worker.authorized_models == expected

    def test_can_call_only_verifier(self, worker: WorkerAgent) -> None:
        assert worker.can_call == frozenset({"verifier"})

    def test_cannot_call_orchestrator(self, worker: WorkerAgent) -> None:
        with pytest.raises(AgentContractViolation):
            worker.validate_call("orchestrator")


class TestVerifierContract:
    """Verify the VerifierAgent's contract declaration."""

    def test_name(self, verifier: VerifierAgent) -> None:
        assert verifier.name == "verifier"

    def test_role(self, verifier: VerifierAgent) -> None:
        assert verifier.role == AgentRole.VERIFIER

    def test_authorized_models_claude_only(self, verifier: VerifierAgent) -> None:
        assert verifier.authorized_models == frozenset({ModelId.CLAUDE_SONNET})

    def test_can_call_empty(self, verifier: VerifierAgent) -> None:
        """Verifier is a terminal node — cannot call any agent."""
        assert verifier.can_call == frozenset()

    def test_cannot_call_worker(self, verifier: VerifierAgent) -> None:
        with pytest.raises(AgentContractViolation):
            verifier.validate_call("worker")

    def test_cannot_call_orchestrator(self, verifier: VerifierAgent) -> None:
        with pytest.raises(AgentContractViolation):
            verifier.validate_call("orchestrator")


class TestOrchestratorContract:
    """Verify the OrchestratorAgent's contract declaration."""

    def test_name(self, orchestrator: OrchestratorAgent) -> None:
        assert orchestrator.name == "orchestrator"

    def test_role(self, orchestrator: OrchestratorAgent) -> None:
        assert orchestrator.role == AgentRole.ORCHESTRATOR

    def test_no_authorized_models(self, orchestrator: OrchestratorAgent) -> None:
        """Orchestrator does not call models directly."""
        assert orchestrator.authorized_models == frozenset()

    def test_can_call_only_worker(self, orchestrator: OrchestratorAgent) -> None:
        assert orchestrator.can_call == frozenset({"worker"})

    def test_cannot_call_verifier(self, orchestrator: OrchestratorAgent) -> None:
        with pytest.raises(AgentContractViolation):
            orchestrator.validate_call("verifier")


# ── Cross-Agent Whitelist Tests ──────────────────────────────────────────────


class TestCrossAgentWhitelist:
    """Validate that the call-graph is acyclic: orchestrator→worker→verifier."""

    def test_orchestrator_can_call_worker(
        self, orchestrator: OrchestratorAgent
    ) -> None:
        orchestrator.validate_call("worker")  # should not raise

    def test_worker_can_call_verifier(self, worker: WorkerAgent) -> None:
        worker.validate_call("verifier")  # should not raise

    def test_verifier_is_terminal(self, verifier: VerifierAgent) -> None:
        """Verifier cannot call anything — validates the DAG constraint."""
        for target in ("orchestrator", "worker", "verifier", "anything"):
            with pytest.raises(AgentContractViolation):
                verifier.validate_call(target)

    def test_worker_cannot_call_self(self, worker: WorkerAgent) -> None:
        with pytest.raises(AgentContractViolation):
            worker.validate_call("worker")
