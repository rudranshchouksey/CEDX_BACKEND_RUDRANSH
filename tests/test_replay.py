import json
from pathlib import Path

import pytest

from cedx_pipeline.agents.contracts import ModelId, WorkerDraft
from cedx_pipeline.agents.model_router import ModelGateway, ModelRouter
from cedx_pipeline.agents.orchestrator import OrchestratorAgent
from cedx_pipeline.agents.queues import ApprovedQueue, ExceptionQueue
from cedx_pipeline.agents.trace import SpanCollector
from cedx_pipeline.agents.transcript import TranscriptBundle
from cedx_pipeline.amendment import Amendment
from cedx_pipeline.config import REPLAY_LLM
from cedx_pipeline.detectors.models import Anomaly
from cedx_pipeline.errors import MissingTranscriptError
from cedx_pipeline.intake.registry import DataRegistry, Record


@pytest.fixture
def mock_record():
    return Record(
        id="REC-REPLAY-1",
        source_format="json",
        owner="Alice",
        deadline="2026-12-31",
        amount=100.0,
        notes="Clean record",
        source_version_hash="abcdef1234567890",
        payload={},
    )


@pytest.fixture
def transcripts_dir(tmp_path):
    d = tmp_path / "transcripts"
    d.mkdir()
    return d


def test_recording_mode(mock_record, transcripts_dir, monkeypatch):
    """Ensure REPLAY_LLM=False saves a transcript."""
    monkeypatch.setattr("cedx_pipeline.agents.model_router.REPLAY_LLM", False)
    monkeypatch.setattr("cedx_pipeline.agents.model_router.TRANSCRIPTS_DIR", transcripts_dir)

    gateway = ModelGateway()
    result = gateway.infer(
        agent="worker",
        model=ModelId.GEMINI_FLASH,
        prompt="Test prompt",
        record=mock_record,
    )

    files = list(transcripts_dir.glob("*.json"))
    assert len(files) == 1

    bundle = TranscriptBundle.load(files[0])
    assert bundle.agent == "worker"
    assert bundle.request == "Test prompt"
    assert bundle.model == "gemini-1.5-flash"
    assert bundle.raw_response == result.output


def test_replay_mode(mock_record, transcripts_dir, monkeypatch):
    """Ensure REPLAY_LLM=True loads from the transcript without network."""
    monkeypatch.setattr("cedx_pipeline.agents.model_router.REPLAY_LLM", True)
    monkeypatch.setattr("cedx_pipeline.agents.model_router.TRANSCRIPTS_DIR", transcripts_dir)

    bundle = TranscriptBundle(
        request="Test prompt 2",
        raw_response="Replayed output",
        response_hash="hash1234567890abcdef",
        model=ModelId.GPT4O_MINI.value,
        prompt_version="1.0.0",
        agent="worker",
    )
    bundle.save(transcripts_dir)

    gateway = ModelGateway()
    result = gateway.infer(
        agent="worker",
        model=ModelId.GPT4O_MINI,
        prompt="Test prompt 2",
        record=mock_record,
    )

    assert result.output == "Replayed output"


def test_missing_transcript(mock_record, transcripts_dir, monkeypatch):
    """Ensure REPLAY_LLM=True raises an error if no match is found."""
    monkeypatch.setattr("cedx_pipeline.agents.model_router.REPLAY_LLM", True)
    monkeypatch.setattr("cedx_pipeline.agents.model_router.TRANSCRIPTS_DIR", transcripts_dir)

    gateway = ModelGateway()
    with pytest.raises(MissingTranscriptError):
        gateway.infer(
            agent="worker",
            model=ModelId.GPT4O_MINI,
            prompt="Unknown prompt",
            record=mock_record,
        )


def test_hallucination_replay(mock_record, transcripts_dir, monkeypatch):
    """Test full offline orchestration with injected hallucination edge case."""
    monkeypatch.setattr("cedx_pipeline.agents.model_router.REPLAY_LLM", True)
    monkeypatch.setattr("cedx_pipeline.agents.model_router.TRANSCRIPTS_DIR", transcripts_dir)

    from cedx_pipeline.agents.worker import WorkerAgent
    from cedx_pipeline.agents.contracts import AgentContext
    dummy_worker = WorkerAgent(None, None, None)
    
    # 1. Create a hallucinated Worker transcript
    context_0 = AgentContext(
        record_id=mock_record.id,
        record_fields={"owner": mock_record.owner},
        anomalies=[],
        past_failures=0,
        amendment_role="compliance",
        amendment_threshold=1000
    )
    worker_prompt_0 = dummy_worker._build_prompt(mock_record, context_0)
    hallucinated_assembly = {
        "record_id": mock_record.id,
        "owner": mock_record.owner,
        "deadline": mock_record.deadline,
        "amount": "999999",  # HALLUCINATION
        "summary": "Processed",
        "source_hash": mock_record.source_version_hash[:16],
        "assembly_version": "1.0.0",
    }
    TranscriptBundle(
        request=worker_prompt_0,
        raw_response=json.dumps(hallucinated_assembly, indent=2),
        response_hash="worker0",
        model=ModelId.GEMINI_FLASH.value,
        prompt_version="1.0.0",
        agent="worker",
    ).save(transcripts_dir)

    # 2. Create the Verifier transcript for the hallucinated draft
    verifier_prompt_0 = (
        f"Verify draft for record {mock_record.id} against source:\n"
        f"{mock_record}\nDraft:\n{hallucinated_assembly}"
    )
    TranscriptBundle(
        request=verifier_prompt_0,
        raw_response=json.dumps({"verdict": "fail", "reason": "Verifier found 1 issue(s)."}),
        response_hash="verifier0",
        model=ModelId.CLAUDE_SONNET.value,
        prompt_version="1.0.0",
        agent="verifier",
    ).save(transcripts_dir)

    # 3. Create the Retry Worker transcript (correct amount)
    context_1 = AgentContext(
        record_id=mock_record.id,
        record_fields={"owner": mock_record.owner},
        anomalies=[],
        past_failures=1,
        amendment_role="compliance",
        amendment_threshold=1000
    )
    worker_prompt_1 = dummy_worker._build_prompt(mock_record, context_1)
    correct_assembly = {
        "record_id": mock_record.id,
        "owner": mock_record.owner,
        "deadline": mock_record.deadline,
        "amount": str(mock_record.amount),
        "summary": "Processed retry",
        "source_hash": mock_record.source_version_hash[:16],
        "assembly_version": "1.0.0",
    }
    TranscriptBundle(
        request=worker_prompt_1,
        raw_response=json.dumps(correct_assembly, indent=2),
        response_hash="worker1",
        model=ModelId.GPT4O.value,  # Escalate on retry
        prompt_version="1.0.0",
        agent="worker",
    ).save(transcripts_dir)

    # 4. Create the Retry Verifier transcript
    verifier_prompt_1 = (
        f"Verify draft for record {mock_record.id} against source:\n"
        f"{mock_record}\nDraft:\n{correct_assembly}"
    )
    TranscriptBundle(
        request=verifier_prompt_1,
        raw_response=json.dumps({"verdict": "pass", "reason": "Draft passed all validation checks."}),
        response_hash="verifier1",
        model=ModelId.CLAUDE_SONNET.value,
        prompt_version="1.0.0",
        agent="verifier",
    ).save(transcripts_dir)

    # Setup orchestrator
    registry = DataRegistry()
    registry.register(mock_record)
    amendment = Amendment("CEDX-TEST", "compliance", 1000, "abcdef")

    orchestrator = OrchestratorAgent(registry, [], amendment)
    result = orchestrator.run()

    # Verify the outcome
    assert mock_record.id in [r.record_id for r in result.approved.snapshot()]
    
    # Verify trace spans match our expectations
    spans = result.collector.spans_for_record(mock_record.id)
    
    # 4 spans: Worker (fail), Verifier (fail), Worker (retry pass), Verifier (retry pass)
    assert len(spans) == 4
    
    # Check the first Verifier span caught the hallucination
    v_span_0 = spans[1]
    assert v_span_0.agent == "verifier"
    assert v_span_0.verdict.value == "fail"
    
    # Check retry succeeded
    v_span_1 = spans[3]
    assert v_span_1.agent == "verifier"
    assert v_span_1.verdict.value == "pass"
    assert v_span_1.retries == 1
