import logging
import sys
from decimal import Decimal

from cedx_pipeline.agents.contracts import ModelId, WorkerDraft, VerifierVerdict
from cedx_pipeline.agents.model_router import ModelGateway
from cedx_pipeline.agents.trace import SpanCollector
from cedx_pipeline.agents.verifier import VerifierAgent
from cedx_pipeline.amendment import Amendment
from cedx_pipeline.governance.state_machine import ReviewStateMachine, DeliveryBlockedError
from cedx_pipeline.intake.registry import Record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_probe():
    logger.info("Starting probe: Agent Failure / Hallucination Intercept")
    
    # 1. Setup
    amendment = Amendment(case_id="PROBE-CASE", role="compliance", threshold=1000, digest="x")
    record = Record(
        id="PROBE-REC-2",
        source_format="json",
        owner="Alice",
        deadline="2026-12-31",
        amount=Decimal("15000"),
        notes="High value record",
        source_version_hash="abcdef",
        payload={}
    )
    
    # 2. Worker submits a hallucinated draft
    draft = WorkerDraft(
        record_id=record.id,
        assembly={
            "record_id": record.id,
            "owner": record.owner,
            "deadline": record.deadline,
            "amount": "99999",  # hallucinated amount
            "summary": "Hallucinated data.",
        },
        model_used=ModelId.GEMINI_FLASH,
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.001,
        latency_ms=5.0,
    )
    
    # 3. Verifier checks it
    collector = SpanCollector()
    gateway = ModelGateway()
    verifier = VerifierAgent(collector, gateway)
    result = verifier.evaluate(draft, record)
    
    if result.verdict != VerifierVerdict.FAIL:
        logger.error(f"Probe Failed: Verifier did not fail the hallucination. Verdict was {result.verdict}")
        sys.exit(1)
        
    logger.info(f"Verifier correctly identified hallucination with issues: {result.issues}")
    
    # 4. State Machine should block delivery due to AGENT_HALLUCINATION
    machine = ReviewStateMachine(record, amendment)
    machine.transition_to_in_review()
    machine.transition_to_approved()
    
    for issue in result.issues:
        if issue.startswith("AGENT_HALLUCINATION"):
            machine.add_reason_code("AGENT_HALLUCINATION")
        elif issue.startswith("AGENT_MALFORMED"):
            machine.add_reason_code("AGENT_MALFORMED")
        
    try:
        machine.transition_to_delivered()
        logger.error("Probe Failed: Record was delivered despite agent failure!")
        sys.exit(1)
    except DeliveryBlockedError as e:
        logger.info(f"Probe Success: Delivery safely blocked by reason codes: {e}")
        sys.exit(0)


if __name__ == "__main__":
    run_probe()
