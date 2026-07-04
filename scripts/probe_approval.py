import logging
import sys

from cedx_pipeline.amendment import Amendment
from cedx_pipeline.governance.state_machine import ReviewStateMachine, LiveAmendmentGateError
from cedx_pipeline.intake.registry import Record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_probe():
    logger.info("Starting probe: Approval Gate Bypass")
    
    # 1. Setup an amendment with a threshold of 1000
    amendment = Amendment(case_id="PROBE-CASE", role="compliance", threshold=1000, digest="x")
    
    # 2. Setup a record with amount 5000 (exceeds threshold)
    record = Record(
        id="PROBE-REC-1",
        source_format="json",
        owner="Alice",
        deadline="2026-12-31",
        amount=5000.0,
        notes="High value record",
        source_version_hash="abcdef",
        payload={}
    )
    
    machine = ReviewStateMachine(record, amendment)
    
    # 3. Try to push it through to delivered without valid signature
    machine.transition_to_in_review()
    machine.transition_to_approved()
    
    try:
        machine.transition_to_delivered()
        logger.error("Probe Failed: Record was delivered without required signature!")
        sys.exit(1)
    except LiveAmendmentGateError as e:
        logger.info(f"Probe Success: Delivery safely blocked: {e}")
        sys.exit(0)


if __name__ == "__main__":
    run_probe()
