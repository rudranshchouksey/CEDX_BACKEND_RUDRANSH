import logging
import sys
from unittest import mock

from cedx_pipeline.agents.orchestrator import OrchestratorAgent
from cedx_pipeline.amendment import compute_amendment
from cedx_pipeline.intake.registry import DataRegistry, Record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_probe():
    logger.info("Starting probe: Budget Enforcement")
    
    registry = DataRegistry()
    registry.register(
        Record(
            id="PROBE-BUDGET-1",
            source_format="json",
            owner="Alice",
            deadline="2026-12-31",
            amount=100.0,
            notes="Clean record",
            source_version_hash="abcdef",
            payload={}
        )
    )

    orch = OrchestratorAgent(registry, [], compute_amendment("TEST-BUDGET"))

    with mock.patch("cedx_pipeline.agents.orchestrator.MAX_COST_USD_PER_RECORD", 0.00000001):
        result = orch.run()

    exceptions = result.exceptions.snapshot()
    if not exceptions:
        logger.error("Probe Failed: Record was not routed to exceptions!")
        sys.exit(1)
        
    reason_codes = exceptions[0].reason_codes
    if "BUDGET_EXCEEDED" not in reason_codes:
        logger.error(f"Probe Failed: BUDGET_EXCEEDED not in reason codes: {reason_codes}")
        sys.exit(1)
        
    logger.info("Probe Success: BUDGET_EXCEEDED successfully caught.")
    sys.exit(0)


if __name__ == "__main__":
    run_probe()
