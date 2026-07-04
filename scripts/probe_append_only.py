import logging
import sys
from dataclasses import FrozenInstanceError

from cedx_pipeline.amendment import Amendment
from cedx_pipeline.governance.audit import AuditEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_probe():
    logger.info("Starting probe: Append-Only Immutable Event Stream")
    
    amendment = Amendment(case_id="PROBE-CASE", role="compliance", threshold=1000, digest="x")
    engine = AuditEngine(amendment)
    
    engine.append_event("TEST_EVENT", {"data": "test"})
    
    events = getattr(engine, "_events")
    event = events[0]
    
    try:
        event.seq = 999
        logger.error("Probe Failed: Was able to mutate a frozen audit event!")
        sys.exit(1)
    except FrozenInstanceError:
        logger.info("Probe Success: AuditEvent mutation successfully blocked by FrozenInstanceError.")
        sys.exit(0)


if __name__ == "__main__":
    run_probe()
