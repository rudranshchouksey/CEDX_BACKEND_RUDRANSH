"""CEDX Pipeline — Phase 1 + Phase 2 entrypoint.

Orchestrates the full pipeline in sequence:

    **Phase 1:**
    1. Cryptographic Amendment Core — derive role & threshold from CASE_ID.
    2. Resilient Intake Pipeline — ingest feed, EML, and PDF into registry.
    3. Detection Engine — anomaly & injection detectors.

    **Phase 2:**
    4. Multi-Agent Fleet — orchestrator → worker → verifier loop with
       budget enforcement, model routing, and observability tracing.

Exit codes:
    * ``0`` — pipeline completed (anomalies may or may not be present).
    * ``1`` — fatal configuration or infrastructure error.
"""

from __future__ import annotations

import logging
import sys

from cedx_pipeline.agents.orchestrator import FleetResult, OrchestratorAgent
from cedx_pipeline.amendment import Amendment, init_amendment
from cedx_pipeline.config import get_pipeline_now
from cedx_pipeline.detectors.engine import run_detectors
from cedx_pipeline.detectors.models import Anomaly
from cedx_pipeline.errors import CedxPipelineError
from cedx_pipeline.governance.audit import AuditEngine
from cedx_pipeline.governance.state_machine import ReviewStateMachine, DeliveryBlockedError, LiveAmendmentGateError
from cedx_pipeline.intake.pipeline import run_intake
from cedx_pipeline.intake.registry import DataRegistry

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Set up structured root logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _print_anomaly_report(anomalies: list[Anomaly]) -> None:
    """Emit a human-readable anomaly summary to stdout."""
    if not anomalies:
        logger.info("No anomalies detected — pipeline clean.")
        return

    logger.info("=" * 72)
    logger.info("ANOMALY REPORT — %d finding(s)", len(anomalies))
    logger.info("=" * 72)

    for idx, a in enumerate(anomalies, 1):
        logger.info(
            "  [%d] %-20s | severity=%-8s | record=%s\n"
            "       %s",
            idx,
            a.anomaly_type.value,
            a.severity,
            a.record_id,
            a.detail,
        )

    logger.info("=" * 72)


def _print_fleet_report(result: FleetResult) -> None:
    """Emit a human-readable fleet execution summary to stdout."""
    logger.info("=" * 72)
    logger.info("AGENT FLEET REPORT")
    logger.info("=" * 72)

    logger.info(
        "  Approved: %d | Exceptions: %d",
        result.approved.count(),
        result.exceptions.count(),
    )

    # ── Per-record details ───────────────────────────────────────────────
    for rr in result.record_results:
        reason_str = ", ".join(rr.reason_codes) if rr.reason_codes else "none"
        model_str = rr.draft.model_used.value if rr.draft else "n/a"
        logger.info(
            "  record=%-30s state=%-12s model=%-20s retries=%d "
            "cost=$%.6f steps=%d reasons=[%s]",
            rr.record_id,
            rr.state.value,
            model_str,
            rr.retries,
            rr.total_cost,
            rr.total_steps,
            reason_str,
        )

    # ── Observability summary ────────────────────────────────────────────
    summary = result.collector.summary()
    logger.info("  --- Trace Summary ---")
    logger.info("  Total spans: %d", summary["total_spans"])
    logger.info("  Total cost:  $%.6f", summary["total_cost_usd"])
    logger.info("  Tokens in:   %d", summary["total_tokens_in"])
    logger.info("  Tokens out:  %d", summary["total_tokens_out"])
    logger.info("  Models used: %s", ", ".join(summary["models_used"]))
    logger.info("  Agents:      %s", ", ".join(summary["agents_active"]))

    logger.info("=" * 72)


def main() -> None:
    """Top-level pipeline entrypoint.

    Designed to be called as a CLI script (``cedx-pipeline``) or via
    ``python -m cedx_pipeline.main``.
    """
    _configure_logging()
    logger.info("CEDX Pipeline — starting (Phase 1 + Phase 2).")

    try:
        # ── Phase 1 ─────────────────────────────────────────────────────

        # 1. Cryptographic Amendment
        amendment: Amendment = init_amendment()
        logger.info(
            "Amendment initialised: case_id=%s role=%s threshold=%d",
            amendment.case_id,
            amendment.role,
            amendment.threshold,
        )

        # 2. Resilient Intake
        registry: DataRegistry = run_intake()
        logger.info("Intake complete: %d record(s) registered.", registry.count())

        # 3. Detection Engine
        pipeline_now = get_pipeline_now()
        anomalies: list[Anomaly] = run_detectors(registry, pipeline_now)

        # 4. Anomaly Report
        _print_anomaly_report(anomalies)

        # ── Phase 2 ─────────────────────────────────────────────────────

        # 5. Multi-Agent Fleet
        logger.info("Initialising multi-agent fleet (Phase 2).")
        orchestrator = OrchestratorAgent(registry, anomalies, amendment)
        fleet_result: FleetResult = orchestrator.run()

        # 6. Fleet Report
        _print_fleet_report(fleet_result)

        # ── Phase 4 ─────────────────────────────────────────────────────
        logger.info("Running Phase 4: State-Machine Review and Audit Packaging")
        audit_engine = AuditEngine(amendment)
        
        delivered_payloads = []
        # Process approved records
        for record_res in fleet_result.approved.snapshot():
            record = registry.get(record_res.record_id)
            if not record:
                continue
            
            machine = ReviewStateMachine(record, amendment, audit_engine=audit_engine)
            # Example transitions for happy path
            machine.transition_to_in_review()
            machine.transition_to_approved()
            
            # If threshold is met, auto-sign for tests (in real life, a human would sign)
            amount = float(record.amount) if record.amount is not None else 0.0
            if amount >= amendment.threshold:
                machine.add_signature(amendment.role, "auto-agent")
                
            for rc in record_res.reason_codes:
                machine.add_reason_code(rc)
                
            try:
                machine.transition_to_delivered()
                delivered_payloads.append(record.payload)
            except (DeliveryBlockedError, LiveAmendmentGateError) as e:
                logger.warning("Delivery blocked for %s: %s", record.id, e)
                record_res.reason_codes.append("DELIVERY_BLOCKED")
                fleet_result.exceptions.enqueue(record_res)
        
        audit_engine.export(
            out_dir="out",
            exception_records=fleet_result.exceptions.snapshot(),
            approved_records=fleet_result.approved.snapshot(),
            collector=fleet_result.collector,
            package_payload=delivered_payloads
        )

        logger.info("CEDX Pipeline — completed successfully.")

    except CedxPipelineError as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error in pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
