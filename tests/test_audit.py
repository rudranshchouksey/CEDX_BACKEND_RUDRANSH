import json
from pathlib import Path

import jsonschema
import pytest

from cedx_pipeline.agents.contracts import FleetRecordResult, RecordState
from cedx_pipeline.agents.trace import SpanCollector, TraceSpan
from cedx_pipeline.amendment import Amendment
from cedx_pipeline.governance.audit import AuditEngine


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture
def audit_schema():
    schema_path = Path(__file__).parent.parent / "audit.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_audit_engine_sequencing():
    amendment = Amendment("CEDX-123", "risk_officer", 5000, "abc123digest")
    engine = AuditEngine(amendment)
    
    engine.append_event("PROCESS_STARTED", {"records": 10})
    engine.append_event("PROCESS_COMPLETED", {"status": "success"})
    
    assert len(engine._events) == 2
    assert engine._events[0].seq == 0
    assert engine._events[0].event_type == "PROCESS_STARTED"
    assert engine._events[1].seq == 1
    assert engine._events[1].event_type == "PROCESS_COMPLETED"


def test_audit_engine_export_and_validation(out_dir, audit_schema):
    amendment = Amendment("CEDX-123", "risk_officer", 5000, "abc123digest")
    engine = AuditEngine(amendment)
    
    engine.append_event("EXPORT_INIT", {"destination": str(out_dir)})
    
    # Mock records
    approved = FleetRecordResult(
        record_id="REC-1",
        state=RecordState.APPROVED,
        reason_codes=[],
        total_cost=0.01,
        total_steps=3,
    )
    exception = FleetRecordResult(
        record_id="REC-2",
        state=RecordState.EXCEPTION,
        reason_codes=["AGENT_HALLUCINATION"],
        total_cost=0.005,
        total_steps=2,
    )
    
    # Mock spans
    collector = SpanCollector()
    collector.record(
        TraceSpan(
            record_id="REC-1",
            agent="worker",
            model="gpt-4o",
            prompt_version="1.0.0",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.01,
            latency_ms=10.0,
            retries=0,
            status="success",
        )
    )
    
    package_payload = [{"record_id": "REC-1", "amount": 1000}]
    
    engine.export(
        out_dir=out_dir,
        exception_records=[exception],
        approved_records=[approved],
        collector=collector,
        package_payload=package_payload,
        branded_package_name="test_package.json"
    )
    
    # Check files exist
    assert (out_dir / "audit.json").exists()
    assert (out_dir / "exception_queue.json").exists()
    assert (out_dir / "test_package.json").exists()
    
    # Verify exception queue
    exceptions = json.loads((out_dir / "exception_queue.json").read_text(encoding="utf-8"))
    assert len(exceptions) == 1
    assert exceptions[0]["record_id"] == "REC-2"
    assert "AGENT_HALLUCINATION" in exceptions[0]["reason_codes"]
    
    # Verify audit.json schema compliance
    # We validate only the "trace_span" definitions as the schema doesn't define the top level object itself
    # but we will validate what it has defined.
    # We can validate trace spans against the $defs/trace_span schema
    audit_data = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
    
    assert audit_data["cost"] == 0.01
    assert audit_data["amendment"]["role"] == "risk_officer"
    
    trace_span_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": audit_schema["$defs"],
        "$ref": "#/$defs/trace_span"
    }
    
    # Validate the generated spans
    for span in audit_data["agent_trace"]["REC-1"]:
        jsonschema.validate(instance=span, schema=trace_span_schema)
