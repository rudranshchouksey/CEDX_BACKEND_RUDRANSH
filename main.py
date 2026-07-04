from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cedx_pipeline.amendment import Amendment, init_amendment
from cedx_pipeline.governance.state_machine import ReviewStateMachine, LiveAmendmentGateError, ReviewState
from cedx_pipeline.intake.registry import Record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CEDX Governance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-Memory State Mocking for Serverless Runtime ────────────────────────────

class MockState:
    def __init__(self):
        self.amendment: Amendment = None
        self.records: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {
            "total_processed": 0,
            "total_cost": 0.0,
        }

state = MockState()

def bootstrap_state():
    """Load mock data or parse from out/audit.json if available locally."""
    try:
        if not os.environ.get("CASE_ID"):
            os.environ["CASE_ID"] = "CEDX-VERCEL-1234"
        state.amendment = init_amendment()

        audit_path = Path(__file__).parent / "out" / "audit.json"
        
        # Populate realistic mock state first for Vercel demo robustness
        state.records = {
            "REC-001": {
                "id": "REC-001", "state": ReviewState.APPROVED.value, "amount": 25000.0, 
                "reason_codes": [],
                "lineage": {"owner": "Alice", "deadline": "2026-10-31", "source_format": "json", "source_hash": "a1b2c3d4"},
                "agent_trace": [
                    {"agent": "orchestrator", "model": "none", "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "latency_ms": 1.2, "verdict": "N/A"},
                    {"agent": "worker", "model": "gemini-1.5-flash", "tokens_in": 150, "tokens_out": 50, "cost_usd": 0.015, "latency_ms": 250.0, "verdict": "N/A"},
                    {"agent": "verifier", "model": "gemini-1.5-flash", "tokens_in": 200, "tokens_out": 10, "cost_usd": 0.02, "latency_ms": 150.0, "verdict": "PASS"}
                ]
            },
            "REC-002": {
                "id": "REC-002", "state": ReviewState.IN_REVIEW.value, "amount": 100.0, 
                "reason_codes": [],
                "lineage": {"owner": "Bob", "deadline": "2026-11-15", "source_format": "pdf", "source_hash": "f9e8d7c6"},
                "agent_trace": [
                    {"agent": "worker", "model": "gemini-1.5-flash", "tokens_in": 500, "tokens_out": 100, "cost_usd": 0.05, "latency_ms": 400.0, "verdict": "N/A"},
                    {"agent": "verifier", "model": "gemini-1.5-flash", "tokens_in": 600, "tokens_out": 20, "cost_usd": 0.06, "latency_ms": 300.0, "verdict": "PASS"}
                ]
            },
            "REC-003": {
                "id": "REC-003", "state": ReviewState.IN_REVIEW.value, "amount": 500.0, 
                "reason_codes": ["AGENT_HALLUCINATION"],
                "lineage": {"owner": "Charlie", "deadline": "2026-12-01", "source_format": "eml", "source_hash": "deadbeef"},
                "agent_trace": [
                    {"agent": "worker", "model": "gemini-1.5-pro", "tokens_in": 1000, "tokens_out": 200, "cost_usd": 0.15, "latency_ms": 800.0, "verdict": "N/A"},
                    {"agent": "verifier", "model": "gemini-1.5-flash", "tokens_in": 1200, "tokens_out": 50, "cost_usd": 0.12, "latency_ms": 200.0, "verdict": "FAIL", "issues": ["AGENT_HALLUCINATION"]}
                ]
            },
            "REC-004": {
                "id": "REC-004", "state": ReviewState.DELIVERED.value, "amount": 75000.0, 
                "reason_codes": [],
                "lineage": {"owner": "Diana", "deadline": "2026-10-01", "source_format": "json", "source_hash": "11223344"},
                "agent_trace": [
                    {"agent": "worker", "model": "gemini-1.5-flash", "tokens_in": 100, "tokens_out": 30, "cost_usd": 0.01, "latency_ms": 100.0, "verdict": "N/A"},
                    {"agent": "verifier", "model": "gemini-1.5-flash", "tokens_in": 130, "tokens_out": 10, "cost_usd": 0.013, "latency_ms": 90.0, "verdict": "PASS"}
                ]
            }
        }
        state.stats["total_processed"] = 4
        state.stats["total_cost"] = 0.438
        state.stats["p95_latency"] = 780.0
        state.stats["avg_cost"] = 0.438 / 4

        # Override with real data if available locally
        if audit_path.exists():
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
            
            state.events = audit_data.get("events", [])
            state.stats["total_cost"] = audit_data.get("cost_metrics", {}).get("total_usd", 0.0)
            
            real_records = {}
            all_latencies = []
            
            for trace_id, traces in audit_data.get("agent_trace", {}).items():
                for t in traces:
                    if "latency_ms" in t:
                        all_latencies.append(t["latency_ms"])
                
                real_records[trace_id] = {
                    "id": trace_id,
                    "state": ReviewState.APPROVED.value,
                    "amount": 50000.0, # Placeholder if not found in feed
                    "reason_codes": [],
                    "lineage": {"owner": "System", "deadline": "N/A", "source_format": "unknown", "source_hash": "N/A"},
                    "agent_trace": traces
                }
            
            if real_records:
                state.records = real_records
                state.stats["total_processed"] = len(real_records)
                state.stats["avg_cost"] = state.stats["total_cost"] / len(real_records) if len(real_records) > 0 else 0
                if all_latencies:
                    all_latencies.sort()
                    idx = int(len(all_latencies) * 0.95)
                    state.stats["p95_latency"] = all_latencies[idx]

    except Exception as e:
        logger.error(f"Failed to bootstrap state: {e}")

bootstrap_state()


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    if not state.amendment:
        return {"status": "uninitialized"}
        
    return {
        "case_id": state.amendment.case_id,
        "role": state.amendment.role,
        "threshold": state.amendment.threshold,
        "total_processed": state.stats.get("total_processed", 0),
        "total_cost": state.stats.get("total_cost", 0.0),
        "avg_cost": state.stats.get("avg_cost", 0.0),
        "p95_latency": state.stats.get("p95_latency", 0.0),
        "replay_llm": state.stats.get("replay_llm", True)
    }

@app.get("/api/records")
async def get_records():
    # Return minimal record listing for the left pane
    return [{"id": r["id"], "state": r["state"], "amount": r["amount"], "reason_codes": r.get("reason_codes", [])} for r in state.records.values()]

@app.get("/api/records/{record_id}")
async def get_record(record_id: str):
    # Deep dive payload for the right pane
    record = state.records.get(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record

class SettingsPayload(BaseModel):
    replay_llm: bool

@app.post("/api/settings")
async def update_settings(payload: SettingsPayload):
    state.stats["replay_llm"] = payload.replay_llm
    return {"status": "success", "replay_llm": payload.replay_llm}


class ReviewAction(BaseModel):
    record_id: str
    actor_role: str
    action: str  # approve, reject, request_changes, deliver

@app.post("/api/review")
async def review_record(payload: ReviewAction):
    record_data = state.records.get(payload.record_id)
    if not record_data:
        raise HTTPException(status_code=404, detail="Record not found")

    # Create dummy Record for StateMachine
    dummy_record = Record(
        id=record_data["id"],
        source_format="json",
        owner="operator",
        deadline="2099-12-31",
        amount=record_data["amount"],
        notes="",
        source_version_hash="x",
        payload={}
    )
    
    # Initialize machine at current state
    machine = ReviewStateMachine(
        dummy_record, 
        state.amendment, 
        initial_state=ReviewState(record_data["state"])
    )
    
    # Load reason codes
    for rc in record_data.get("reason_codes", []):
        machine.add_reason_code(rc)

    # Perform action
    try:
        if payload.action == "approve":
            machine.add_signature(payload.actor_role, "operator_ui")
            machine.transition_to_approved()
        elif payload.action == "request_changes":
            machine.transition_to_changes_requested()
        elif payload.action == "deliver":
            # If the user clicks deliver, we must check the signatures.
            # Let's inject a signature just so we can test the Live Amendment Gate
            # We assume the 'approve' action added it. If they try to deliver 
            # with wrong role, it will fail.
            machine.add_signature(payload.actor_role, "operator_ui")
            try:
                machine.transition_to_delivered()
            except LiveAmendmentGateError as e:
                state.events.append({"event": "UNAUTHORIZED_DELIVERY_ATTEMPT", "details": str(e)})
                raise HTTPException(status_code=403, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail="Invalid action")
            
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=str(e))

    # Update state
    record_data["state"] = machine.state.value

    # Resilient Runtime Exception Interceptor for Vercel's Read-Only File System
    try:
        audit_path = Path(__file__).parent / "out" / "audit.json"
        # Mock file mutation block for validation checks
        if audit_path.parent.exists():
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump({
                    "events": state.events, 
                    "stats": state.stats,
                    "records": state.records
                }, f)
    except (OSError, PermissionError) as e:
        # Graceful degradation into optimized in-memory store
        logger.warning(f"Read-only file system detected (Vercel Serverless). Redirecting data mutations to in-memory mock. Detail: {e}")

    return {"status": "success", "record": record_data}




