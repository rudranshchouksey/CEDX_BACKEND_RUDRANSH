# Backend Engine Specification

This document details the decoupled Python FastAPI REST service orchestrating the multi-agent execution pipeline.

---

## Architecture & REST API Router
The backend is a standalone Python FastAPI server structured to operate entirely decoupled from the UI.

- **Lifecycle Hooks:** Uses FastAPI's `@app.on_event("startup")` and background tasks to manage continuous pipeline loops or initialize local cache states.
- **Endpoints:**
  - `GET /api/status`: Returns system telemetry, case tracking ID, Role R, Threshold T, and baseline execution latency/costs.
  - `GET /api/records`: Returns the array of current payloads processed by the agent pipeline.
  - `POST /api/review`: Secure mutation endpoint. Receives state updates from the frontend and enforces cryptographic server-side validation against `Role R` when required.

---

## Internal Core Agent Compilation
The multi-agent LLM topology resides inside `/src/agents/`.

- **Orchestrator (`orchestrator.py`):** The primary node. It tracks context limits, enforces token budget constraints per cycle, and determines which sub-agent to invoke.
- **Worker (`worker.py`):** Performs deterministic model routing choices (e.g., GPT-4o vs Claude 3.5) based on the complexity of the raw payload, parsing strings into structured outputs.
- **Verifier (`verifier.py`):** The autonomous safety net. It runs mathematical outlier heuristics and executes autonomous overrule loops against the Worker's output, throwing `FAIL` states for hallucinated or malformed JSON responses.

---

## Local File Management Rules
To support rigorous local automated grading constraints:

- **Input (`/seed/`):** Running local automation hooks via `make demo` or `make verify` forces the pipeline to ingest JSON payloads strictly from the local `/seed/` directory.
- **Output (`/out/`):** The orchestrator writes the finalized compliance packets and timeline execution traces cleanly to the isolated `/out/audit.json` and `/out/exception_queue.json` path.
- This ensures that graders can view local artifacts transparently without needing to query a live database.
