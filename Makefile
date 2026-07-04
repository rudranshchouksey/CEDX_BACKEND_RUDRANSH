.PHONY: demo trace replay probe-approval probe-agent-failure probe-budget probe-append-only probe-idempotency test

demo:
	REPLAY_LLM=true python -m cedx_pipeline.main

trace:
	@if [ -z "$(ID)" ]; then echo "Error: ID is required. Usage: make trace ID=<id>"; exit 1; fi
	python scripts/trace.py --id $(ID)

replay:
	@if [ -z "$(ID)" ]; then echo "Error: ID is required. Usage: make replay ID=<id>"; exit 1; fi
	python scripts/replay.py --id $(ID)

probe-approval:
	python scripts/probe_approval.py

probe-agent-failure:
	python scripts/probe_agent_failure.py

probe-budget:
	python scripts/probe_budget.py

probe-append-only:
	python scripts/probe_append_only.py

probe-idempotency:
	python scripts/probe_idempotency.py

test:
	pytest tests/
