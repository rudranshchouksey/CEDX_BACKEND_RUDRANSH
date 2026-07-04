"""Tests for the Observability Backbone — TraceSpan and SpanCollector.

Validates span schema compliance with audit.schema.json, per-record
grouping, cost aggregation, and the summary report.
"""

from __future__ import annotations

import pytest

from cedx_pipeline.agents.contracts import VerifierVerdict
from cedx_pipeline.agents.trace import SpanCollector, TraceSpan


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_span(
    record_id: str = "REC-001",
    agent: str = "worker",
    cost: float = 0.005,
    verdict: VerifierVerdict | None = None,
) -> TraceSpan:
    return TraceSpan(
        record_id=record_id,
        agent=agent,
        model="gemini-1.5-flash",
        prompt_version="1.0.0",
        tokens_in=100,
        tokens_out=50,
        cost_usd=cost,
        latency_ms=12.5,
        retries=0,
        status="success",
        verdict=verdict,
    )


# ── TraceSpan Tests ──────────────────────────────────────────────────────────


class TestTraceSpan:
    """Tests for the TraceSpan data model."""

    def test_schema_fields_present(self) -> None:
        """All audit.schema.json required fields must be present."""
        span = _make_span()
        d = span.to_dict()

        required_fields = {
            "agent", "model", "prompt_version", "tokens_in",
            "tokens_out", "cost_usd", "latency_ms", "retries", "status",
        }
        assert required_fields.issubset(d.keys())

    def test_verdict_none_by_default(self) -> None:
        span = _make_span(agent="worker")
        assert span.verdict is None

    def test_verdict_serialises_as_string(self) -> None:
        span = _make_span(agent="verifier", verdict=VerifierVerdict.PASS)
        d = span.to_dict()
        assert d["verdict"] == "pass"

    def test_frozen(self) -> None:
        span = _make_span()
        with pytest.raises(AttributeError):
            span.status = "tampered"  # type: ignore[misc]


# ── SpanCollector Tests ──────────────────────────────────────────────────────


class TestSpanCollector:
    """Tests for the SpanCollector aggregation."""

    def test_record_and_retrieve(self) -> None:
        collector = SpanCollector()
        span = _make_span(record_id="C-001")
        collector.record(span)

        retrieved = collector.spans_for_record("C-001")
        assert len(retrieved) == 1
        assert retrieved[0] is span

    def test_per_record_grouping(self) -> None:
        collector = SpanCollector()
        collector.record(_make_span(record_id="A"))
        collector.record(_make_span(record_id="A"))
        collector.record(_make_span(record_id="B"))

        assert len(collector.spans_for_record("A")) == 2
        assert len(collector.spans_for_record("B")) == 1
        assert len(collector.spans_for_record("C")) == 0

    def test_all_spans(self) -> None:
        collector = SpanCollector()
        for i in range(5):
            collector.record(_make_span(record_id=f"R-{i}"))

        assert len(collector.all_spans()) == 5

    def test_total_cost_aggregation(self) -> None:
        collector = SpanCollector()
        collector.record(_make_span(cost=0.010))
        collector.record(_make_span(cost=0.025))
        collector.record(_make_span(cost=0.005))

        assert abs(collector.total_cost() - 0.040) < 1e-9

    def test_cost_for_record(self) -> None:
        collector = SpanCollector()
        collector.record(_make_span(record_id="X", cost=0.01))
        collector.record(_make_span(record_id="X", cost=0.02))
        collector.record(_make_span(record_id="Y", cost=0.05))

        assert abs(collector.cost_for_record("X") - 0.03) < 1e-9
        assert abs(collector.cost_for_record("Y") - 0.05) < 1e-9

    def test_count(self) -> None:
        collector = SpanCollector()
        assert collector.count() == 0
        collector.record(_make_span())
        collector.record(_make_span())
        assert collector.count() == 2

    def test_summary_structure(self) -> None:
        collector = SpanCollector()
        collector.record(_make_span(agent="worker", cost=0.01))
        collector.record(
            _make_span(agent="verifier", cost=0.02, verdict=VerifierVerdict.PASS)
        )

        summary = collector.summary()

        assert summary["total_spans"] == 2
        assert abs(summary["total_cost_usd"] - 0.03) < 1e-9
        assert "worker" in summary["agents_active"]
        assert "verifier" in summary["agents_active"]
        assert summary["total_tokens_in"] == 200
        assert summary["total_tokens_out"] == 100

    def test_empty_collector(self) -> None:
        collector = SpanCollector()
        assert collector.total_cost() == 0.0
        assert collector.all_spans() == []
        assert collector.count() == 0
        summary = collector.summary()
        assert summary["total_spans"] == 0
