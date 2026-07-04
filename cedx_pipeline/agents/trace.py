"""Observability backbone — span collector for agent tracing.

Every agent interaction must append a :class:`TraceSpan` to the
:class:`SpanCollector`.  The span schema precisely matches the
``trace_span`` definition in ``audit.schema.json``.

The verifier agent's span is the **only** span that populates the
``verdict`` field.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from cedx_pipeline.agents.contracts import VerifierVerdict

logger = logging.getLogger(__name__)


# ── Trace Span ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TraceSpan:
    """A single observability span.  Fields match ``audit.schema.json``.

    Attributes:
        record_id:      ID of the record this span relates to.
        agent:          Name of the agent that produced this span.
        model:          Model identifier used for inference.
        prompt_version: Semantic version of the prompt template.
        tokens_in:      Number of input tokens.
        tokens_out:     Number of output tokens.
        cost_usd:       Inference cost in USD.
        latency_ms:     Wall-clock latency in milliseconds.
        retries:        Number of retries attempted before this span.
        status:         Outcome status (``"success"``, ``"error"``,
                        ``"timeout"``).
        verdict:        Only set by the verifier agent.
    """

    record_id: str
    agent: str
    model: str
    prompt_version: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    retries: int
    status: str
    verdict: VerifierVerdict | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict matching the JSON schema."""
        d = asdict(self)
        if d["verdict"] is not None:
            d["verdict"] = d["verdict"].value if hasattr(d["verdict"], "value") else d["verdict"]
        return d


# ── Span Collector ──────────────────────────────────────────────────────────


class SpanCollector:
    """Thread-safe, append-only span collector.

    Spans are grouped by ``record_id`` for per-record cost aggregation
    and audit trail reconstruction.

    Usage::

        collector = SpanCollector()
        collector.record(span)
        print(collector.total_cost())
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: dict[str, list[TraceSpan]] = {}

    def record(self, span: TraceSpan) -> None:
        """Append *span* to the collector.

        This method is safe to call from any thread.
        """
        with self._lock:
            if span.record_id not in self._spans:
                self._spans[span.record_id] = []
            self._spans[span.record_id].append(span)

        logger.debug(
            "Span recorded: agent=%s model=%s record=%s cost=$%.6f",
            span.agent,
            span.model,
            span.record_id,
            span.cost_usd,
        )

    def spans_for_record(self, record_id: str) -> list[TraceSpan]:
        """Return all spans for a given *record_id*.

        Returns a shallow copy — safe to iterate without holding the lock.
        """
        with self._lock:
            return list(self._spans.get(record_id, []))

    def all_spans(self) -> list[TraceSpan]:
        """Return every span across all records as a flat list."""
        with self._lock:
            return [
                span
                for spans in self._spans.values()
                for span in spans
            ]

    def total_cost(self) -> float:
        """Return the cumulative cost across all spans."""
        with self._lock:
            return sum(
                span.cost_usd
                for spans in self._spans.values()
                for span in spans
            )

    def cost_for_record(self, record_id: str) -> float:
        """Return the cumulative cost for a specific record."""
        with self._lock:
            return sum(
                span.cost_usd
                for span in self._spans.get(record_id, [])
            )

    def count(self) -> int:
        """Return the total number of spans collected."""
        with self._lock:
            return sum(len(spans) for spans in self._spans.values())

    def summary(self) -> dict[str, Any]:
        """Return an aggregate summary dict for reporting."""
        all_spans = self.all_spans()
        return {
            "total_spans": len(all_spans),
            "total_cost_usd": round(self.total_cost(), 6),
            "total_tokens_in": sum(s.tokens_in for s in all_spans),
            "total_tokens_out": sum(s.tokens_out for s in all_spans),
            "records_traced": len(self._spans),
            "agents_active": sorted({s.agent for s in all_spans}),
            "models_used": sorted({s.model for s in all_spans}),
        }
