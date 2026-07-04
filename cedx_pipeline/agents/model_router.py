"""Model Gateway and complexity-based Model Router.

The :class:`ModelGateway` abstracts model inference behind a uniform
interface.  In Phase 2 it **simulates** responses — Phase 3 will swap in
real API clients (OpenAI, Gemini, Anthropic) without touching agent code.

The :class:`ModelRouter` inspects record complexity (anomaly count, amount
magnitude, past failures) to select the most cost-effective model that can
handle the record's risk profile.

Routing table (no hardcoded dollar thresholds):

    ============================================  ====================
    Condition                                     Model Selected
    ============================================  ====================
    Clean record, no anomalies, no prior fails    gemini-1.5-flash
    1–2 anomalies OR moderate complexity          gpt-4o-mini
    3+ anomalies OR INJECTION OR prior failures   gemini-1.5-pro
    Retry after verifier rejection                gpt-4o
    ============================================  ====================
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from cedx_pipeline.agents.contracts import ModelId
from cedx_pipeline.agents.transcript import TranscriptBundle
from cedx_pipeline.config import MODEL_COST_PER_1K_TOKENS, REPLAY_LLM, TRANSCRIPTS_DIR
from cedx_pipeline.detectors.models import AnomalyType
from cedx_pipeline.errors import MissingTranscriptError
from cedx_pipeline.intake.registry import Record

logger = logging.getLogger(__name__)


# ── Inference Result ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Immutable result from a model inference call.

    Attributes:
        output:     The model's text output (simulated in Phase 2).
        tokens_in:  Input token count.
        tokens_out: Output token count.
        cost_usd:   Inference cost in USD.
        latency_ms: Wall-clock latency in milliseconds.
    """

    output: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float


# ── Model Gateway (Simulated) ───────────────────────────────────────────────


class ModelGateway:
    """Simulated model inference gateway.

    Produces deterministic, structurally valid outputs based on the input
    record.  Real API integration is a Phase 3 concern — the interface
    contract (``infer(model, prompt, record)``) will remain identical.
    """

    def infer(
        self,
        agent: str,
        model: ModelId,
        prompt: str,
        record: Record,
        simulated_response: str | None = None,
    ) -> InferenceResult:
        """Run (simulated) inference and return the result.

        The simulation:
            * Generates a deterministic assembly draft from the record.
            * Computes token counts from prompt/output lengths.
            * Derives cost from the model's per-1K-token rate.
            * Adds a small simulated latency based on model tier.

        Args:
            agent:  The name of the agent calling inference.
            model:  The model to invoke.
            prompt: The formatted prompt string.
            record: The source record (used for deterministic output).
            simulated_response: If provided, bypasses the internal simulation
                                (useful for verifier verdicts).

        Returns:
            An :class:`InferenceResult` with simulated metrics.
        """
        start = time.monotonic()

        if REPLAY_LLM:
            return self._replay_inference(agent, model, prompt)

        # ── Deterministic simulated output ───────────────────────────────
        if simulated_response is not None:
            output_text = simulated_response
        else:
            assembly = self._build_simulated_assembly(record)
            output_text = json.dumps(assembly, indent=2)

        # ── Token estimation (1 token ≈ 4 chars) ────────────────────────
        tokens_in = max(1, len(prompt) // 4)
        tokens_out = max(1, len(output_text) // 4)

        # ── Cost calculation ─────────────────────────────────────────────
        rate = MODEL_COST_PER_1K_TOKENS.get(model.value, 0.005)
        cost_usd = rate * (tokens_in + tokens_out) / 1000.0

        elapsed_ms = (time.monotonic() - start) * 1000.0

        result = InferenceResult(
            output=output_text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(elapsed_ms, 2),
        )

        # ── Save Transcript ──────────────────────────────────────────────
        response_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        bundle = TranscriptBundle(
            request=prompt,
            raw_response=output_text,
            response_hash=response_hash,
            model=model.value,
            prompt_version="1.0.0",
            agent=agent,
        )
        bundle.save(TRANSCRIPTS_DIR)

        return result

    def _replay_inference(
        self, agent: str, model: ModelId, prompt: str
    ) -> InferenceResult:
        """Scan transcripts and return a matched InferenceResult."""
        if not TRANSCRIPTS_DIR.exists():
            raise MissingTranscriptError(
                f"Transcripts directory {TRANSCRIPTS_DIR} does not exist."
            )

        for path in TRANSCRIPTS_DIR.glob("*.json"):
            try:
                bundle = TranscriptBundle.load(path)
                if bundle.request == prompt and bundle.agent == agent:
                    tokens_in = max(1, len(prompt) // 4)
                    tokens_out = max(1, len(bundle.raw_response) // 4)
                    rate = MODEL_COST_PER_1K_TOKENS.get(bundle.model, 0.005)
                    cost_usd = rate * (tokens_in + tokens_out) / 1000.0
                    return InferenceResult(
                        output=bundle.raw_response,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=round(cost_usd, 6),
                        latency_ms=10.0,
                    )
            except Exception as e:
                logger.warning("Failed to load transcript %s: %s", path, e)

        raise MissingTranscriptError(
            f"No transcript found for agent '{agent}' matching the prompt."
        )

    @staticmethod
    def _build_simulated_assembly(record: Record) -> dict[str, Any]:
        """Build a deterministic assembly payload from *record* fields."""
        return {
            "record_id": record.id,
            "owner": record.owner,
            "deadline": record.deadline,
            "amount": str(record.amount) if record.amount is not None else None,
            "summary": f"Processed record {record.id} from {record.source_format} source.",
            "source_hash": record.source_version_hash[:16],
            "assembly_version": "1.0.0",
        }


# ── Model Router ────────────────────────────────────────────────────────────


class ModelRouter:
    """Complexity-based model selection.

    Inspects the record's anomaly profile and past failure count to route
    to the most cost-effective model that can handle the risk level.
    """

    def select_model(
        self,
        record: Record,
        anomaly_types: list[str],
        past_failures: int,
    ) -> ModelId:
        """Choose the appropriate model for *record*.

        Args:
            record:         The record to process.
            anomaly_types:  List of anomaly type strings from Phase 1.
            past_failures:  Number of prior verifier rejections.

        Returns:
            The selected :class:`ModelId`.
        """
        # ── Retry escalation (highest priority) ─────────────────────────
        if past_failures > 0:
            selected = ModelId.GPT4O
            logger.info(
                "Router: record=%s past_failures=%d -> %s (retry escalation)",
                record.id,
                past_failures,
                selected.value,
            )
            return selected

        # ── Injection or high anomaly count → premium ───────────────────
        has_injection = AnomalyType.INJECTION_BLOCKED.value in anomaly_types
        anomaly_count = len(anomaly_types)

        if has_injection or anomaly_count >= 3:
            selected = ModelId.GEMINI_PRO
            logger.info(
                "Router: record=%s anomalies=%d injection=%s -> %s (premium)",
                record.id,
                anomaly_count,
                has_injection,
                selected.value,
            )
            return selected

        # ── Moderate complexity → mid-tier ──────────────────────────────
        if anomaly_count >= 1:
            selected = ModelId.GPT4O_MINI
            logger.info(
                "Router: record=%s anomalies=%d -> %s (mid-tier)",
                record.id,
                anomaly_count,
                selected.value,
            )
            return selected

        # ── Clean record → cheapest ─────────────────────────────────────
        selected = ModelId.GEMINI_FLASH
        logger.info(
            "Router: record=%s anomalies=0 -> %s (cost-effective)",
            record.id,
            selected.value,
        )
        return selected
