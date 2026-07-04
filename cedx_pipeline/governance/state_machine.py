"""State-Machine Operator Review Surface.

Builds an explicit, state-checked review engine executing the lifecycle:
draft -> in_review -> changes_requested -> approved -> delivered.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from cedx_pipeline.amendment import Amendment
from cedx_pipeline.intake.registry import Record

# Hardcoded delivery blocking reason codes (Class-A or Agent-Failure).
DELIVERY_BLOCK_CODES = frozenset({
    "AGENT_HALLUCINATION",
    "AGENT_MALFORMED",
    "CLASS_A_RISK",
})


class ReviewState(str, Enum):
    """Lifecycle states for the review engine."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    DELIVERED = "delivered"


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""


class DeliveryBlockedError(Exception):
    """Raised when the delivery boundary is blocked by reason codes."""


class LiveAmendmentGateError(Exception):
    """Raised when a record fails the live amendment gate check."""


class ReviewStateMachine:
    """Manages explicit state transitions for records."""

    def __init__(
        self,
        record: Record,
        amendment: Amendment,
        initial_state: ReviewState = ReviewState.DRAFT,
        audit_engine: Any | None = None,
    ):
        self._record = record
        self._amendment = amendment
        self._state = initial_state
        self._approval_trail: list[dict[str, Any]] = []
        self._reason_codes: list[str] = []
        self._audit_engine = audit_engine

    @property
    def state(self) -> ReviewState:
        """Current state of the review machine."""
        return self._state

    def _log_transition(self, old_state: ReviewState, new_state: ReviewState) -> None:
        if self._audit_engine:
            self._audit_engine.append_event(
                "STATE_TRANSITION",
                {
                    "record_id": self._record.id,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                },
            )

    def add_signature(self, role: str, actor: str) -> None:
        """Add a validation signature to the approval trail."""
        self._approval_trail.append({"role": role, "actor": actor})

    def add_reason_code(self, code: str) -> None:
        """Add a reason code to the record."""
        self._reason_codes.append(code)

    def transition_to_in_review(self) -> None:
        """Transition from draft to in_review."""
        if self._state not in {ReviewState.DRAFT, ReviewState.CHANGES_REQUESTED}:
            raise InvalidTransitionError(f"Cannot transition from {self._state} to IN_REVIEW")
        old = self._state
        self._state = ReviewState.IN_REVIEW
        self._log_transition(old, self._state)

    def transition_to_changes_requested(self) -> None:
        """Transition from in_review to changes_requested."""
        if self._state != ReviewState.IN_REVIEW:
            raise InvalidTransitionError(f"Cannot transition from {self._state} to CHANGES_REQUESTED")
        old = self._state
        self._state = ReviewState.CHANGES_REQUESTED
        self._log_transition(old, self._state)

    def transition_to_approved(self) -> None:
        """Transition from in_review to approved."""
        if self._state != ReviewState.IN_REVIEW:
            raise InvalidTransitionError(f"Cannot transition from {self._state} to APPROVED")
        old = self._state
        self._state = ReviewState.APPROVED
        self._log_transition(old, self._state)

    def transition_to_delivered(self) -> None:
        """Transition from approved to delivered.

        This checks the Live Amendment Gate and the Delivery Boundary.
        """
        if self._state != ReviewState.APPROVED:
            raise InvalidTransitionError(f"Cannot transition from {self._state} to DELIVERED")

        # 1. Delivery Boundary Check
        blocked = set(self._reason_codes).intersection(DELIVERY_BLOCK_CODES)
        if blocked:
            raise DeliveryBlockedError(f"Delivery blocked by reason codes: {blocked}")

        # 2. Live Amendment Gate
        amount = float(self._record.amount) if self._record.amount is not None else 0.0
        if amount >= self._amendment.threshold:
            required_role = self._amendment.role
            has_valid_signature = any(
                sig.get("role") == required_role for sig in self._approval_trail
            )
            if not has_valid_signature:
                raise LiveAmendmentGateError(
                    f"Amount {amount} >= threshold {self._amendment.threshold}. "
                    f"Missing signature from required role: {required_role}"
                )

        old = self._state
        self._state = ReviewState.DELIVERED
        self._log_transition(old, self._state)
