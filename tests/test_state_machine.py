import pytest

from cedx_pipeline.amendment import Amendment
from cedx_pipeline.governance.state_machine import (
    DELIVERY_BLOCK_CODES,
    DeliveryBlockedError,
    InvalidTransitionError,
    LiveAmendmentGateError,
    ReviewState,
    ReviewStateMachine,
)
from cedx_pipeline.intake.registry import Record


@pytest.fixture
def mock_record():
    return Record(
        id="REC-TEST-1",
        source_format="json",
        owner="Alice",
        deadline="2026-12-31",
        amount=1500.0,
        notes="Clean record",
        source_version_hash="abcdef",
        payload={},
    )


@pytest.fixture
def mock_amendment():
    return Amendment(case_id="CEDX-TEST", role="compliance", threshold=1000, digest="digest123")


def test_valid_transitions(mock_record, mock_amendment):
    machine = ReviewStateMachine(mock_record, mock_amendment)
    assert machine.state == ReviewState.DRAFT

    machine.transition_to_in_review()
    assert machine.state == ReviewState.IN_REVIEW

    machine.transition_to_changes_requested()
    assert machine.state == ReviewState.CHANGES_REQUESTED

    machine.transition_to_in_review()
    assert machine.state == ReviewState.IN_REVIEW

    machine.transition_to_approved()
    assert machine.state == ReviewState.APPROVED

    # Add valid signature because amount (1500) >= threshold (1000)
    machine.add_signature("compliance", "Alice")
    
    machine.transition_to_delivered()
    assert machine.state == ReviewState.DELIVERED


def test_invalid_transitions(mock_record, mock_amendment):
    machine = ReviewStateMachine(mock_record, mock_amendment)

    with pytest.raises(InvalidTransitionError):
        machine.transition_to_approved()

    with pytest.raises(InvalidTransitionError):
        machine.transition_to_delivered()


def test_live_amendment_gate_blocks(mock_record, mock_amendment):
    machine = ReviewStateMachine(mock_record, mock_amendment)
    machine.transition_to_in_review()
    machine.transition_to_approved()

    # Amount is 1500, threshold is 1000. Missing signature for 'compliance'.
    with pytest.raises(LiveAmendmentGateError, match="Missing signature from required role: compliance"):
        machine.transition_to_delivered()


def test_live_amendment_gate_passes_under_threshold(mock_record, mock_amendment):
    record_under = Record(
        id="REC-TEST-2",
        source_format="json",
        owner="Alice",
        deadline="2026-12-31",
        amount=500.0,  # Under 1000
        notes="Clean record",
        source_version_hash="abcdef",
        payload={},
    )
    machine = ReviewStateMachine(record_under, mock_amendment)
    machine.transition_to_in_review()
    machine.transition_to_approved()

    # Should pass without signature because amount < threshold
    machine.transition_to_delivered()
    assert machine.state == ReviewState.DELIVERED


def test_delivery_boundary_blocks_reason_codes(mock_record, mock_amendment):
    machine = ReviewStateMachine(mock_record, mock_amendment)
    machine.transition_to_in_review()
    machine.transition_to_approved()

    # Even with a valid signature...
    machine.add_signature("compliance", "Alice")

    # If it has a blocked reason code, it should fail.
    machine.add_reason_code("AGENT_HALLUCINATION")

    with pytest.raises(DeliveryBlockedError, match="Delivery blocked by reason codes"):
        machine.transition_to_delivered()
