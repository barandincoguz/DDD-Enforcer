"""Schema-strict tests for Entity, ValueObject, Aggregate, BoundedContext.

Phase A adds: required Entity.confidence, required Entity.justification,
required Aggregate.members, optional BoundedContext.supporting_sentence_ids,
optional BoundedContext.business_rules.

Phase D1 tightens Entity.evidence_sentence_indices from optional
(default empty list) to required (min_length=1).
"""

import pytest
from pydantic import ValidationError
from core.schemas import (
    Entity,
    Aggregate,
    BoundedContext,
    DomainModel,
    UbiquitousLanguage,
    ProjectMetadata,
)


def test_entity_requires_confidence_field():
    """confidence must be supplied; the old default 0.5 is gone."""
    with pytest.raises(ValidationError):
        Entity(name="Customer", description="Buys things")


def test_entity_requires_justification_field():
    with pytest.raises(ValidationError):
        Entity(name="Customer", description="Buys things", confidence=0.8)


def test_entity_rejects_missing_evidence_sentence_indices_in_phase_d():
    """Phase D1: evidence_sentence_indices is required (min_length=1)."""
    with pytest.raises(ValidationError):
        Entity(
            name="Customer",
            description="A buyer",
            confidence=0.9,
            justification="Mentioned in 4 SRS sentences",
        )


def test_entity_accepts_phase_d_with_evidence():
    e = Entity(
        name="Customer",
        description="A buyer",
        confidence=0.9,
        justification="Mentioned in 4 SRS sentences",
        evidence_sentence_indices=[2, 5, 9],
    )
    assert e.evidence_sentence_indices == [2, 5, 9]


def test_aggregate_requires_members_field():
    with pytest.raises(ValidationError):
        Aggregate(name="Order", description="Order consistency boundary")


def test_aggregate_accepts_explicit_members():
    a = Aggregate(
        name="Order",
        description="Order consistency boundary",
        members=["Order", "OrderLine"],
    )
    assert a.members == ["Order", "OrderLine"]


def test_bounded_context_accepts_business_rules():
    bc = BoundedContext(
        context_name="OrderMgmt",
        description="Manages orders",
        ubiquitous_language=UbiquitousLanguage(
            entities=[
                Entity(
                    name="Order",
                    description="An order",
                    confidence=0.95,
                    justification="Primary entity",
                    evidence_sentence_indices=[0],
                )
            ],
            value_objects=[],
            domain_events=[],
        ),
        business_rules=["Orders must have at least one item"],
    )
    assert bc.business_rules == ["Orders must have at least one item"]


def test_domain_model_rejects_empty_bounded_contexts():
    """An empty bounded_contexts list must not validate (FM-04 prep)."""
    with pytest.raises(ValidationError):
        DomainModel(
            project_name="X",
            project_metadata=ProjectMetadata(version="1.0", generated_at="2026-05-18"),
            bounded_contexts=[],
            global_rules=None,
        )
