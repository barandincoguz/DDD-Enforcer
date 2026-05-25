import pytest
from pydantic import ValidationError
from core.schemas import ContextRelationship, ContextMap, DomainModel


def _rel(**kw):
    base = dict(context_a="Ordering", context_b="Inventory",
                relationship_type="CUSTOMER_SUPPLIER", upstream="Inventory",
                rationale="Ordering consumes stock levels from Inventory.")
    base.update(kw)
    return ContextRelationship(**base)


def test_directional_requires_upstream_member():
    r = _rel()
    assert r.upstream == "Inventory"
    with pytest.raises(ValidationError):
        _rel(upstream=None)
    with pytest.raises(ValidationError):
        _rel(upstream="Nonexistent")


def test_mutual_rejects_upstream():
    r = _rel(relationship_type="PARTNERSHIP", upstream=None)
    assert r.relationship_type == "PARTNERSHIP"
    with pytest.raises(ValidationError):
        _rel(relationship_type="PARTNERSHIP", upstream="Ordering")


def test_separate_ways_and_bbom_reject_upstream():
    assert _rel(relationship_type="SEPARATE_WAYS", upstream=None)
    assert _rel(relationship_type="BIG_BALL_OF_MUD", upstream=None)
    with pytest.raises(ValidationError):
        _rel(relationship_type="SEPARATE_WAYS", upstream="Ordering")


def test_distinct_contexts_required():
    with pytest.raises(ValidationError):
        _rel(context_a="X", context_b="X", relationship_type="PARTNERSHIP", upstream=None)


def test_context_map_defaults_and_domain_field():
    cm = ContextMap(model_id="gemini-3.1-pro-preview")
    assert cm.relationships == [] and cm.warnings == [] and cm.error is None


def test_domain_model_context_map_optional_backward_compat():
    from core.schemas import (
        BoundedContext, UbiquitousLanguage, ProjectMetadata, Entity,
    )
    m = DomainModel(
        project_name="P",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ordering",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                )],
                value_objects=None,
                domain_events=None,
            ),
        )],
        global_rules=None,
    )
    assert m.context_map is None
