"""Pure-function merge of List[SpecialistAnalysis] → DomainModel skeleton.

The deterministic merge MUST:
- preserve every entity by name and attribute set
- group entities under their originating bounded context
- preserve confidence + justification + evidence_sentence_indices
- never fabricate entities not present in input
- pass aggregate.members referential check (D8)
"""

import pytest

from core.synthesizer.merge import build_deterministic_skeleton
from core.pipeline_contracts import (
    SpecialistAnalysis, ContextHypothesis,
)
from core.schemas import Entity, ValueObject, Aggregate, DomainEvent


def _make_analysis(ctx_name: str, entities: list, **extra):
    ctx = ContextHypothesis(context_name=ctx_name, description=f"{ctx_name} context")
    return SpecialistAnalysis(context=ctx, entities=entities, **extra)


def _make_entity(name: str):
    return Entity(
        name=name,
        description=f"{name} entity",
        confidence=0.9,
        justification=f"cited in sentences about {name}",
        evidence_sentence_indices=[1, 2],
    )


def test_merge_preserves_entity_count():
    analyses = [
        _make_analysis("Sales", [_make_entity("Order"), _make_entity("Customer")]),
        _make_analysis("Inventory", [_make_entity("Product")]),
    ]
    model = build_deterministic_skeleton(analyses, project_name="TestModel")
    total = sum(len(bc.ubiquitous_language.entities) for bc in model.bounded_contexts)
    assert total == 3


def test_merge_preserves_entity_name_and_fields():
    analyses = [
        _make_analysis("Sales", [_make_entity("Order")]),
    ]
    model = build_deterministic_skeleton(analyses, project_name="TestModel")
    sales = next(bc for bc in model.bounded_contexts if bc.context_name == "Sales")
    e = sales.ubiquitous_language.entities[0]
    assert e.name == "Order"
    assert e.description == "Order entity"
    assert e.confidence == 0.9
    assert e.evidence_sentence_indices == [1, 2]


def test_merge_creates_one_bounded_context_per_analysis():
    analyses = [
        _make_analysis("Sales", []),
        _make_analysis("Inventory", []),
        _make_analysis("Customer", []),
    ]
    model = build_deterministic_skeleton(analyses, project_name="TestModel")
    assert len(model.bounded_contexts) == 3
    names = {bc.context_name for bc in model.bounded_contexts}
    assert names == {"Sales", "Inventory", "Customer"}


def test_merge_carries_value_objects_and_aggregates():
    vo = ValueObject(name="Money", attributes=["amount", "currency"], description="x")
    agg = Aggregate(name="OrderRoot", description="Order consistency", members=["Order"])
    analyses = [
        _make_analysis(
            "Sales",
            [_make_entity("Order")],
            value_objects=[vo],
            aggregates=[agg],
        ),
    ]
    model = build_deterministic_skeleton(analyses, project_name="TestModel")
    sales = model.bounded_contexts[0]
    assert len(sales.ubiquitous_language.value_objects) == 1
    assert sales.ubiquitous_language.value_objects[0].name == "Money"
    assert len(sales.ubiquitous_language.aggregates) == 1
    assert sales.ubiquitous_language.aggregates[0].members == ["Order"]


def test_merge_emits_default_global_rules_and_metadata():
    analyses = [_make_analysis("Sales", [_make_entity("Order")])]
    model = build_deterministic_skeleton(analyses, project_name="TestModel")
    assert model.global_rules.naming_convention == "PascalCase"
    assert "Manager" in (model.global_rules.banned_global_terms or [])
    assert model.project_metadata.version == "1.0"


def test_merge_with_zero_analyses_raises():
    """No analyses → DomainModel._non_empty validator raises ValidationError
    because bounded_contexts must be non-empty. The Synthesizer caller is
    responsible for guarding against zero analyses before calling merge."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        build_deterministic_skeleton([], project_name="EmptyModel")


def test_merge_does_not_invoke_llm():
    """The merge module must be pure Python — no LLM calls."""
    import core.synthesizer.merge as merge_mod
    src = merge_mod.__file__
    with open(src) as f:
        text = f.read()
    assert "llm_client" not in text, "merge.py must not reference llm_client"
    assert "structured_output" not in text, "merge.py must not call LLM"
    assert "client.chat" not in text, "merge.py must not call LLM"
