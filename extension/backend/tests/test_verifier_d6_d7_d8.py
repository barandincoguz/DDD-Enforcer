"""D6/D7/D8 hard-fail invariants on the deterministic Synthesizer output.

D6 — entity count preservation: sum(analyses entities) == sum(model entities).
D7 — entity name traceability: every model entity traces by name to an analysis.
D8 — aggregate member referential integrity: aggregate.members ⊆ context entities.

These are HARD-FAIL invariants (code-bug detectors), not Refiner-loopable.
"""

import pytest
from unittest.mock import patch

from core.verifier.checks_semantic_d6_d7_d8 import (
    check_d6_entity_count_preservation,
    check_d7_entity_name_traceability,
    check_d8_aggregate_member_referential_integrity,
)
from core.pipeline_contracts import SpecialistAnalysis, ContextHypothesis
from core.schemas import Entity, Aggregate, DomainModel, BoundedContext, UbiquitousLanguage
from core.synthesizer.errors import SynthesizerInvariantError
from core.synthesizer import synthesize_domain_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(name: str) -> Entity:
    return Entity(
        name=name,
        description=f"{name} entity",
        confidence=0.9,
        justification=f"cited in sentences about {name}",
        evidence_sentence_indices=[1],
    )


def _make_analysis(ctx_name: str, entity_names: list[str]) -> SpecialistAnalysis:
    ctx = ContextHypothesis(context_name=ctx_name, description=f"{ctx_name} context")
    return SpecialistAnalysis(
        context=ctx,
        entities=[_make_entity(n) for n in entity_names],
    )


def _make_model(ctx_entity_map: dict[str, list[str]]) -> DomainModel:
    """Build a minimal DomainModel from a {context_name: [entity_names]} dict."""
    from core.synthesizer.metadata import build_default_metadata, build_default_global_rules

    bcs = []
    for ctx_name, entity_names in ctx_entity_map.items():
        ul = UbiquitousLanguage(
            entities=[_make_entity(n) for n in entity_names],
            value_objects=None,
            domain_events=None,
        )
        bcs.append(BoundedContext(
            context_name=ctx_name,
            description=f"{ctx_name} context",
            ubiquitous_language=ul,
        ))
    return DomainModel(
        project_name="TestModel",
        project_metadata=build_default_metadata(),
        bounded_contexts=bcs,
        global_rules=build_default_global_rules(),
    )


# ---------------------------------------------------------------------------
# D6 tests
# ---------------------------------------------------------------------------

def test_d6_passes_when_counts_match():
    analyses = [
        _make_analysis("Sales", ["Order", "Customer"]),
        _make_analysis("Inventory", ["Product"]),
    ]
    model = _make_model({"Sales": ["Order", "Customer"], "Inventory": ["Product"]})
    issues = check_d6_entity_count_preservation(analyses, model)
    assert issues == []


def test_d6_fails_when_synthesizer_drops_entities():
    analyses = [
        _make_analysis("Sales", ["Order", "Customer"]),
        _make_analysis("Inventory", ["Product"]),
    ]
    # Model is missing "Customer" — simulates a buggy merge
    model = _make_model({"Sales": ["Order"], "Inventory": ["Product"]})
    issues = check_d6_entity_count_preservation(analyses, model)
    assert len(issues) == 1
    assert issues[0].check_id == "D6"
    assert issues[0].severity == "ERROR"


# ---------------------------------------------------------------------------
# D7 tests
# ---------------------------------------------------------------------------

def test_d7_passes_when_every_model_entity_traces_to_analyses():
    analyses = [
        _make_analysis("Sales", ["Order", "Customer"]),
    ]
    model = _make_model({"Sales": ["Order", "Customer"]})
    issues = check_d7_entity_name_traceability(analyses, model)
    assert issues == []


def test_d7_fails_when_model_has_fabricated_entity():
    analyses = [
        _make_analysis("Sales", ["Order"]),
    ]
    # "Invoice" was not in any analysis — fabrication
    model = _make_model({"Sales": ["Order", "Invoice"]})
    issues = check_d7_entity_name_traceability(analyses, model)
    assert len(issues) == 1
    assert issues[0].check_id == "D7"
    assert issues[0].severity == "ERROR"
    assert "Invoice" in issues[0].message
    assert "Sales" in issues[0].target


def test_d7_passes_with_case_insensitive_name_match():
    analyses = [
        _make_analysis("Sales", ["Order"]),
    ]
    # Model uses "order" (lowercase) — should still pass
    model = _make_model({"Sales": ["order"]})
    issues = check_d7_entity_name_traceability(analyses, model)
    assert issues == []


# ---------------------------------------------------------------------------
# D8 tests
# ---------------------------------------------------------------------------

def _make_model_with_aggregate(
    ctx_name: str,
    entity_names: list[str],
    agg_name: str,
    members: list[str],
) -> DomainModel:
    from core.synthesizer.metadata import build_default_metadata, build_default_global_rules

    agg = Aggregate(
        name=agg_name,
        description="consistency boundary",
        members=members,
    )
    ul = UbiquitousLanguage(
        entities=[_make_entity(n) for n in entity_names],
        value_objects=None,
        aggregates=[agg],
        domain_events=None,
    )
    bc = BoundedContext(
        context_name=ctx_name,
        description=f"{ctx_name} context",
        ubiquitous_language=ul,
    )
    return DomainModel(
        project_name="TestModel",
        project_metadata=build_default_metadata(),
        bounded_contexts=[bc],
        global_rules=build_default_global_rules(),
    )


def test_d8_passes_when_all_aggregate_members_exist():
    model = _make_model_with_aggregate(
        ctx_name="Sales",
        entity_names=["Order", "Customer"],
        agg_name="OrderRoot",
        members=["Order", "Customer"],
    )
    issues = check_d8_aggregate_member_referential_integrity(model)
    assert issues == []


def test_d8_fails_when_aggregate_references_nonexistent_entity():
    model = _make_model_with_aggregate(
        ctx_name="Sales",
        entity_names=["Order"],
        agg_name="OrderRoot",
        members=["Order", "Payment"],  # Payment doesn't exist
    )
    issues = check_d8_aggregate_member_referential_integrity(model)
    assert len(issues) == 1
    assert issues[0].check_id == "D8"
    assert issues[0].severity == "ERROR"
    assert "Payment" in issues[0].message
    assert "Sales" in issues[0].target


def test_d8_passes_with_case_insensitive_member_match():
    """D8 must match entity names case-insensitively, mirroring D7."""
    model = _make_model_with_aggregate(
        ctx_name="Sales",
        entity_names=["Order"],
        agg_name="OrderRoot",
        members=["order"],  # lowercase — entity is "Order"
    )
    issues = check_d8_aggregate_member_referential_integrity(model)
    assert issues == []


# ---------------------------------------------------------------------------
# Integration: synthesize_domain_model raises SynthesizerInvariantError
# ---------------------------------------------------------------------------

def test_synthesize_raises_on_d6_failure():
    """Monkeypatch build_deterministic_skeleton to inject a D6 bug, then
    confirm that synthesize_domain_model raises SynthesizerInvariantError."""
    from core.synthesizer.metadata import build_default_metadata, build_default_global_rules

    analyses = [
        _make_analysis("Sales", ["Order", "Customer"]),
    ]

    # Buggy skeleton: drops Customer
    buggy_ul = UbiquitousLanguage(
        entities=[_make_entity("Order")],
        value_objects=None,
        domain_events=None,
    )
    buggy_bc = BoundedContext(
        context_name="Sales",
        description="Sales context",
        ubiquitous_language=buggy_ul,
    )
    buggy_model = DomainModel(
        project_name="TestModel",
        project_metadata=build_default_metadata(),
        bounded_contexts=[buggy_bc],
        global_rules=build_default_global_rules(),
    )

    with patch(
        "core.synthesizer.build_deterministic_skeleton",
        return_value=buggy_model,
    ):
        with pytest.raises(SynthesizerInvariantError) as exc_info:
            synthesize_domain_model(
                analyses,
                llm_client=None,
                skip_enrich=True,
            )

    err = exc_info.value
    assert "D6" in err.check_id
    assert err.details  # non-empty list of issue dicts
