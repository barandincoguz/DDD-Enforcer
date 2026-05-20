"""Replay test: historical Mar-13 Specialist intermediate → deterministic Synthesizer.

Loads core/intermediate/20260313_221928_3_specialist.json (a pre-D1-patch dump
that lacks description/confidence/justification/evidence_sentence_indices) and
runs it through the new deterministic Synthesizer via synthesize_domain_model().

An adapter (_legacy_to_typed_analysis) fills the missing D1 fields with safe
stub values so Pydantic strict validation passes. The tests assert:
  - Named entities (User, Product) are preserved in their expected contexts.
  - D6 entity-count invariant holds (sum of input entity counts == output count).

These tests catch FM-LOST-style regressions without spending live LLM tokens.
"""

import json
import pathlib
import pytest

from core.pipeline_contracts import ContextHypothesis, SpecialistAnalysis
from core.synthesizer import synthesize_domain_model
from core.schemas import Entity, ValueObject, Aggregate, DomainEvent

# ---------------------------------------------------------------------------
# Path to historical dump
# ---------------------------------------------------------------------------

_DUMP_PATH = (
    pathlib.Path(__file__).parent.parent
    / "core"
    / "intermediate"
    / "20260313_221928_3_specialist.json"
)


# ---------------------------------------------------------------------------
# Adapter: pre-D1 dict → typed SpecialistAnalysis
# ---------------------------------------------------------------------------

def _adapt_entity(raw: dict) -> Entity:
    """Convert a pre-D1 entity dict (name + attributes only) to a typed Entity."""
    return Entity(
        name=raw["name"],
        description=raw.get("description", f"Stub description for {raw['name']}"),
        confidence=raw.get("confidence", 0.5),
        justification=raw.get("justification", "Adapted from historical dump; no LLM justification available."),
        evidence_sentence_indices=raw.get("evidence_sentence_indices", [0]),
    )


def _adapt_value_object(raw: dict) -> ValueObject:
    return ValueObject(
        name=raw["name"],
        attributes=raw.get("attributes", []),
        description=raw.get("description", None),
    )


def _adapt_aggregate(raw: dict) -> Aggregate:
    return Aggregate(
        name=raw["name"],
        description=raw.get("description", f"Stub description for {raw['name']}"),
        members=raw.get("members", []),
    )


def _adapt_domain_event(raw: dict) -> DomainEvent:
    return DomainEvent(
        name=raw["name"],
        description=raw.get("description", None),
    )


def _legacy_to_typed_analysis(item: dict) -> SpecialistAnalysis:
    """Convert one analyses[] entry from the historical dump into a SpecialistAnalysis."""
    context_name = item["context"]
    analysis = item.get("analysis", {})

    context = ContextHypothesis(context_name=context_name)

    entities = [_adapt_entity(e) for e in analysis.get("entities", [])]
    value_objects = [_adapt_value_object(v) for v in analysis.get("value_objects", [])]
    aggregates = [_adapt_aggregate(a) for a in analysis.get("aggregates", [])]
    domain_events = [_adapt_domain_event(ev) for ev in analysis.get("domain_events", [])]

    return SpecialistAnalysis(
        context=context,
        entities=entities,
        value_objects=value_objects,
        aggregates=aggregates,
        domain_events=domain_events,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def historical_analyses():
    """Load and adapt the Mar-13 historical Specialist dump."""
    if not _DUMP_PATH.exists():
        pytest.skip(f"Historical dump not found: {_DUMP_PATH}")
    raw = json.loads(_DUMP_PATH.read_text(encoding="utf-8"))
    return [_legacy_to_typed_analysis(item) for item in raw.get("analyses", [])]


@pytest.fixture(scope="module")
def synthesized_model(historical_analyses):
    """Run the deterministic Synthesizer on the adapted analyses (no LLM calls)."""
    return synthesize_domain_model(
        historical_analyses,
        llm_client=None,
        project_name="HistoricalReplayTest",
        skip_enrich=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_replay_mar13_preserves_user_and_product_entities(synthesized_model):
    """User entity in UserManagement and Product in ProductCatalog must survive the merge."""
    contexts_by_name = {bc.context_name: bc for bc in synthesized_model.bounded_contexts}

    assert "UserManagement" in contexts_by_name, (
        "BoundedContext 'UserManagement' missing from synthesized model"
    )
    assert "ProductCatalog" in contexts_by_name, (
        "BoundedContext 'ProductCatalog' missing from synthesized model"
    )

    user_mgmt_entities = {
        e.name for e in contexts_by_name["UserManagement"].ubiquitous_language.entities
    }
    assert "User" in user_mgmt_entities, (
        f"Entity 'User' missing from UserManagement; found: {user_mgmt_entities}"
    )

    product_catalog_entities = {
        e.name for e in contexts_by_name["ProductCatalog"].ubiquitous_language.entities
    }
    assert "Product" in product_catalog_entities, (
        f"Entity 'Product' missing from ProductCatalog; found: {product_catalog_entities}"
    )


def test_replay_total_entity_count_matches_input(historical_analyses, synthesized_model):
    """D6 invariant: total entity count in output equals sum of input entity counts."""
    input_count = sum(len(a.entities) for a in historical_analyses)
    output_count = sum(
        len(bc.ubiquitous_language.entities)
        for bc in synthesized_model.bounded_contexts
    )
    assert output_count == input_count, (
        f"D6 entity-count invariant violated: input={input_count}, output={output_count}"
    )
