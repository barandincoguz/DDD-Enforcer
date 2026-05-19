"""Mocked-LLM tests for narrow enrichment."""

import json
from unittest.mock import MagicMock

from core.synthesizer import synthesize_domain_model
from core.synthesizer.enrich import enrich_synonyms_and_dependencies
from core.synthesizer.merge import build_deterministic_skeleton
from core.pipeline_contracts import SpecialistAnalysis, ContextHypothesis
from core.schemas import Entity


def _e(name):
    return Entity(
        name=name, description=f"{name} entity", confidence=0.9,
        justification="cited", evidence_sentence_indices=[1],
    )


def _ctx_and_analysis(ctx_name, entities):
    ctx = ContextHypothesis(context_name=ctx_name, description=f"{ctx_name} ctx")
    return SpecialistAnalysis(context=ctx, entities=entities)


def _mock_chat_response(text):
    resp = MagicMock()
    resp.content = text
    return resp


def test_enrich_populates_synonyms_to_avoid():
    analyses = [_ctx_and_analysis("Sales", [_e("Order"), _e("Customer")])]
    skeleton = build_deterministic_skeleton(analyses, project_name="X")
    client = MagicMock()
    client.chat.return_value = _mock_chat_response(json.dumps({
        "entities": [
            {"name": "Order", "synonyms_to_avoid": ["Purchase", "Cart"]},
            {"name": "Customer", "synonyms_to_avoid": ["Client", "Buyer"]},
        ]
    }))
    result = enrich_synonyms_and_dependencies(skeleton, analyses, client)
    sales = result.bounded_contexts[0]
    assert sales.ubiquitous_language.entities[0].synonyms_to_avoid == ["Purchase", "Cart"]
    assert sales.ubiquitous_language.entities[1].synonyms_to_avoid == ["Client", "Buyer"]


def test_enrich_does_not_touch_entity_data():
    """Enrichment must NOT modify name, description, confidence,
    justification, or evidence_sentence_indices."""
    analyses = [_ctx_and_analysis("Sales", [_e("Order")])]
    skeleton = build_deterministic_skeleton(analyses, project_name="X")
    original_entity = skeleton.bounded_contexts[0].ubiquitous_language.entities[0].model_dump()

    client = MagicMock()
    client.chat.return_value = _mock_chat_response(json.dumps({"entities": []}))
    enrich_synonyms_and_dependencies(skeleton, analyses, client)

    enriched_entity = skeleton.bounded_contexts[0].ubiquitous_language.entities[0]
    for field in ("name", "description", "confidence", "justification", "evidence_sentence_indices"):
        assert getattr(enriched_entity, field) == original_entity[field]


def test_enrich_failure_does_not_crash_synthesis():
    """If the LLM call raises, enrichment is logged and skipped;
    synthesis still produces a valid DomainModel."""
    analyses = [_ctx_and_analysis("Sales", [_e("Order")])]
    skeleton = build_deterministic_skeleton(analyses, project_name="X")
    client = MagicMock()
    client.chat.side_effect = RuntimeError("API unavailable")
    result = enrich_synonyms_and_dependencies(skeleton, analyses, client)
    # No crash; entity still valid; synonyms_to_avoid stays None
    assert result.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"


def test_enrich_infers_cross_context_dependencies():
    """allowed_dependencies populated by scanning description+justification
    for mentions of other context names."""
    sales_order = Entity(
        name="Order", description="Order references the Customer entity",
        confidence=0.9, justification="cited", evidence_sentence_indices=[1],
    )
    customer = Entity(
        name="Customer", description="A buyer in the system",
        confidence=0.9, justification="cited", evidence_sentence_indices=[1],
    )
    sales = _ctx_and_analysis("Sales", [sales_order])
    customer_ctx = _ctx_and_analysis("Customer", [customer])
    skeleton = build_deterministic_skeleton([sales, customer_ctx], project_name="X")

    client = MagicMock()
    client.chat.return_value = _mock_chat_response(json.dumps({"entities": []}))
    result = enrich_synonyms_and_dependencies(skeleton, [sales, customer_ctx], client)

    sales_bc = next(bc for bc in result.bounded_contexts if bc.context_name == "Sales")
    assert sales_bc.allowed_dependencies == ["Customer"]


def test_synthesize_domain_model_skip_enrich():
    """skip_enrich=True returns the skeleton without any LLM calls."""
    analyses = [_ctx_and_analysis("Sales", [_e("Order")])]
    client = MagicMock()  # Should NOT be called
    result = synthesize_domain_model(
        analyses, llm_client=client, project_name="X", skip_enrich=True,
    )
    assert result.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"
    client.chat.assert_not_called()
