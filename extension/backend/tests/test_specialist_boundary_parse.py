"""Defensive list-or-dict parsing at the Specialist boundary.

The live pipeline crashed at architect.py:692 because Gemini-Pro
occasionally emits a top-level JSON array instead of the prompted
object. These tests cover the defensive unwrap + Pydantic validation
that converts that crash mode into a typed retry signal.
"""

import pytest

from core.architect import DomainArchitect
from core.orchestration.errors import SpecialistShapeError
from core.pipeline_contracts import SpecialistAnalysis, ContextHypothesis


def _well_formed_dict():
    return {
        "context": "Sales",
        "entities": [
            {
                "name": "Order",
                "description": "A customer purchase",
                "attributes": ["id", "total"],
                "confidence": 0.9,
                "justification": "cited in 3 sentences",
                "evidence_sentence_indices": [1, 2, 3],
            }
        ],
        "value_objects": [],
        "services": [],
        "aggregates": [],
        "domain_events": [],
        "business_rules": [],
    }


def test_dict_passes_through_unwrap_unchanged():
    """A well-formed dict input is returned as-is."""
    payload = _well_formed_dict()
    result = DomainArchitect._unwrap_singleton_list(payload)
    assert result is payload  # identity preserved


def test_singleton_list_is_unwrapped_to_dict():
    """[{...}] is unwrapped to {...} — the exact Gemini-Pro quirk."""
    payload = _well_formed_dict()
    result = DomainArchitect._unwrap_singleton_list([payload])
    assert result == payload


def test_empty_list_raises():
    """An empty top-level list cannot be unwrapped; should raise."""
    with pytest.raises(ValueError, match="empty"):
        DomainArchitect._unwrap_singleton_list([])


def test_multi_element_list_raises():
    """[{a}, {b}] is ambiguous — refuse to silently pick one."""
    with pytest.raises(ValueError, match="multiple"):
        DomainArchitect._unwrap_singleton_list([{"a": 1}, {"b": 2}])


def test_non_dict_non_list_raises():
    """Strings, numbers, None — none are valid Specialist outputs."""
    with pytest.raises(ValueError, match="unexpected type"):
        DomainArchitect._unwrap_singleton_list("just a string")


def test_boundary_validation_produces_typed_analysis():
    """The full boundary: dict → SpecialistAnalysis with strict fields."""
    payload = _well_formed_dict()
    ctx = ContextHypothesis(context_name="Sales", description="Order flow")
    analysis = DomainArchitect._validate_specialist_payload(payload, ctx)
    assert isinstance(analysis, SpecialistAnalysis)
    assert analysis.context.context_name == "Sales"
    assert analysis.entities[0].name == "Order"


def test_boundary_validation_raises_specialist_shape_error_on_list():
    """A top-level list payload triggers SpecialistShapeError after
    the unwrap helper rejects the empty/multi-element cases."""
    ctx = ContextHypothesis(context_name="Sales", description="x")
    with pytest.raises(SpecialistShapeError) as exc_info:
        DomainArchitect._validate_specialist_payload([], ctx)
    assert "Sales" in str(exc_info.value)


def test_boundary_validation_raises_on_missing_description():
    """Entity without `description` violates strict schema → shape error."""
    payload = _well_formed_dict()
    del payload["entities"][0]["description"]
    ctx = ContextHypothesis(context_name="Sales", description="x")
    with pytest.raises(SpecialistShapeError):
        DomainArchitect._validate_specialist_payload(payload, ctx)
