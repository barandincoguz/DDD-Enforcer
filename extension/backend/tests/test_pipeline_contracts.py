"""Smoke tests for core.pipeline_contracts.

Each envelope must round-trip cleanly, reject malformed input via
ValidationError, and default sensible empty collections.
"""

import pytest
from pydantic import ValidationError

from core.pipeline_contracts import (
    SectionedSentence,
    ChunkMetadata,
    ScoutOutput,
    ContextHypothesis,
    ArchitectOutput,
    Ambiguity,
    SpecialistAnalysis,
    VerifierIssue,
    VerifierResult,
)


def test_sectioned_sentence_construct():
    s = SectionedSentence(index=0, text="hello", section="Intro")
    assert s.index == 0
    assert s.text == "hello"
    assert s.section == "Intro"


def test_sectioned_sentence_rejects_negative_index():
    with pytest.raises(ValidationError):
        SectionedSentence(index=-1, text="x")


def test_scout_output_defaults():
    out = ScoutOutput(
        sentences=[],
        chunk_metadata=ChunkMetadata(chunk_count=0, total_chars=0),
    )
    assert out.sentences == []
    assert out.chunk_metadata.truncated_chunks == 0


def test_architect_output_open_questions_default_empty():
    out = ArchitectOutput(contexts=[])
    assert out.open_questions == []


def test_specialist_analysis_default_empty_collections():
    ctx = ContextHypothesis(context_name="Sales", description="Order flow")
    a = SpecialistAnalysis(context=ctx)
    assert a.entities == []
    assert a.value_objects == []
    assert a.services == []
    assert a.aggregates == []
    assert a.domain_events == []
    assert a.business_rules == []
    assert a.ambiguities == []


def test_specialist_analysis_carries_entities():
    """A SpecialistAnalysis with strict-schema Entity objects round-trips."""
    from core.schemas import Entity
    ctx = ContextHypothesis(context_name="Sales", description="Order flow")
    e = Entity(
        name="Order",
        description="A customer purchase",
        confidence=0.9,
        justification="Cited in 3 SRS sentences",
        evidence_sentence_indices=[1, 2, 3],
    )
    a = SpecialistAnalysis(context=ctx, entities=[e])
    assert len(a.entities) == 1
    assert a.entities[0].name == "Order"
    assert a.entities[0].evidence_sentence_indices == [1, 2, 3]


def test_specialist_analysis_validates_from_dict():
    """model_validate accepts a plain dict (this is the boundary path)."""
    payload = {
        "context": {"context_name": "Sales", "description": "Order flow"},
        "entities": [
            {
                "name": "Order",
                "description": "A customer purchase",
                "confidence": 0.9,
                "justification": "Cited in 3 SRS sentences",
                "evidence_sentence_indices": [1, 2, 3],
            }
        ],
    }
    a = SpecialistAnalysis.model_validate(payload)
    assert a.entities[0].name == "Order"


def test_specialist_analysis_rejects_list_input():
    """Validation fails if a list is passed where a dict is expected.
    This is the exact crash mode at architect.py:692."""
    with pytest.raises(ValidationError):
        SpecialistAnalysis.model_validate([{"entities": []}])


def test_verifier_issue_construct():
    issue = VerifierIssue(
        severity="ERROR", check_id="D6", target="entity_count",
        message="2 entities lost during synthesis",
    )
    assert issue.severity == "ERROR"
    assert issue.check_id == "D6"


def test_verifier_result_ok_when_no_issues():
    r = VerifierResult(ok=True)
    assert r.ok is True
    assert r.issues == []
    assert r.error_count() == 0


def test_unresolved_extra_field_raises():
    """LLMs occasionally emit `_unresolved` keys (refiner feedback signal).
    These must NOT be silently swallowed by Pydantic; they should raise
    so the retry loop can act on them. Other extras are tolerated."""
    ctx = ContextHypothesis(context_name="Sales", description="x")
    with pytest.raises(ValidationError):
        SpecialistAnalysis.model_validate({
            "context": ctx.model_dump(),
            "_unresolved": "could not classify entity X",
        })
