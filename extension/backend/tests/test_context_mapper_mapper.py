from core.context_mapper.types import ProposedRelationship, ContextMapResponse
from core.context_mapper.errors import ContextMapperError

def test_proposed_relationship_minimal():
    pr = ProposedRelationship(context_a="A", context_b="B",
                              relationship_type="SEPARATE_WAYS", rationale="r")
    assert pr.upstream is None and pr.evidence_sentence_indices == []

def test_response_defaults():
    resp = ContextMapResponse()
    assert resp.analysis == "" and resp.relationships == []

def test_error_is_exception():
    e = ContextMapperError(reason="json_failed")
    assert "json_failed" in str(e)
