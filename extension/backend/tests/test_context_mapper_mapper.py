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


def test_error_reason_coerced_to_str():
    class _Weird:
        def __str__(self): return "weird-reason"
    e = ContextMapperError(reason=_Weird())  # non-str reason must not break consumers
    assert isinstance(e.reason, str) and e.reason == "weird-reason"


import pytest
from core.context_mapper.mapper import run_context_mapper
from core.context_mapper.errors import ContextMapperError
from core.context_mapper.types import ContextMapResponse, ProposedRelationship
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


class _Resp:
    def __init__(self, parsed=None, failed=False, reason=None):
        self.parsed = parsed
        self.json_failed = failed
        self.json_fail_reason = reason


class _Client:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []
    def structured_output(self, **kw):
        self.calls.append(kw)
        return self._resp


class _Cfg:
    model_id = "gemini-3.1-pro-preview"
    temperature = 0.05
    seed = 42


def _ul():
    return UbiquitousLanguage(entities=[], value_objects=[], domain_events=[])


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=_ul()),
            BoundedContext(context_name="Inventory", ubiquitous_language=_ul()),
        ],
        global_rules=None)


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="s0"), SectionedSentence(index=1, text="s1")],
        chunk_metadata=ChunkMetadata(chunk_count=2, total_chars=4, truncated_chunks=0))


def test_maps_relationships_and_sets_model_id():
    resp = _Resp(parsed=ContextMapResponse(relationships=[ProposedRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type="CUSTOMER_SUPPLIER",
        upstream="Inventory", rationale="r", evidence_sentence_indices=[0])]))
    cm = run_context_mapper(_model(), _scout(), None, client=_Client(resp), stage_cfg=_Cfg())
    assert cm.model_id == "gemini-3.1-pro-preview"
    assert len(cm.relationships) == 1
    assert cm.relationships[0].relationship_type == "CUSTOMER_SUPPLIER"


def test_evidence_trimmed_to_scout_or_minus_one():
    resp = _Resp(parsed=ContextMapResponse(relationships=[ProposedRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type="PARTNERSHIP",
        rationale="r", evidence_sentence_indices=[0, 1, 99, -1, 0])]))
    cm = run_context_mapper(_model(), _scout(), None, client=_Client(resp), stage_cfg=_Cfg())
    assert sorted(cm.relationships[0].evidence_sentence_indices) == [-1, 0, 1]


def test_invalid_relationship_dropped_not_fatal():
    resp = _Resp(parsed=ContextMapResponse(relationships=[
        ProposedRelationship(context_a="Ordering", context_b="Inventory",
            relationship_type="CONFORMIST", upstream="Ghost", rationale="bad"),
        ProposedRelationship(context_a="Ordering", context_b="Inventory",
            relationship_type="PARTNERSHIP", rationale="ok"),
    ]))
    cm = run_context_mapper(_model(), _scout(), None, client=_Client(resp), stage_cfg=_Cfg())
    assert len(cm.relationships) == 1 and cm.relationships[0].relationship_type == "PARTNERSHIP"


def test_json_failed_raises():
    with pytest.raises(ContextMapperError):
        run_context_mapper(_model(), _scout(), None,
                           client=_Client(_Resp(failed=True, reason="schema")), stage_cfg=_Cfg())


def test_feedback_path_uses_remap(monkeypatch):
    from core.schemas import ContextMap
    import core.context_mapper.mapper as m
    seen = {}
    monkeypatch.setattr(m, "build_remap_prompt", lambda *a, **k: seen.setdefault("remap", True) or "REMAP")
    resp = _Resp(parsed=ContextMapResponse(relationships=[]))
    run_context_mapper(_model(), _scout(), [object()], client=_Client(resp), stage_cfg=_Cfg(),
                       _prev_map=ContextMap(model_id="m"))
    assert seen.get("remap")
