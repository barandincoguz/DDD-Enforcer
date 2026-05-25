from core.orchestration.pipeline import _apply_context_map, PipelineDeps
from core.context_mapper.errors import ContextMapperError
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage,
                          ProjectMetadata, ContextMap, ContextRelationship)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


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
    return ScoutOutput(sentences=[SectionedSentence(index=0, text="s")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=1, truncated_chunks=0))


def _deps(context_mapper):
    return PipelineDeps(scout=lambda t: None, architect=lambda s: None,
        architect_with_feedback=lambda s, i: None, specialist=lambda a, s: None,
        synthesizer=lambda x: None, verifier=lambda s: None, context_mapper=context_mapper)


def test_noop_when_mapper_none():
    m = _model()
    out = _apply_context_map(m, _deps(None), _scout())
    assert out.context_map is None and out is m


def test_purity_input_unchanged_and_deps_derived():
    def mapper(model, scout, feedback):
        return ContextMap(model_id="m", relationships=[ContextRelationship(
            context_a="Ordering", context_b="Inventory",
            relationship_type="CUSTOMER_SUPPLIER", upstream="Inventory", rationale="r")])
    m = _model()
    out = _apply_context_map(m, _deps(mapper), _scout())
    assert m.context_map is None  # input untouched (deep copy)
    assert out.context_map is not None
    ord_ctx = next(b for b in out.bounded_contexts if b.context_name == "Ordering")
    inv_ctx = next(b for b in out.bounded_contexts if b.context_name == "Inventory")
    assert ord_ctx.allowed_dependencies == ["Inventory"]
    assert inv_ctx.allowed_dependencies is None


def test_failure_keeps_baseline_and_records_error():
    def mapper(model, scout, feedback):
        raise ContextMapperError(reason="json_failed")
    m = _model()
    m.bounded_contexts[0].allowed_dependencies = ["Inventory"]
    out = _apply_context_map(m, _deps(mapper), _scout())
    assert out.context_map is not None and out.context_map.error == "json_failed"
    assert out.bounded_contexts[0].allowed_dependencies == ["Inventory"]


def test_feedback_forwarded():
    captured = {}
    def mapper(model, scout, feedback):
        captured["fb"] = feedback
        return ContextMap(model_id="m")
    _apply_context_map(_model(), _deps(mapper), _scout(), feedback=["finding"])
    assert captured["fb"] == ["finding"]
