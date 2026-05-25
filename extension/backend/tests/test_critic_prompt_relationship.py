from core.critic.prompt import build_critique_prompt
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata,
                          ContextMap, ContextRelationship)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


def _ul():
    return UbiquitousLanguage(entities=[], value_objects=[], domain_events=[])


def _model_with_map():
    m = DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=_ul()),
            BoundedContext(context_name="Inventory", ubiquitous_language=_ul())],
        global_rules=None)
    m.context_map = ContextMap(model_id="m", relationships=[ContextRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type="CONFORMIST",
        upstream="Inventory", rationale="r")])
    return m


def _scout():
    return ScoutOutput(sentences=[SectionedSentence(index=0, text="s")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=1, truncated_chunks=0))


def test_prompt_includes_context_map_and_relationship_instruction():
    p = build_critique_prompt(_model_with_map(), _scout(), [])
    assert "CONFORMIST" in p
    assert "relationship" in p.lower()
    assert "WRONG_RELATIONSHIP_TYPE" in p or "relationship:" in p


def test_prompt_without_map_still_builds():
    m = _model_with_map(); m.context_map = None
    p = build_critique_prompt(m, _scout(), [])
    assert "Ordering" in p
