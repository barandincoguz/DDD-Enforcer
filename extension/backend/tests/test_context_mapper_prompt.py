from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage,
                          ProjectMetadata, CritiqueFinding, ContextMap, ContextRelationship)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata
from core.context_mapper.prompt import build_map_prompt, build_remap_prompt


def _ul() -> UbiquitousLanguage:
    return UbiquitousLanguage(entities=[], value_objects=[], domain_events=[])


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", description="orders",
                           ubiquitous_language=_ul()),
            BoundedContext(context_name="Inventory", description="stock",
                           ubiquitous_language=_ul()),
        ],
        global_rules=None)


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="Orders reduce stock.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=10, truncated_chunks=0))


def test_map_prompt_lists_contexts_and_taxonomy():
    p = build_map_prompt(_model(), _scout())
    assert "Ordering" in p and "Inventory" in p
    assert "CUSTOMER_SUPPLIER" in p and "ANTI_CORRUPTION_LAYER" in p and "SEPARATE_WAYS" in p
    assert "[0]" in p
    assert "import" not in p.lower()


def test_remap_prompt_includes_feedback_and_prior_map():
    prev = ContextMap(model_id="m", relationships=[ContextRelationship(
        context_a="Ordering", context_b="Inventory",
        relationship_type="CONFORMIST", upstream="Inventory", rationale="x")])
    # WRONG_RELATIONSHIP_TYPE is not yet in CritiqueFinding.finding_type; use "OTHER"
    finding = CritiqueFinding(
        finding_type="OTHER", priority="high",
        target_ref="relationship:Ordering->Inventory",
        rationale="Ordering translates Inventory's model; this is ACL not Conformist.",
        proposed_revision="Use ANTI_CORRUPTION_LAYER.")
    p = build_remap_prompt(_model(), _scout(), prev, [finding])
    assert "CONFORMIST" in p and "ANTI_CORRUPTION_LAYER" in p
    assert "translates" in p
