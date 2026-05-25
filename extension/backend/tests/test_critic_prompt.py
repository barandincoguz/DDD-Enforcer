from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata
from core.critic.types import CritiqueCycleMemory
from core.critic.prompt import build_critique_prompt


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ordering",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                )],
                value_objects=None, domain_events=None,
            ),
        )],
        global_rules=None,
    )


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="A customer places an order.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=27),
    )


def test_prompt_includes_model_scout_and_schema_directive():
    prompt = build_critique_prompt(_model(), _scout(), history=[])
    assert "Ordering" in prompt
    assert "A customer places an order." in prompt
    assert "0" in prompt
    assert "findings" in prompt
    assert "step" in prompt.lower()


def test_prompt_includes_reflexion_history_when_present():
    history = [CritiqueCycleMemory(
        cycle=0, findings_summary=["high ANEMIC_ENTITY entity:Ordering.Order"],
        diff_summary="Ordering: entities added: Payment",
    )]
    prompt = build_critique_prompt(_model(), _scout(), history=history)
    assert "PREVIOUS CYCLES" in prompt
    assert "ANEMIC_ENTITY" in prompt
    assert "entities added: Payment" in prompt
