"""_generate_once returns the model plus intermediates; critic=None unchanged."""
from core.orchestration.pipeline import run_pipeline, _generate_once, PipelineDeps
from core.pipeline_contracts import (
    ScoutOutput, ArchitectOutput, ContextHypothesis, SpecialistAnalysis,
    SectionedSentence, ChunkMetadata,
)
from core.schemas import DomainModel, Entity
from core.verifier.types import VerifierResult
from unittest.mock import MagicMock


def _deps():
    def scout_fn(t):
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        )

    def architect_fn(scout):
        return ArchitectOutput(contexts=[ContextHypothesis(context_name="Ord", description="x")])

    def specialist_fn(arch, scout):
        return [SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[Entity(name="Order", description="An order.", confidence=0.9,
                             justification="c", evidence_sentence_indices=[0])],
        )]

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(analyses, llm_client=MagicMock(),
                                       project_name="T", skip_enrich=True)

    return PipelineDeps(
        scout=scout_fn, architect=architect_fn,
        architect_with_feedback=lambda s, i: architect_fn(s),
        specialist=specialist_fn, synthesizer=synthesizer_fn,
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
    )


def test_generate_once_returns_model_and_intermediates():
    deps = _deps()
    scout = deps.scout("x")
    model, arch, specialist = _generate_once(scout, deps, srs_path="x")
    assert isinstance(model, DomainModel)
    assert arch.contexts[0].context_name == "Ord"
    assert specialist[0].entities[0].name == "Order"


def test_critic_none_pipeline_unchanged():
    deps = _deps()
    model = run_pipeline(srs_text="x", deps=deps)
    assert model.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"
