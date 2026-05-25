"""Full pipeline → critique loop integration with a fake critic dep."""
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.pipeline_contracts import (
    ScoutOutput, ArchitectOutput, ContextHypothesis, SpecialistAnalysis,
    SectionedSentence, ChunkMetadata,
)
from core.schemas import (
    DomainModel, Entity, CritiqueFinding, CriticReport, CriticLoopTrace,
)
from core.verifier.types import VerifierResult


def _deps_with_critic(critic):
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
        scout=lambda t: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8)),
        architect=architect_fn,
        architect_with_feedback=lambda s, i: architect_fn(s),
        specialist=specialist_fn,
        specialist_with_feedback=lambda a, s, prev, issues: specialist_fn(a, s),
        synthesizer=synthesizer_fn,
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
        critic=critic,
    )


def test_pipeline_with_clean_critic_attaches_converged_report():
    def critic(model, scout, history):
        return CriticReport(model_id="m", findings=[],
                            loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))

    model = run_pipeline(srs_text="x", deps=_deps_with_critic(critic))
    assert isinstance(model, DomainModel)
    assert model.critic_report is not None
    assert model.critic_report.loop.outcome == "converged"
    assert model.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"


def test_pipeline_content_finding_drives_one_revision_then_converges():
    calls = [0]

    def critic(model, scout, history):
        calls[0] += 1
        if calls[0] == 1:
            return CriticReport(model_id="m", findings=[CritiqueFinding(
                finding_type="ANEMIC_ENTITY", priority="high",
                target_ref="entity:Ord.Order", rationale="r", proposed_revision="p")],
                loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))
        return CriticReport(model_id="m", findings=[],
                            loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))

    model = run_pipeline(srs_text="x", deps=_deps_with_critic(critic))
    assert calls[0] == 2
    assert model.critic_report.loop.outcome == "converged"
    assert model.critic_report.loop.best_cycle == 1
