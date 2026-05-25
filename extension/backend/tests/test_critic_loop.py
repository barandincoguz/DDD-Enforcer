import os
from unittest.mock import MagicMock
from core.critic.loop import run_critique_loop, critique_score, findings_signature
from core.orchestration.pipeline import PipelineDeps
from core.pipeline_contracts import (
    ScoutOutput, ArchitectOutput, ContextHypothesis, SpecialistAnalysis,
    SectionedSentence, ChunkMetadata,
)
from core.schemas import DomainModel, Entity, CritiqueFinding, CriticReport, CriticLoopTrace
from core.verifier.types import VerifierResult


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="An order.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
    )


def _base_deps():
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
        scout=lambda t: _scout(), architect=architect_fn,
        architect_with_feedback=lambda s, i: architect_fn(s),
        specialist=specialist_fn,
        specialist_with_feedback=lambda a, s, prev, issues: specialist_fn(a, s),
        synthesizer=synthesizer_fn,
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
    )


def _high_finding():
    return CritiqueFinding(finding_type="ANEMIC_ENTITY", priority="high",
                           target_ref="entity:Ord.Order", rationale="r", proposed_revision="p")


def _report(findings, score=0.0):
    return CriticReport(model_id="m", findings=findings, score=score,
                        loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))


def test_score_orders_by_severity():
    assert critique_score([_high_finding()]) == 3.0
    assert critique_score([]) == 0.0


def test_converged_when_cycle0_clean():
    deps = _base_deps()
    deps.critic = lambda model, scout, history: _report([])
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.outcome == "converged"
    assert model.critic_report.loop.cycles_used == 1


def test_keep_best_returns_lowest_score_cycle():
    deps = _base_deps()
    calls = [0]

    def critic(model, scout, history):
        calls[0] += 1
        return _report([_high_finding()]) if calls[0] == 1 else _report([_high_finding(), _high_finding()])

    deps.critic = critic
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.best_cycle == 0
    assert model.critic_report.score == 3.0


def test_flap_stops_loop():
    deps = _base_deps()

    def critic(model, scout, history):
        return _report([_high_finding()])

    deps.critic = critic
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.outcome == "flapped"


def test_failure_is_non_fatal_returns_best_so_far():
    from core.critic.errors import CriticError
    deps = _base_deps()
    calls = [0]

    def critic(model, scout, history):
        calls[0] += 1
        if calls[0] == 1:
            return _report([_high_finding()])
        raise CriticError(reason="boom", cycle=calls[0])

    deps.critic = critic
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.outcome == "failed"
    assert model.critic_report.error is not None
    assert isinstance(model, DomainModel)
