"""5-stage pipeline driver tests. Mock LLM, mock verifier, mock refiner."""

import pytest
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.verifier.types import VerifierResult, VerifierIssue, IssueSeverity
from core.orchestration.errors import (
    ArchitectExtractionError,
    PipelineError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
)
from core.pipeline_contracts import (
    ScoutOutput,
    ArchitectOutput,
    ContextHypothesis,
    SpecialistAnalysis,
    SectionedSentence,
    ChunkMetadata,
)
from core.schemas import DomainModel, Entity


def _ok():
    return VerifierResult(ok=True, issues=[])


def _make_typed_deps():
    """Build PipelineDeps with typed envelope stubs."""

    def scout_fn(srs_text: str) -> ScoutOutput:
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order is placed by a customer.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=45),
        )

    def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="Manages orders"),
        ])

    def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput):
        return [
            SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="A purchase order placed by a customer.",
                    confidence=0.9,
                    justification="Cited in sentence 0",
                    evidence_sentence_indices=[0],
                )],
            )
        ]

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    def verifier_fn(snapshot):
        return _ok()

    return PipelineDeps(
        scout=scout_fn,
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
    )


def test_pipeline_happy_path_produces_domain_model():
    deps = _make_typed_deps()
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert isinstance(model, DomainModel)
    assert len(model.bounded_contexts) == 1
    assert model.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"


def test_pipeline_propagates_architect_extraction_error():
    deps = _make_typed_deps()
    deps.architect = MagicMock(side_effect=ArchitectExtractionError(srs_path="x"))
    with pytest.raises(ArchitectExtractionError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_propagates_specialist_failure():
    deps = _make_typed_deps()
    deps.specialist = MagicMock(side_effect=SpecialistFailureError(context_name="OrderMgmt"))
    with pytest.raises(SpecialistFailureError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_invokes_refiner_when_verifier_finds_issues():
    """Verifier returns an issue on first call; pipeline refines (re-runs Specialist)
    then verifier returns ok. specialist must be called twice."""
    specialist_mock = MagicMock()

    call_count = [0]

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="Manages orders"),
        ])

    def specialist_fn(arch, scout):
        call_count[0] += 1
        return [
            SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="A purchase order.",
                    confidence=0.9,
                    justification="Cited in sentence 0",
                    evidence_sentence_indices=[0],
                )],
            )
        ]

    verifier_calls = [0]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="specialist:OrderMgmt.entities[0]",
                issue_type="missing_evidence",
                severity=IssueSeverity.ERROR,
                message="missing evidence",
            )])
        return VerifierResult(ok=True, issues=[])

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    deps = PipelineDeps(
        scout=lambda text: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        ),
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
    )
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    # Refiner re-runs Specialist once → total 2 specialist calls
    assert call_count[0] == 2
    assert model is not None


# =============================================================================
# WP-CORE-5b — SynthesizerEmptyModelError guard placement + srs_path
# =============================================================================


def test_pipeline_raises_synthesizer_empty_model_error_when_specialist_returns_empty():
    """T-EMPTY-1: initial-empty Specialist DI path raises SynthesizerEmptyModelError,
    NOT pydantic.ValidationError. Verifies pre-call guard + PipelineError taxonomy.
    Closes Codex N-2 (merged v1's T-EMPTY-1 + T-EMPTY-2 into one assertion)."""
    deps = _make_typed_deps()
    deps.specialist = MagicMock(return_value=[])
    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert isinstance(exc_info.value, SynthesizerEmptyModelError)


def test_pipeline_synthesizer_not_invoked_when_specialist_empty():
    """T-EMPTY-2: pre-call guard short-circuits before deps.synthesizer is called."""
    deps = _make_typed_deps()
    deps.specialist = MagicMock(return_value=[])
    synth_mock = MagicMock()
    deps.synthesizer = synth_mock
    with pytest.raises(SynthesizerEmptyModelError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert synth_mock.call_count == 0, (
        "Pre-call guard must short-circuit before deps.synthesizer is invoked."
    )


def test_pipeline_raises_synthesizer_empty_model_error_when_refiner_rerun_returns_empty():
    """T-EMPTY-3 (Codex W-1): refiner-success-path edge — first Specialist call
    returns non-empty, verifier fails once, rerun returns [], verifier accepts.
    refined_specialist becomes []; pre-call guard raises SynthesizerEmptyModelError."""
    specialist_calls = [0]

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="x"),
        ])

    def specialist_fn(arch, scout):
        specialist_calls[0] += 1
        if specialist_calls[0] == 1:
            return [SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="An order.",
                    confidence=0.9,
                    justification="cited",
                    evidence_sentence_indices=[0],
                )],
            )]
        return []  # Rerun returns empty.

    verifier_calls = [0]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="specialist:OrderMgmt.entities[0]",
                issue_type="missing_evidence",
                severity=IssueSeverity.ERROR,
                message="m",
            )])
        return VerifierResult(ok=True, issues=[])

    deps = PipelineDeps(
        scout=lambda text: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        ),
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=MagicMock(),  # Should never be invoked (pre-call guard).
        verifier=verifier_fn,
    )

    with pytest.raises(SynthesizerEmptyModelError):
        run_pipeline(srs_text="x", deps=deps)
    assert specialist_calls[0] == 2, (
        "Refiner should have invoked Specialist twice (initial + one rerun)."
    )


def test_pipeline_post_call_check_catches_injected_synthesizer_returning_empty_model():
    """T-EMPTY-4 (Codex W-3): belt-and-suspenders for injected synthesizers that
    bypass Pydantic via DomainModel.model_construct (which skips validators)."""
    from core.schemas import ProjectMetadata

    deps = _make_typed_deps()

    def injected_synthesizer(analyses):
        # model_construct bypasses Pydantic validation, allowing empty bounded_contexts.
        return DomainModel.model_construct(
            project_name="Test",
            project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
            bounded_contexts=[],
            global_rules=None,
        )

    deps.synthesizer = injected_synthesizer

    with pytest.raises(SynthesizerEmptyModelError) as exc_info:
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert "bypassed Pydantic" in str(exc_info.value), (
        "Post-call check should emit a distinct message from the pre-call guard."
    )
