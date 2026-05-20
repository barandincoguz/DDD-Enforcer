"""5-stage pipeline driver tests. Mock LLM, mock verifier, mock refiner."""

import pytest
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.verifier.types import VerifierResult, VerifierIssue, IssueSeverity
from core.orchestration.errors import (
    ArchitectExtractionError,
    SpecialistFailureError,
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
