"""Phase B1: PipelineError hierarchy."""

from core.orchestration.errors import (
    PipelineError,
    ScoutChunkParseError,
    ArchitectExtractionError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
    RefinementExhaustedError,
    InsufficientGroundingError,
)


def test_all_pipeline_errors_subclass_pipeline_error():
    for cls in [
        ScoutChunkParseError,
        ArchitectExtractionError,
        SpecialistFailureError,
        SynthesizerEmptyModelError,
        RefinementExhaustedError,
        InsufficientGroundingError,
    ]:
        assert issubclass(cls, PipelineError), f"{cls.__name__} must subclass PipelineError"


def test_scout_chunk_parse_error_carries_chunk_id_and_attempts():
    e = ScoutChunkParseError(chunk_id="3.1", attempts=5)
    assert e.chunk_id == "3.1"
    assert e.attempts == 5
    assert "3.1" in str(e)


def test_architect_extraction_error_carries_srs_path():
    e = ArchitectExtractionError(srs_path="inputs/SRS.docx")
    assert e.srs_path == "inputs/SRS.docx"
    assert "SRS.docx" in str(e)


def test_specialist_failure_error_carries_context_name():
    e = SpecialistFailureError(context_name="OrderMgmt")
    assert e.context_name == "OrderMgmt"


def test_synthesizer_empty_model_error_carries_input_summary():
    e = SynthesizerEmptyModelError(input_summary="0 analyses")
    assert "0 analyses" in str(e)


def test_refinement_exhausted_error_carries_issues():
    e = RefinementExhaustedError(issues=[{"stage": "specialist", "issue_type": "missing_evidence"}])
    assert len(e.issues) == 1
    assert e.issues[0]["stage"] == "specialist"


def test_insufficient_grounding_error_carries_entity_name():
    e = InsufficientGroundingError(entity_name="GhostEntity")
    assert "GhostEntity" in str(e)
