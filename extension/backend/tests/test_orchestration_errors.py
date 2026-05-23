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


# =============================================================================
# WP-CORE-7 — ArchitectGroundingError (F-22 mode C hybrid)
# =============================================================================


def test_architect_grounding_error_carries_srs_path_issues_cycles():
    """T-AGE-1 (WP-CORE-7): ArchitectGroundingError exposes srs_path, issues,
    residual_issues, cycles_attempted; message includes srs_path and counts.

    Import is inside the test body (not at module top) per Codex W-2: keeps
    the RED commit collectable. The test fails by ImportError at body
    execution time, which pytest reports as a test failure (not a collection
    error)."""
    from core.orchestration.errors import (
        ArchitectGroundingError,
        PipelineError,
    )

    fake_arch_issue = {"check_id": "D1", "target": "architect:contexts[X]", "message": "no IDs"}
    fake_residual = {"check_id": "D2", "target": "specialist:X.entities[0]", "message": "no evidence"}

    e = ArchitectGroundingError(
        srs_path="inputs/SRS.docx",
        issues=[fake_arch_issue],
        residual_issues=[fake_residual],
        cycles_attempted=1,
    )

    assert isinstance(e, PipelineError)
    assert e.srs_path == "inputs/SRS.docx"
    assert e.issues == [fake_arch_issue]
    assert e.residual_issues == [fake_residual]
    assert e.cycles_attempted == 1
    msg = str(e)
    assert "SRS.docx" in msg
    assert "1" in msg  # cycle count present
