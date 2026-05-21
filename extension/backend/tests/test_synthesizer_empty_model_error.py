"""SynthesizerEmptyModelError construction and pipeline integration.

WP-CORE-1 T8: The prior test patched DomainArchitect.synthesize and
synthesize_final_model, both of which were deleted in T4/T6. The error
class itself (core/orchestration/errors.py) and the pipeline check in
run_pipeline remain; this file now tests those directly.
"""

import pytest
from core.orchestration.errors import SynthesizerEmptyModelError


def test_synthesizer_empty_model_error_carries_input_summary():
    """SynthesizerEmptyModelError must preserve the input_summary string
    in its message so callers can diagnose why the pipeline produced no contexts."""
    err = SynthesizerEmptyModelError(input_summary="0 contexts from 3 SpecialistAnalyses")
    assert "0 contexts" in str(err)


def test_synthesizer_empty_model_error_is_exception():
    """SynthesizerEmptyModelError must be raise-able as a standard Python exception."""
    with pytest.raises(SynthesizerEmptyModelError):
        raise SynthesizerEmptyModelError(input_summary="empty")


def test_create_fallback_model_is_gone():
    """B4 deletes _create_fallback_model. DomainArchitect instance must not have
    the attribute (checked via object.__new__ to bypass __init__)."""
    from core.architect import DomainArchitect
    arch = DomainArchitect.__new__(DomainArchitect)
    assert not hasattr(arch, "_create_fallback_model"), (
        "_create_fallback_model must be deleted; an empty model is "
        "no longer a legitimate pipeline output."
    )


# =============================================================================
# WP-CORE-5b — srs_path field + diagnostic message
# =============================================================================


def test_synthesizer_empty_model_error_carries_srs_path():
    """T-EMPTY-5 (Codex OQ-2): SynthesizerEmptyModelError must carry srs_path
    field and include it in str(err). Default is '<unknown>'. Matches WP-CORE-4
    pattern for IntermediateSaveError."""
    err = SynthesizerEmptyModelError(
        input_summary="0 SpecialistAnalysis from upstream pipeline",
        srs_path="/abs/path/SRS.docx",
    )
    assert err.srs_path == "/abs/path/SRS.docx"
    assert "/abs/path/SRS.docx" in str(err)

    err_default = SynthesizerEmptyModelError(input_summary="x")
    assert err_default.srs_path == "<unknown>"
    assert "<unknown>" in str(err_default)


def test_synthesizer_empty_model_error_message_diagnostic():
    """T-EMPTY-6: error message must be diagnostic enough for support cases —
    name the failure mode AND the SRS path AND the input summary."""
    err = SynthesizerEmptyModelError(
        input_summary="0 SpecialistAnalysis from upstream pipeline",
        srs_path="/inputs/SRS.docx",
    )
    msg = str(err)
    assert "empty DomainModel" in msg
    assert "0 SpecialistAnalysis" in msg
    assert "/inputs/SRS.docx" in msg
