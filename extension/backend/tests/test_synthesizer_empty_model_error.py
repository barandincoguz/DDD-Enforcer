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
