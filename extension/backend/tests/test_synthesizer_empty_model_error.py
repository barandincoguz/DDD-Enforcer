"""Phase B4: Synthesizer must raise SynthesizerEmptyModelError when the
LLM returns an empty bounded_contexts list. _create_fallback_model is
deleted; the bare except is already narrowed (A5).
"""

import pytest
from unittest.mock import patch
from core.architect import DomainArchitect
from core.orchestration.errors import SynthesizerEmptyModelError


def _arch():
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.run_timestamp = "20260518_000000"
    return a


def test_synthesize_final_model_raises_when_empty_bounded_contexts():
    arch = _arch()
    empty_payload = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [],
        "global_rules": None,
    }
    with patch.object(arch, "synthesize", return_value=empty_payload):
        with pytest.raises(SynthesizerEmptyModelError):
            arch.synthesize_final_model(analyses=[])


def test_create_fallback_model_is_gone():
    """B4 deletes _create_fallback_model. Instance must not have the attribute."""
    arch = _arch()
    assert not hasattr(arch, "_create_fallback_model"), (
        "_create_fallback_model must be deleted in B4; an empty model is "
        "no longer a legitimate pipeline output."
    )
