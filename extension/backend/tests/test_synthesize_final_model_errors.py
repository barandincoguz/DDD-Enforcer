"""Phase A: synthesize_final_model must propagate Pydantic validation errors
instead of returning an empty model via a bare except.

Phase B4 refinement: the empty-bounded_contexts path is now an explicit
SynthesizerEmptyModelError (see test_synthesizer_empty_model_error.py).
This test pins the remaining contract: non-bounded_contexts ValidationErrors
(e.g. malformed project_metadata) must propagate unwrapped.
"""

import pytest
from pydantic import ValidationError
from unittest.mock import patch

from core.architect import DomainArchitect


def _make_arch():
    """Bypass __init__ to avoid needing a real API key in this unit test."""
    arch = DomainArchitect.__new__(DomainArchitect)
    arch.model_name = "gemini-3.1-pro-preview"
    arch.last_request_time = 0
    arch.min_delay = 0
    arch.request_count = 0
    import threading
    arch._rate_limit_lock = threading.Lock()
    arch.scout_max_workers = 1
    from core.token_tracker import TokenTracker
    arch.token_tracker = TokenTracker.get_instance()
    arch.progress_callback = None
    arch.run_timestamp = "20260518_000000"
    return arch


def test_synthesize_final_model_propagates_validation_error():
    """When synthesize() returns a payload whose error is OUTSIDE
    bounded_contexts (e.g. project_metadata shape failure), the Pydantic
    ValidationError must propagate unwrapped. B4 only converts errors
    that mention bounded_contexts into SynthesizerEmptyModelError.
    """
    arch = _make_arch()
    with patch.object(arch, "synthesize") as mock_synth:
        # bounded_contexts is non-empty (so the early SynthesizerEmptyModelError
        # check passes), but project_metadata is malformed — version is an int
        # not a str, which will fail Pydantic validation on ProjectMetadata.
        mock_synth.return_value = {
            "project_name": "X",
            "project_metadata": {"version": 123, "generated_at": ["not", "a", "string"]},
            "bounded_contexts": [
                {
                    "context_name": "C1",
                    "description": "desc",
                    "ubiquitous_language": {
                        "entities": [
                            {
                                "name": "E",
                                "description": "d",
                                "confidence": 0.9,
                                "justification": "j",
                                "evidence_sentence_indices": [0],
                            }
                        ],
                        "value_objects": None,
                        "domain_events": None,
                    },
                }
            ],
            "global_rules": None,
        }
        with pytest.raises(ValidationError):
            arch.synthesize_final_model(analyses=[])
