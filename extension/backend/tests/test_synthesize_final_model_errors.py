"""Phase A: synthesize_final_model must propagate Pydantic validation errors
instead of returning an empty model via a bare except.
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
    """When synthesize() returns invalid JSON shape, the Pydantic error must
    propagate; the bare except path is gone.

    Before the fix, this test would see ValidationError caught by the bare
    except, which then tries to return a fallback model. The fallback also
    fails Pydantic validation (FM-04 forbids empty bounded_contexts), so a
    second ValidationError bubbles up.

    After the fix, the first ValidationError propagates directly.
    """
    arch = _make_arch()
    with patch.object(arch, "synthesize") as mock_synth:
        # Return a dict missing required fields — DomainModel construction will fail
        mock_synth.return_value = {"project_name": "X"}
        with pytest.raises(ValidationError):
            arch.synthesize_final_model(analyses=[])
