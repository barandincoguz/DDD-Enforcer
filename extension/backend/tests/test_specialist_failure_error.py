"""Phase B3: Specialist must raise SpecialistFailureError when retries
exhaust, not return [{"context": ctx, "analysis": {"error": "parse_failed"}}]."""

import pytest
from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.orchestration.errors import SpecialistFailureError


def _arch():
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.last_request_time = 0
    a.min_delay = 0
    a.request_count = 0
    import threading
    a._rate_limit_lock = threading.Lock()
    from core.token_tracker import TokenTracker
    a.token_tracker = TokenTracker.get_instance()
    a.progress_callback = None
    a.run_timestamp = "20260518_000000"
    a.client = MagicMock()
    return a


def test_specialist_raises_on_parse_failure_after_retries():
    arch = _arch()
    bad_response = MagicMock()
    bad_response.candidates = [MagicMock()]
    bad_response.candidates[0].finish_reason = "STOP"
    bad_response.text = "garbage"
    arch.client.models.generate_content.return_value = bad_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_parse_json_response", return_value={"error": "json_parse_failed"}):
        with pytest.raises(SpecialistFailureError):
            arch.extract_all_contexts_details(
                contexts=["OrderMgmt"], domain_sentences=["s1", "s2"]
            )


def test_specialist_raises_on_exception_after_retries():
    arch = _arch()
    arch.client.models.generate_content.side_effect = RuntimeError("boom")

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_is_quota_error_and_backoff", return_value=False):
        with pytest.raises(SpecialistFailureError):
            arch.extract_all_contexts_details(
                contexts=["OrderMgmt"], domain_sentences=["s1", "s2"]
            )
