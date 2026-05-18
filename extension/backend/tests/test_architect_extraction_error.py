"""Phase B2: Architect must raise ArchitectExtractionError when retries
exhaust, not return ['CoreDomain']."""

import pytest
from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.orchestration.errors import ArchitectExtractionError


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


def test_architect_raises_when_response_parse_fails_all_retries():
    arch = _arch()
    bad_response = MagicMock()
    bad_response.candidates = [MagicMock()]
    bad_response.candidates[0].finish_reason = "STOP"
    bad_response.text = "not valid json"
    arch.client.models.generate_content.return_value = bad_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"):
        with pytest.raises(ArchitectExtractionError):
            arch.identify_contexts(domain_sentences=["one", "two"])


def test_architect_raises_when_response_is_empty_list():
    arch = _arch()
    empty_response = MagicMock()
    empty_response.candidates = [MagicMock()]
    empty_response.candidates[0].finish_reason = "STOP"
    empty_response.text = '{"contexts": []}'
    arch.client.models.generate_content.return_value = empty_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_parse_json_response", return_value={"contexts": []}):
        with pytest.raises(ArchitectExtractionError):
            arch.identify_contexts(domain_sentences=["one", "two"])
