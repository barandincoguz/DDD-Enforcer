"""Phase C7b: Specialist now runs one LLM call per bounded context, not
one omnibus call for all (FM-23 context-blending fix).
"""

from unittest.mock import MagicMock, patch
from core.architect import DomainArchitect


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
    a.scout_max_workers = 1
    return a


def test_extract_per_context_makes_one_llm_call_per_context():
    arch = _arch()
    # Post-WP-01a-commit-6: DomainArchitect calls self.client.chat(...) which
    # returns an LLMResponse, then wraps it in _LLMResponseAdapter so legacy
    # helpers (_safe_response_text, _check_response_completion, token tracker)
    # keep working. We stub a fake LLMResponse-shaped object.
    from core.llm.base import LLMResponse, TokenUsage
    ok_response = LLMResponse(
        content=(
            '{"context": "X", "entities": [{"name": "E", "attributes": [], '
            '"confidence": 0.9, "justification": "t", "evidence_sentence_indices": [0]}], '
            '"value_objects": [], "services": [], "aggregates": [], '
            '"domain_events": [], "business_rules": []}'
        ),
        parsed=None,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        json_failed=False,
    )
    arch.client.chat.return_value = ok_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch.token_tracker, "track_api_call"), \
         patch.object(
             arch, "_parse_json_response",
             return_value={
                 "context": "X",
                 "entities": [{"name": "E", "description": "An entity.", "attributes": [], "confidence": 0.9, "justification": "t", "evidence_sentence_indices": [0]}],
                 "value_objects": [], "services": [], "aggregates": [], "domain_events": [], "business_rules": [],
             },
         ):
        contexts = ["OrderMgmt", "Billing", "Inventory"]
        sentences = ["s0", "s1", "s2"]
        results = arch.extract_per_context_details(contexts, sentences)
    # 3 contexts → 3 calls
    assert arch.client.chat.call_count == 3
    assert len(results) == 3
