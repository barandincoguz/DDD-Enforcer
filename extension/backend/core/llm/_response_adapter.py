"""Legacy compatibility shim — surfaces an LLMResponse with the same
attribute shape as the google-genai SDK response (.text, .usage_metadata
with prompt_token_count / candidates_token_count / total_token_count /
cached_content_token_count, .candidates[].finish_reason). Drop once
all legacy helpers read LLMResponse fields directly.
"""

from typing import Any


class _GenaiCompatUsage:
    __slots__ = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
    )

    def __init__(self, llm_usage: Any):
        self.prompt_token_count = llm_usage.prompt_tokens
        self.candidates_token_count = llm_usage.completion_tokens
        self.total_token_count = llm_usage.total_tokens
        self.cached_content_token_count = llm_usage.cached_tokens


class _GenaiCompatCandidate:
    __slots__ = ("finish_reason",)

    def __init__(self, finish_reason: str):
        self.finish_reason = finish_reason


class LLMResponseAdapter:
    """Wraps an LLMResponse so legacy genai-shaped helpers keep working.

    Surfaces `.text`, `.usage_metadata` (with prompt_token_count /
    candidates_token_count / total_token_count / cached_content_token_count),
    and `.candidates[0].finish_reason` so the existing helpers
    (_safe_response_text, _check_response_completion,
    token_tracker.track_api_call) keep working unchanged.

    Temporary shim — a future cleanup commit can migrate those helpers
    to read LLMResponse fields directly and drop this adapter.
    """

    def __init__(self, llm_response: Any):
        self._llm = llm_response
        self.text = llm_response.content
        self.usage_metadata = _GenaiCompatUsage(llm_response.usage)
        finish_reason = "OTHER" if llm_response.json_failed else "STOP"
        self.candidates = [_GenaiCompatCandidate(finish_reason)]
