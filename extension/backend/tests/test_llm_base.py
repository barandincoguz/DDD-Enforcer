"""Smoke tests for core.llm scaffold (commit 1)."""

import pytest
from core.llm import (
    LLMClient,
    LLMResponse,
    TokenUsage,
    LLMError,
    RateLimitError,
    AuthError,
    SchemaError,
    RetryExhausted,
)


def test_token_usage_dataclass_constructs():
    u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert u.cached_tokens == 0


def test_llm_response_dataclass_constructs():
    r = LLMResponse(
        content="hello",
        parsed=None,
        usage=TokenUsage(1, 1, 2),
        model_id="m",
        provider="p",
    )
    assert r.json_failed is False
    assert r.raw_response == {}


def test_llm_client_is_abstract():
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_error_hierarchy():
    for cls in [RateLimitError, AuthError, SchemaError, RetryExhausted]:
        assert issubclass(cls, LLMError)


def test_rate_limit_error_carries_provider():
    e = RateLimitError(provider="ollama")
    assert e.provider == "ollama"
    assert e.status_code == 429


def test_retry_exhausted_carries_attempt_count():
    e = RetryExhausted(attempt_count=3, last_exception=ValueError("boom"))
    assert e.attempt_count == 3
    assert isinstance(e.last_exception, ValueError)
