"""Smoke tests for core.llm scaffold (commit 1) and factories (commit 5)."""

from unittest.mock import patch

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
    get_client,
    get_client_for_model,
)
from core.llm.gemini import GeminiClient
from core.llm.ollama import OllamaClient


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


def test_get_client_gemini_returns_gemini_client():
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake"}):
        with patch("core.llm.gemini.genai.Client"):
            client = get_client("gemini")
            assert isinstance(client, GeminiClient)


def test_get_client_ollama_returns_ollama_client():
    with patch.dict("os.environ", {"OLLAMA_API_KEYS": "k1,k2"}):
        client = get_client("ollama")
        assert isinstance(client, OllamaClient)


def test_get_client_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_client("anthropic")  # type: ignore[arg-type]


def test_get_client_for_model_resolves_via_registry():
    with patch.dict("os.environ", {"OLLAMA_API_KEYS": "k1,k2"}):
        client = get_client_for_model("gpt-oss:120b-cloud")
        assert isinstance(client, OllamaClient)
