"""Smoke tests for core.llm.retry — decorator semantics with mocked sleep."""

import pytest
from unittest.mock import patch

from core.llm.errors import AuthError, RateLimitError, RetryExhausted
from core.llm.retry import with_retry_and_rotation


def test_success_on_first_attempt_does_not_rotate():
    calls = []

    @with_retry_and_rotation(keys=["k0", "k1", "k2"])
    def fn(api_key):
        calls.append(api_key)
        return "ok"

    assert fn() == "ok"
    assert calls == ["k0"]


def test_rotation_through_keys_on_rate_limit():
    calls = []

    @with_retry_and_rotation(keys=["k0", "k1", "k2"], max_retries=3)
    def fn(api_key):
        calls.append(api_key)
        if api_key != "k2":
            raise RateLimitError(provider="ollama")
        return "ok"

    with patch("core.llm.retry.time.sleep") as mock_sleep:
        assert fn() == "ok"
    assert calls == ["k0", "k1", "k2"]
    mock_sleep.assert_not_called()  # within-key-pool rotation = no sleep


def test_backoff_after_all_keys_cycle():
    @with_retry_and_rotation(keys=["k0", "k1"], max_retries=5, base_delay=1.0)
    def fn(api_key):
        raise RateLimitError(provider="ollama")

    with patch("core.llm.retry.time.sleep") as mock_sleep:
        with pytest.raises(RetryExhausted) as excinfo:
            fn()
    # 5 attempts: 2 keys × first pass + 3 backoff attempts (re-cycling keys)
    assert excinfo.value.attempt_count == 5
    # Backoff sleeps fire only after the first key-pool pass
    assert mock_sleep.call_count == 3


def test_auth_error_propagates_without_retry():
    calls = []

    @with_retry_and_rotation(keys=["k0", "k1"])
    def fn(api_key):
        calls.append(api_key)
        raise AuthError(provider="gemini", status_code=401)

    with pytest.raises(AuthError):
        fn()
    assert calls == ["k0"]  # no rotation


def test_retry_exhausted_carries_last_exception_and_keys_tried():
    @with_retry_and_rotation(keys=["k0", "k1"], max_retries=2)
    def fn(api_key):
        raise RateLimitError(provider="ollama", status_code=429)

    with patch("core.llm.retry.time.sleep"):
        with pytest.raises(RetryExhausted) as excinfo:
            fn()
    assert excinfo.value.attempt_count == 2
    assert isinstance(excinfo.value.last_exception, RateLimitError)
    assert excinfo.value.keys_tried == ["k0", "k1"]


def test_empty_keys_list_raises_value_error():
    with pytest.raises(ValueError):
        with_retry_and_rotation(keys=[])


def test_single_key_uses_backoff_each_retry():
    @with_retry_and_rotation(keys=["only-key"], max_retries=3, base_delay=1.0)
    def fn(api_key):
        raise RateLimitError(provider="ollama")

    with patch("core.llm.retry.time.sleep") as mock_sleep:
        with pytest.raises(RetryExhausted):
            fn()
    # 3 attempts: first is the initial call, the next 2 sleep before retry
    assert mock_sleep.call_count == 2
