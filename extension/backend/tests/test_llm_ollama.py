"""Smoke tests for core.llm.ollama — OpenAI SDK mocked."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core.llm.ollama import OllamaClient
from core.llm.errors import AuthError


class _ToyEntity(BaseModel):
    name: str
    confidence: float


def _mock_completion(content: str, finish: str = "stop"):
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = content
    m.choices[0].finish_reason = finish
    m.usage.prompt_tokens = 10
    m.usage.completion_tokens = 5
    m.usage.total_tokens = 15
    m.model_dump = MagicMock(return_value={"id": "test"})
    return m


def test_ollama_client_requires_keys():
    with patch.dict("os.environ", {"OLLAMA_API_KEYS": ""}, clear=False):
        with pytest.raises(ValueError):
            OllamaClient(keys=None)


def test_ollama_client_reads_env_var_keys():
    with patch.dict("os.environ", {"OLLAMA_API_KEYS": "k1, k2,k3"}):
        c = OllamaClient()
        assert c._keys == ["k1", "k2", "k3"]


def test_ollama_chat_happy_path_returns_llmresponse():
    client = OllamaClient(keys=["k0"])
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = _mock_completion("hello")
    with patch.object(client, "_make_client", return_value=fake_openai):
        resp = client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-oss:120b-cloud",
        )
    assert resp.content == "hello"
    assert resp.provider == "ollama"
    assert resp.model_id == "gpt-oss:120b-cloud"
    assert resp.usage.total_tokens == 15
    assert resp.json_failed is False


def test_ollama_structured_output_parses_valid_json():
    client = OllamaClient(keys=["k0"])
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = _mock_completion(
        '{"name": "Customer", "confidence": 0.9}'
    )
    with patch.object(client, "_make_client", return_value=fake_openai):
        resp = client.structured_output(
            messages=[{"role": "user", "content": "extract"}],
            schema=_ToyEntity,
            model="gpt-oss:120b-cloud",
        )
    assert resp.json_failed is False
    assert isinstance(resp.parsed, _ToyEntity)
    assert resp.parsed.name == "Customer"


def test_ollama_structured_output_sets_json_failed_on_invalid_json():
    client = OllamaClient(keys=["k0"])
    fake_openai = MagicMock()
    fake_openai.chat.completions.create.return_value = _mock_completion("not valid json {{")
    with patch.object(client, "_make_client", return_value=fake_openai):
        resp = client.structured_output(
            messages=[{"role": "user", "content": "extract"}],
            schema=_ToyEntity,
            model="gpt-oss:120b-cloud",
        )
    assert resp.json_failed is True
    assert resp.json_fail_reason is not None
    assert "invalid_json" in resp.json_fail_reason
    assert resp.parsed is None


def test_ollama_structured_output_sets_json_failed_on_schema_mismatch():
    client = OllamaClient(keys=["k0"])
    fake_openai = MagicMock()
    # Valid JSON but missing required field 'confidence'
    fake_openai.chat.completions.create.return_value = _mock_completion(
        '{"name": "Customer"}'
    )
    with patch.object(client, "_make_client", return_value=fake_openai):
        resp = client.structured_output(
            messages=[{"role": "user", "content": "extract"}],
            schema=_ToyEntity,
            model="gpt-oss:120b-cloud",
        )
    assert resp.json_failed is True
    assert resp.json_fail_reason is not None
    assert "schema_mismatch" in resp.json_fail_reason
    assert resp.parsed is None


def test_ollama_auth_error_propagates_without_retry():
    import openai as _openai_module
    client = OllamaClient(keys=["k0", "k1"])
    fake_openai = MagicMock()

    # openai SDK exceptions need a response object with a real Request
    # attached. Build minimally with MagicMock so we exercise the
    # translation path without depending on private SDK internals.
    fake_response = MagicMock()
    fake_response.status_code = 401
    fake_response.request = MagicMock()
    fake_openai.chat.completions.create.side_effect = _openai_module.AuthenticationError(
        message="bad key",
        response=fake_response,
        body=None,
    )

    with patch.object(client, "_make_client", return_value=fake_openai):
        with pytest.raises(AuthError):
            client.chat(messages=[{"role": "user", "content": "x"}], model="gpt-oss:120b-cloud")
