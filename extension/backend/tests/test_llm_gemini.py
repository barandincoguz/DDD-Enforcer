"""Smoke tests for core.llm.gemini — google-genai SDK mocked."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core.llm.gemini import GeminiClient
from core.llm.errors import AuthError


class _ToyEntity(BaseModel):
    name: str
    confidence: float


def _mock_response(text: str = "", parsed: object = None, usage_kwargs=None):
    r = MagicMock()
    r.text = text
    r.parsed = parsed
    r.usage_metadata.prompt_token_count = (usage_kwargs or {}).get("prompt", 10)
    r.usage_metadata.candidates_token_count = (usage_kwargs or {}).get("completion", 5)
    r.usage_metadata.total_token_count = (usage_kwargs or {}).get("total", 15)
    r.usage_metadata.cached_content_token_count = 0
    return r


def test_gemini_client_requires_api_key():
    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
        with pytest.raises(ValueError):
            GeminiClient(api_key=None)


def test_gemini_client_chat_returns_llmresponse():
    with patch("core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _mock_response(
            text="hello"
        )
        client = GeminiClient(api_key="fake-key")
        resp = client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.1-pro-preview",
        )
        assert resp.content == "hello"
        assert resp.provider == "gemini"
        assert resp.model_id == "gemini-3.1-pro-preview"
        assert resp.usage.total_tokens == 15


def test_gemini_client_falls_back_g2_to_2_5_flash():
    """G2 (gemini-3.1-flash-lite) silently resolves to gemini-2.5-flash."""
    captured_model = []

    def _capture(**kwargs):
        captured_model.append(kwargs.get("model"))
        return _mock_response(text="ok")

    with patch("core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = _capture
        client = GeminiClient(api_key="fake-key")
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="gemini-3.1-flash-lite",
        )
    assert captured_model == ["gemini-2.5-flash"]


def test_gemini_structured_output_uses_sdk_parsed_when_available():
    parsed_instance = _ToyEntity(name="Customer", confidence=0.9)
    with patch("core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _mock_response(
            text='{"name": "Customer", "confidence": 0.9}',
            parsed=parsed_instance,
        )
        client = GeminiClient(api_key="fake-key")
        resp = client.structured_output(
            messages=[{"role": "user", "content": "extract"}],
            schema=_ToyEntity,
            model="gemini-3.1-pro-preview",
        )
        assert resp.json_failed is False
        assert isinstance(resp.parsed, _ToyEntity)
        assert resp.parsed.name == "Customer"


def test_gemini_structured_output_falls_back_to_manual_parse():
    """When SDK doesn't surface `parsed`, we json.loads the text and validate."""
    with patch("core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _mock_response(
            text='{"name": "Customer", "confidence": 0.9}',
            parsed=None,
        )
        client = GeminiClient(api_key="fake-key")
        resp = client.structured_output(
            messages=[{"role": "user", "content": "extract"}],
            schema=_ToyEntity,
            model="gemini-3.1-pro-preview",
        )
        assert resp.json_failed is False
        assert isinstance(resp.parsed, _ToyEntity)


def test_gemini_structured_output_sets_json_failed_on_invalid_json():
    with patch("core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _mock_response(
            text="not valid json {{",
            parsed=None,
        )
        client = GeminiClient(api_key="fake-key")
        resp = client.structured_output(
            messages=[{"role": "user", "content": "extract"}],
            schema=_ToyEntity,
            model="gemini-3.1-pro-preview",
        )
        assert resp.json_failed is True
        assert resp.json_fail_reason is not None
        assert "invalid_json" in resp.json_fail_reason
        assert resp.parsed is None


def test_gemini_translates_unauthorized_to_auth_error():
    with patch("core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError(
            "401 Unauthorized: invalid API key"
        )
        client = GeminiClient(api_key="fake-key")
        with pytest.raises(AuthError):
            client.chat(
                messages=[{"role": "user", "content": "x"}],
                model="gemini-3.1-pro-preview",
            )
