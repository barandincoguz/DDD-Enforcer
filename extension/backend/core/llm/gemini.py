"""GeminiClient — Google AI Gemini via google-genai SDK.

Reads GEMINI_API_KEY env var; the retry decorator wraps a single-key
pool with exponential backoff (rotation is a no-op since only one key
is supplied). The SDK exposes structured output via response_schema +
response_mime_type="application/json".

Supported D1 models (see core.llm.registry):
- gemini-3.1-pro-preview (G1)
- gemini-3.1-flash-lite (G2 — falls back to gemini-2.5-flash if
  3.1-flash-lite is unavailable at runtime)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Type

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from core.llm.base import LLMClient, LLMResponse, TokenUsage
from core.llm.errors import AuthError, RateLimitError
from core.llm.retry import with_retry_and_rotation


# Models that should silently fall back to a different identifier if
# the requested name is rejected by the provider as unknown/unavailable.
_RUNTIME_FALLBACKS: Dict[str, str] = {
    "gemini-3.1-flash-lite": "gemini-2.5-flash",
}


def _translate_genai_exception(exc: BaseException, provider_label: str) -> BaseException:
    """Map google-genai SDK exceptions to core.llm.errors.

    google-genai raises google.api_core / google.genai exceptions with
    a `code` attribute or message string. We inspect both because the
    SDK's exception hierarchy is in flux across versions.
    """
    msg = str(exc).lower()
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (401,) or "unauthorized" in msg or "invalid api key" in msg:
        return AuthError(provider=provider_label, status_code=401, message=str(exc))
    if status_code in (429,) or "rate limit" in msg or "quota" in msg or "resource_exhausted" in msg:
        return RateLimitError(provider=provider_label, status_code=429, message=str(exc))
    if status_code in (403,) or "permission" in msg:
        return RateLimitError(provider=provider_label, status_code=403, message=str(exc))
    if status_code in (500, 502, 503, 504) or "unavailable" in msg or "internal" in msg:
        return RateLimitError(
            provider=provider_label,
            status_code=status_code if isinstance(status_code, int) else 503,
            message=str(exc),
        )
    return exc


class GeminiClient(LLMClient):
    """LLM client for Google Gemini via google-genai SDK."""

    PROVIDER_LABEL = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        """Construct with explicit api_key or read GEMINI_API_KEY env var.

        Args:
            api_key: Optional explicit key. If None, reads
                     GEMINI_API_KEY env var.
        """
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "GeminiClient requires an API key. Set GEMINI_API_KEY env "
                "var or pass api_key=."
            )
        self._key = key
        self._client = genai.Client(api_key=self._key)

    def _resolve_model(self, model: str) -> str:
        """Return the effective model id, applying runtime fallbacks."""
        return _RUNTIME_FALLBACKS.get(model, model)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        resolved_model = self._resolve_model(model)
        config_kwargs: Dict[str, Any] = {
            "temperature": kwargs.get("temperature", 0.05),
            "seed": kwargs.get("seed", 42),
        }
        # Architect-style callers ask for JSON without strict schema.
        # Pass through response_mime_type so the model emits JSON text
        # rather than prose; callers parse manually.
        rmt = kwargs.get("response_mime_type")
        if rmt:
            config_kwargs["response_mime_type"] = rmt
        max_tokens = kwargs.get("max_output_tokens") or kwargs.get("max_tokens")
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        config = genai_types.GenerateContentConfig(**config_kwargs)
        contents = self._messages_to_contents(messages)

        @with_retry_and_rotation(keys=[self._key])
        def _call(api_key: str) -> Any:  # noqa: ARG001 — api_key supplied for decorator contract
            try:
                return self._client.models.generate_content(
                    model=resolved_model,
                    contents=contents,
                    config=config,
                )
            except BaseException as e:
                translated = _translate_genai_exception(e, self.PROVIDER_LABEL)
                if translated is e:
                    raise
                raise translated from e

        start = time.monotonic()
        response = _call()
        latency_ms = (time.monotonic() - start) * 1000.0

        content = self._safe_response_text(response)
        usage = self._extract_usage(response)
        return LLMResponse(
            content=content,
            parsed=None,
            usage=usage,
            model_id=resolved_model,
            provider=self.PROVIDER_LABEL,
            json_failed=False,
            json_fail_reason=None,
            latency_ms=latency_ms,
            raw_response={},
        )

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        resolved_model = self._resolve_model(model)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=kwargs.get("temperature", 0.05),
            seed=kwargs.get("seed", 42),
        )
        contents = self._messages_to_contents(messages)

        @with_retry_and_rotation(keys=[self._key])
        def _call(api_key: str) -> Any:  # noqa: ARG001
            try:
                return self._client.models.generate_content(
                    model=resolved_model,
                    contents=contents,
                    config=config,
                )
            except BaseException as e:
                translated = _translate_genai_exception(e, self.PROVIDER_LABEL)
                if translated is e:
                    raise
                raise translated from e

        start = time.monotonic()
        response = _call()
        latency_ms = (time.monotonic() - start) * 1000.0

        content = self._safe_response_text(response)
        usage = self._extract_usage(response)
        parsed: Optional[BaseModel] = None
        json_failed = False
        json_fail_reason: Optional[str] = None

        # Prefer the SDK's `.parsed` attribute if it surfaced a Pydantic
        # instance directly; otherwise fall back to manual parse.
        sdk_parsed = getattr(response, "parsed", None)
        if isinstance(sdk_parsed, BaseModel):
            parsed = sdk_parsed
        else:
            try:
                payload = json.loads(content) if content else None
                if payload is None:
                    json_failed = True
                    json_fail_reason = "invalid_json: empty response"
                else:
                    parsed = schema.model_validate(payload)
            except json.JSONDecodeError as e:
                json_failed = True
                json_fail_reason = f"invalid_json: {e}"
            except ValidationError as e:
                json_failed = True
                json_fail_reason = f"schema_mismatch: {e}"

        return LLMResponse(
            content=content,
            parsed=parsed,
            usage=usage,
            model_id=resolved_model,
            provider=self.PROVIDER_LABEL,
            json_failed=json_failed,
            json_fail_reason=json_fail_reason,
            latency_ms=latency_ms,
            raw_response={},
        )

    def count_tokens(self, text: str, model: str) -> int:
        """Return Gemini's token count for the given text under the
        given model. Thin wrapper around google-genai's count_tokens.
        """
        resolved_model = self._resolve_model(model)
        try:
            resp = self._client.models.count_tokens(
                model=resolved_model,
                contents=text,
            )
        except BaseException as e:
            translated = _translate_genai_exception(e, self.PROVIDER_LABEL)
            if translated is e:
                raise
            raise translated from e
        return getattr(resp, "total_tokens", 0) or 0

    @staticmethod
    def _messages_to_contents(messages: List[Dict[str, str]]) -> str:
        """Flatten a chat-style messages list into a single content
        string for google-genai's generate_content API. System messages
        get a SYSTEM: prefix; user/assistant alternations are
        concatenated.
        """
        parts = []
        for m in messages:
            role = m.get("role", "user").upper()
            text = m.get("content", "")
            parts.append(f"{role}: {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _safe_response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
        # Older SDK shapes may put text under candidates[0].content.parts[0].text
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            if content is not None:
                parts = getattr(content, "parts", None) or []
                if parts:
                    return getattr(parts[0], "text", "") or ""
        return ""

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return TokenUsage(0, 0, 0)
        return TokenUsage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            total_tokens=getattr(meta, "total_token_count", 0) or 0,
            cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
        )
