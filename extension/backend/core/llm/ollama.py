"""OllamaClient — Ollama Cloud via OpenAI-compatible API at ollama.com/v1.

Ollama Cloud exposes an OpenAI-API-compatible endpoint, so this client
uses the openai Python SDK with a custom base_url. Reads keys from the
OLLAMA_API_KEYS env var (comma-separated). The retry decorator handles
rotation across the key pool on RateLimitError.

Supported D1 models (see core.llm.registry):
- gpt-oss:120b-cloud
- qwen3-coder-next:cloud
- minimax-m2:cloud
- gemma4:31b-cloud
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
import openai
from pydantic import BaseModel, ValidationError

from core.llm.base import LLMClient, LLMResponse, TokenUsage
from core.llm.errors import AuthError, RateLimitError
from core.llm.retry import with_retry_and_rotation


OLLAMA_BASE_URL = "https://ollama.com/v1"


def _record_to_emitter_if_active(response: LLMResponse, operation: str) -> None:
    """WP-CORE-20 (F-9): record this call onto the active StageEmitter if any.

    Same contract as GeminiClient's helper — instrumentation lives at the
    LLMResponse construction boundary (Codex W-1). No-op if no emitter in
    context (CLI / init / schema_probe).
    """
    try:
        from core.observability.emitter import get_current_emitter
        emitter = get_current_emitter()
        if emitter is not None:
            emitter.record_llm_call(response, operation=operation)
    except Exception:
        # Never let observability bring down the LLM client.
        pass


def _translate_openai_exception(exc: BaseException, provider_label: str) -> BaseException:
    """Map openai SDK exceptions to core.llm.errors so the retry
    decorator can act on them. 403 is retryable per spec; 401 is not.
    Other openai exceptions are returned unwrapped so they surface.
    """
    if isinstance(exc, openai.AuthenticationError):
        return AuthError(provider=provider_label, status_code=401, message=str(exc))
    if isinstance(exc, openai.PermissionDeniedError):
        # 403 is in RETRYABLE_STATUS per spec — treat as a transient
        # rate-limit so the retry decorator rotates keys.
        return RateLimitError(provider=provider_label, status_code=403, message=str(exc))
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(provider=provider_label, status_code=429, message=str(exc))
    if isinstance(exc, openai.InternalServerError):
        return RateLimitError(provider=provider_label, status_code=500, message=str(exc))
    # Any other openai.APIStatusError with a retryable status code → RateLimitError.
    if isinstance(exc, openai.APIStatusError):
        status_code = getattr(exc, "status_code", 0)
        if status_code in {429, 403, 500, 502, 503, 504}:
            return RateLimitError(
                provider=provider_label, status_code=status_code, message=str(exc)
            )
    return exc


class OllamaClient(LLMClient):
    """LLM client for Ollama Cloud via OpenAI-compatible API."""

    PROVIDER_LABEL = "ollama"

    def __init__(self, keys: Optional[List[str]] = None):
        """Construct with explicit keys or read OLLAMA_API_KEYS env var.

        Args:
            keys: Optional explicit key list. If None, parses
                  OLLAMA_API_KEYS env var (comma-separated).
        """
        if keys is None:
            raw = os.environ.get("OLLAMA_API_KEYS", "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ValueError(
                "OllamaClient requires at least one API key. Set "
                "OLLAMA_API_KEYS env var (comma-separated) or pass keys=."
            )
        self._keys = keys

    def _make_client(self, key: str) -> OpenAI:
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key=key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        # Architect-style callers can pass response_mime_type="application/json"
        # to coax JSON output without a strict schema. OpenAI-compatible
        # endpoints accept response_format={"type": "json_object"}.
        rmt = kwargs.get("response_mime_type")
        response_format = None
        if rmt == "application/json":
            response_format = {"type": "json_object"}

        @with_retry_and_rotation(keys=self._keys)
        def _call(api_key: str) -> Any:
            client = self._make_client(api_key)
            create_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.05),
                "seed": kwargs.get("seed", 42),
                "max_tokens": kwargs.get("max_tokens"),
            }
            if response_format is not None:
                create_kwargs["response_format"] = response_format
            try:
                return client.chat.completions.create(**create_kwargs)
            except BaseException as e:
                translated = _translate_openai_exception(e, self.PROVIDER_LABEL)
                if translated is e:
                    raise
                raise translated from e

        start = time.monotonic()
        response = _call()
        latency_ms = (time.monotonic() - start) * 1000.0

        content = response.choices[0].message.content or ""
        usage = self._extract_usage(response)
        llm_response = LLMResponse(
            content=content,
            parsed=None,
            usage=usage,
            model_id=model,
            provider=self.PROVIDER_LABEL,
            json_failed=False,
            json_fail_reason=None,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )
        _record_to_emitter_if_active(llm_response, kwargs.get("operation", "chat"))
        return llm_response

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        @with_retry_and_rotation(keys=self._keys)
        def _call(api_key: str) -> Any:
            client = self._make_client(api_key)
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": schema.model_json_schema(),
                            "strict": True,
                        },
                    },
                    temperature=kwargs.get("temperature", 0.05),
                    seed=kwargs.get("seed", 42),
                )
            except BaseException as e:
                translated = _translate_openai_exception(e, self.PROVIDER_LABEL)
                if translated is e:
                    raise
                raise translated from e

        start = time.monotonic()
        response = _call()
        latency_ms = (time.monotonic() - start) * 1000.0

        content = response.choices[0].message.content or ""
        usage = self._extract_usage(response)
        parsed: Optional[BaseModel] = None
        json_failed = False
        json_fail_reason: Optional[str] = None
        try:
            payload = json.loads(content)
            parsed = schema.model_validate(payload)
        except json.JSONDecodeError as e:
            json_failed = True
            json_fail_reason = f"invalid_json: {e}"
        except ValidationError as e:
            json_failed = True
            json_fail_reason = f"schema_mismatch: {e}"

        llm_response = LLMResponse(
            content=content,
            parsed=parsed,
            usage=usage,
            model_id=model,
            provider=self.PROVIDER_LABEL,
            json_failed=json_failed,
            json_fail_reason=json_fail_reason,
            latency_ms=latency_ms,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )
        _record_to_emitter_if_active(
            llm_response, kwargs.get("operation", "structured_output")
        )
        return llm_response

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage:
        u = getattr(response, "usage", None)
        if u is None:
            return TokenUsage(0, 0, 0)
        return TokenUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            total_tokens=getattr(u, "total_tokens", 0) or 0,
        )
