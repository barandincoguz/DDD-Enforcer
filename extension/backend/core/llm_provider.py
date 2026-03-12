"""
Provider abstraction for structured LLM calls.

The current implementation ships with a Gemini adapter, but the interface is
kept intentionally small so model/provider comparisons can reuse the same
validation and generation services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import os
from typing import Any, Dict, Optional, Type

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

from core.research_metrics import ResearchMetricsStore
from core.token_tracker import TokenTracker

load_dotenv()


@dataclass
class LLMUsage:
    """Normalized usage metadata across providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def billable_prompt_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_tokens)

    @property
    def total_tokens(self) -> int:
        return self.billable_prompt_tokens + self.completion_tokens


@dataclass
class LLMCallResult:
    """Normalized result envelope for a provider call."""

    provider: str
    model: str
    text: str
    usage: LLMUsage
    finish_reason: Optional[str] = None
    parsed: Optional[Any] = None
    parse_success: bool = False
    raw_response: Any = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract provider interface for structured text generation."""

    provider_name = "unknown"

    @abstractmethod
    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        stage: str,
        operation: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        response_mime_type: str = "application/json",
        retry_count: int = 0,
    ) -> LLMCallResult:
        """Generate a structured response and normalize metadata."""

    @abstractmethod
    def count_tokens(self, *, model: str, text: str) -> int:
        """Count tokens using the provider's native API when available."""


class GeminiLLMProvider(LLMProvider):
    """Gemini-backed implementation of the provider interface."""

    provider_name = "gemini"

    def __init__(self, api_key: Optional[str] = None):
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=resolved_api_key)
        self.token_tracker = TokenTracker.get_instance()
        self.research_metrics = ResearchMetricsStore.get_instance()

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        stage: str,
        operation: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        response_mime_type: str = "application/json",
        retry_count: int = 0,
    ) -> LLMCallResult:
        response_config = {
            "response_mime_type": response_mime_type,
            "temperature": temperature,
        }
        if seed is not None:
            response_config["seed"] = seed
        if response_schema is not None:
            response_config["response_schema"] = response_schema

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**response_config),
        )

        usage = self._extract_usage(response)
        parsed, parse_success = self._parse_response_text(
            text=response.text or "",
            response_schema=response_schema,
        )
        finish_reason = None
        if getattr(response, "candidates", None):
            finish_reason = self._normalize_finish_reason(
                getattr(response.candidates[0], "finish_reason", None)
            )

        cost_usd = self.token_tracker.track_usage(
            provider=self.provider_name,
            model=model,
            stage=stage,
            operation=operation,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_tokens=usage.cached_tokens,
            parse_success=parse_success,
            retry_count=retry_count,
        )

        self.research_metrics.record_provider_call(
            {
                "provider": self.provider_name,
                "model": model,
                "stage": stage,
                "operation": operation,
                "prompt_tokens": usage.billable_prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "cached_tokens": usage.cached_tokens,
                "parse_success": parse_success,
                "retry_count": retry_count,
                "finish_reason": finish_reason,
                "cost_usd": round(cost_usd, 8),
            }
        )

        return LLMCallResult(
            provider=self.provider_name,
            model=model,
            text=response.text or "",
            usage=usage,
            finish_reason=finish_reason,
            parsed=parsed,
            parse_success=parse_success,
            raw_response=response,
            retry_count=retry_count,
            metadata={"cost_usd": round(cost_usd, 8)},
        )

    def count_tokens(self, *, model: str, text: str) -> int:
        response = self.client.models.count_tokens(model=model, contents=text)
        return getattr(response, "total_tokens", 0) or 0

    def _normalize_finish_reason(self, finish_reason: Any) -> Optional[str]:
        if finish_reason is None:
            return None
        normalized = getattr(finish_reason, "name", None) or str(finish_reason)
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        return normalized.strip()

    def _extract_usage(self, response: Any) -> LLMUsage:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return LLMUsage()
        return LLMUsage(
            prompt_tokens=getattr(usage, "prompt_token_count", None) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", None) or 0,
            cached_tokens=getattr(usage, "cached_content_token_count", None) or 0,
        )

    def _parse_response_text(
        self,
        *,
        text: str,
        response_schema: Optional[Type[BaseModel]],
    ) -> tuple[Optional[Any], bool]:
        if not text:
            return None, False
        try:
            if response_schema is not None:
                return response_schema.model_validate_json(text), True
            return json.loads(text), True
        except Exception:
            try:
                cleaned = text.replace("```json", "").replace("```", "").strip()
                if response_schema is not None:
                    return response_schema.model_validate_json(cleaned), True
                return json.loads(cleaned), True
            except Exception:
                return None, False


class StaticJSONProvider(LLMProvider):
    """Deterministic provider for tests and offline experiment smoke runs."""

    provider_name = "static-json"

    def __init__(self, responses: Dict[str, Dict[str, Any]], model_name: str = "static-json-model"):
        self.responses = responses
        self.model_name = model_name
        self.token_tracker = TokenTracker.get_instance()
        self.research_metrics = ResearchMetricsStore.get_instance()

    def generate_json(
        self,
        *,
        model: str,
        prompt: str,
        stage: str,
        operation: str,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        response_mime_type: str = "application/json",
        retry_count: int = 0,
    ) -> LLMCallResult:
        payload = self.responses.get(operation) or self.responses.get(stage) or {"is_violation": False, "violations": []}
        text = json.dumps(payload)
        parse_target = payload
        parse_success = True
        if response_schema is not None:
            parsed = response_schema.model_validate(payload)
        else:
            parsed = payload
        prompt_tokens = max(1, round(len(prompt) / 4))
        completion_tokens = max(1, round(len(text) / 4))
        cost_usd = self.token_tracker.track_usage(
            provider=self.provider_name,
            model=model,
            stage=stage,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=0,
            parse_success=parse_success,
            retry_count=retry_count,
        )
        self.research_metrics.record_provider_call(
            {
                "provider": self.provider_name,
                "model": model,
                "stage": stage,
                "operation": operation,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cached_tokens": 0,
                "parse_success": True,
                "retry_count": retry_count,
                "finish_reason": "STOP",
                "cost_usd": round(cost_usd, 8),
            }
        )
        return LLMCallResult(
            provider=self.provider_name,
            model=model,
            text=text,
            usage=LLMUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cached_tokens=0),
            finish_reason="STOP",
            parsed=parsed if parse_target is not None else None,
            parse_success=True,
            retry_count=retry_count,
            metadata={"cost_usd": round(cost_usd, 8)},
        )

    def count_tokens(self, *, model: str, text: str) -> int:
        return max(1, round(len(text) / 4))
