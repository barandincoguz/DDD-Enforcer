"""Provider-agnostic LLM client package.

Replaces the legacy single-provider core.llm_client. Exposes a uniform
LLMClient ABC plus typed exceptions so the rest of the codebase calls
LLMs without knowing whether the underlying provider is Gemini or
Ollama (or anything else added later).
"""

from core.llm.base import LLMClient, LLMResponse, TokenUsage
from core.llm.errors import (
    LLMError,
    RateLimitError,
    AuthError,
    SchemaError,
    RetryExhausted,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "LLMError",
    "RateLimitError",
    "AuthError",
    "SchemaError",
    "RetryExhausted",
]
