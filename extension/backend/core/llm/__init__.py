"""Provider-agnostic LLM client package.

Replaces the legacy single-provider core.llm_client. Exposes a uniform
LLMClient ABC plus typed exceptions so the rest of the codebase calls
LLMs without knowing whether the underlying provider is Gemini or
Ollama (or anything else added later).

Use get_client(provider) to instantiate the right concrete client, or
get_client_for_model(model_id) to let the registry resolve it.
"""

from typing import Literal

from core.llm.base import LLMClient, LLMResponse, TokenUsage
from core.llm.errors import (
    LLMError,
    RateLimitError,
    AuthError,
    SchemaError,
    RetryExhausted,
)
from core.llm.registry import MODELS, ModelSpec, model_spec


Provider = Literal["gemini", "ollama"]


def get_client(provider: Provider) -> LLMClient:
    """Construct a concrete LLMClient for the given provider.

    Lazy imports keep the package importable when only one provider's
    SDK is installed in a given runtime.
    """
    if provider == "gemini":
        from core.llm.gemini import GeminiClient
        return GeminiClient()
    if provider == "ollama":
        from core.llm.ollama import OllamaClient
        return OllamaClient()
    raise ValueError(f"Unknown provider: {provider!r}")


def get_client_for_model(model_id: str) -> LLMClient:
    """Look up the model's provider in the registry and return its client."""
    spec = model_spec(model_id)
    return get_client(spec.provider)


__all__ = [
    "LLMClient",
    "LLMResponse",
    "TokenUsage",
    "LLMError",
    "RateLimitError",
    "AuthError",
    "SchemaError",
    "RetryExhausted",
    "MODELS",
    "ModelSpec",
    "model_spec",
    "get_client",
    "get_client_for_model",
]
