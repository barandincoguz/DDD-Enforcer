"""LLMClient ABC + response/usage dataclasses.

The LLMResponse contract is the key abstraction: every provider client
returns the same shape regardless of underlying SDK. Downstream
consumers (architect, validator) only depend on this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


@dataclass
class TokenUsage:
    """Normalized token usage across providers."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0


@dataclass
class LLMResponse:
    """Provider-agnostic response wrapper.

    `parsed` is None when structured_output was requested but the model
    emitted text that failed Pydantic parsing. In that case json_failed
    is True and json_fail_reason carries the failure category.
    """
    content: str
    parsed: Optional[BaseModel]
    usage: TokenUsage
    model_id: str
    provider: str
    json_failed: bool = False
    json_fail_reason: Optional[str] = None
    latency_ms: float = 0.0
    raw_response: Dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Abstract base for every provider client.

    Implementations must support both plain chat completion and a
    schema-enforced variant. Schema enforcement is best-effort:
    providers that lack native JSON-schema support should set
    json_failed=True instead of crashing when the model output cannot
    be parsed into the requested Pydantic class.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Plain chat completion (no schema enforcement)."""

    @abstractmethod
    def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Schema-enforced output.

        Sets json_failed=True on the returned LLMResponse if the model
        violates the schema or emits unparseable JSON; does not raise
        in that case so the caller can decide whether to retry, count
        the failure, or skip.
        """
