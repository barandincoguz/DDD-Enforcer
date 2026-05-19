"""Typed exceptions for the LLM client package.

LLMError is the root; each subclass corresponds to a category that
callers may want to branch on (e.g. retry vs raise vs skip). The
retry decorator in core.llm.retry catches RateLimitError + transient
5xx wrappers and rotates keys / backs off; AuthError + SchemaError
propagate immediately (no retry value).
"""

from typing import List, Optional


class LLMError(Exception):
    """Base for every LLM client failure."""


class RateLimitError(LLMError):
    """429 from a provider. Retryable; trigger key rotation or backoff."""

    def __init__(self, provider: str, status_code: int = 429, message: Optional[str] = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message or f"{provider} rate limit (status {status_code})")


class AuthError(LLMError):
    """401/403 — bad API key or insufficient permissions. NOT retryable."""

    def __init__(self, provider: str, status_code: int, message: Optional[str] = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message or f"{provider} auth failure (status {status_code})")


class SchemaError(LLMError):
    """Model emitted output that failed schema validation. NOT retryable by default."""

    def __init__(self, model_id: str, reason: str, raw_output: Optional[str] = None):
        self.model_id = model_id
        self.reason = reason
        self.raw_output = raw_output
        super().__init__(f"{model_id} schema failure: {reason}")


class RetryExhausted(LLMError):
    """All retry attempts and key rotations exhausted."""

    def __init__(
        self,
        attempt_count: int,
        last_exception: Optional[BaseException] = None,
        keys_tried: Optional[List[str]] = None,
    ):
        self.attempt_count = attempt_count
        self.last_exception = last_exception
        self.keys_tried = keys_tried or []
        last_summary = (
            f"; last error: {type(last_exception).__name__}: {last_exception}"
            if last_exception is not None
            else ""
        )
        super().__init__(
            f"Retry exhausted after {attempt_count} attempt(s) "
            f"(keys_tried={len(self.keys_tried)}){last_summary}"
        )
