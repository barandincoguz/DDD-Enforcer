"""Retry + key-rotation decorator for LLM client transport failures.

Algorithm (per D1 retry contract):
  1. attempt 0 uses keys[0]
  2. RateLimitError → rotate to next key with NO delay
  3. all keys exhausted with retryable errors → exponential backoff
     (base_delay * 2 ** step) and reuse keys round-robin
  4. AuthError / SchemaError / any other LLMError → propagate immediately
  5. max_retries exhausted → RetryExhausted(last_exception, keys_tried)

Provider clients (GeminiClient, OllamaClient) MUST translate every
transient transport failure (429, 403, 5xx, network timeout, etc.)
into RateLimitError before raising; the decorator only catches that
typed class so programming bugs and SDK contract failures still
surface unwrapped.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, List, Optional, TypeVar, cast

from core.llm.errors import RateLimitError, RetryExhausted


T = TypeVar("T")


# Status codes that providers should map to RateLimitError before the
# decorator sees them. Documented here for cross-reference; the
# decorator itself does not inspect status codes.
RETRYABLE_STATUS = {429, 403, 500, 502, 503, 504}


def with_retry_and_rotation(
    *,
    keys: List[str],
    max_retries: Optional[int] = None,
    base_delay: float = 1.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a function `fn(api_key, ...) -> T` so it retries with
    rotation on RateLimitError.

    Args:
        keys: Pool of API keys to rotate through (at least one entry).
              If only one key is supplied, all retries reuse it with
              exponential backoff.
        max_retries: Total attempt ceiling. Defaults to len(keys) * 2
                     (so each key is tried twice — once immediately,
                     once after backoff) with a floor of 3.
        base_delay: Seconds for the first backoff step. Subsequent
                    backoffs double (1s, 2s, 4s, …).

    Returns:
        A decorator. The decorated function will be called repeatedly
        with the next api_key on each retry until either it returns
        successfully or RetryExhausted is raised.
    """
    if not keys:
        raise ValueError("with_retry_and_rotation requires at least one key")

    effective_max_retries = (
        max_retries if max_retries is not None else max(len(keys) * 2, 3)
    )

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[BaseException] = None
            keys_tried: List[str] = []
            for attempt in range(effective_max_retries):
                key = keys[attempt % len(keys)]
                keys_tried.append(key)
                try:
                    return fn(key, *args, **kwargs)
                except RateLimitError as e:
                    last_exception = e
                    next_attempt = attempt + 1
                    if next_attempt >= effective_max_retries:
                        break
                    if next_attempt < len(keys):
                        # Still have an unused key — rotate immediately.
                        continue
                    # All keys cycled at least once — back off before retrying.
                    backoff_step = attempt - len(keys) + 1
                    time.sleep(base_delay * (2 ** backoff_step))
            raise RetryExhausted(
                attempt_count=effective_max_retries,
                last_exception=last_exception,
                keys_tried=keys_tried,
            )

        return cast(Callable[..., T], wrapper)

    return decorator
